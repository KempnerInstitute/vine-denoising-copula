#!/usr/bin/env python3
"""Prepare the UCI density-estimation benchmarks used in the VDC paper.

POWER, GAS, HEPMASS and MINIBOONE are the standard preprocessed splits of Papamakarios et al. (2017),
"Masked Autoregressive Flow for Density Estimation", rebuilt from their Zenodo archive (record 1161203,
data.tar.gz, CC BY 4.0) with the preprocessing of github.com/gpapamak/maf (datasets/*.py). The archive
and every member read from it are checked against recorded SHA-256 hashes, and the output arrays are
compared with those of the paper's runs.

    python scripts/download_uci_datasets.py                       # download, verify, prepare all four
    python scripts/download_uci_datasets.py --archive data.tar.gz # use a local copy of the archive
    export DATA_ROOT=<out>                                        # vdc.data.tabular.maybe_load_uci then
                                                                  # reads <out>/uci/<name>.npz

Each <out>/uci/<name>.npz holds float64 arrays train, validate and test, and <out>/uci/manifest.json
records the hashes. The default <out> is <data root>/uci_maf, where <data root> is $DATA_ROOT, else
$OUTPUT_BASE/datasets, else <repo>/data (vdc.data.paths.data_root).

The Combined Cycle Power Plant data (UCI 294) are a different dataset. Earlier versions of this script
saved them as uci/power.npz; the paper's sampling-quality and imputation experiments use them under the
name CCPP. Prepare them with --ccpp, which writes <data root>/uci/ccpp.npz with the earlier processing
(first sheet, float32, 80/20 split with seed 42).

GAS and HEPMASS need pandas (GAS is distributed as a pandas pickle) and CCPP needs openpyxl:
pip install "vine-denoising-copula[data]".
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

MAF_URLS = (
    "https://zenodo.org/api/records/1161203/files/data.tar.gz/content",
    "https://zenodo.org/records/1161203/files/data.tar.gz?download=1",
)
MAF_ARCHIVE_SHA256 = "e56bef4cf33c9bcd2d45e84b78c73374e85e506b76468c4cd026a4c6e7d353d3"

# Archive members read by the preprocessing: name -> (sha256, size in bytes).
MAF_MEMBERS = {
    "data/power/data.npy": ("4aec1dfe82ccdf98b67d2d0742f2a702252765a313d0669d95653aaf4ad9af82", 131_154_000),
    "data/gas/ethylene_CO.pickle": ("25daab120cdca3bb32e2e66349fdc37ba27f1d2b136c43abc2077375a80a7cd8", 168_331_262),
    "data/hepmass/1000_train.csv": ("83d57fa9cf518cd66ea43051d5919f531faeac340815730424c24f4e03ddd6f7", 500_058_506),
    "data/hepmass/1000_test.csv": ("6cbfbcac39d1b6b4d987b2d2efabf2e97ca57adb63d06221ae657e229fbee673", 250_026_686),
    "data/miniboone/data.npy": ("22f76243ec6148cfc7178801312d06290922219d182da76d3b60b300e71fb0d9", 12_551_952),
}
DATASET_MEMBERS = {
    "power": ("data/power/data.npy",),
    "gas": ("data/gas/ethylene_CO.pickle",),
    "hepmass": ("data/hepmass/1000_train.csv", "data/hepmass/1000_test.csv"),
    "miniboone": ("data/miniboone/data.npy",),
}
MAF_DATASETS = tuple(DATASET_MEMBERS)

# Published split sizes (train, validate, test, d).
EXPECTED_SPLITS = {
    "power": (1_659_917, 184_435, 204_928, 6),
    "gas": (852_174, 94_685, 105_206, 8),
    "hepmass": (315_123, 35_013, 174_987, 21),
    "miniboone": (29_556, 3_284, 3_648, 43),
}
# SHA-256 of the C-ordered bytes of every array in the NPZ files of the paper's UCI runs, and of the files.
EXPECTED_ARRAYS = {
    "power": {"train": "a3c6eee0ffee8de3059925f1778afc1e09a384c14503b1e5e1dc5f186c46dcad",
              "validate": "39ada11da37acf8e93782ae9d55ea15605bf41102354469504d20f97a6fee9d1",
              "test": "68dab933584464a90156b733f20d907e83462fb06c100a1c2ee8b70068ec7d69"},
    "gas": {"train": "70136e10e909ed59bfdead8e8251ecf9925ec5215abd1e8852c1f2ebd3ddab05",
            "validate": "526c98c716f205df62f69f05c5581afcde602d2ace6214216e4e9b08f4cdb646",
            "test": "c52361fc20ff1dc236cae616dc689f8f1789843bd461d4d89e22d5212459f211"},
    "hepmass": {"train": "7eb15894622c90eb1e1b4433521feb3d81ffae834e7e07030b6918f0601a9a96",
                "validate": "15261b83fef5a18e05595af0f7d1fa4d7e5968a7c757660cb0afc68563afe1a5",
                "test": "613a470cdc390f3ca3bd4aa22c3d1b9eb3cf755e2fd6a8010b6370491134d423"},
    "miniboone": {"train": "aafbb8e1cb55685729ce36329d159f6b8f5bf869c4b61a1651cae41a0fba0f19",
                  "validate": "25a54623319dff5facb62eca5548371962c669acfe03eb3cdb7e255a2eb019ad",
                  "test": "ff61c966cd4f99049eecf6e2ecb880ba52afe3605f6b925f406efcf418ef30b7"},
}
EXPECTED_NPZ_SHA256 = {
    "power": "015d1af935e7408f97ac7390186e34affbcd1aa74215bc4842412321bc524ab7",
    "gas": "4eecc1be83a76dcf523f42bcd6eb46fe54c42cacba85ee7eab6a54e113b2d9da",
    "hepmass": "e5192380081d49c4829783f707867feee769a925efaefbf3ee4325d9dc954d16",
    "miniboone": "6187ca09ef5f7d14549c3e157a71416140ac9ee94529dc6f080368a56e25a903",
}

CCPP_URLS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
    "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip",
)
CCPP_XLSX_NAME = "Folds5x2_pp.xlsx"
CCPP_XLSX_SHA256 = "ccd490981db2a2f079963b3d9f0aea30d9d338900a0285428dfc6385396f4651"
# Arrays (and file) of the CCPP power.npz that earlier versions wrote and the paper's CCPP experiments read.
CCPP_EXPECTED_ARRAYS = {"train": "1056c0c79b59b647d21f81efa6d339525ff5b44b2d89fc27dbc4a150cf7cab7e",
                        "test": "87ac9b96d2479e405c9622a9c85177de8ee4a0f913496da605d6954d100027e7"}
CCPP_EXPECTED_NPZ_SHA256 = "892c8cb8f29c5997bc5fad8c38a531d8a247fdd88c4307f48780f7426e6469d1"


class HashMismatch(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------
# Hashing, downloading and safe extraction
# ----------------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def verify_file(path: Path, expected: str, what: str) -> str:
    got = sha256_file(path)
    if got != expected:
        raise HashMismatch(f"{what}: sha256 {got} does not match the recorded {expected} ({path})")
    return got


def download(urls, dest: Path, expected_sha256: str | None = None) -> Path:
    """Download the first reachable URL to dest (through dest.part), checking the hash when one is given."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    errors = []
    for url in urls:
        print(f"  downloading {url}", flush=True)
        h = hashlib.sha256()
        try:
            with urllib.request.urlopen(url, timeout=60) as r, part.open("wb") as f:
                for block in iter(lambda: r.read(1 << 22), b""):
                    f.write(block)
                    h.update(block)
        except OSError as exc:
            errors.append(f"{url}: {exc}")
            part.unlink(missing_ok=True)
            continue
        if expected_sha256 is not None and h.hexdigest() != expected_sha256:
            part.unlink(missing_ok=True)
            raise HashMismatch(f"{url}: sha256 {h.hexdigest()} does not match the recorded {expected_sha256}")
        os.replace(part, dest)
        return dest
    raise RuntimeError("download failed:\n  " + "\n  ".join(errors))


def safe_member_path(name: str, dest: Path) -> Path:
    """Destination of an archive member inside dest; rejects absolute and parent-relative names."""
    p = PurePosixPath(name)
    if not name or p.is_absolute() or name.startswith("\\") or "\\" in name or ".." in p.parts:
        raise ValueError(f"unsafe archive member name: {name!r}")
    root = Path(dest).resolve()
    out = (root / Path(*p.parts)).resolve()
    if out != root and root not in out.parents:
        raise ValueError(f"archive member escapes the extraction directory: {name!r}")
    return out


def extract_maf_members(archive: Path, dest: Path, members) -> dict:
    """Stream the archive once and extract the requested members, checking names, types and hashes.

    Every member must have a safe relative name and be a regular file or a directory (no links or
    devices). Only the requested members are written; macOS resource-fork entries (._*) are ignored.
    """
    wanted = {m: MAF_MEMBERS[m] for m in members}
    found = {}
    with tarfile.open(archive, "r|gz") as tar:
        for m in tar:
            out = safe_member_path(m.name, dest)
            if m.issym() or m.islnk():
                raise ValueError(f"link in archive: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise ValueError(f"unexpected member type in archive: {m.name!r}")
            if m.name not in wanted:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            src = tar.extractfile(m)
            with out.open("wb") as f:
                for block in iter(lambda: src.read(1 << 23), b""):
                    f.write(block)
                    h.update(block)
            sha, size = wanted[m.name]
            if h.hexdigest() != sha or out.stat().st_size != size:
                out.unlink()
                raise HashMismatch(f"archive member {m.name}: sha256 {h.hexdigest()} does not match the recorded {sha}")
            found[m.name] = out
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise RuntimeError(f"archive is missing members: {missing}")
    return found


# ----------------------------------------------------------------------------------------------
# MAF preprocessing (numerics identical to github.com/gpapamak/maf datasets/*.py)
# ----------------------------------------------------------------------------------------------

def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError('GAS and HEPMASS need pandas: pip install "vine-denoising-copula[data]"') from exc
    return pd


def _load_legacy_pandas_pickle(path: Path):
    """Read the pandas-0.x pickle shipped in the MAF archive with current pandas.

    Unpickling runs code from the file, so callers pass only archive members whose hash was verified.
    """
    pd = _pandas()
    remap = {"pandas.core.index": "pandas.core.indexes.base", "pandas.indexes.base": "pandas.core.indexes.base",
             "pandas.indexes.range": "pandas.core.indexes.range"}

    class LegacyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name in ("Int64Index", "Float64Index", "UInt64Index"):
                return pd.Index
            return super().find_class(remap.get(module, module), name)

    with Path(path).open("rb") as f:
        return LegacyUnpickler(f, encoding="latin1").load()


def prepare_power(root: Path):
    rng = np.random.RandomState(42)
    data = np.load(Path(root) / "power" / "data.npy")
    rng.shuffle(data)
    n = data.shape[0]
    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)
    voltage_noise = 0.01 * rng.rand(n, 1)
    gap_noise = 0.001 * rng.rand(n, 1)
    sm_noise = rng.rand(n, 3)
    time_noise = np.zeros((n, 1))
    data = data + np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    n_test = int(0.1 * data.shape[0])
    test, data = data[-n_test:], data[:-n_test]
    n_val = int(0.1 * data.shape[0])
    val, train = data[-n_val:], data[:-n_val]
    both = np.vstack((train, val))
    mu, s = both.mean(axis=0), both.std(axis=0)
    return (train - mu) / s, (val - mu) / s, (test - mu) / s


def prepare_gas(root: Path):
    data = _load_legacy_pandas_pickle(Path(root) / "gas" / "ethylene_CO.pickle")
    for col in ("Meth", "Eth", "Time"):
        data = data.drop(col, axis=1)

    def corr_numbers(frame):
        return (frame.corr() > 0.98).to_numpy().sum(axis=1)

    counts = corr_numbers(data)
    while np.any(counts > 1):
        data = data.drop(data.columns[np.where(counts > 1)[0][0]], axis=1)
        counts = corr_numbers(data)
    data = ((data - data.mean()) / data.std()).to_numpy()
    n_test = int(0.1 * data.shape[0])
    test, train = data[-n_test:], data[:-n_test]
    n_val = int(0.1 * train.shape[0])
    val, train = train[-n_val:], train[:-n_val]
    return train, val, test


def prepare_hepmass(root: Path):
    pd = _pandas()
    tr = pd.read_csv(Path(root) / "hepmass" / "1000_train.csv", index_col=False)
    te = pd.read_csv(Path(root) / "hepmass" / "1000_test.csv", index_col=False)
    tr = tr[tr[tr.columns[0]] == 1].drop(tr.columns[0], axis=1)
    te = te[te[te.columns[0]] == 1].drop(te.columns[0], axis=1)
    te = te.drop(te.columns[-1], axis=1)  # as in the reference code ("the data set is messed up")
    mu, s = tr.mean(), tr.std()
    tr, te = ((tr - mu) / s).to_numpy(), ((te - mu) / s).to_numpy()
    remove = []
    for i, feature in enumerate(tr.T):
        counts = Counter(feature)
        if np.array([v for _, v in sorted(counts.items())])[0] > 5:
            remove.append(i)
    keep = np.array([i for i in range(tr.shape[1]) if i not in remove])
    tr, te = tr[:, keep], te[:, keep]
    n_val = int(tr.shape[0] * 0.1)
    return tr[:-n_val], tr[-n_val:], te


def prepare_miniboone(root: Path):
    data = np.load(Path(root) / "miniboone" / "data.npy")
    n_test = int(0.1 * data.shape[0])
    test, data = data[-n_test:], data[:-n_test]
    n_val = int(0.1 * data.shape[0])
    val, train = data[-n_val:], data[:-n_val]
    both = np.vstack((train, val))
    mu, s = both.mean(axis=0), both.std(axis=0)
    return (train - mu) / s, (val - mu) / s, (test - mu) / s


PREPARE = {"power": prepare_power, "gas": prepare_gas, "hepmass": prepare_hepmass, "miniboone": prepare_miniboone}


# ----------------------------------------------------------------------------------------------
# Writing and checking outputs
# ----------------------------------------------------------------------------------------------

def npz_array_hashes(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: sha256_array(z[k]) for k in z.files}


def guard_existing(path: Path, expected_arrays: dict, force: bool) -> bool:
    """True if path already holds the expected arrays; refuses to replace a different file unless forced."""
    if not path.exists():
        return False
    try:
        current = npz_array_hashes(path)
    except Exception:  # unreadable: treat as a foreign file
        current = None
    if current == expected_arrays:
        return True
    if not force:
        raise FileExistsError(
            f"{path} exists and does not hold the expected arrays (it may be the Combined Cycle Power Plant "
            "file that earlier versions of this script saved as uci/power.npz). Choose another --out, or pass "
            "--force to replace it.")
    return False


def write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def check_arrays(name: str, got: dict, expected: dict, allow_mismatch: bool) -> bool:
    for k in expected:
        print(f"    {name}.{k}: " + ("matches the paper's array" if got.get(k) == expected[k] else
                                     "DIFFERS from the paper's array"))
    ok = got == expected
    if not ok and not allow_mismatch:
        raise HashMismatch(f"{name}: prepared arrays differ from the paper's (see above); rerun with "
                           "--allow-hash-mismatch to keep them anyway")
    return ok


def prepare_maf(datasets, out: Path, archive: Path | None, cache_dir: Path, *, keep_extracted: bool = False,
                force: bool = False, allow_mismatch: bool = False) -> dict:
    """Write out/uci/<name>.npz for the requested datasets and return the manifest."""
    uci_dir = Path(out) / "uci"
    todo = [d for d in datasets if not guard_existing(uci_dir / f"{d}.npz", EXPECTED_ARRAYS[d], force)]
    for d in sorted(set(datasets) - set(todo)):
        print(f"  {d}: {uci_dir / (d + '.npz')} already holds the paper's arrays; skipping")
    manifest_path = uci_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not todo:
        return manifest
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if archive is None:
        archive = cache_dir / "data.tar.gz"
        if not (archive.exists() and sha256_file(archive) == MAF_ARCHIVE_SHA256):
            download(MAF_URLS, archive, MAF_ARCHIVE_SHA256)
    print(f"  verifying {archive}", flush=True)
    try:
        verify_file(archive, MAF_ARCHIVE_SHA256, "MAF archive")
    except HashMismatch:
        if not allow_mismatch:
            raise
        print("  WARNING: the archive hash differs; members are still checked one by one")
    members = [m for d in todo for m in DATASET_MEMBERS[d]]
    extract_root = cache_dir / "extracted" if keep_extracted else Path(tempfile.mkdtemp(prefix="extract_", dir=cache_dir))
    try:
        print(f"  extracting {len(members)} members to {extract_root}", flush=True)
        extract_maf_members(archive, extract_root, members)
        manifest.update({
            "source": "Zenodo record 1161203, data.tar.gz (Papamakarios et al. 2017, CC BY 4.0)",
            "urls": list(MAF_URLS), "archive_sha256": sha256_file(archive),
            "preprocessing": "github.com/gpapamak/maf datasets/{power,gas,hepmass,miniboone}.py, as ported in "
                             "scripts/download_uci_datasets.py",
            "numpy": np.__version__,
        })
        manifest.setdefault("datasets", {})
        for d in todo:
            print(f"  preparing {d}", flush=True)
            train, val, test = (np.asarray(a, dtype=np.float64) for a in PREPARE[d](extract_root / "data"))
            split = (train.shape[0], val.shape[0], test.shape[0], train.shape[1])
            if split != EXPECTED_SPLITS[d]:
                raise RuntimeError(f"{d}: split sizes {split} differ from the published {EXPECTED_SPLITS[d]}")
            arrays = {"train": train, "validate": val, "test": test}
            hashes = {k: sha256_array(v) for k, v in arrays.items()}
            ok = check_arrays(d, hashes, EXPECTED_ARRAYS[d], allow_mismatch)
            path = uci_dir / f"{d}.npz"
            write_npz(path, **arrays)
            file_sha = sha256_file(path)
            manifest["datasets"][d] = {"rows_train_validate_test": list(split[:3]), "d": split[3],
                                       "npz_sha256": file_sha, "array_sha256": hashes,
                                       "matches_paper_arrays": ok,
                                       "matches_paper_file": file_sha == EXPECTED_NPZ_SHA256[d]}
            if d in ("gas", "hepmass"):
                manifest["pandas"] = _pandas().__version__
            print(f"  wrote {path} ({split[0]}/{split[1]}/{split[2]} rows, d={split[3]})", flush=True)
    finally:
        if not keep_extracted:
            shutil.rmtree(extract_root, ignore_errors=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ----------------------------------------------------------------------------------------------
# Combined Cycle Power Plant (UCI 294)
# ----------------------------------------------------------------------------------------------

def read_ccpp_xlsx(data: bytes) -> np.ndarray:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError('CCPP needs openpyxl: pip install "vine-denoising-copula[data]"') from exc
    ws = load_workbook(io.BytesIO(data)).active
    rows = [[float(x) for x in row] for row in ws.iter_rows(min_row=2, values_only=True) if row[0] is not None]
    return np.array(rows, dtype=np.float32)


def split_ccpp(data: np.ndarray):
    """The split of the earlier downloader: legacy global seed 42, 80% train."""
    idx = np.random.RandomState(42).permutation(len(data))
    cut = int(0.8 * len(data))
    return data[idx[:cut]], data[idx[cut:]]


def prepare_ccpp(out: Path, archive: Path | None, cache_dir: Path, *, force: bool = False,
                 allow_mismatch: bool = False) -> Path:
    """Write out/uci/ccpp.npz (float32 train/test) from the UCI CCPP zip."""
    path = Path(out) / "uci" / "ccpp.npz"
    if guard_existing(path, CCPP_EXPECTED_ARRAYS, force):
        print(f"  ccpp: {path} already holds the paper's arrays; skipping")
        return path
    if archive is None:
        archive = Path(cache_dir) / "CCPP.zip"
        if not archive.exists():
            download(CCPP_URLS, archive)
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if PurePosixPath(n).name == CCPP_XLSX_NAME]
        if len(names) != 1:
            raise RuntimeError(f"{archive}: expected one {CCPP_XLSX_NAME}, found {names}")
        xlsx = z.read(names[0])
    got = hashlib.sha256(xlsx).hexdigest()
    if got != CCPP_XLSX_SHA256:
        msg = f"{names[0]}: sha256 {got} does not match the recorded {CCPP_XLSX_SHA256}"
        if not allow_mismatch:
            raise HashMismatch(msg)
        print("  WARNING: " + msg)
    train, test = split_ccpp(read_ccpp_xlsx(xlsx))
    check_arrays("ccpp", {"train": sha256_array(train), "test": sha256_array(test)}, CCPP_EXPECTED_ARRAYS,
                 allow_mismatch)
    write_npz(path, train=train, test=test)
    same_file = sha256_file(path) == CCPP_EXPECTED_NPZ_SHA256
    print(f"  wrote {path} (train {train.shape}, test {test.shape}; file "
          + ("identical to" if same_file else "differs from") + " the earlier uci/power.npz)")
    return path


# ----------------------------------------------------------------------------------------------

def default_data_root() -> Path:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from vdc.data.paths import data_root

    return data_root()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=MAF_DATASETS, default=list(MAF_DATASETS),
                    help="MAF benchmarks to prepare (default: all four)")
    ap.add_argument("--out", type=Path, default=None,
                    help="data root for the MAF splits; files go to OUT/uci/<name>.npz (default: <data root>/uci_maf)")
    ap.add_argument("--archive", type=Path, default=None, help="local copy of the Zenodo data.tar.gz")
    ap.add_argument("--cache-dir", type=Path, default=None, help="downloads and extraction (default: OUT/raw)")
    ap.add_argument("--keep-extracted", action="store_true", help="keep the extracted archive members")
    ap.add_argument("--ccpp", action="store_true",
                    help="prepare only the Combined Cycle Power Plant data (UCI 294), used by the paper's "
                         "sampling-quality and imputation experiments, as <data root>/uci/ccpp.npz")
    ap.add_argument("--ccpp-archive", type=Path, default=None, help="local copy of the UCI CCPP zip")
    ap.add_argument("--ccpp-out", type=Path, default=None, help="data root for ccpp.npz (default: <data root>)")
    ap.add_argument("--force", action="store_true", help="replace existing output files that differ")
    ap.add_argument("--allow-hash-mismatch", action="store_true",
                    help="keep outputs whose arrays differ from the paper's (archive members are always checked)")
    a = ap.parse_args(argv)

    if a.ccpp:
        out = (a.ccpp_out or default_data_root()).expanduser()
        cache = (a.cache_dir or out / "raw" / "ccpp").expanduser()
        print(f"CCPP -> {out / 'uci' / 'ccpp.npz'}")
        prepare_ccpp(out, a.ccpp_archive, cache, force=a.force, allow_mismatch=a.allow_hash_mismatch)
        return 0

    root = default_data_root()
    out = (a.out or root / "uci_maf").expanduser()
    if out.resolve() == root.resolve():
        ap.error("--out must not be the default data root, whose uci/power.npz may be the CCPP file used "
                 "elsewhere; choose a dedicated directory such as <data root>/uci_maf")
    cache = (a.cache_dir or out / "raw").expanduser()
    print(f"MAF splits -> {out / 'uci'} (set DATA_ROOT={out} to load them with vdc.data.tabular.maybe_load_uci)")
    prepare_maf(a.datasets, out, a.archive, cache, keep_extracted=a.keep_extracted, force=a.force,
                allow_mismatch=a.allow_hash_mismatch)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, HashMismatch) as exc:
        raise SystemExit(f"error: {exc}")
