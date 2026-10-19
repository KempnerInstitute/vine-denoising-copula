"""Tests for scripts/download_uci_datasets.py on tiny synthetic inputs (no network, no real data)."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_uci_datasets.py"


@pytest.fixture(scope="module")
def dl():
    spec = importlib.util.spec_from_file_location("download_uci_datasets", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, a)
    return buf.getvalue()


def _tar_gz(path: Path, members: dict) -> Path:
    """members: name -> bytes (regular file) or a configured TarInfo (links, odd names)."""
    with tarfile.open(path, "w:gz") as t:
        for name, payload in members.items():
            if isinstance(payload, tarfile.TarInfo):
                t.addfile(payload)
            else:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                t.addfile(info, io.BytesIO(payload))
    return path


def _power_raw(n=100):
    return np.random.RandomState(0).rand(n, 8)


def _miniboone_raw(n=60):
    return np.random.RandomState(1).randn(n, 43)


def test_power_is_deterministic_float64_and_standardized(dl, tmp_path):
    (tmp_path / "power").mkdir()
    np.save(tmp_path / "power" / "data.npy", _power_raw())
    a, b = dl.prepare_power(tmp_path), dl.prepare_power(tmp_path)
    assert [x.shape for x in a] == [(81, 6), (9, 6), (10, 6)]
    for x, y in zip(a, b):
        assert x.dtype == np.float64 and x.tobytes() == y.tobytes()
    both = np.vstack(a[:2])
    np.testing.assert_allclose(both.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(both.std(axis=0), 1.0, atol=1e-12)


def test_miniboone_split_sizes(dl, tmp_path):
    (tmp_path / "miniboone").mkdir()
    np.save(tmp_path / "miniboone" / "data.npy", _miniboone_raw())
    train, val, test = dl.prepare_miniboone(tmp_path)
    assert (train.shape, val.shape, test.shape) == ((49, 43), (5, 43), (6, 43))


def test_gas_drops_correlated_columns_and_standardizes(dl, tmp_path):
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(2)
    s1 = rng.randn(100)
    frame = pd.DataFrame({"Time": np.arange(100.0), "Meth": rng.rand(100), "Eth": rng.rand(100),
                          "s1": s1, "s2": 2 * s1 + 1e-3 * rng.randn(100), "s3": rng.randn(100),
                          "s4": rng.randn(100)})
    (tmp_path / "gas").mkdir()
    frame.to_pickle(tmp_path / "gas" / "ethylene_CO.pickle")
    train, val, test = dl.prepare_gas(tmp_path)
    assert (train.shape, val.shape, test.shape) == ((81, 3), (9, 3), (10, 3))
    data = np.vstack((train, val, test))
    np.testing.assert_allclose(data.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(data.std(axis=0, ddof=1), 1.0, atol=1e-12)
    expected = frame[["s2", "s3", "s4"]]
    np.testing.assert_allclose(data, ((expected - expected.mean()) / expected.std()).to_numpy(), rtol=0, atol=0)


def test_hepmass_keeps_signal_rows_and_drops_discrete_feature(dl, tmp_path):
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(3)

    def frame(n):
        label = np.tile([1, 0], n // 2)
        mass = np.where(np.arange(n) % 4 < 2, 500.0, 1000.0)  # discrete, smallest value repeated
        return pd.DataFrame({"# label": label, "f0": rng.randn(n), "f1": rng.randn(n), "f2": rng.randn(n),
                             "mass": mass})

    (tmp_path / "hepmass").mkdir()
    frame(120).to_csv(tmp_path / "hepmass" / "1000_train.csv", index=False)
    frame(40).to_csv(tmp_path / "hepmass" / "1000_test.csv", index=False)
    train, val, test = dl.prepare_hepmass(tmp_path)
    assert (train.shape, val.shape, test.shape) == ((54, 3), (6, 3), (20, 3))
    np.testing.assert_allclose(np.vstack((train, val)).mean(axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize("name", ["/etc/passwd", "../evil", "data/../../evil", "data/power/../../x",
                                  "data\\power\\x", ""])
def test_safe_member_path_rejects_unsafe_names(dl, tmp_path, name):
    with pytest.raises(ValueError):
        dl.safe_member_path(name, tmp_path)


def test_safe_member_path_accepts_relative_names(dl, tmp_path):
    assert dl.safe_member_path("data/power/data.npy", tmp_path) == (tmp_path / "data/power/data.npy").resolve()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_extraction_rejects_links(dl, tmp_path, kind):
    info = tarfile.TarInfo("data/power/data.npy")
    info.type, info.linkname = kind, "/etc/passwd"
    archive = _tar_gz(tmp_path / "a.tar.gz", {"link": info})
    with pytest.raises(ValueError):
        dl.extract_maf_members(archive, tmp_path / "x", ["data/power/data.npy"])
    assert not (tmp_path / "x" / "data" / "power" / "data.npy").exists()


def test_extraction_rejects_absolute_and_parent_members(dl, tmp_path):
    for i, name in enumerate(["/abs/file", "data/../../escape"]):
        archive = _tar_gz(tmp_path / f"b{i}.tar.gz", {name: b"x"})
        with pytest.raises(ValueError):
            dl.extract_maf_members(archive, tmp_path / "x", ["data/power/data.npy"])


def test_extraction_rejects_member_with_wrong_hash(dl, tmp_path):
    archive = _tar_gz(tmp_path / "c.tar.gz", {"data/miniboone/data.npy": _npy_bytes(_miniboone_raw())})
    with pytest.raises(dl.HashMismatch):
        dl.extract_maf_members(archive, tmp_path / "x", ["data/miniboone/data.npy"])
    assert not (tmp_path / "x" / "data" / "miniboone" / "data.npy").exists()


def test_verify_file_reports_mismatch(dl, tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert dl.verify_file(p, _sha(b"abc"), "f") == _sha(b"abc")
    with pytest.raises(dl.HashMismatch):
        dl.verify_file(p, "0" * 64, "f")


def test_refuses_to_replace_a_foreign_power_npz(dl, tmp_path):
    (tmp_path / "uci").mkdir()
    np.savez(tmp_path / "uci" / "power.npz", train=np.zeros((3, 5), np.float32), test=np.zeros((2, 5), np.float32))
    with pytest.raises(FileExistsError):
        dl.prepare_maf(["power"], tmp_path, None, tmp_path / "raw")
    assert not (tmp_path / "raw").exists()  # nothing was downloaded


def test_write_npz_is_byte_deterministic(dl, tmp_path):
    arrays = dict(train=np.arange(6.0).reshape(3, 2), validate=np.ones((1, 2)), test=np.zeros((2, 2)))
    dl.write_npz(tmp_path / "a.npz", **arrays)
    dl.write_npz(tmp_path / "b.npz", **arrays)
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes()


def test_prepare_maf_end_to_end_on_a_synthetic_archive(dl, tmp_path, monkeypatch):
    power, mini = _npy_bytes(_power_raw()), _npy_bytes(_miniboone_raw())
    junk = tarfile.TarInfo("data/power/._data.npy")
    junk.size = 0
    archive = _tar_gz(tmp_path / "data.tar.gz", {"data/power/data.npy": power, "data/miniboone/data.npy": mini,
                                                  "data/cifar10/test_batch": b"unused", "junk": junk})
    monkeypatch.setattr(dl, "MAF_ARCHIVE_SHA256", dl.sha256_file(archive))
    monkeypatch.setitem(dl.MAF_MEMBERS, "data/power/data.npy", (_sha(power), len(power)))
    monkeypatch.setitem(dl.MAF_MEMBERS, "data/miniboone/data.npy", (_sha(mini), len(mini)))
    monkeypatch.setitem(dl.EXPECTED_SPLITS, "power", (81, 9, 10, 6))
    monkeypatch.setitem(dl.EXPECTED_SPLITS, "miniboone", (49, 5, 6, 43))
    ref = tmp_path / "ref"
    (ref / "power").mkdir(parents=True)
    (ref / "miniboone").mkdir()
    np.save(ref / "power" / "data.npy", _power_raw())
    np.save(ref / "miniboone" / "data.npy", _miniboone_raw())
    for name in ("power", "miniboone"):
        arrays = dict(zip(("train", "validate", "test"), getattr(dl, f"prepare_{name}")(ref)))
        monkeypatch.setitem(dl.EXPECTED_ARRAYS, name, {k: dl.sha256_array(v) for k, v in arrays.items()})

    out = tmp_path / "out"
    manifest = dl.prepare_maf(["power", "miniboone"], out, archive, tmp_path / "cache")
    for name in ("power", "miniboone"):
        assert manifest["datasets"][name]["matches_paper_arrays"]
        with np.load(out / "uci" / f"{name}.npz") as z:
            assert z.files == ["train", "validate", "test"] and z["train"].dtype == np.float64
    assert json.loads((out / "uci" / "manifest.json").read_text())["archive_sha256"] == dl.MAF_ARCHIVE_SHA256
    assert not any((tmp_path / "cache").iterdir())  # temporary extraction removed
    # A second run finds the prepared files and does not touch the archive.
    archive.unlink()
    dl.prepare_maf(["power", "miniboone"], out, archive, tmp_path / "cache")


def test_prepare_maf_rejects_wrong_output_arrays(dl, tmp_path, monkeypatch):
    mini = _npy_bytes(_miniboone_raw())
    archive = _tar_gz(tmp_path / "data.tar.gz", {"data/miniboone/data.npy": mini})
    monkeypatch.setattr(dl, "MAF_ARCHIVE_SHA256", dl.sha256_file(archive))
    monkeypatch.setitem(dl.MAF_MEMBERS, "data/miniboone/data.npy", (_sha(mini), len(mini)))
    monkeypatch.setitem(dl.EXPECTED_SPLITS, "miniboone", (49, 5, 6, 43))
    with pytest.raises(dl.HashMismatch):  # recorded hashes are those of the real data
        dl.prepare_maf(["miniboone"], tmp_path / "out", archive, tmp_path / "cache")
    manifest = dl.prepare_maf(["miniboone"], tmp_path / "out2", archive, tmp_path / "cache", allow_mismatch=True)
    assert manifest["datasets"]["miniboone"]["matches_paper_arrays"] is False


def test_prepare_ccpp_end_to_end_on_a_synthetic_workbook(dl, tmp_path, monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    rows = np.round(np.random.RandomState(5).rand(20, 5) * 100, 2)
    wb = openpyxl.Workbook()
    wb.active.append(["AT", "V", "AP", "RH", "PE"])
    for r in rows:
        wb.active.append([float(x) for x in r])
    buf = io.BytesIO()
    wb.save(buf)
    xlsx = buf.getvalue()
    archive = tmp_path / "CCPP.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("CCPP/Folds5x2_pp.xlsx", xlsx)
    train, test = dl.split_ccpp(rows.astype(np.float32))
    monkeypatch.setattr(dl, "CCPP_XLSX_SHA256", _sha(xlsx))
    monkeypatch.setattr(dl, "CCPP_EXPECTED_ARRAYS", {"train": dl.sha256_array(train), "test": dl.sha256_array(test)})
    path = dl.prepare_ccpp(tmp_path / "root", archive, tmp_path / "cache")
    assert path == tmp_path / "root" / "uci" / "ccpp.npz"
    with np.load(path) as z:
        assert z["train"].dtype == np.float32
        np.testing.assert_array_equal(z["train"], train)
        np.testing.assert_array_equal(z["test"], test)
    assert not (tmp_path / "root" / "uci" / "power.npz").exists()


def test_ccpp_split_matches_the_legacy_global_seed(dl):
    data = np.random.RandomState(4).rand(50, 5).astype(np.float32)
    state = np.random.get_state()
    try:
        np.random.seed(42)  # the earlier downloader's procedure
        idx = np.random.permutation(len(data))
    finally:
        np.random.set_state(state)
    train, test = dl.split_ccpp(data)
    np.testing.assert_array_equal(train, data[idx[:40]])
    np.testing.assert_array_equal(test, data[idx[40:]])
