#!/usr/bin/env python3
"""Upload a staged pretrained-model bundle to Hugging Face using git-lfs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def run(cmd, *, cwd=None, env=None) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def create_repo(repo_id: str, token: str, private: bool) -> None:
    owner, name = (repo_id.split("/", 1) if "/" in repo_id else ("", repo_id))
    payload = {
        "name": name,
        "private": bool(private),
        "type": "model",
    }
    if owner:
        payload["organization"] = owner
    req = urllib.request.Request(
        "https://huggingface.co/api/repos/create",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req):
            return
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            return
        raise


def make_askpass_script(tmpdir: Path) -> Path:
    script = tmpdir / "git_askpass.sh"
    script.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  *Username*) printf '%s' \"hf_user\" ;;\n"
        "  *Password*) printf '%s' \"$HF_TOKEN\" ;;\n"
        "  *) printf '' ;;\n"
        "esac\n"
    )
    script.chmod(0o700)
    return script


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a staged pretrained model bundle to Hugging Face")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--token-env", type=str, default="HF_TOKEN")
    parser.add_argument("--create-repo", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", type=str, default="Upload pretrained VDC model bundle")
    args = parser.parse_args()

    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit(
            f"Missing Hugging Face token in environment variable {args.token_env}"
        )
    bundle_dir = args.bundle_dir.expanduser().resolve()
    if not bundle_dir.exists():
        raise SystemExit(f"Bundle directory does not exist: {bundle_dir}")

    if args.create_repo:
        create_repo(args.repo_id, token, private=args.private)

    with tempfile.TemporaryDirectory(prefix="vdc_hf_upload_") as tmp_raw:
        tmpdir = Path(tmp_raw)
        askpass = make_askpass_script(tmpdir)
        env = os.environ.copy()
        env["GIT_ASKPASS"] = str(askpass)
        env["HF_TOKEN"] = token

        run(["git-lfs", "install"], env=env)
        run(["git", "clone", f"https://huggingface.co/{args.repo_id}", str(tmpdir / "repo")], env=env)
        repo_dir = tmpdir / "repo"

        for child in bundle_dir.iterdir():
            target = repo_dir / child.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)

        run(["git", "add", "."], cwd=repo_dir, env=env)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        if not status.stdout.strip():
            print("No changes to upload.")
            return

        run(["git", "commit", "-m", args.commit_message], cwd=repo_dir, env=env)
        run(["git", "push", "origin", args.revision], cwd=repo_dir, env=env)
        print(f"Uploaded bundle to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
