"""Idempotent, resumable download + extraction of KuaiRand-Pure. Writes
OUTSIDE the repo (KR_DATA_ROOT, default a sibling _kuairand_data/ directory)
so the dataset never risks being committed, and records a manifest with the
downloaded file's size and sha256 for reproducibility.
"""
import hashlib
import json
import os
import sys
import tarfile

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.env_loader import load_env

load_env()

URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz?download=1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_FILES = [
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "log_random_4_22_to_5_08_pure.csv",
    "user_features_pure.csv",
    "video_features_basic_pure.csv",
    "video_features_statistic_pure.csv",
]


def _default_target_dir():
    kr_root = os.environ.get("KR_DATA_ROOT")
    if kr_root:
        # KR_DATA_ROOT points at .../KuaiRand-Pure/data -- the archive extracts
        # to .../KuaiRand-Pure, i.e. one level up.
        return os.path.dirname(os.path.dirname(kr_root))
    return os.path.join(os.path.dirname(ROOT), "_kuairand_data")


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    archive_path = os.path.join(dest_dir, "KuaiRand-Pure.tar.gz")
    resume_from = os.path.getsize(archive_path) if os.path.exists(archive_path) else 0

    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    with requests.get(URL, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        mode = "ab" if resume_from and r.status_code == 206 else "wb"
        with open(archive_path, mode) as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return archive_path


def extract(archive_path, dest_dir):
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest_dir, filter="data")


def verify(data_dir):
    missing = [f for f in EXPECTED_FILES if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        raise SystemExit(f"missing expected files after extraction: {missing}")


def main():
    target_dir = _default_target_dir()
    data_dir = os.path.join(target_dir, "KuaiRand-Pure", "data")

    if os.path.isdir(data_dir):
        try:
            verify(data_dir)
            print(f"KuaiRand-Pure already present and complete at {data_dir}")
            return
        except SystemExit:
            print("existing extraction incomplete, re-downloading/extracting...")

    print(f"downloading to {target_dir} ...")
    archive_path = download(target_dir)
    size = os.path.getsize(archive_path)
    digest = _sha256_file(archive_path)
    print(f"downloaded {size:,} bytes, sha256={digest}")

    print("extracting ...")
    extract(archive_path, target_dir)
    verify(data_dir)

    manifest = {"url": URL, "archive_size_bytes": size, "archive_sha256": digest, "data_dir": data_dir}
    with open(os.path.join(ROOT, "artifacts", "data_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"done: {data_dir}")
    print(f"set KR_DATA_ROOT={data_dir} in your .env if it differs from the current setting")


if __name__ == "__main__":
    main()
