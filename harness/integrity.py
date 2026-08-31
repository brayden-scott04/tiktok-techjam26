"""sha256 manifest verification for kit/ and harness/. Run before starting a
scored agent run and re-checked around every node execution -- if either
directory's tracked files have changed since the manifest was recorded, the
run aborts with a `tamper` event rather than silently scoring against a
different evaluate.py or a weakened guard.
"""
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT_MANIFEST = os.path.join(ROOT, "kit", "KIT_SHA256.txt")

# harness/ files whose integrity matters for the leakage/compliance guarantees.
# Not the whole directory (agent/, scripts/, tests/ change during normal
# development) -- just the files a compromised copy of which would silently
# break a safety property.
HARNESS_CRITICAL_FILES = [
    "harness/dataset.py",
    "harness/context.py",
    "harness/guards.py",
    "harness/eval_client.py",
    "harness/eval_server.py",
    "harness/node_entry.py",
    "harness/runner.py",
    "harness/sealed.py",
    "harness/task_spec.py",
]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_kit_manifest():
    manifest = {}
    with open(KIT_MANIFEST, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            digest, name = line.split(" *", 1)
            manifest[name] = digest
    return manifest


def verify_kit():
    """Raises RuntimeError naming the first file whose hash no longer matches
    kit/KIT_SHA256.txt. Returns True if every tracked file is unchanged."""
    manifest = load_kit_manifest()
    for name, expected in manifest.items():
        path = os.path.join(ROOT, "kit", name)
        if not os.path.exists(path):
            raise RuntimeError(f"TAMPER: kit/{name} is missing")
        actual = _sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"TAMPER: kit/{name} hash mismatch (expected {expected}, got {actual})")
    return True


def write_harness_manifest(path=None):
    path = path or os.path.join(ROOT, "harness", "HARNESS_SHA256.txt")
    with open(path, "w", encoding="utf-8") as fh:
        for rel in HARNESS_CRITICAL_FILES:
            digest = _sha256_file(os.path.join(ROOT, rel))
            fh.write(f"{digest} *{rel}\n")
    return path


def verify_harness(path=None):
    path = path or os.path.join(ROOT, "harness", "HARNESS_SHA256.txt")
    if not os.path.exists(path):
        raise RuntimeError(f"TAMPER: harness manifest {path} does not exist -- run write_harness_manifest() first")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            expected, rel = line.split(" *", 1)
            actual = _sha256_file(os.path.join(ROOT, rel))
            if actual != expected:
                raise RuntimeError(f"TAMPER: {rel} hash mismatch (expected {expected}, got {actual})")
    return True


def verify_all():
    verify_kit()
    verify_harness()
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--write-harness-manifest":
        p = write_harness_manifest()
        print(f"wrote {p}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_all()
        print("integrity OK: kit/ and harness/ match their recorded manifests")
    else:
        print("usage: python -m harness.integrity --write-harness-manifest | --verify")
