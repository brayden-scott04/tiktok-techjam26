import os

import pytest

from agent.validate import validate_candidate
from harness import dataset as ds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
CACHE_PATH = os.path.join(ROOT, "artifacts", "cache", "kuairand_pure")
SANITIZED_PATH = os.path.join(ROOT, "artifacts", "cache", "kuairand_pure_sanitized")


@pytest.fixture(scope="module")
def valid_labels():
    if not os.path.exists(CACHE_PATH + ".npz"):
        pytest.skip("cache not built -- run `python -m scripts.build_cache` first")
    auth_cache, _ = ds.load_cache(CACHE_PATH, skip_raw_meta=True)
    valid_c = auth_cache["valid"]
    return list(valid_c["user_id_raw"]), [int(x) for x in valid_c["long_view"]]


def _read(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as fh:
        return fh.read()


def test_banned_import_rejected_at_static_stage(valid_labels):
    uids, labels = valid_labels
    result = validate_candidate(_read("banned_import.py"), set(), SANITIZED_PATH, uids, labels, ROOT)
    assert result.passed is False
    assert result.stage == "banned_import"


def test_bad_syntax_rejected_at_static_stage(valid_labels):
    uids, labels = valid_labels
    result = validate_candidate(_read("bad_syntax.py"), set(), SANITIZED_PATH, uids, labels, ROOT)
    assert result.passed is False
    assert result.stage == "syntax_error"


def test_duplicate_rejected_before_smoke(valid_labels):
    uids, labels = valid_labels
    code = _read("duplicate_a.py")
    first = validate_candidate(code, set(), SANITIZED_PATH, uids, labels, ROOT)
    assert first.passed is True
    second = validate_candidate(code, {first.ast_hash}, SANITIZED_PATH, uids, labels, ROOT)
    assert second.passed is False
    assert second.stage == "dedup"


def test_n0000_solution_passes_full_gate(valid_labels):
    uids, labels = valid_labels
    with open(os.path.join(ROOT, "solutions", "n0000", "solution.py"), encoding="utf-8") as fh:
        code = fh.read()
    result = validate_candidate(code, set(), SANITIZED_PATH, uids, labels, ROOT, smoke_timeout_sec=60)
    assert result.passed is True, result.detail
    assert result.smoke_result.status == "ok"
    assert result.ast_hash is not None
