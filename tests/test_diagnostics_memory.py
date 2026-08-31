import os

import numpy as np
import pytest

from harness import dataset as ds
from harness.diagnostics import compute_diagnostics, render_diagnostics_markdown
from agent import memory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(ROOT, "artifacts", "cache", "kuairand_pure")


@pytest.fixture(scope="module")
def valid_data():
    if not os.path.exists(CACHE_PATH + ".npz"):
        pytest.skip("cache not built -- run `python -m scripts.build_cache` first")
    auth_cache, _ = ds.load_cache(CACHE_PATH, skip_raw_meta=True)
    return auth_cache["valid"]


def test_compute_diagnostics_shape(valid_data):
    rng = np.random.default_rng(0)
    scores = rng.normal(size=len(valid_data["date"])).astype(np.float32)
    diag = compute_diagnostics(
        list(valid_data["user_id_raw"]), [int(x) for x in valid_data["long_view"]], scores,
        dates=list(valid_data["date"]),
    )
    assert "overall" in diag
    assert 0.0 <= diag["overall"]["primary"] <= 1.0
    assert len(diag["by_impression_bucket"]) > 0
    assert diag["per_date_primary"] is not None
    assert 0.0 <= diag["degenerate_constant_score_user_frac"] <= 1.0


def test_render_diagnostics_markdown_is_a_string(valid_data):
    rng = np.random.default_rng(0)
    scores = rng.normal(size=len(valid_data["date"])).astype(np.float32)
    diag = compute_diagnostics(
        list(valid_data["user_id_raw"]), [int(x) for x in valid_data["long_view"]], scores,
    )
    md = render_diagnostics_markdown(diag)
    assert isinstance(md, str) and "primary" in md.lower()


def test_static_prefix_is_deterministic():
    a = memory.static_prefix()
    b = memory.static_prefix()
    assert a == b
    assert "fit_predict" in a
    assert "GAUC" in a


def test_render_journal_empty():
    assert "no prior" in memory.render_journal([])


def test_render_journal_nonempty():
    nodes = [
        {"node_id": "n0000", "parent_node_id": None, "action": "root", "direction": "baseline",
         "hypothesis": "official FM baseline", "status": "ok",
         "metrics": {"primary": 0.6016}, "delta_vs_baseline": 0.0, "seconds": 20.3, "est_runtime_sec": 40},
    ]
    md = memory.render_journal(nodes)
    assert "n0000" in md and "0.6016" in md


def test_build_draft_messages_structure():
    msgs = memory.build_draft_messages("journal", "diag", "budget", used_directions=["loss", "arch"])
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "loss" in msgs[1]["content"] and "arch" in msgs[1]["content"]


def test_build_improve_messages_includes_parent_code():
    msgs = memory.build_improve_messages("journal", "diag", "budget", "def fit_predict(ctx): pass", "some hypothesis")
    assert "def fit_predict(ctx): pass" in msgs[1]["content"]
    assert "some hypothesis" in msgs[1]["content"]
