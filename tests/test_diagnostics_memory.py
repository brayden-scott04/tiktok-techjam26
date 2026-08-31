import os

import numpy as np
import pytest

from harness import dataset as ds
from harness.diagnostics import (
    compute_diagnostics, render_diagnostics_markdown,
    compute_comparative_diagnostics, render_comparative_diagnostics_markdown,
)
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


def test_compute_comparative_diagnostics_detects_a_real_improvement(valid_data):
    labels = [int(x) for x in valid_data["long_view"]]
    uids = list(valid_data["user_id_raw"])
    # "candidate" scores the true label directly (near-oracle); incumbent is random.
    # This should show up as a large positive overall_delta and per-bucket deltas.
    candidate = np.asarray(labels, dtype=np.float32) + np.random.default_rng(0).normal(0, 0.01, len(labels))
    incumbent = np.random.default_rng(1).normal(size=len(labels)).astype(np.float32)
    comp = compute_comparative_diagnostics(candidate, incumbent, uids, labels, dates=list(valid_data["date"]))
    assert comp["overall_delta"] > 0.1
    assert len(comp["bucket_deltas"]) > 0
    for d in comp["bucket_deltas"].values():
        assert abs(d["delta"] - (d["candidate_primary"] - d["incumbent_primary"])) < 1e-9


def test_compute_comparative_diagnostics_identical_scores_zero_delta(valid_data):
    labels = [int(x) for x in valid_data["long_view"]]
    uids = list(valid_data["user_id_raw"])
    scores = np.random.default_rng(0).normal(size=len(labels)).astype(np.float32)
    comp = compute_comparative_diagnostics(scores, scores, uids, labels)
    assert abs(comp["overall_delta"]) < 1e-9
    for d in comp["bucket_deltas"].values():
        assert abs(d["delta"]) < 1e-9


def test_render_comparative_diagnostics_markdown_is_a_string(valid_data):
    labels = [int(x) for x in valid_data["long_view"]]
    uids = list(valid_data["user_id_raw"])
    a = np.random.default_rng(0).normal(size=len(labels)).astype(np.float32)
    b = np.random.default_rng(1).normal(size=len(labels)).astype(np.float32)
    comp = compute_comparative_diagnostics(a, b, uids, labels)
    md = render_comparative_diagnostics_markdown(comp)
    assert isinstance(md, str) and "delta" in md.lower() and "incumbent" in md.lower()


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


def test_build_improve_messages_without_comparison_omits_section():
    msgs = memory.build_improve_messages("journal", "diag", "budget", "code", "hyp")
    assert "wasn't adopted" not in msgs[1]["content"]


def test_build_improve_messages_with_comparison_includes_it():
    msgs = memory.build_improve_messages("journal", "diag", "budget", "code", "hyp", comparison_md="| bucket | delta |\n|---|---|\n| 1-5 | -0.01 |")
    assert "wasn't adopted" in msgs[1]["content"]
    assert "1-5" in msgs[1]["content"]


def test_render_journal_shows_calibration_columns():
    nodes = [
        {"node_id": "n0001", "parent_node_id": "n0000", "action": "draft", "direction": "loss",
         "hypothesis": "try BPR", "status": "ok", "metrics": {"primary": 0.60}, "delta_vs_baseline": -0.001,
         "seconds": 20.0, "est_runtime_sec": 30, "expected_effect": {"delta": 0.01, "confidence": 0.5},
         "calibration_error": -0.0095},
    ]
    md = memory.render_journal(nodes)
    assert "+0.0100" in md  # predicted delta
    assert "-0.0095" in md  # calibration error


def test_calibration_summary_empty_below_two_points():
    assert memory.render_calibration_summary([]) == ""
    assert memory.render_calibration_summary([(0.01, -0.005)]) == ""


def test_calibration_summary_detects_overconfidence():
    pairs = [(0.01, -0.009), (0.02, -0.018), (0.015, -0.014)]
    summary = memory.render_calibration_summary(pairs)
    assert "overconfident" in summary


def test_calibration_summary_detects_underconfidence():
    pairs = [(0.001, 0.008), (0.001, 0.007), (0.002, 0.006)]
    summary = memory.render_calibration_summary(pairs)
    assert "underconfident" in summary


def test_calibration_summary_well_calibrated():
    pairs = [(0.005, 0.0001), (0.004, -0.0002), (0.006, 0.0001)]
    summary = memory.render_calibration_summary(pairs)
    assert "well-calibrated" in summary
