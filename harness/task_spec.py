"""Single source of truth for task constants. Re-exports from the vendored kit
where possible so there is exactly one place each fact is defined. Never redefine
a value that kit/ already defines — import it instead.
"""
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from kit.data import LABEL, SPLITS, FIELDS  # noqa: E402

with open(os.path.join(_ROOT, "kit", "baseline_scores.json"), encoding="utf-8") as fh:
    _BASELINE_SCORES = json.load(fh)

with open(os.path.join(_ROOT, "config", "run.json"), encoding="utf-8") as fh:
    RUN_CONFIG = json.load(fh)

# Official reference numbers, pulled from the organizer-shipped baseline_scores.json
# rather than re-typed, so a kit update can't silently desync from our constants.
BASELINE_VALID_PRIMARY = _BASELINE_SCORES["scores"]["fm_official"]["valid"]["primary"]
BASELINE_TEST_PRIMARY = _BASELINE_SCORES["scores"]["fm_official"]["test"]["primary"]
BASELINE_VALID = _BASELINE_SCORES["scores"]["fm_official"]["valid"]
BASELINE_TEST = _BASELINE_SCORES["scores"]["fm_official"]["test"]
RANDOM_TEST_PRIMARY = _BASELINE_SCORES["scores"]["random"]["test"]["primary"]
RANDOM_VALID_PRIMARY = _BASELINE_SCORES["scores"]["random"]["valid"]["primary"]
POP_TEST_PRIMARY = _BASELINE_SCORES["scores"]["item_popularity"]["test"]["primary"]
POP_VALID_PRIMARY = _BASELINE_SCORES["scores"]["item_popularity"]["valid"]["primary"]
ORACLE_VALID_PRIMARY = _BASELINE_SCORES["scores"]["oracle_ceiling"]["valid"]["primary"]
ORACLE_TEST_PRIMARY = _BASELINE_SCORES["scores"]["oracle_ceiling"]["test"]["primary"]

# Expected split sizes; a mismatch means the load path or encoding differs from
# what the organizers describe, and everything downstream is untrustworthy.
EXPECTED_SPLIT_SIZES = {"train": 1_141_112, "valid": 124_909, "test": 170_588}

# Outcome columns that must never reach the agent for a non-train row.
# `long_view` is the label; the rest are post-impression feedback signals
# (legitimate as train-only auxiliary targets, illegal as input features).
OUTCOME_COLUMNS = [
    "long_view", "is_click", "is_like", "is_follow",
    "is_comment", "is_forward", "is_hate", "play_time_ms",
]

SUBMISSION_HEADER = ["row_id", "user_id", "video_id", "score"]

CONVERGENCE = RUN_CONFIG["convergence"]
CAPS = RUN_CONFIG["caps"]
