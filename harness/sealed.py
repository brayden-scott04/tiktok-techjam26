"""One-shot hidden-test scoring. Deliberately NOT imported anywhere in agent/
or harness/node_entry.py -- a unit test (tests/test_isolation.py) asserts this
module is unreachable from agent/'s import graph, so "the agent never computed
a test score" is a structural property of the codebase, not a policy.

Run only by a human, only once, via scripts/final_test_score.py, after the
agent run has converged. Refuses to run twice (a lock file next to the
authoritative cache) so a re-run can't be used to peek and retry.
"""
import json
import os

from kit.evaluate import evaluate as kit_evaluate
from kit.submit import write_submission, read_submission

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_PATH = os.path.join(ROOT, "artifacts", ".test_scored")


class AlreadyScoredError(Exception):
    pass


def score_test(test_user_ids, test_labels, test_scores, force=False):
    """Compute the ONE hidden-test score for this submission. Raises
    AlreadyScoredError if this has already been called once for this run,
    unless force=True (only ever used deliberately, by a human, and logged)."""
    if os.path.exists(LOCK_PATH) and not force:
        with open(LOCK_PATH, encoding="utf-8") as fh:
            prior = json.load(fh)
        raise AlreadyScoredError(
            f"Test set was already scored at {prior.get('timestamp')}. "
            "Refusing to score again -- this is a one-shot operation. "
            "Pass force=True if you deliberately intend to re-score."
        )

    metrics = kit_evaluate(test_user_ids, test_labels, test_scores)

    import time

    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    with open(LOCK_PATH, "w", encoding="utf-8") as fh:
        json.dump({"timestamp": time.time(), "metrics": metrics}, fh, indent=2)

    return metrics


def write_and_verify_submission(path, rows, scores):
    """Write the submission via the UNMODIFIED kit.submit writer, then
    immediately read it back via the unmodified kit.submit reader (full
    alignment validation) -- an independent check of our own writer using the
    organizers' own code, not just our own logic checking itself."""
    write_submission(path, rows, scores)
    return read_submission(path, rows)
