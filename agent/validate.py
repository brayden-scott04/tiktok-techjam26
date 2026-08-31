"""Pre-execution gate for a candidate solution: compile -> AST guard -> dedup
-> smoke run. Nothing reaches a full (multi-seed, full-timeout) run without
passing all four. This is what keeps a bad LLM response from burning a full
900s x 3-seed slot on something that was never going to work.
"""
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from harness import guards
from harness.runner import run_node, NodeResult


@dataclass
class ValidationResult:
    passed: bool
    stage: str  # syntax_error | banned_import | banned_name | peek_string | label_access_attempt | dedup | smoke | ok
    detail: str = ""
    ast_hash: Optional[str] = None
    smoke_result: Optional[NodeResult] = None


def validate_candidate(
    code,
    known_hashes,
    sanitized_cache_path,
    valid_user_ids,
    valid_labels,
    repo_root,
    smoke_timeout_sec=120,
):
    """Runs the four-stage gate on `code` (a full solution.py source string).
    `known_hashes` is the set of ast_normalized_hash values for every node
    already in the tree -- callers should add the returned ast_hash to it on
    success, so the next candidate is checked against an up-to-date set.
    """
    try:
        guards.check_source(code)
    except guards.GuardViolation as e:
        return ValidationResult(passed=False, stage=e.kind, detail=e.detail)

    ast_hash = guards.ast_normalized_hash(code)
    if ast_hash in known_hashes:
        return ValidationResult(passed=False, stage="dedup", detail=f"duplicate of an existing node (hash {ast_hash[:12]}...)")

    with tempfile.TemporaryDirectory(prefix="kr_validate_") as tmp:
        solution_path = os.path.join(tmp, "solution.py")
        with open(solution_path, "w", encoding="utf-8") as fh:
            fh.write(code)
        out_dir = os.path.join(tmp, "out")

        smoke_result = run_node(
            solution_path=solution_path,
            out_dir=out_dir,
            sanitized_cache_path=sanitized_cache_path,
            authoritative_valid_user_ids=valid_user_ids,
            authoritative_valid_labels=valid_labels,
            seed=0,
            smoke=True,
            timeout_sec=smoke_timeout_sec,
            repo_root=repo_root,
        )

    if smoke_result.status != "ok":
        return ValidationResult(passed=False, stage="smoke", detail=smoke_result.traceback or smoke_result.status, smoke_result=smoke_result)

    return ValidationResult(passed=True, stage="ok", ast_hash=ast_hash, smoke_result=smoke_result)
