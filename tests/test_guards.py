import os

import pytest

from harness import guards

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _read(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as fh:
        return fh.read()


def test_clean_solution_passes():
    src = _read("duplicate_a.py")
    assert guards.check_source(src) is True


def test_bad_syntax_rejected():
    with pytest.raises(guards.GuardViolation) as exc:
        guards.check_source(_read("bad_syntax.py"))
    assert exc.value.kind == "syntax_error"


def test_banned_import_rejected():
    with pytest.raises(guards.GuardViolation) as exc:
        guards.check_source(_read("banned_import.py"))
    assert exc.value.kind == "banned_import"
    assert exc.value.detail == "os"


def test_peek_string_rejected():
    with pytest.raises(guards.GuardViolation) as exc:
        guards.check_source(_read("peek_test.py"))
    # 'open' is also a banned name, and banned-name check runs before the
    # peek-string check, so either is an acceptable rejection reason here.
    assert exc.value.kind in ("banned_name", "peek_string")


def test_train_on_valid_rejected_by_static_guard_too():
    # This fixture explicitly accesses ctx.splits['valid'].long_view -- a
    # legitimate solution never would, since it's always -1. Defense-in-depth
    # layer 3 (the static peek-attempt scan) catches this even before the
    # STRUCTURAL defense (the value being -1) would render it harmless.
    with pytest.raises(guards.GuardViolation) as exc:
        guards.check_source(_read("train_on_valid.py"))
    assert exc.value.kind == "label_access_attempt"


def test_eval_budget_fixture_is_guard_clean():
    assert guards.check_source(_read("eval_budget_exceeded.py")) is True


def test_infinite_loop_fixture_is_guard_clean():
    assert guards.check_source(_read("infinite_loop.py")) is True


def test_oom_fixture_is_guard_clean():
    assert guards.check_source(_read("oom.py")) is True


def test_nan_scores_fixture_is_guard_clean():
    assert guards.check_source(_read("nan_scores.py")) is True


def test_wrong_length_fixture_is_guard_clean():
    assert guards.check_source(_read("wrong_length.py")) is True


def test_ast_hash_insensitive_to_comments_and_whitespace():
    h_a = guards.ast_normalized_hash(_read("duplicate_a.py"))
    h_b = guards.ast_normalized_hash(_read("duplicate_b.py"))
    assert h_a == h_b


def test_ast_hash_sensitive_to_real_logic_change():
    h_a = guards.ast_normalized_hash(_read("duplicate_a.py"))
    h_other = guards.ast_normalized_hash(_read("wrong_length.py"))
    assert h_a != h_other
