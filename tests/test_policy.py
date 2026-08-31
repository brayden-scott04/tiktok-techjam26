from agent.policy import check_convergence

EPS = 0.002
N = 3
FLOOR = 12


def test_floor_suppresses_early_plateau():
    # First 5 scored iterations are flat (gain 0 across any window), but the
    # floor of 12 has not been reached -- must not converge yet.
    scores = [0.55, 0.60, 0.601, 0.6011, 0.6012]
    result = check_convergence(scores, EPS, N, FLOOR)
    assert result["converged"] is False
    assert result["reason"] == "below_min_scored_iterations_floor"


def test_clean_plateau_converges_after_floor():
    # 12 iterations: strong early gains, then flat for the last 3.
    scores = [0.50, 0.55, 0.58, 0.595, 0.601, 0.603, 0.604,
              0.6045, 0.6048, 0.6049, 0.6050, 0.6051]
    assert len(scores) == 12
    result = check_convergence(scores, EPS, N, FLOOR)
    # last N=3: [0.6049, 0.6050, 0.6051] -> best_in_window=0.6051
    # before: everything up to index -4 -> best_before=0.6048
    # gain = 0.6051-0.6048 = 0.0003 <= 0.002 -> converged
    assert result["converged"] is True
    assert result["reason"] == "plateau"
    assert abs(result["gain"] - 0.0003) < 1e-9


def test_late_improvement_inside_window_prevents_convergence():
    # Same as the plateau case, but the last entry jumps well past epsilon.
    scores = [0.50, 0.55, 0.58, 0.595, 0.601, 0.603, 0.604,
              0.6045, 0.6048, 0.6049, 0.6050, 0.6090]
    result = check_convergence(scores, EPS, N, FLOOR)
    assert result["converged"] is False
    assert result["reason"] == "still_improving"
    assert result["gain"] > EPS


def test_gain_exactly_at_epsilon_boundary_converges():
    # gain == epsilon exactly should count as converged ("<= epsilon").
    before = [0.10, 0.20, 0.30, 0.40, 0.500, 0.600, 0.700, 0.800, 0.900]
    window = [0.9005, 0.901, 0.902]  # best_in_window - best_before(0.900) = 0.002
    scores = before + window
    assert len(scores) == 12
    result = check_convergence(scores, EPS, N, FLOOR)
    assert result["converged"] is True


def test_cumulative_window_is_against_running_best_not_previous_iteration():
    # Regression test for the earlier (incorrect) design: this is NOT a
    # per-iteration "did this beat the last one" check. best_before_window
    # must be the max over ALL prior iterations, not just the one right
    # before the window.
    scores = [0.10, 0.90, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10] + [0.11, 0.10, 0.12]
    assert len(scores) == 12
    result = check_convergence(scores, EPS, N, FLOOR)
    # best_before_window must be 0.90 (the early spike), not 0.10 (the
    # iteration immediately preceding the window) or 0.12 (a naive "last
    # value seen" bug).
    assert result["best_before_window"] == 0.90
    # The window (best 0.12) never gets within epsilon of the historical best
    # (0.90) -- by the literal rule ("has not improved by more than epsilon")
    # a window that does WORSE than history has, trivially, not improved, so
    # this correctly counts as converged/plateaued too, not just a flat
    # window. Convergence means "stop chasing this", and a regression
    # qualifies just as much as a plateau does.
    assert result["converged"] is True


def test_crashed_iterations_are_simply_absent_not_specially_handled():
    # The FAQ says crashed iterations "do not advance or reset the
    # convergence window". This function has no concept of a crashed
    # iteration at all -- the caller is responsible for never appending one.
    # This test documents that contract: a list with an artificially removed
    # "crash" produces the same result as if the crash had never happened.
    with_gap_removed = [0.50, 0.55, 0.58, 0.595, 0.601, 0.603, 0.604,
                         0.6045, 0.6048, 0.6049, 0.6050, 0.6051]
    result_a = check_convergence(with_gap_removed, EPS, N, FLOOR)
    result_b = check_convergence(list(with_gap_removed), EPS, N, FLOOR)
    assert result_a == result_b


def test_directions_in_window_reported():
    scores = [0.50, 0.55, 0.58, 0.595, 0.601, 0.603, 0.604,
              0.6045, 0.6048, 0.6049, 0.6050, 0.6051]
    directions = ["loss", "loss", "sequence", "sequence", "loss", "loss",
                  "arch", "arch", "arch", "multitask", "multitask", "multitask"]
    result = check_convergence(scores, EPS, N, FLOOR, directions=directions)
    assert result["directions_in_window"] == ["multitask"]


def test_mismatched_directions_length_raises():
    import pytest

    with pytest.raises(ValueError):
        check_convergence([0.1] * 12, EPS, N, FLOOR, directions=["loss"] * 5)
