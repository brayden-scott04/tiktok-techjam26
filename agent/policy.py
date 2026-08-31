"""Convergence policy and (later) node-selection policy for the solution tree.

The convergence check is the function that decides when a scored competition
run ends, so it gets its own careful unit tests (tests/test_policy.py) rather
than being inlined into agent/loop.py.
"""


def check_convergence(scored_primaries, epsilon, N, min_scored_iterations, directions=None):
    """Implements the organizer FAQ 2.9 cumulative-window rule, with a
    team-declared minimum-iteration floor (also sanctioned by the same FAQ):

        converged  iff  len(scored_primaries) >= min_scored_iterations
                    and best(primary over the last N scored iterations)
                        - best(primary over everything before that window)
                        <= epsilon

    `scored_primaries` must already have crashed/debug iterations excluded by
    the caller -- per the FAQ, those "do not advance or reset the convergence
    window", which this function implements simply by never seeing them.

    `directions`, if given, is a same-length list of each scored iteration's
    direction tag; the return includes `directions_in_window` for the
    anti-stall diversity property (see agent/loop.py's plateau branch-switch,
    which is what actually enforces the diversity -- this function only
    reports it).

    Returns a dict; `converged` is always present.
    """
    n = len(scored_primaries)
    if directions is not None and len(directions) != n:
        raise ValueError("directions must be the same length as scored_primaries")

    if n < min_scored_iterations:
        return {
            "converged": False,
            "reason": "below_min_scored_iterations_floor",
            "scored_iterations": n,
            "min_scored_iterations": min_scored_iterations,
        }

    if n <= N:
        # Not enough history before the window to compare against.
        return {
            "converged": False,
            "reason": "insufficient_history_before_window",
            "scored_iterations": n,
        }

    window = scored_primaries[-N:]
    before = scored_primaries[:-N]
    best_before = max(before)
    best_in_window = max(window)
    gain = best_in_window - best_before
    # A tiny numerical tolerance so a gain that is mathematically exactly
    # epsilon doesn't flip to "still improving" on float64 rounding noise
    # (e.g. 0.902 - 0.900 == 0.0020000000000000018, not 0.002). Real primary
    # scores will never legitimately land within 1e-9 of epsilon by chance.
    converged = gain <= epsilon + 1e-9

    result = {
        "converged": converged,
        "reason": "plateau" if converged else "still_improving",
        "scored_iterations": n,
        "best_before_window": best_before,
        "best_in_window": best_in_window,
        "gain": gain,
        "epsilon": epsilon,
        "N": N,
        "window": window,
    }
    if directions is not None:
        result["directions_in_window"] = sorted(set(directions[-N:]))
    return result
