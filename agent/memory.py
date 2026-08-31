"""Journal rendering and prompt assembly. The static prefix (system.md +
task_brief.md) is kept byte-identical and first in the message list across
every call so OpenAI's prompt caching actually applies -- see agent/llm.py's
module docstring for the measured cache_write_tokens/cached_tokens fields
this is meant to exploit.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(ROOT, "agent", "prompts")


def _read_prompt(name):
    with open(os.path.join(PROMPTS_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def static_prefix():
    """system.md + task_brief.md, concatenated once. Must never have anything
    call-specific (a timestamp, a counter) interpolated into it -- that would
    break prompt caching for every subsequent call."""
    return _read_prompt("system.md") + "\n\n---\n\n" + _read_prompt("task_brief.md")


def render_journal(nodes):
    """`nodes` is a list of dicts, oldest first, each with at minimum:
    node_id, parent_node_id, action, direction, hypothesis, status,
    metrics (or None), delta_vs_baseline (or None), seconds, est_runtime_sec,
    expected_effect (or None), calibration_error (or None).
    """
    if not nodes:
        return "(no prior iterations yet -- this is the first one)"

    lines = ["| node | parent | action | direction | hypothesis | primary | delta | pred Δ | calib err | status | time (est) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in nodes:
        hyp = (n.get("hypothesis") or "")[:80]
        primary = f"{n['metrics']['primary']:.4f}" if n.get("metrics") else "-"
        delta = f"{n['delta_vs_baseline']:+.4f}" if n.get("delta_vs_baseline") is not None else "-"
        ee = n.get("expected_effect") or {}
        pred = f"{ee['delta']:+.4f}" if ee.get("delta") is not None else "-"
        calib = f"{n['calibration_error']:+.4f}" if n.get("calibration_error") is not None else "-"
        secs = n.get("seconds")
        est = n.get("est_runtime_sec")
        time_str = f"{secs:.0f}s ({est:.0f}s est)" if secs is not None else "-"
        lines.append(
            f"| {n['node_id']} | {n.get('parent_node_id', '-')} | {n['action']} | {n.get('direction', '-')} "
            f"| {hyp} | {primary} | {delta} | {pred} | {calib} | {n['status']} | {time_str} |"
        )
    return "\n".join(lines)


def render_calibration_summary(recent_pred_actual_pairs):
    """`recent_pred_actual_pairs` is a list of (predicted_delta, calibration_error)
    tuples for the last few scored LLM nodes. calibration_error = actual_gain -
    predicted_delta, so a negative mean error means the model has been
    overconfident (actual gains fall short of what it predicted)."""
    if len(recent_pred_actual_pairs) < 2:
        return ""
    preds = [p for p, _ in recent_pred_actual_pairs]
    errs = [e for _, e in recent_pred_actual_pairs]
    mean_pred = sum(preds) / len(preds)
    mean_err = sum(errs) / len(errs)
    if mean_err < -0.0005:
        verdict = "you have been overconfident -- actual gains have fallen short of your predictions"
    elif mean_err > 0.0005:
        verdict = "you have been underconfident -- actual gains have exceeded your predictions"
    else:
        verdict = "your predictions have been reasonably well-calibrated"
    return (
        f"Calibration over your last {len(recent_pred_actual_pairs)} predictions: "
        f"mean predicted Δ={mean_pred:+.4f}, mean calibration error={mean_err:+.4f} -> {verdict}."
    )


def render_budget_block(state):
    return (
        f"Nodes used: {state['total_iterations']}/{state['max_total_iterations']}. "
        f"Scored iterations: {state['scored_iterations']} (floor for convergence: {state['min_scored_iterations']}). "
        f"Wall-clock used: {state['wall_seconds']:.0f}s / {state['max_wall_seconds']:.0f}s. "
        f"Spend so far: ${state['cost_usd']:.2f} / ${state['max_cost_usd']:.2f}."
    )


def forbidden_directions_note(used_directions):
    if not used_directions:
        return ""
    return f"\n\nDirections already attempted (pick a different one for a `draft`): {', '.join(sorted(set(used_directions)))}."


def build_draft_messages(journal_md, diagnostics_md, budget_md, used_directions):
    action_prompt = _read_prompt("draft.md") + forbidden_directions_note(used_directions)
    user_content = (
        f"# Journal so far\n{journal_md}\n\n"
        f"# Diagnostics on the current best node\n{diagnostics_md}\n\n"
        f"# Budget\n{budget_md}\n\n"
        f"---\n\n{action_prompt}"
    )
    return [
        {"role": "system", "content": static_prefix()},
        {"role": "user", "content": user_content},
    ]


def build_improve_messages(journal_md, diagnostics_md, budget_md, parent_code, parent_hypothesis, comparison_md=None):
    action_prompt = _read_prompt("improve.md")
    comparison_section = ""
    if comparison_md:
        comparison_section = (
            f"# Why this parent wasn't adopted as the leading approach last time\n"
            f"This is the exact bucketed comparison against whatever was the best node at the time this parent "
            f"was scored. Use it to target a specific, informed fix rather than guessing.\n\n{comparison_md}\n\n"
        )
    user_content = (
        f"# Journal so far\n{journal_md}\n\n"
        f"# Diagnostics on the current best node\n{diagnostics_md}\n\n"
        f"# Budget\n{budget_md}\n\n"
        f"{comparison_section}"
        f"# Parent solution (original hypothesis: {parent_hypothesis})\n```python\n{parent_code}\n```\n\n"
        f"---\n\n{action_prompt}"
    )
    return [
        {"role": "system", "content": static_prefix()},
        {"role": "user", "content": user_content},
    ]


def build_debug_messages(failing_code, error_summary, hypothesis):
    action_prompt = _read_prompt("debug.md")
    user_content = (
        f"# Original hypothesis (unchanged)\n{hypothesis}\n\n"
        f"# Failing solution\n```python\n{failing_code}\n```\n\n"
        f"# Failure detail\n{error_summary}\n\n"
        f"---\n\n{action_prompt}"
    )
    return [
        {"role": "system", "content": static_prefix()},
        {"role": "user", "content": user_content},
    ]


def build_repair_syntax_messages(failing_code, syntax_error):
    template = _read_prompt("repair_syntax.md").format(error=syntax_error)
    user_content = f"```python\n{failing_code}\n```\n\n{template}"
    return [{"role": "user", "content": user_content}]


def build_reformat_messages(malformed_text):
    template = _read_prompt("reformat.md")
    user_content = f"{template}\n\nYour previous raw response:\n{malformed_text}"
    return [{"role": "user", "content": user_content}]
