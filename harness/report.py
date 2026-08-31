"""Renders artifacts/run_log.jsonl into a human-readable report.md and a
standalone results_table.md. Pure post-processing -- reads the log, writes
markdown, no side effects on the run itself.
"""
import json
import os

from harness import task_spec as spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_run_log(path):
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _scored(records):
    return [r for r in records if r.get("metrics")]


def render_results_table(records, test_metrics=None):
    scored = _scored(records)
    best = max(scored, key=lambda r: r["metrics"]["primary"], default=None)
    lines = [
        "| | valid GAUC | valid nDCG@5 | valid primary | test GAUC | test nDCG@5 | test primary | Δ primary vs baseline |",
        "|---|---|---|---|---|---|---|---|",
        f"| random | - | - | {spec.RANDOM_VALID_PRIMARY:.4f} | - | - | {spec.RANDOM_TEST_PRIMARY:.4f} | {spec.RANDOM_TEST_PRIMARY - spec.BASELINE_TEST_PRIMARY:+.4f} |",
        f"| item popularity | - | - | {spec.POP_VALID_PRIMARY:.4f} | - | - | {spec.POP_TEST_PRIMARY:.4f} | {spec.POP_TEST_PRIMARY - spec.BASELINE_TEST_PRIMARY:+.4f} |",
        f"| official FM baseline | {spec.BASELINE_VALID['GAUC']:.4f} | {spec.BASELINE_VALID['nDCG@5']:.4f} | {spec.BASELINE_VALID_PRIMARY:.4f} | {spec.BASELINE_TEST['GAUC']:.4f} | {spec.BASELINE_TEST['nDCG@5']:.4f} | {spec.BASELINE_TEST_PRIMARY:.4f} | 0.0000 |",
    ]
    if best:
        m = best["metrics"]
        if test_metrics:
            tline = (
                f"| **our converged best ({best['node_id']})** | {m['GAUC']:.4f} | {m['nDCG@5']:.4f} | {m['primary']:.4f} "
                f"| {test_metrics['GAUC']:.4f} | {test_metrics['nDCG@5']:.4f} | {test_metrics['primary']:.4f} "
                f"| {test_metrics['primary'] - spec.BASELINE_TEST_PRIMARY:+.4f} |"
            )
        else:
            tline = (
                f"| **our best so far ({best['node_id']})** | {m['GAUC']:.4f} | {m['nDCG@5']:.4f} | {m['primary']:.4f} "
                f"| (not yet sealed-scored) | | | |"
            )
        lines.append(tline)
    oracle_frac = None
    if best and test_metrics:
        headroom = spec.ORACLE_TEST_PRIMARY - spec.BASELINE_TEST_PRIMARY
        oracle_frac = (test_metrics["primary"] - spec.BASELINE_TEST_PRIMARY) / headroom if headroom else None
    if oracle_frac is not None:
        lines.append(f"\n% of oracle headroom captured: {oracle_frac:.1%} (oracle ceiling: test primary {spec.ORACLE_TEST_PRIMARY:.4f})")
    return "\n".join(lines)


def render_trajectory(records):
    scored = _scored(records)
    if not scored:
        return "(no scored iterations yet)"
    lines = ["| scored # | node | direction | valid primary | vs baseline |", "|---|---|---|---|---|"]
    for i, r in enumerate(scored, 1):
        p = r["metrics"]["primary"]
        lines.append(f"| {i} | {r['node_id']} | {r.get('direction', '-')} | {p:.4f} | {p - spec.BASELINE_VALID_PRIMARY:+.4f} |")
    return "\n".join(lines)


def render_directions(records):
    scored = _scored(records)
    by_dir = {}
    for r in scored:
        d = r.get("direction", "unknown")
        s = by_dir.setdefault(d, {"attempts": 0, "best": -1.0})
        s["attempts"] += 1
        s["best"] = max(s["best"], r["metrics"]["primary"])
    if not by_dir:
        return "(none yet)"
    lines = ["| direction | attempts | best primary | Δ vs baseline |", "|---|---|---|---|"]
    for d, s in sorted(by_dir.items(), key=lambda kv: -kv[1]["best"]):
        lines.append(f"| {d} | {s['attempts']} | {s['best']:.4f} | {s['best'] - spec.BASELINE_VALID_PRIMARY:+.4f} |")
    return "\n".join(lines)


def render_events_table(records):
    rows = []
    for r in records:
        for e in r.get("events", []) or []:
            rows.append((r.get("node_id", "-"), e.get("type", "-"), str(e.get("reason") or e.get("detail") or "")[:100]))
    if not rows:
        return "(no error/recovery events recorded)"
    lines = ["| node | event | detail |", "|---|---|---|"]
    for node, etype, detail in rows:
        lines.append(f"| {node} | {etype} | {detail} |")
    return "\n".join(lines)


def render_resource_summary(records):
    total_in = sum(r.get("usage", {}).get("input_tokens", 0) for r in records)
    total_out = sum(r.get("usage", {}).get("output_tokens", 0) for r in records)
    total_cost = sum(r.get("usage", {}).get("cost_usd", 0.0) for r in records)
    n_iter = len(records)
    n_scored = len(_scored(records))
    interventions_path = os.path.join(ROOT, "artifacts", "interventions.jsonl")
    n_interventions = 0
    if os.path.exists(interventions_path):
        with open(interventions_path, encoding="utf-8") as fh:
            n_interventions = sum(1 for _ in fh)
    return (
        f"- Total iterations: {n_iter} (scored: {n_scored})\n"
        f"- Total LLM tokens: {total_in:,} in + {total_out:,} out = {total_in + total_out:,}\n"
        f"- Total cost: ${total_cost:.2f}\n"
        f"- Manual interventions: {n_interventions}\n"
    )


def render_report(run_log_path, out_path, results_table_path=None, test_metrics=None):
    records = load_run_log(run_log_path)
    sections = [
        "# Agent Run Report\n",
        render_resource_summary(records),
        "\n## Results\n",
        render_results_table(records, test_metrics=test_metrics),
        "\n## Score trajectory\n",
        render_trajectory(records),
        "\n## Directions explored\n",
        render_directions(records),
        "\n## Errors and recovery events\n",
        render_events_table(records),
    ]
    report = "\n".join(sections)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    if results_table_path:
        with open(results_table_path, "w", encoding="utf-8") as fh:
            fh.write(render_results_table(records, test_metrics=test_metrics))
    return report


if __name__ == "__main__":
    run_log = os.path.join(ROOT, "artifacts", "run_log.jsonl")
    out = os.path.join(ROOT, "artifacts", "report.md")
    table = os.path.join(ROOT, "artifacts", "results_table.md")
    render_report(run_log, out, table)
    print(f"wrote {out} and {table}")
