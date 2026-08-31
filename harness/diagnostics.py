"""Computes the diagnostics evidence pack shown to the agent each iteration.
Runs in the PARENT process (it needs the true validation labels), never in
the sandbox -- this is what turns "guess a technique" into "read the failure
mode, target it." Scope note: this is a deliberately focused first cut
(impression-count buckets, within-user score degeneracy, per-date stability)
rather than an exhaustive diagnostic suite -- these three are the ones with
the clearest causal link to a next research direction.
"""
import collections

import numpy as np

from kit.evaluate import evaluate

IMPRESSION_BUCKETS = [(1, 5), (6, 20), (21, 100), (101, float("inf"))]


def _bucket_label(lo, hi):
    return f"{lo}-{hi}" if hi != float("inf") else f"{lo}+"


def compute_diagnostics(user_ids, labels, scores, dates=None):
    by_user = collections.defaultdict(list)
    for i, u in enumerate(user_ids):
        by_user[u].append(i)

    by_impression_bucket = {}
    for lo, hi in IMPRESSION_BUCKETS:
        idxs = [i for u, rows in by_user.items() if lo <= len(rows) <= hi for i in rows]
        if not idxs:
            continue
        sub_uids = [user_ids[i] for i in idxs]
        sub_labels = [labels[i] for i in idxs]
        sub_scores = [scores[i] for i in idxs]
        m = evaluate(sub_uids, sub_labels, sub_scores)
        by_impression_bucket[_bucket_label(lo, hi)] = {
            "users": m["users"], "rows": m["rows"],
            "GAUC": m["GAUC"], "nDCG@5": m["nDCG@5"], "primary": m["primary"],
        }

    within_user_stds = []
    for u, rows in by_user.items():
        if len(rows) > 1:
            within_user_stds.append(float(np.std([scores[i] for i in rows])))
    score_within_user_std_mean = float(np.mean(within_user_stds)) if within_user_stds else 0.0
    degenerate_user_frac = (
        float(np.mean([s < 1e-9 for s in within_user_stds])) if within_user_stds else 0.0
    )

    per_date_primary = None
    if dates is not None:
        by_date = collections.defaultdict(list)
        for i, d in enumerate(dates):
            by_date[int(d)].append(i)
        per_date_primary = {}
        for d, idxs in sorted(by_date.items()):
            sub_uids = [user_ids[i] for i in idxs]
            sub_labels = [labels[i] for i in idxs]
            sub_scores = [scores[i] for i in idxs]
            m = evaluate(sub_uids, sub_labels, sub_scores)
            per_date_primary[d] = round(m["primary"], 4)

    overall = evaluate(user_ids, labels, scores)

    return {
        "overall": {"GAUC": overall["GAUC"], "nDCG@5": overall["nDCG@5"], "primary": overall["primary"]},
        "by_impression_bucket": by_impression_bucket,
        "score_within_user_std_mean": score_within_user_std_mean,
        "degenerate_constant_score_user_frac": degenerate_user_frac,
        "per_date_primary": per_date_primary,
    }


def compute_comparative_diagnostics(candidate_scores, incumbent_scores, user_ids, labels, dates=None):
    """Per-impression-bucket delta between a candidate and the current
    incumbent, plus whether the candidate's within-user score variance
    collapsed relative to the incumbent's. Shown for the just-completed
    node against whatever it was compared against -- strictly more
    information (bucketed data the harness already computes both sides of),
    never a diagnosis of *why* in prose, so it doesn't hand over content.
    """
    cand = compute_diagnostics(user_ids, labels, candidate_scores, dates=dates)
    incu = compute_diagnostics(user_ids, labels, incumbent_scores, dates=dates)

    bucket_deltas = {}
    for label in set(cand["by_impression_bucket"]) | set(incu["by_impression_bucket"]):
        c = cand["by_impression_bucket"].get(label)
        i = incu["by_impression_bucket"].get(label)
        if c is None or i is None:
            continue
        bucket_deltas[label] = {
            "candidate_primary": c["primary"], "incumbent_primary": i["primary"],
            "delta": c["primary"] - i["primary"], "users": c["users"],
        }

    return {
        "candidate_overall": cand["overall"],
        "incumbent_overall": incu["overall"],
        "overall_delta": cand["overall"]["primary"] - incu["overall"]["primary"],
        "bucket_deltas": bucket_deltas,
        "candidate_within_user_std": cand["score_within_user_std_mean"],
        "incumbent_within_user_std": incu["score_within_user_std_mean"],
        "candidate_degenerate_frac": cand["degenerate_constant_score_user_frac"],
    }


def render_comparative_diagnostics_markdown(comp):
    lines = [
        f"Overall primary: candidate {comp['candidate_overall']['primary']:.4f} vs incumbent "
        f"{comp['incumbent_overall']['primary']:.4f} (delta {comp['overall_delta']:+.4f})",
        "",
        "| bucket | candidate primary | incumbent primary | delta | users |",
        "|---|---|---|---|---|",
    ]
    for label, d in sorted(comp["bucket_deltas"].items()):
        lines.append(
            f"| {label} | {d['candidate_primary']:.4f} | {d['incumbent_primary']:.4f} "
            f"| {d['delta']:+.4f} | {d['users']} |"
        )
    lines.append("")
    lines.append(
        f"Within-user score std: candidate {comp['candidate_within_user_std']:.4f} vs incumbent "
        f"{comp['incumbent_within_user_std']:.4f}"
    )
    if comp["candidate_degenerate_frac"] > 0.05:
        lines.append(
            f"Note: {comp['candidate_degenerate_frac']:.1%} of the candidate's users got a "
            f"near-constant score across their impressions (little to no within-user discrimination)."
        )
    return "\n".join(lines)


def render_diagnostics_markdown(diag):
    lines = ["| bucket | users | GAUC | nDCG@5 | primary |", "|---|---|---|---|---|"]
    for label, d in diag["by_impression_bucket"].items():
        lines.append(f"| {label} impressions | {d['users']} | {d['GAUC']:.4f} | {d['nDCG@5']:.4f} | {d['primary']:.4f} |")
    md = "\n".join(lines)
    md += f"\n\nWithin-user score std (mean across users with >1 impression): {diag['score_within_user_std_mean']:.4f}"
    md += f"\nFraction of users with a degenerate (near-constant) score across their impressions: {diag['degenerate_constant_score_user_frac']:.2%}"
    if diag.get("per_date_primary"):
        vals = list(diag["per_date_primary"].values())
        md += f"\nPer-date primary range across the validation window: {min(vals):.4f} - {max(vals):.4f} (std {np.std(vals):.4f})"
    return md
