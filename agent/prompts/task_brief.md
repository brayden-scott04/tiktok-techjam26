# Task

Rank each user's logged video impressions by predicted relevance. Relevance label: `long_view` (binary, native column). This is **within-user ranking over already-logged impressions**, not full-catalog retrieval — you never need to consider videos outside what's already in a split.

# Metrics — pinned conventions (do not deviate)

Primary score = mean(GAUC, nDCG@5). Exact scoring semantics:

- **GAUC**: computed only over users with `0 < positive_count < impression_count` (i.e. users with at least one positive and at least one negative among their impressions), weighted by each user's positive count. Users who are all-positive or all-negative contribute nothing to GAUC.
- **nDCG@5**: computed for every user, including those with zero positives — a zero-positive user contributes **0.0** to the average (not excluded). Gain function is `2^relevance - 1` (identity for a binary label).
- Final primary = mean of the two.

# Headroom — where you actually stand (validation split numbers)

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (sanity floor) | 0.4993 | 0.4675 | 0.4834 |
| item popularity | 0.6387 | 0.5227 | 0.5807 |
| **official FM baseline (your starting point)** | **0.6674** | **0.5357** | **0.6016** |
| oracle (true labels used as scores) | 1.0000 | 0.6968 | 0.8484 |

The ceiling is **0.8484, not 1.0** — 27.1% of users in this dataset have zero positive impressions (their nDCG is capped at 0 no matter what any model does), and 9.2% are all-positive (excluded from GAUC entirely). Realistic headroom above the baseline is about **0.25 of primary score**, not 0.45. Judge your own progress against 0.8484, not against 1.0.

# What node 0 (your starting point) already does

A factorization machine over 5 categorical fields (`user_id`, `video_id`, `author_id`, `tab`, a duration bucket derived from a 10-way train-quantile split of `duration_ms`), jointly embedded in one flat table, trained with **pointwise binary cross-entropy** on `long_view`, Adam-optimized, early-stopped on validation primary with patience 4.

# Already tried, and measured not to help (don't re-spend iterations rediscovering these)

- **Adding more static feature columns** (`music_id`, `video_type`, `upload_type`, plus 5 bucketed user-side columns like `follow_user_num_range`) to the same pointwise-FM setup: primary 0.5940 vs 0.5950 for the plain 5-field version on an earlier internal split — a wash, possibly slightly negative.
- **Embedding dimension** k=8 / 16 / 32: 0.5895 / 0.5902 / 0.5887 on that same internal split — essentially flat. Capacity is not the bottleneck here.
- **A structural fact, not just an empirical one**: because ranking happens *within each user*, any feature that is constant across a single user's rows cannot change that user's ranking at all — its coefficient affects every one of that user's candidate scores identically. This means **pure user-side features contribute exactly zero in first order**. They can only matter through an *interaction* with an item-side feature (e.g. a user-activity-level × video-type cross term), never on their own.

# Data available to you (train split only — see hard rules)

Per-row log columns: `user_id`, `video_id`, `date`, `hourmin`, `time_ms`, `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `long_view`, `play_time_ms`, `duration_ms`, `tab`. `duration_ms` is the video's own length (known at ranking time, safe to use everywhere). The six `is_*` flags and `play_time_ms` are **post-impression outcomes** — safe to use as auxiliary *training targets* on train rows, but they did not happen yet at ranking time, so using them as *input features* anywhere (including on train, if your inference-time code would need them) is a modeling error, not just a rule violation.

Static side info: per-video (`author_id`, `video_type`, `upload_type`, `music_id`, duration) and per-user (bucketed activity/follower/registration-age ranges) features are available via `ctx.video_feats` / `ctx.user_feats`.

# Training-data rule (hard constraint, structurally enforced)

You may only train on the `train` split. `valid` and `test` carry no labels in this process — not because you're asked not to look, but because they are not there. Do not design around a plan that requires refitting on validation data before a final prediction; it is not available to you at any point, including a hypothetical "final" iteration.
