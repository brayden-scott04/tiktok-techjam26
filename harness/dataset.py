"""Authoritative data cache.

kit.data.load() only parses 7 columns (date, user_id, video_id, author_id, tab,
duration_ms, long_view) — not enough for the richer solution contract. This
module re-reads the raw CSVs independently to pick up every column, but is
built to produce **identical split membership and row order** to
kit.data.load(): same two log files, same file order, same per-row date filter.
scripts/build_cache.py asserts this equivalence element-wise; nothing downstream
is trusted until that check passes (Phase 1 exit criterion 5).

This is harness-side preparation code — it runs with full file access, before
any sandboxed solution executes. It is never imported by harness/node_entry.py.
"""
import csv
import os

import numpy as np

from kit.data import SPLITS, LABEL

LOG_FILES = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")

# Columns kit.data.load() does not parse but the solution contract exposes.
OUTCOME_INT_FIELDS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate"]

# Every column that carries outcome information and must be masked to -1 for
# any row outside the train split. `long_view` is the label; the rest are
# post-impression feedback that is not knowable at ranking time.
ALL_OUTCOME_FIELDS = OUTCOME_INT_FIELDS + ["long_view", "play_time_ms"]

# Curated static feature columns, matching kit/ablation_features.py's own
# USER_FE / VID_FE lists (already identified by the starter kit as the ones
# worth exposing). Fit across the FULL feature file, not train-only: these are
# fixed categorical dictionaries (e.g. activity-level buckets), not identity
# vocabularies, so using the full file leaks no split information.
USER_FEAT_COLS = [
    "follow_user_num_range", "register_days_range", "fans_user_num_range",
    "friend_user_num_range", "user_active_degree",
]
VIDEO_FEAT_COLS = ["music_id", "video_type", "upload_type"]


def _read_video_meta(data_dir):
    vid2author, vid2feat = {}, {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
            vid2feat[r["video_id"]] = {
                "video_type": r["video_type"],
                "upload_type": r["upload_type"],
                "music_id": r["music_id"],
                "video_duration": r["video_duration"],
            }
    return vid2author, vid2feat


def _read_user_features(data_dir):
    out = {}
    with open(os.path.join(data_dir, "user_features_pure.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["user_id"]] = r
    return out


def _fit_vocab(values):
    vocab = {}
    for v in values:
        if v not in vocab:
            vocab[v] = len(vocab)
    return vocab


def _build_user_feat_arrays(user_feats_raw, user_vocab, n_users):
    """Static demographic-style user features (bucketed activity/follower
    ranges), matching kit/ablation_features.py's USER_FE list. Column vocabs
    are fit across the FULL feature file (fixed categorical dictionaries, not
    identity vocabularies), so this leaks no split information. Users with no
    row in user_features_pure.csv (shouldn't happen for KuaiRand-Pure, but
    defensive) get each column's UNK slot."""
    arrays, vocab_sizes = {}, {}
    for col in USER_FEAT_COLS:
        vocab = _fit_vocab([row[col] for row in user_feats_raw.values()])
        arr = np.full(n_users, len(vocab), dtype=np.int32)
        for uid, uidx in user_vocab.items():
            row = user_feats_raw.get(uid)
            if row is not None:
                arr[uidx] = vocab.get(row[col], len(vocab))
        arrays[col] = arr
        vocab_sizes[col] = len(vocab) + 1
    return arrays, vocab_sizes


def _build_video_feat_arrays(vid2feat, video_vocab, n_videos):
    """Static video features (music/type/upload_type + duration), matching
    kit/ablation_features.py's VID_FE list (minus author_id, already its own
    identity field). Same full-file vocab-fitting rationale as user feats."""
    arrays, vocab_sizes = {}, {}
    for col in VIDEO_FEAT_COLS:
        vocab = _fit_vocab([f[col] for f in vid2feat.values()])
        arr = np.full(n_videos, len(vocab), dtype=np.int32)
        for vid, vidx in video_vocab.items():
            feat = vid2feat.get(vid)
            if feat is not None:
                arr[vidx] = vocab.get(feat[col], len(vocab))
        arrays[col] = arr
        vocab_sizes[col] = len(vocab) + 1

    dur = np.zeros(n_videos, dtype=np.float32)
    for vid, vidx in video_vocab.items():
        feat = vid2feat.get(vid)
        if feat is not None:
            try:
                dur[vidx] = float(feat["video_duration"])
            except (KeyError, ValueError):
                pass
    arrays["video_duration"] = dur
    return arrays, vocab_sizes


def build_cache(data_dir):
    """Read both log_standard files in file order, split by date exactly as
    kit.data.SPLITS defines (disjoint ranges, so first-match assignment is
    equivalent to kit.data.load()'s independent per-split filter), and encode
    into numpy arrays. Vocabularies are fit on TRAIN ONLY with a trailing UNK
    slot, matching kit/data.py's convention.
    """
    vid2author, vid2feat = _read_video_meta(data_dir)

    by_split = {name: [] for name in SPLITS}
    for fname in LOG_FILES:
        with open(os.path.join(data_dir, fname), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                d = int(r["date"])
                for name, (lo, hi) in SPLITS.items():
                    if lo <= d <= hi:
                        by_split[name].append(r)
                        break

    train_rows = by_split["train"]
    user_vocab = _fit_vocab([r["user_id"] for r in train_rows])
    video_vocab = _fit_vocab([r["video_id"] for r in train_rows])
    author_vocab = _fit_vocab([vid2author.get(r["video_id"], "UNK") for r in train_rows])
    tab_vocab = _fit_vocab([r["tab"] for r in train_rows])

    def idx(vocab, key):
        return vocab.get(key, len(vocab))  # unseen -> UNK slot at len(vocab)

    cache = {}
    for name, rws in by_split.items():
        c = {
            "date": np.array([int(r["date"]) for r in rws], dtype=np.int32),
            "user_idx": np.array([idx(user_vocab, r["user_id"]) for r in rws], dtype=np.int32),
            "video_idx": np.array([idx(video_vocab, r["video_id"]) for r in rws], dtype=np.int32),
            "author_idx": np.array(
                [idx(author_vocab, vid2author.get(r["video_id"], "UNK")) for r in rws], dtype=np.int32
            ),
            "tab_idx": np.array([idx(tab_vocab, r["tab"]) for r in rws], dtype=np.int32),
            "duration_ms": np.array([float(r["duration_ms"]) for r in rws], dtype=np.float32),
            "hourmin": np.array([int(r["hourmin"]) for r in rws], dtype=np.int32),
            "time_ms": np.array([int(r["time_ms"]) for r in rws], dtype=np.int64),
            "play_time_ms": np.array([float(r["play_time_ms"]) for r in rws], dtype=np.float32),
            # int32, not int8: kit.evaluate.auc() computes npos*(npos+1), which
            # overflows an int8 accumulator for any user with >~11 positives
            # under numpy 2.x's NEP 50 type-promotion rules. Discovered via the
            # eval_valid round-trip check below -- see harness/eval_server.py.
            "long_view": np.array([1 if r[LABEL] != "0" else 0 for r in rws], dtype=np.int32),
            # raw strings, kept only for the kit.data.load() equivalence check
            # and for writing the submission (row_id/user_id/video_id alignment)
            "user_id_raw": np.array([r["user_id"] for r in rws], dtype=object),
            "video_id_raw": np.array([r["video_id"] for r in rws], dtype=object),
            "author_id_raw": np.array([vid2author.get(r["video_id"], "UNK") for r in rws], dtype=object),
            "tab_raw": np.array([r["tab"] for r in rws], dtype=object),
        }
        for f in OUTCOME_INT_FIELDS:
            c[f] = np.array([int(r[f]) for r in rws], dtype=np.int32)  # see long_view comment above
        cache[name] = c

    n_users, n_videos = len(user_vocab) + 1, len(video_vocab) + 1
    user_feats_raw = _read_user_features(data_dir)
    user_feat_arrays, user_feat_vocab_sizes = _build_user_feat_arrays(user_feats_raw, user_vocab, n_users)
    video_feat_arrays, video_feat_vocab_sizes = _build_video_feat_arrays(vid2feat, video_vocab, n_videos)

    meta = {
        "n_users": n_users,
        "n_videos": n_videos,
        "n_authors": len(author_vocab) + 1,
        "n_tabs": len(tab_vocab) + 1,
        "vid2feat": vid2feat,
        "user_feats_raw": user_feats_raw,
        "user_vocab_size": len(user_vocab),
        "video_vocab_size": len(video_vocab),
        "author_vocab_size": len(author_vocab),
        "tab_vocab_size": len(tab_vocab),
        "user_feat_arrays": user_feat_arrays,
        "video_feat_arrays": video_feat_arrays,
        "user_feat_vocab_sizes": user_feat_vocab_sizes,
        "video_feat_vocab_sizes": video_feat_vocab_sizes,
    }
    return cache, meta


def sanitize_cache(cache):
    """Return a deep copy in which every outcome column for every non-train
    row is overwritten with -1. This is what harness/node_entry.py loads —
    the child process never has a copy of the cache that ever held a true
    valid/test label."""
    sanitized = {}
    for split, c in cache.items():
        c2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in c.items()}
        if split != "train":
            n = len(c2["date"])
            for f in ALL_OUTCOME_FIELDS:
                c2[f] = np.full(n, -1, dtype=c2[f].dtype)
        sanitized[split] = c2
    return sanitized


def verify_against_kit(cache, data_dir):
    """Assert element-wise equivalence with the organizers' own kit.data.load()
    across all rows. This is Phase 1 exit criterion 5 and licenses using the
    fast cache everywhere downstream instead of re-parsing with the kit loader.
    Raises AssertionError with a specific message on any mismatch.
    """
    from kit.data import load as kit_load

    kit_splits = kit_load(data_dir)
    for name, kit_rows in kit_splits.items():
        c = cache[name]
        assert len(kit_rows) == len(c["date"]), (
            f"{name}: kit has {len(kit_rows)} rows, our cache has {len(c['date'])}"
        )
        for i, (date, user_id, video_id, author_id, tab, duration_ms, label) in enumerate(kit_rows):
            assert date == c["date"][i], f"{name}[{i}] date mismatch: {date} vs {c['date'][i]}"
            assert user_id == c["user_id_raw"][i], f"{name}[{i}] user_id mismatch"
            assert video_id == c["video_id_raw"][i], f"{name}[{i}] video_id mismatch"
            assert author_id == c["author_id_raw"][i], f"{name}[{i}] author_id mismatch"
            assert tab == c["tab_raw"][i], f"{name}[{i}] tab mismatch"
            assert abs(duration_ms - float(c["duration_ms"][i])) < 1e-6, f"{name}[{i}] duration_ms mismatch"
            assert label == int(c["long_view"][i]), f"{name}[{i}] label mismatch"
    return True


CACHE_ARRAY_KEYS = [
    "date", "user_idx", "video_idx", "author_idx", "tab_idx", "duration_ms",
    "hourmin", "time_ms", "play_time_ms", "long_view",
    "user_id_raw", "video_id_raw", "author_id_raw", "tab_raw",
] + OUTCOME_INT_FIELDS


def save_cache(cache, meta, path):
    """Persist to a single .npz so later scripts/nodes don't re-parse CSVs
    (1.4M rows is a few seconds of pure-Python CSV parsing per split build)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {}
    for split in cache:
        for key in CACHE_ARRAY_KEYS:
            payload[f"{split}__{key}"] = cache[split][key]
    for col, arr in meta["user_feat_arrays"].items():
        payload[f"userfeat__{col}"] = arr
    for col, arr in meta["video_feat_arrays"].items():
        payload[f"videofeat__{col}"] = arr
    np.savez_compressed(path + ".npz", **payload)

    import json
    meta_json = {
        k: v for k, v in meta.items()
        if k not in ("vid2feat", "user_feats_raw", "user_feat_arrays", "video_feat_arrays")
    }
    with open(path + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta_json, fh)
    with open(path + ".vidfeat.json", "w", encoding="utf-8") as fh:
        json.dump(meta["vid2feat"], fh)
    with open(path + ".userfeat.json", "w", encoding="utf-8") as fh:
        json.dump(meta["user_feats_raw"], fh)


def load_cache(path, skip_raw_meta=False):
    """skip_raw_meta=True skips loading vid2feat/user_feats_raw (the two large
    per-id dict-of-dicts), which node_entry.py never needs — context.py builds
    ctx.user_feats/video_feats from the userfeat__*/videofeat__* npz arrays."""
    data = np.load(path + ".npz", allow_pickle=True)
    splits = set(k.split("__", 1)[0] for k in data.files if k.split("__", 1)[0] in SPLITS)
    cache = {split: {key: data[f"{split}__{key}"] for key in CACHE_ARRAY_KEYS} for split in splits}

    user_feat_arrays = {k[len("userfeat__"):]: data[k] for k in data.files if k.startswith("userfeat__")}
    video_feat_arrays = {k[len("videofeat__"):]: data[k] for k in data.files if k.startswith("videofeat__")}

    import json
    with open(path + ".meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["user_feat_arrays"] = user_feat_arrays
    meta["video_feat_arrays"] = video_feat_arrays
    if not skip_raw_meta:
        with open(path + ".vidfeat.json", encoding="utf-8") as fh:
            meta["vid2feat"] = json.load(fh)
        with open(path + ".userfeat.json", encoding="utf-8") as fh:
            meta["user_feats_raw"] = json.load(fh)
    return cache, meta
