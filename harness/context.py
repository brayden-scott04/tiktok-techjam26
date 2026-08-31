"""Builds the Ctx object handed to a solution's fit_predict(ctx).

Everything here runs INSIDE the sandboxed child process (harness/node_entry.py
calls build_ctx once at startup). The child has only ever loaded the SANITIZED
cache (harness/dataset.sanitize_cache output) — outcome columns for valid/test
rows are physically -1 before this module ever sees them. ctx.eval_valid is a
network call to harness/eval_server.EvalServer, which runs in the PARENT
process and is the only thing in the whole system holding true valid labels.

This module itself may import anything (it's trusted harness code); the
AST allowlist in harness/guards.py restricts solution.py, not this file.
"""
import os
import time
from dataclasses import dataclass, field

import numpy as np

from harness.dataset import ALL_OUTCOME_FIELDS
from harness.eval_client import eval_valid_client


@dataclass
class Split:
    date: np.ndarray
    user_idx: np.ndarray
    video_idx: np.ndarray
    author_idx: np.ndarray
    tab_idx: np.ndarray
    duration_ms: np.ndarray
    hourmin: np.ndarray
    time_ms: np.ndarray
    # Outcome columns. For valid/test these are all -1 (sanitized) by construction.
    long_view: np.ndarray
    is_click: np.ndarray
    is_like: np.ndarray
    is_follow: np.ndarray
    is_comment: np.ndarray
    is_forward: np.ndarray
    is_hate: np.ndarray
    play_time_ms: np.ndarray

    def __len__(self):
        return len(self.date)


def _split_from_cache(c):
    return Split(
        date=c["date"], user_idx=c["user_idx"], video_idx=c["video_idx"],
        author_idx=c["author_idx"], tab_idx=c["tab_idx"], duration_ms=c["duration_ms"],
        hourmin=c["hourmin"], time_ms=c["time_ms"], long_view=c["long_view"],
        is_click=c["is_click"], is_like=c["is_like"], is_follow=c["is_follow"],
        is_comment=c["is_comment"], is_forward=c["is_forward"], is_hate=c["is_hate"],
        play_time_ms=c["play_time_ms"],
    )


class EvalBudgetExceeded(Exception):
    pass


class Ctx:
    def __init__(self, splits, n_users, n_videos, n_authors, n_tabs,
                 user_feats, video_feats, user_feat_vocab_sizes, video_feat_vocab_sizes,
                 seed, smoke, max_seconds, eval_host, eval_port, eval_token,
                 log_fn):
        self.splits = splits
        self.n_users = n_users
        self.n_videos = n_videos
        self.n_authors = n_authors
        self.n_tabs = n_tabs
        self.user_feats = user_feats
        self.video_feats = video_feats
        self.user_feat_vocab_sizes = user_feat_vocab_sizes
        self.video_feat_vocab_sizes = video_feat_vocab_sizes
        self.seed = seed
        self.smoke = smoke
        self.max_seconds = max_seconds
        self._eval_host = eval_host
        self._eval_port = eval_port
        self._eval_token = eval_token
        self._log_fn = log_fn
        self._eval_calls = 0
        self._eval_time = 0.0
        self._start_time = time.time()

    def log(self, msg):
        self._log_fn(msg)

    def elapsed_seconds(self):
        return time.time() - self._start_time

    def eval_valid(self, scores):
        """Score `scores` (len == len(ctx.splits['valid'])) against the TRUE
        validation labels, which this process never holds. Returns
        {'GAUC':.., 'nDCG@5':.., 'primary':..}. Raises EvalBudgetExceeded past
        the per-run call cap (this is a normal, loggable debug signal, not a
        crash — it means "evaluate per epoch, not per batch")."""
        t0 = time.time()
        try:
            result = eval_valid_client(self._eval_host, self._eval_port, self._eval_token, scores)
        except RuntimeError as e:
            if "call cap" in str(e):
                raise EvalBudgetExceeded(str(e)) from e
            raise
        self._eval_calls += 1
        self._eval_time += time.time() - t0
        return result


def build_ctx(sanitized_cache, meta, seed, smoke, max_seconds, eval_host, eval_port, eval_token, log_fn):
    splits = {name: _split_from_cache(c) for name, c in sanitized_cache.items()}
    return Ctx(
        splits=splits,
        n_users=meta["n_users"], n_videos=meta["n_videos"],
        n_authors=meta["n_authors"], n_tabs=meta["n_tabs"],
        user_feats=meta["user_feat_arrays"], video_feats=meta["video_feat_arrays"],
        user_feat_vocab_sizes=meta["user_feat_vocab_sizes"],
        video_feat_vocab_sizes=meta["video_feat_vocab_sizes"],
        seed=seed, smoke=smoke, max_seconds=max_seconds,
        eval_host=eval_host, eval_port=eval_port, eval_token=eval_token,
        log_fn=log_fn,
    )


def assert_sanitized(sanitized_cache):
    """Defensive self-check, run once at child startup: every outcome column
    for every non-train row must be exactly -1. If this ever fails it means a
    harness bug bypassed sanitize_cache(), and the child aborts rather than
    proceeding with a solution that could see real labels."""
    for split_name, c in sanitized_cache.items():
        if split_name == "train":
            continue
        for f in ALL_OUTCOME_FIELDS:
            arr = c[f]
            if not np.all(arr == -1):
                raise RuntimeError(
                    f"SANITIZER FAILURE: {split_name}.{f} contains non -1 values. Aborting."
                )
