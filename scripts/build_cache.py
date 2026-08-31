"""Phase 1: build the authoritative cache from raw CSVs and verify it is
element-wise identical to kit.data.load(). Run with: python -X utf8 -m scripts.build_cache
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import dataset as ds
from harness import task_spec as spec
from harness import context as ctxmod
from harness.eval_server import EvalServer, eval_valid_client


def main():
    data_dir = os.environ.get("KR_DATA_ROOT")
    if not data_dir or not os.path.isdir(data_dir):
        raise SystemExit(
            f"KR_DATA_ROOT is not set to a valid directory (got {data_dir!r}). "
            "Set it in .env or the environment."
        )

    print(f"Building cache from {data_dir} ...")
    t0 = time.time()
    cache, meta = ds.build_cache(data_dir)
    build_s = time.time() - t0
    print(f"  built in {build_s:.1f}s")

    sizes = {name: int(len(cache[name]["date"])) for name in cache}
    print(f"  split sizes: {sizes}")
    expected = spec.EXPECTED_SPLIT_SIZES
    size_ok = sizes == expected
    print(f"  expected:    {expected}  -> {'OK' if size_ok else 'MISMATCH'}")

    print("Verifying element-wise equivalence with kit.data.load() ...")
    t0 = time.time()
    ds.verify_against_kit(cache, data_dir)
    verify_s = time.time() - t0
    print(f"  verified OK in {verify_s:.1f}s")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(root, "artifacts", "cache", "kuairand_pure")
    ds.save_cache(cache, meta, cache_path)
    print(f"Saved authoritative cache to {cache_path}.npz")

    print("Building sanitized cache (valid/test outcome columns -> -1) ...")
    sanitized = ds.sanitize_cache(cache)
    ctxmod.assert_sanitized(sanitized)
    print("  sanitizer self-check: PASS (every non-train outcome column is -1)")
    sanitized_meta = {k: v for k, v in meta.items() if k not in ("vid2feat", "user_feats_raw")}
    sanitized_path = os.path.join(root, "artifacts", "cache", "kuairand_pure_sanitized")
    ds.save_cache(sanitized, {**meta, **sanitized_meta}, sanitized_path)
    print(f"Saved sanitized cache to {sanitized_path}.npz  <- this is the ONLY path node_entry.py may read")

    # Feature columns must survive sanitization unchanged.
    feature_cols = ["date", "user_idx", "video_idx", "author_idx", "tab_idx", "duration_ms", "hourmin", "time_ms"]
    for split in ("valid", "test"):
        for col in feature_cols:
            assert (cache[split][col] == sanitized[split][col]).all(), f"{split}.{col} was altered by sanitization!"
    print("  feature columns verified unchanged by sanitization")

    print("Verifying ctx.eval_valid() round-trip against kit.evaluate.evaluate() directly ...")
    from kit.evaluate import evaluate as kit_evaluate
    import numpy as np

    valid_c = cache["valid"]
    fm_scores = np.random.default_rng(0).normal(size=len(valid_c["date"])).astype(np.float32)
    direct = kit_evaluate(
        list(valid_c["user_id_raw"]), [int(x) for x in valid_c["long_view"]], fm_scores
    )

    server = EvalServer(list(valid_c["user_id_raw"]), list(valid_c["long_view"]), token="verify-token", call_cap=5)
    port = server.start()
    try:
        via_server = eval_valid_client("127.0.0.1", port, "verify-token", fm_scores)
    finally:
        server.stop()

    eval_ok = (
        abs(direct["GAUC"] - via_server["GAUC"]) < 1e-12
        and abs(direct["nDCG@5"] - via_server["nDCG@5"]) < 1e-12
        and abs(direct["primary"] - via_server["primary"]) < 1e-12
    )
    print(f"  direct={direct['primary']:.6f} via_server={via_server['primary']:.6f} -> {'OK' if eval_ok else 'MISMATCH'}")
    assert eval_ok, "eval_valid server round-trip does not match kit.evaluate.evaluate() directly"

    result = {
        "split_sizes": sizes,
        "expected_split_sizes": expected,
        "split_sizes_ok": size_ok,
        "kit_equivalence_ok": True,
        "build_seconds": build_s,
        "verify_seconds": verify_s,
        "n_users": meta["n_users"],
        "n_videos": meta["n_videos"],
        "n_authors": meta["n_authors"],
        "n_tabs": meta["n_tabs"],
        "sanitizer_self_check_ok": True,
        "sanitizer_features_unchanged_ok": True,
        "eval_valid_server_matches_direct_ok": eval_ok,
        "eval_valid_direct_primary": direct["primary"],
        "eval_valid_via_server_primary": via_server["primary"],
    }
    os.makedirs(os.path.join(root, "artifacts"), exist_ok=True)
    out_path = os.path.join(root, "artifacts", "phase1_verification.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            existing = json.load(fh)
    existing["build_cache"] = result
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)

    if not size_ok:
        raise SystemExit("Split sizes did not match expected values. Stop here.")
    print("Phase 1 cache build + verification: PASS")


if __name__ == "__main__":
    main()
