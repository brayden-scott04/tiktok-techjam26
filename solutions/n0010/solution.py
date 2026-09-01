import numpy as np

# hypothesis: Train a single factorization machine to regress the signed log-watchtime margin relative to an empirically learned duration-dependent long-view threshold.
# changes: Added split-column access compatible with both mapping-style and Split-object inputs.

def fit_predict(ctx) -> dict:
    def col(split, name):
        try:
            return split[name]
        except (TypeError, KeyError, AttributeError):
            return getattr(split, name)

    rng = np.random.default_rng(int(ctx.seed))
    tr, va, te = ctx.splits["train"], ctx.splits["valid"], ctx.splits["test"]
    n_all = len(col(tr, "date"))
    if ctx.smoke and n_all > 60000:
        keep = rng.choice(n_all, size=60000, replace=False)
    else:
        keep = np.arange(n_all, dtype=np.int64)

    y = np.asarray(col(tr, "long_view"))[keep].astype(np.float32)
    play = np.nan_to_num(np.asarray(col(tr, "play_time_ms"))[keep].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    dur = np.nan_to_num(np.asarray(col(tr, "duration_ms"))[keep].astype(np.float64), nan=1.0, posinf=1.0, neginf=1.0)
    play = np.maximum(play, 0.0)
    dur = np.maximum(dur, 1.0)

    edges = np.unique(np.quantile(dur, np.linspace(0, 1, 33)[1:-1])) if len(dur) else np.empty(0)
    db = np.searchsorted(edges, dur, side="right")
    nb = len(edges) + 1
    cuts = np.empty(nb, dtype=np.float64)
    fallback = max(100.0, float(np.median(np.minimum(dur, 18000.0))))
    for b in range(nb):
        ii = np.flatnonzero(db == b)
        if not len(ii):
            cuts[b] = fallback
            continue
        order = np.argsort(play[ii], kind="stable")
        ps, ys = play[ii][order], y[ii][order].astype(np.int64)
        pp = np.r_[0, np.cumsum(ys)]
        nn = np.arange(len(ii) + 1) - pp
        starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
        splits = np.r_[starts, len(ii)]
        errors = pp[splits] + (nn[-1] - nn[splits])
        k = int(splits[int(np.argmin(errors))])
        if k == 0:
            cut = max(1.0, float(ps[0]) - 1.0)
        elif k == len(ii):
            cut = float(ps[-1]) + 1.0
        else:
            cut = 0.5 * (float(ps[k - 1]) + float(ps[k]))
        cuts[b] = max(1.0, cut)

    margin = np.log1p(play) - np.log1p(cuts[db])
    mag = np.clip(np.abs(margin), 0.05, 3.0)
    target = np.where(y > 0.5, mag, -mag).astype(np.float32)

    dedges = np.unique(np.quantile(dur, np.linspace(0, 1, 11)[1:-1])) if len(dur) else np.empty(0)
    duration_bucket = np.searchsorted(dedges, dur, side="right")
    nd = len(dedges) + 1
    ou = 0
    ov = ou + int(ctx.n_users)
    oa = ov + int(ctx.n_videos)
    ot = oa + int(ctx.n_authors)
    od = ot + int(ctx.n_tabs)
    nf = od + nd
    x = np.column_stack((
        np.asarray(col(tr, "user_idx"))[keep].astype(np.int64) + ou,
        np.asarray(col(tr, "video_idx"))[keep].astype(np.int64) + ov,
        np.asarray(col(tr, "author_idx"))[keep].astype(np.int64) + oa,
        np.asarray(col(tr, "tab_idx"))[keep].astype(np.int64) + ot,
        duration_bucket.astype(np.int64) + od))

    gv = float(np.var(target)) + 1e-4
    scale = np.empty(nd, dtype=np.float32)
    for b in range(nd):
        z = target[duration_bucket == b]
        v = (len(z) * float(np.var(z)) + 200.0 * gv) / (len(z) + 200.0) if len(z) else gv
        scale[b] = np.float32(np.clip(np.sqrt(v), 0.25, 3.0))

    kdim = 16
    w = np.zeros(nf, dtype=np.float32)
    V = rng.normal(0, 0.02, (nf, kdim)).astype(np.float32)
    mw, vw = np.zeros_like(w), np.zeros_like(w)
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    bias = float(np.mean(target)) if len(target) else 0.0
    mb = vb = 0.0
    beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 0.012
    step = 0
    best_primary = -1.0
    best_w = best_V = best_valid = None
    best_bias = bias
    best_std = False
    patience = 0
    max_epochs = 3 if ctx.smoke else 14
    batch_size = 4096 if ctx.smoke else 8192

    def predict(xx, bias0, w0, V0):
        vr = V0[xx]
        sv = np.sum(vr, axis=1)
        return bias0 + np.sum(w0[xx], axis=1) + 0.5 * np.sum(sv * sv - np.sum(vr * vr, axis=1), axis=1)

    for epoch in range(max_epochs):
        order = rng.permutation(len(target))
        batches = 0
        for start in range(0, len(target), batch_size):
            if ctx.elapsed_seconds() > max(1.0, float(ctx.max_seconds) - 8.0):
                break
            ids = order[start:start + batch_size]
            xb, tb = x[ids], target[ids]
            vr = V[xb]
            sv = np.sum(vr, axis=1)
            pred = bias + np.sum(w[xb], axis=1) + 0.5 * np.sum(sv * sv - np.sum(vr * vr, axis=1), axis=1)
            ds = np.clip(pred - tb, -0.75, 0.75).astype(np.float32) / max(1, len(ids))
            step += 1
            c1, c2 = 1 - beta1 ** step, 1 - beta2 ** step
            gb = float(np.sum(ds))
            mb = beta1 * mb + (1 - beta1) * gb; vb = beta2 * vb + (1 - beta2) * gb * gb
            bias -= lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
            for f in range(5):
                idsf = xb[:, f]
                uniq, inv = np.unique(idsf, return_inverse=True)
                gw = np.bincount(inv, weights=ds, minlength=len(uniq)).astype(np.float32) + 2e-6 * w[uniq]
                local = ds[:, None] * (sv - vr[:, f])
                gV = np.empty((len(uniq), kdim), dtype=np.float32)
                for j in range(kdim):
                    gV[:, j] = np.bincount(inv, weights=local[:, j], minlength=len(uniq))
                gV += 2e-6 * V[uniq]
                mw[uniq] = beta1 * mw[uniq] + (1-beta1) * gw; vw[uniq] = beta2 * vw[uniq] + (1-beta2) * gw * gw
                mV[uniq] = beta1 * mV[uniq] + (1-beta1) * gV; vV[uniq] = beta2 * vV[uniq] + (1-beta2) * gV * gV
                w[uniq] -= lr * (mw[uniq] / c1) / (np.sqrt(vw[uniq] / c2) + eps)
                V[uniq] -= lr * (mV[uniq] / c1) / (np.sqrt(vV[uniq] / c2) + eps)
                
                
            batches += 1
        vd = np.maximum(np.nan_to_num(np.asarray(col(va, "duration_ms")), nan=1.0, posinf=1.0, neginf=1.0), 1.0)
        vdb = np.searchsorted(dedges, vd, side="right")
        vx = np.column_stack((np.asarray(col(va,"user_idx"),dtype=np.int64)+ou, np.asarray(col(va,"video_idx"),dtype=np.int64)+ov, np.asarray(col(va,"author_idx"),dtype=np.int64)+oa, np.asarray(col(va,"tab_idx"),dtype=np.int64)+ot, vdb+od))
        scores = np.nan_to_num(predict(vx, bias, w, V), nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)
        raw = ctx.eval_valid(scores); std = ctx.eval_valid(scores / scale[vdb])
        use_std = std["primary"] > raw["primary"]; met = std if use_std else raw; chosen = scores / scale[vdb] if use_std else scores
        if met["primary"] > best_primary:
            best_primary = float(met["primary"]); best_w=w.copy(); best_V=V.copy(); best_bias=bias; best_std=use_std; best_valid=chosen.copy(); patience=0
        else: patience += 1
        if patience >= 4 or ctx.elapsed_seconds() > max(1.0, float(ctx.max_seconds)-8.0): break

    if best_w is None: best_w, best_V, best_bias, best_valid = w, V, bias, scores
    td = np.maximum(np.nan_to_num(np.asarray(col(te,"duration_ms")), nan=1.0, posinf=1.0, neginf=1.0), 1.0)
    tdb = np.searchsorted(dedges, td, side="right")
    tx = np.column_stack((np.asarray(col(te,"user_idx"),dtype=np.int64)+ou, np.asarray(col(te,"video_idx"),dtype=np.int64)+ov, np.asarray(col(te,"author_idx"),dtype=np.int64)+oa, np.asarray(col(te,"tab_idx"),dtype=np.int64)+ot, tdb+od))
    test = predict(tx, best_bias, best_w, best_V)
    if best_std: test = test / scale[tdb]
    return {"valid": np.asarray(best_valid, dtype=np.float32), "test": np.nan_to_num(test, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)}
