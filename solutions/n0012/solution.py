import numpy as np


def fit_predict(ctx) -> dict:
    def col(sp, name):
        try:
            return getattr(sp, name)
        except AttributeError:
            return sp[name]

    train = ctx.splits["train"]
    valid = ctx.splits["valid"]
    test = ctx.splits["test"]
    rng = np.random.default_rng(int(ctx.seed))

    y_all = np.asarray(col(train, "long_view"), dtype=np.float64)
    n_all = y_all.size
    dur_train = np.asarray(col(train, "duration_ms"), dtype=np.float64)
    finite = dur_train[np.isfinite(dur_train)]
    qs = np.quantile(finite, np.arange(1, 10) / 10.0) if finite.size else np.arange(1, 10, dtype=np.float64)

    sizes = [int(ctx.n_users), int(ctx.n_videos), int(ctx.n_authors), int(ctx.n_tabs), 10]
    offsets = np.zeros(5, dtype=np.int64)
    offsets[1:] = np.cumsum(np.asarray(sizes[:-1], dtype=np.int64))
    n_cat = int(sum(sizes))

    def make_ids(sp):
        vals = [
            np.clip(np.asarray(col(sp, "user_idx"), dtype=np.int64), 0, sizes[0] - 1),
            np.clip(np.asarray(col(sp, "video_idx"), dtype=np.int64), 0, sizes[1] - 1),
            np.clip(np.asarray(col(sp, "author_idx"), dtype=np.int64), 0, sizes[2] - 1),
            np.clip(np.asarray(col(sp, "tab_idx"), dtype=np.int64), 0, sizes[3] - 1),
        ]
        db = np.clip(np.searchsorted(qs, np.asarray(col(sp, "duration_ms"), dtype=np.float64), side="right"), 0, 9)
        return np.column_stack(vals + [db.astype(np.int64)]) + offsets

    ids = {"train": make_ids(train), "valid": make_ids(valid), "test": make_ids(test)}
    if ctx.smoke and n_all > 50000:
        rows = rng.choice(n_all, 50000, replace=False)
    else:
        rows = np.arange(n_all, dtype=np.int64)
    tr_ids, y = ids["train"][rows], y_all[rows]
    k = 8 if ctx.smoke else 16
    w = np.zeros(n_cat, dtype=np.float64)
    V = rng.normal(0, 0.025, (n_cat, k))
    mw = np.zeros_like(w); vw = np.zeros_like(w)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    b0 = float(np.log((y.mean() + 1e-4) / (1 - y.mean() + 1e-4)))
    mb = vb = 0.0
    t = 0
    best = (-1.0, w.copy(), V.copy(), b0)
    stale = 0
    epochs = 2 if ctx.smoke else 12

    def score(ii):
        out = np.empty(ii.shape[0], dtype=np.float64)
        for s in range(0, ii.shape[0], 50000):
            z = ii[s:s + 50000]
            q = V[z]; sm = q.sum(1)
            out[s:s + z.shape[0]] = b0 + w[z].sum(1) + 0.5 * np.sum(sm * sm - (q * q).sum(1), 1)
        return out

    for ep in range(epochs):
        order = rng.permutation(y.size)
        for start in range(0, y.size, 4096):
            take = order[start:start + 4096]; ii = tr_ids[take]; yy = y[take]
            q = V[ii]; sm = q.sum(1)
            r = b0 + w[ii].sum(1) + 0.5 * np.sum(sm * sm - (q * q).sum(1), 1)
            p = np.empty_like(r); pos = r >= 0
            p[pos] = 1 / (1 + np.exp(-r[pos]))
            e = np.exp(r[~pos]); p[~pos] = e / (1 + e)
            d = (p - yy) / max(1, yy.size)
            flat = ii.ravel(); uq, inv = np.unique(flat, return_inverse=True)
            gw = np.bincount(inv, np.repeat(d, 5), minlength=uq.size) + 2e-6 * w[uq]
            gvr = d[:, None, None] * (sm[:, None, :] - q)
            gv = np.empty((uq.size, k))
            for j in range(k):
                gv[:, j] = np.bincount(inv, gvr.reshape(-1, k)[:, j], minlength=uq.size)
            gv += 3e-6 * V[uq]
            t += 1; c1 = 1 - .9 ** t; c2 = 1 - .999 ** t
            mw[uq] = .9 * mw[uq] + .1 * gw; vw[uq] = .999 * vw[uq] + .001 * gw * gw
            mV[uq] = .9 * mV[uq] + .1 * gv; vV[uq] = .999 * vV[uq] + .001 * gv * gv
            w[uq] -= .014 * (mw[uq] / c1) / (np.sqrt(vw[uq] / c2) + 1e-8)
            V[uq] -= .014 * (mV[uq] / c1) / (np.sqrt(vV[uq] / c2) + 1e-8)
            gb = float(d.sum()); mb = .9 * mb + .1 * gb; vb = .999 * vb + .001 * gb * gb
            b0 -= .014 * (mb / c1) / (np.sqrt(vb / c2) + 1e-8)
            if ctx.elapsed_seconds() > .76 * ctx.max_seconds:
                break
        sv = score(ids["valid"]); met = ctx.eval_valid(sv); cur = float(met["primary"])
        ctx.log("epoch %d primary %.6f" % (ep + 1, cur))
        if cur > best[0] + 1e-7:
            best = (cur, w.copy(), V.copy(), b0); stale = 0
        else:
            stale += 1
        if stale >= (3 if ctx.smoke else 4) or ctx.elapsed_seconds() > .8 * ctx.max_seconds:
            break

    best_primary, w, V, b0 = best
    base = {n: score(ids[n]) for n in ("train", "valid", "test")}
    valid_scores = base["valid"].copy(); test_scores = base["test"].copy()

    if ctx.elapsed_seconds() < .88 * ctx.max_seconds and n_all > 100:
        cn = min(n_all, 20000 if ctx.smoke else 150000)
        ci = rng.choice(n_all, cn, replace=False) if cn < n_all else np.arange(n_all)
        ld = np.log1p(np.maximum(dur_train, 1.0)); lm = ld[ci].mean(); ls = ld[ci].std() + 1e-6
        tm = np.asarray(col(train, "time_ms"), dtype=np.float64); mm = tm[ci].mean(); ms = tm[ci].std() + 1e-6
        x = np.clip((ld[ci] - lm) / ls, -4, 4); z = base["train"][ci]
        zn = (z - z.mean()) / (z.std() + 1e-6)
        X = np.column_stack([np.ones(cn), x, x*x, zn, (tm[ci] - mm) / ms])
        beta = np.zeros(X.shape[1]); ridge = 300.0 if ctx.smoke else 800.0
        for _ in range(5 if ctx.smoke else 7):
            eta = z + X @ beta; p = 1 / (1 + np.exp(-np.clip(eta, -30, 30))); ww = np.maximum(p * (1-p), 1e-4)
            H = X.T @ (X * ww[:, None]); H.flat[::H.shape[0] + 1] += ridge
            step = np.linalg.solve(H, X.T @ (p - y_all[ci]) + ridge * beta)
            beta -= np.clip(step, -.75, .75)
        def correction(sp, bs):
            dl = np.log1p(np.maximum(np.asarray(col(sp, "duration_ms"), dtype=np.float64), 1.0))
            tt = np.asarray(col(sp, "time_ms"), dtype=np.float64)
            xx = np.clip((dl-lm)/ls, -4, 4); zz = (bs-z.mean())/(z.std()+1e-6)
            return np.clip(np.column_stack([np.ones(xx.size), xx, xx*xx, zz, (tt-mm)/ms]) @ beta, -2, 2)
        for a in (.25, .5, .75, 1.0):
            cand = base["valid"] + a * correction(valid, base["valid"])
            if float(ctx.eval_valid(cand)["primary"]) > best_primary + 1e-7:
                valid_scores = cand; test_scores = base["test"] + a * correction(test, base["test"])
    return {"valid": np.asarray(valid_scores), "test": np.asarray(test_scores)}