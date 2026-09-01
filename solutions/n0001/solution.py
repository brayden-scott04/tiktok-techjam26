import numpy as np

# hypothesis: Replace pointwise BCE with within-user Bayesian Personalized Ranking, using sampled hard negatives after the first epoch.
# changes: Added a column accessor supporting the Split object interface (and mappings for compatibility).

def fit_predict(ctx) -> dict:
    def col(split, name):
        try:
            return split[name]
        except (TypeError, KeyError, IndexError):
            return getattr(split, name)

    train = ctx.splits["train"]
    valid = ctx.splits["valid"]
    test = ctx.splits["test"]
    rng = np.random.default_rng(int(ctx.seed) + 1709)
    y = np.asarray(col(train, "long_view"), dtype=np.float32)
    n_train = y.size
    n_users = int(ctx.n_users)
    n_videos = int(ctx.n_videos)
    n_authors = int(ctx.n_authors)
    n_tabs = int(ctx.n_tabs)

    d = np.asarray(col(train, "duration_ms"), dtype=np.float64)
    finite = d[np.isfinite(d)]
    if finite.size:
        cuts = np.quantile(finite, np.linspace(0.0, 1.0, 13))[1:-1]
        fill = float(np.median(finite))
    else:
        cuts = np.linspace(0.0, 1.0, 13)[1:-1]
        fill = 0.0
    n_duration = 12
    n_hour = 6
    offsets = np.asarray([0, n_videos, n_videos + n_authors,
                          n_videos + n_authors + n_tabs,
                          n_videos + n_authors + n_tabs + n_duration], dtype=np.int64)
    total = n_videos + n_authors + n_tabs + n_duration + n_hour
    nf = 5

    def features(s):
        v = np.clip(np.asarray(col(s, "video_idx"), dtype=np.int64), 0, n_videos - 1)
        a = np.clip(np.asarray(col(s, "author_idx"), dtype=np.int64), 0, n_authors - 1)
        t = np.clip(np.asarray(col(s, "tab_idx"), dtype=np.int64), 0, n_tabs - 1)
        z = np.asarray(col(s, "duration_ms"), dtype=np.float64)
        z = np.nan_to_num(z, nan=fill, posinf=fill, neginf=fill)
        db = np.clip(np.searchsorted(cuts, z, side="right"), 0, n_duration - 1)
        hm = np.nan_to_num(np.asarray(col(s, "hourmin"), dtype=np.float64),
                           nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64)
        hour = np.clip((np.mod(hm // 100, 24) // 4), 0, n_hour - 1)
        return np.column_stack((v, a + offsets[1], t + offsets[2],
                                db + offsets[3], hour + offsets[4])).astype(np.int64)

    ft = features(train)
    users = np.clip(np.asarray(col(train, "user_idx"), dtype=np.int64), 0, n_users - 1)
    k = 8 if ctx.smoke else 12
    U = np.zeros((n_users, k), dtype=np.float32)
    E = rng.normal(0, .018, (total, k)).astype(np.float32)
    B = np.zeros(total, dtype=np.float32)
    for j, size in enumerate((n_videos, n_authors, n_tabs, n_duration, n_hour)):
        ids = ft[:, j] - offsets[j]
        cnt = np.bincount(ids, minlength=size).astype(np.float64)
        sm = np.bincount(ids, weights=y, minlength=size).astype(np.float64)
        rate = (sm + 30.0 * ((y.sum() + 1) / (n_train + 2))) / (cnt + 30.0)
        rate = np.clip(rate, .01, .99)
        g = (y.sum() + 1) / (n_train + 2)
        B[offsets[j]:offsets[j] + size] = (.25 * (np.log(rate / (1-rate)) - np.log(g / (1-g))) * (cnt > 0)).astype(np.float32)

    neg = np.flatnonzero(y < .5).astype(np.int64)
    neg = neg[np.argsort(users[neg], kind="stable")] if neg.size else neg
    nc = np.bincount(users[neg], minlength=n_users).astype(np.int64) if neg.size else np.zeros(n_users, dtype=np.int64)
    ns = np.zeros(n_users, dtype=np.int64)
    if n_users > 1:
        ns[1:] = np.cumsum(nc[:-1])
    pos = np.flatnonzero((y > .5) & (nc[users] > 0)).astype(np.int64)
    au = np.full(U.shape, 1e-3, dtype=np.float32)
    ae = np.full(E.shape, 1e-3, dtype=np.float32)
    ab = np.full(B.shape, 1e-2, dtype=np.float32)

    def update(p, acc, ids, gr, lr, reg):
        if not ids.size:
            return
        o = np.argsort(ids, kind="quicksort")
        q = ids[o]
        st = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1]
        ui = q[st]
        cnt = np.diff(np.r_[st, q.size]).astype(np.float32)
        sm = np.add.reduceat(gr[o], st, axis=0) - reg * (cnt[:, None] * p[ui] if gr.ndim > 1 else cnt * p[ui])
        acc[ui] += sm * sm
        p[ui] += lr * sm / (np.sqrt(acc[ui]) + 1e-7)

    def predict(s):
        us = np.clip(np.asarray(col(s, "user_idx"), dtype=np.int64), 0, n_users - 1)
        f = features(s)
        out = np.empty(us.size, dtype=np.float32)
        for lo in range(0, us.size, 100000):
            hi = min(lo + 100000, us.size)
            x = E[f[lo:hi]].sum(axis=1)
            out[lo:hi] = B[f[lo:hi]].sum(axis=1) + np.einsum("ij,ij->i", U[us[lo:hi]], x)
        return out

    best = (-1.0, None, None, None)
    deadline = float(ctx.max_seconds) - max(4.0, min(20.0, .15 * float(ctx.max_seconds)))
    epochs = 2 if ctx.smoke else 9
    cap = 40000 if ctx.smoke else 1200000
    stale = 0
    for ep in range(epochs):
        if not pos.size or ctx.elapsed_seconds() >= deadline:
            break
        rows = rng.choice(pos, cap, replace=False) if pos.size > cap else rng.permutation(pos)
        cand = 1 if ep == 0 else (2 if ctx.smoke else 3)
        for lo in range(0, rows.size, 4096):
            pr = rows[lo:lo + 4096]
            us = users[pr]
            ro = (rng.random((pr.size, cand)) * nc[us, None]).astype(np.int64)
            cr = neg[ns[us, None] + ro]
            if cand > 1:
                cf = ft[cr.reshape(-1)]
                cv = E[cf].sum(axis=1)
                sc = (B[cf].sum(axis=1) + np.einsum("ij,ij->i", U[np.repeat(us, cand)], cv)).reshape(pr.size, cand)
                cr = cr[np.arange(pr.size), np.argmax(sc, axis=1)]
            else:
                cr = cr[:, 0]
            fp, fn = ft[pr], ft[cr]
            uv = U[us].copy(); xp, xn = E[fp].sum(1), E[fn].sum(1)
            g = 1.0 / (1.0 + np.exp(np.clip((B[fp].sum(1) + np.einsum("ij,ij->i", uv, xp)) - (B[fn].sum(1) + np.einsum("ij,ij->i", uv, xn)), -18, 18)))
            update(U, au, us, g[:, None] * (xp - xn), .038, 1.5e-4)
            gg = g[:, None] * uv
            update(E, ae, np.r_[fp.ravel(), fn.ravel()], np.r_[np.repeat(gg, nf, 0), -np.repeat(gg, nf, 0)], .038, 2e-4)
            update(B, ab, np.r_[fp.ravel(), fn.ravel()], np.r_[np.repeat(g, nf), -np.repeat(g, nf)], .02736, 2e-5)
            if ctx.elapsed_seconds() >= deadline:
                break
        vs = predict(valid); m = ctx.eval_valid(vs); p = float(m["primary"])
        ctx.log("epoch=%d primary=%.6f" % (ep + 1, p))
        if p > best[0] + 1e-6:
            best = (p, U.copy(), E.copy(), B.copy()); stale = 0
        else:
            stale += 1
        if stale >= 3:
            break
    if best[1] is not None:
        U, E, B = best[1], best[2], best[3]
    return {"valid": predict(valid), "test": predict(test)}