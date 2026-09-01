import numpy as np


def fit_predict(ctx) -> dict:
    def col(split, name):
        return np.asarray(getattr(split, name))

    train = ctx.splits["train"]
    valid = ctx.splits["valid"]
    test = ctx.splits["test"]
    rng = np.random.default_rng(int(ctx.seed) + 1709)

    y = col(train, "long_view").astype(np.float32)
    n = y.size
    rate = float(np.clip(np.mean(y) if n else 0.5, 1e-4, 1.0 - 1e-4))
    base = float(np.log(rate / (1.0 - rate)))

    d = col(train, "duration_ms").astype(np.float64)
    finite = d[np.isfinite(d)]
    fill = float(np.median(finite)) if finite.size else 0.0
    d = np.nan_to_num(d, nan=fill, posinf=fill, neginf=0.0)
    edges = np.quantile(d, np.linspace(0.0, 1.0, 11))[1:-1]

    hm = col(train, "hourmin").astype(np.int64)
    hmax = int(np.max(hm)) if hm.size else 0
    mode = 2 if hmax > 1439 else (1 if hmax > 23 else 0)

    sizes = [int(ctx.n_users), int(ctx.n_videos), int(ctx.n_authors),
             int(ctx.n_tabs), 10, 24]
    off = np.zeros(6, dtype=np.int64)
    for i in range(1, 6):
        off[i] = off[i - 1] + sizes[i - 1]
    total = int(off[-1] + sizes[-1])

    def fields(s):
        u = np.clip(col(s, "user_idx").astype(np.int64), 0, sizes[0] - 1)
        v = np.clip(col(s, "video_idx").astype(np.int64), 0, sizes[1] - 1)
        a = np.clip(col(s, "author_idx").astype(np.int64), 0, sizes[2] - 1)
        t = np.clip(col(s, "tab_idx").astype(np.int64), 0, sizes[3] - 1)
        dd = np.nan_to_num(col(s, "duration_ms").astype(np.float64),
                           nan=fill, posinf=fill, neginf=0.0)
        db = np.clip(np.searchsorted(edges, dd, side="right"), 0, 9)
        hh0 = col(s, "hourmin").astype(np.int64)
        if mode == 2:
            hh = hh0 // 100
        elif mode == 1:
            hh = (hh0 * 24) // 1440
        else:
            hh = hh0
        hh = np.clip(hh, 0, 23)
        return np.column_stack((u + off[0], v + off[1], a + off[2],
                                t + off[3], db + off[4], hh + off[5]))

    xt, xv, xs = fields(train), fields(valid), fields(test)
    k = 12 if not ctx.smoke else 8
    V = rng.normal(0.0, 0.035, (total, k)).astype(np.float32)
    W = np.zeros(total, dtype=np.float32)

    def init(ids, field, alpha, scale):
        local = ids - off[field]
        cnt = np.bincount(local, minlength=sizes[field]).astype(np.float64)
        sm = np.bincount(local, weights=y, minlength=sizes[field]).astype(np.float64)
        r = np.clip((sm + alpha * rate) / (cnt + alpha), 1e-4, 1 - 1e-4)
        W[off[field]:off[field] + sizes[field]] = (
            scale * (np.log(r / (1-r)) - base)).astype(np.float32)

    for f, a, sc in ((1, 12, .90), (2, 35, .35), (3, 100, .10),
                     (4, 120, .25), (5, 150, .08)):
        init(xt[:, f], f, a, sc)

    play = np.nan_to_num(col(train, "play_time_ms").astype(np.float64),
                         nan=0.0, posinf=0.0, neginf=0.0)
    play = np.maximum(play, 0.0)
    q = (0.58 * np.clip(play / np.maximum(d, 1.0), 0, 1) +
         0.42 * (1.0 - np.exp(-play / 10000.0))).astype(np.float32)
    users = col(train, "user_idx").astype(np.int64)
    order = np.argsort(users, kind="stable")
    su = users[order]
    cuts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1, n] if n else np.array([0])
    pairs_p, pairs_n = [], []
    for j in range(len(cuts) - 1):
        rows = order[cuts[j]:cuts[j+1]]
        p, z = rows[y[rows] > .5], rows[y[rows] <= .5]
        if p.size and z.size:
            pairs_p.append(p)
            pairs_n.append(z[rng.integers(z.size, size=p.size)])
    if not pairs_p:
        return {"valid": np.zeros(len(xv)), "test": np.zeros(len(xs))}
    pp, pn = np.concatenate(pairs_p), np.concatenate(pairs_n)

    def score(X):
        out = np.empty(len(X), dtype=np.float32)
        for st in range(0, len(X), 50000):
            z = X[st:st+50000]; a = V[z]; s = a.sum(1)
            out[st:st+len(z)] = W[z].sum(1) + .5 * (s*s - (a*a).sum(1)).sum(1)
        return out

    best = score(xv)
    met = ctx.eval_valid(best)
    best_primary = float(met["primary"])
    bestV, bestW = V.copy(), W.copy()
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    step = 0
    epochs = 2 if ctx.smoke else 14
    batch = 2048 if ctx.smoke else 1536
    stale = 0
    for ep in range(epochs):
        if ctx.elapsed_seconds() > float(ctx.max_seconds) - max(5., .08*float(ctx.max_seconds)):
            break
        perm = rng.permutation(pp.size)
        for st in range(0, pp.size, batch):
            if st % (batch * 20) == 0 and ctx.elapsed_seconds() > float(ctx.max_seconds) - 5:
                break
            ix = perm[st:st+batch]; xp, xn = xt[pp[ix]], xt[pn[ix]]
            ap, an = V[xp], V[xn]; sp, sn = ap.sum(1), an.sum(1)
            fp = W[xp].sum(1) + .5*(sp*sp-(ap*ap).sum(1)).sum(1)
            fn = W[xn].sum(1) + .5*(sn*sn-(an*an).sum(1)).sum(1)
            g = -1.0 / (1.0 + np.exp(np.clip(fp-fn, -30, 30))) / max(1, len(ix))
            gp = g[:,None,None]*(sp[:,None,:]-ap); gn = -g[:,None,None]*(sn[:,None,:]-an)
            ids = np.r_[xp.ravel(), xn.ravel()]; gv = np.r_[gp.reshape(-1,k), gn.reshape(-1,k)]
            gw = np.r_[np.repeat(g,6), np.repeat(-g,6)]
            so = np.argsort(ids); sid = ids[so]; starts = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1])+1]
            uid = sid[starts]; gv = np.add.reduceat(gv[so], starts, axis=0) + 2e-5*V[uid]
            gw = np.add.reduceat(gw[so], starts) + 1e-5*W[uid]
            step += 1; b1=.9; b2=.999
            mV[uid]=b1*mV[uid]+(1-b1)*gv; vV[uid]=b2*vV[uid]+(1-b2)*gv*gv
            mW[uid]=b1*mW[uid]+(1-b1)*gw; vW[uid]=b2*vW[uid]+(1-b2)*gw*gw
            V[uid] -= .014*(mV[uid]/(1-b1**step))/(np.sqrt(vV[uid]/(1-b2**step))+1e-7)
            W[uid] -= .018*(mW[uid]/(1-b1**step))/(np.sqrt(vW[uid]/(1-b2**step))+1e-7)
        sc = score(xv); met = ctx.eval_valid(sc)
        if float(met["primary"]) > best_primary:
            best_primary=float(met["primary"]); bestV,bestW=V.copy(),W.copy(); stale=0
        else: stale += 1
        if stale >= 4: break
    V,W=bestV,bestW
    return {"valid": score(xv).astype(np.float64), "test": score(xs).astype(np.float64)}