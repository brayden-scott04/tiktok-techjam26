import numpy as np


def fit_predict(ctx) -> dict:
    # The harness supplies Split objects; normalize both attribute-style and
    # mapping-style interfaces to a small dictionary used below.
    names = ["date", "user_idx", "video_idx", "author_idx", "tab_idx",
             "duration_ms", "time_ms", "long_view"]

    def col(split, name):
        if hasattr(split, name):
            return np.asarray(getattr(split, name))
        return np.asarray(split[name])

    tr0 = ctx.splits["train"]
    va0 = ctx.splits["valid"]
    te0 = ctx.splits["test"]
    tr = {x: col(tr0, x) for x in names}
    va = {x: col(va0, x) for x in names}
    te = {x: col(te0, x) for x in names}

    rng = np.random.default_rng(int(ctx.seed))
    nu, nv, na, nt = (int(ctx.n_users), int(ctx.n_videos),
                       int(ctx.n_authors), int(ctx.n_tabs))
    unk_u = nu - 1
    ntr = len(tr["user_idx"])
    cuts = np.quantile(tr["duration_ms"].astype(np.float64),
                       np.arange(1, 10) / 10.0)

    def dur(x):
        return np.searchsorted(cuts, x.astype(np.float64), side="right").astype(np.int32)

    td, vd, ed = dur(tr["duration_ms"]), dur(va["duration_ms"]), dur(te["duration_ms"])

    def previous(split, ds, initial):
        u = split["user_idx"].astype(np.int32)
        v = split["video_idx"].astype(np.int32)
        a = split["author_idx"].astype(np.int32)
        t = split["tab_idx"].astype(np.int32)
        n = len(u)
        pv = np.full(n, nv, np.int32)
        pa = np.full(n, na, np.int32)
        pt = np.full(n, nt, np.int32)
        pd = np.full(n, 10, np.int32)
        lv, la, lt, ld = [z.copy() for z in initial]
        order = np.lexsort((np.arange(n), split["time_ms"], split["date"], u))
        for j in order:
            j = int(j); q = int(u[j])
            if q == unk_u:
                continue
            pv[j], pa[j], pt[j], pd[j] = lv[q], la[q], lt[q], ld[q]
            lv[q], la[q], lt[q], ld[q] = v[j], a[j], t[j], ds[j]
        return [u, v, a, t, ds, pv, pa, pt, pd], (lv, la, lt, ld)

    init = (np.full(nu, nv, np.int32), np.full(nu, na, np.int32),
            np.full(nu, nt, np.int32), np.full(nu, 10, np.int32))
    tf, state = previous(tr, td, init)
    vf, _ = previous(va, vd, state)
    ef, _ = previous(te, ed, state)

    sizes = [nu, nv, na, nt, 10, nv + 1, na + 1, nt + 1, 11]
    nfields = len(sizes)
    k = 4 if ctx.smoke else 12
    y = tr["long_view"].astype(np.float32)
    sel = np.arange(ntr)
    if ctx.smoke and ntr > 100000:
        sel = np.sort(rng.choice(ntr, 100000, replace=False))
    xf = [z[sel] for z in tf]
    y = y[sel]
    n = len(y)

    E = [(rng.standard_normal((s, k)).astype(np.float32) * .025) for s in sizes]
    W = [np.zeros(s, np.float32) for s in sizes]
    mw = [np.zeros(s, np.float32) for s in sizes]
    vw = [np.zeros(s, np.float32) for s in sizes]
    me = [np.zeros((s, k), np.float32) for s in sizes]
    ve = [np.zeros((s, k), np.float32) for s in sizes]
    p = float(np.clip(y.mean(), 1e-5, 1 - 1e-5))
    bias = float(np.log(p / (1 - p)))
    mb = vb = 0.0
    best = None
    bestscore = -1e30
    bad = 0
    epochs = 3 if ctx.smoke else 30
    lr, b1, b2, eps, l2 = .04, .9, .999, 1e-7, 2e-6

    def score(fields):
        out = np.empty(len(fields[0]), np.float32)
        for lo in range(0, len(out), 200000):
            hi = min(len(out), lo + 200000)
            ss = np.zeros((hi-lo, k), np.float32)
            lin = np.full(hi-lo, bias, np.float32)
            sq = np.zeros(hi-lo, np.float32)
            for f in range(nfields):
                z = E[f][fields[f][lo:hi]]
                ss += z; sq += np.sum(z*z, axis=1); lin += W[f][fields[f][lo:hi]]
            out[lo:hi] = lin + .5 * (np.sum(ss*ss, axis=1) - sq)
        return out

    for ep in range(1, epochs + 1):
        if ep > 1 and ctx.elapsed_seconds() > ctx.max_seconds - max(3., .1*ctx.max_seconds):
            break
        ss = np.zeros((n, k), np.float32); lin = np.full(n, bias, np.float32)
        sq = np.zeros(n, np.float32)
        for f in range(nfields):
            z = E[f][xf[f]]; ss += z; sq += np.sum(z*z, axis=1); lin += W[f][xf[f]]
        q = np.clip(lin + .5*(np.sum(ss*ss, axis=1)-sq), -18, 18)
        pr = 1/(1+np.exp(-q)); d = (pr-y)/max(1,n)
        gb = float(d.sum()); mb=b1*mb+(1-b1)*gb; vb=b2*vb+(1-b2)*gb*gb
        bias -= lr*(mb/(1-b1**ep))/(np.sqrt(vb/(1-b2**ep))+eps)
        for f in range(nfields):
            ids, inv = np.unique(xf[f], return_inverse=True)
            ge = np.zeros((len(ids), k), np.float32)
            np.add.at(ge, inv, d[:,None]*(ss-E[f][xf[f]]))
            gw = np.bincount(inv, weights=d, minlength=len(ids)).astype(np.float32)
            ge += l2*E[f][ids]; gw += l2*W[f][ids]
            me[f][ids]=b1*me[f][ids]+(1-b1)*ge; ve[f][ids]=b2*ve[f][ids]+(1-b2)*ge*ge
            mw[f][ids]=b1*mw[f][ids]+(1-b1)*gw; vw[f][ids]=b2*vw[f][ids]+(1-b2)*gw*gw
            E[f][ids] -= lr*(me[f][ids]/(1-b1**ep))/(np.sqrt(ve[f][ids]/(1-b2**ep))+eps)
            W[f][ids] -= lr*(mw[f][ids]/(1-b1**ep))/(np.sqrt(vw[f][ids]/(1-b2**ep))+eps)
        vs = score(vf); m = ctx.eval_valid(vs); cur=float(m["primary"])
        if cur > bestscore + 1e-5:
            bestscore=cur; best=([z.copy() for z in E],[z.copy() for z in W],bias,vs.copy()); bad=0
        else:
            bad += 1
            if bad >= 5: break
    if best is None:
        best=([z.copy() for z in E],[z.copy() for z in W],bias,score(vf))
    E, W, bias, valid = best
    test = score(ef)
    return {"valid": valid, "test": test}
