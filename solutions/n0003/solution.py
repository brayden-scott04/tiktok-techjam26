import numpy as np


def fit_predict(ctx) -> dict:
    rng = np.random.default_rng(int(ctx.seed) + 913)
    tr = ctx.splits["train"]
    va = ctx.splits["valid"]
    te = ctx.splits["test"]

    u = np.asarray(tr.user_idx, dtype=np.int64)
    v = np.asarray(tr.video_idx, dtype=np.int64)
    a = np.asarray(tr.author_idx, dtype=np.int64)
    tab = np.asarray(tr.tab_idx, dtype=np.int64)
    duration = np.asarray(tr.duration_ms, dtype=np.float64)

    finite = duration[np.isfinite(duration)]
    if finite.size:
        edges = np.unique(np.quantile(finite, np.linspace(.1, .9, 9)))
    else:
        edges = np.empty(0, dtype=np.float64)
    dur = np.searchsorted(edges, np.nan_to_num(duration, nan=0.0), side="right").astype(np.int64)
    n_dur = int(edges.size + 1)

    ys = [
        np.asarray(tr.long_view, dtype=np.float32),
        np.asarray(tr.is_click, dtype=np.float32),
        np.asarray(tr.is_like, dtype=np.float32),
        np.asarray(tr.is_follow, dtype=np.float32),
        np.asarray(tr.is_comment, dtype=np.float32),
        np.asarray(tr.is_forward, dtype=np.float32),
        np.asarray(tr.is_hate, dtype=np.float32),
    ]
    nt = len(ys)
    nu, nv, na, ntab = int(ctx.n_users), int(ctx.n_videos), int(ctx.n_authors), int(ctx.n_tabs)
    k = 8 if ctx.smoke else 16
    init = .035
    U = rng.normal(0, init, (nu, k)).astype(np.float32)
    V = rng.normal(0, init, (nv, k)).astype(np.float32)
    A = rng.normal(0, init, (na, k)).astype(np.float32)
    T = rng.normal(0, init, (ntab, k)).astype(np.float32)
    D = rng.normal(0, init, (n_dur, k)).astype(np.float32)
    W = (1 + rng.normal(0, .015, (nt, k))).astype(np.float32)
    BV = np.zeros((nt, nv), np.float32)
    BA = np.zeros((nt, na), np.float32)
    BT = np.zeros((nt, ntab), np.float32)
    BD = np.zeros((nt, n_dur), np.float32)
    au = np.ones_like(U) * 1e-3
    av = np.ones_like(V) * 1e-3
    aa = np.ones_like(A) * 1e-3
    at = np.ones_like(T) * 1e-3
    ad = np.ones_like(D) * 1e-3
    aw = np.ones_like(W) * 1e-3
    abv = np.ones_like(BV) * 1e-3
    aba = np.ones_like(BA) * 1e-3
    abt = np.ones_like(BT) * 1e-3
    abd = np.ones_like(BD) * 1e-3

    order = np.argsort(u, kind="stable")
    su = u[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    ends = np.r_[starts[1:], order.size]
    caps = [16, 8, 5, 5, 4, 4, 4]
    if ctx.smoke:
        caps = [3, 2, 1, 1, 1, 1, 1]
    pairs = []
    for ti, y in enumerate(ys):
        pp, nn = [], []
        for lo, hi in zip(starts, ends):
            rows = order[lo:hi]
            pos = rows[y[rows] > 0]
            neg = rows[y[rows] <= 0]
            if pos.size and neg.size:
                m = min(caps[ti], max(pos.size, neg.size))
                pp.append(pos[rng.integers(pos.size, size=m)])
                nn.append(neg[rng.integers(neg.size, size=m)])
        pairs.append((np.concatenate(pp) if pp else np.empty(0, np.int64),
                      np.concatenate(nn) if nn else np.empty(0, np.int64)))

    weights = np.asarray([1., .35, .20, .20, .16, .16, .12], np.float32)
    epochs = 2 if ctx.smoke else 13
    batch = 512 if ctx.smoke else 2048
    best = -1e30
    state = None
    stale = 0

    for ep in range(epochs):
        for ti in rng.permutation(nt):
            pp, nn = pairs[ti]
            if not pp.size:
                continue
            perm = rng.permutation(pp.size)
            for z in range(0, pp.size, batch):
                if ctx.elapsed_seconds() > ctx.max_seconds - 6:
                    break
                ix = perm[z:z + batch]
                ip, inn = pp[ix], nn[ix]
                up = U[u[ip]]
                vp, vn = V[v[ip]], V[v[inn]]
                ap, an = A[a[ip]], A[a[inn]]
                tp, tn = T[tab[ip]], T[tab[inn]]
                dp, dn = D[dur[ip]], D[dur[inn]]
                sp = up + vp + ap + tp + dp
                sn = up + vn + an + tn + dn
                hp = .5 * (sp * sp - up * up - vp * vp - ap * ap - tp * tp - dp * dp)
                hn = .5 * (sn * sn - up * up - vn * vn - an * an - tn * tn - dn * dn)
                dh = hp - hn
                delta = (dh * W[ti]).sum(1)
                delta += BV[ti, v[ip]] - BV[ti, v[inn]] + BA[ti, a[ip]] - BA[ti, a[inn]]
                delta += BT[ti, tab[ip]] - BT[ti, tab[inn]] + BD[ti, dur[ip]] - BD[ti, dur[inn]]
                c = weights[ti] / (1 + np.exp(np.clip(delta, -30, 30)))
                cw = c[:, None] * W[ti]
                gu = cw * ((vp + ap + tp + dp) - (vn + an + tn + dn))
                grads = [
                    (U, au, u[ip], gu),
                    (V, av, np.r_[v[ip], v[inn]], np.r_[cw * (sp - vp), -cw * (sn - vn)]),
                    (A, aa, np.r_[a[ip], a[inn]], np.r_[cw * (sp - ap), -cw * (sn - an)]),
                    (T, at, np.r_[tab[ip], tab[inn]], np.r_[cw * (sp - tp), -cw * (sn - tn)]),
                    (D, ad, np.r_[dur[ip], dur[inn]], np.r_[cw * (sp - dp), -cw * (sn - dn)])]
                for par, acc, ids, g in grads:
                    so = np.argsort(ids, kind="stable")
                    ids = ids[so]
                    cut = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
                    q = ids[cut]
                    rg = np.add.reduceat(g[so], cut, axis=0)
                    np.clip(rg, -25, 25, out=rg)
                    acc[q] += rg * rg
                    par[q] += .075 * rg / np.sqrt(acc[q] + 1e-8)
                signed = np.r_[c, -c]
                for par, acc, ids in [(BV[ti], abv[ti], np.r_[v[ip], v[inn]]), (BA[ti], aba[ti], np.r_[a[ip], a[inn]]), (BT[ti], abt[ti], np.r_[tab[ip], tab[inn]]), (BD[ti], abd[ti], np.r_[dur[ip], dur[inn]])]:
                    so = np.argsort(ids, kind="stable")
                    ids = ids[so]
                    cut = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
                    q = ids[cut]
                    rg = np.add.reduceat(signed[so], cut)
                    acc[q] += rg * rg
                    par[q] += .13 * rg / np.sqrt(acc[q] + 1e-8)
                gw = (c[:, None] * dh).sum(0)
                aw[ti] += gw * gw
                W[ti] += .025 * gw / np.sqrt(aw[ti] + 1e-8)
        
        def_score = None
        vu = np.asarray(va.user_idx, np.int64); vv = np.asarray(va.video_idx, np.int64); vaa = np.asarray(va.author_idx, np.int64); vt = np.asarray(va.tab_idx, np.int64)
        vd = np.searchsorted(edges, np.nan_to_num(np.asarray(va.duration_ms, np.float64), nan=0), side="right")
        eu, ev, ea, et, ed = U[vu], V[vv], A[vaa], T[vt], D[vd]
        ss = eu + ev + ea + et + ed
        hh = .5 * (ss * ss - eu * eu - ev * ev - ea * ea - et * et - ed * ed)
        def_score = (hh * W[0]).sum(1) + BV[0, vv] + BA[0, vaa] + BT[0, vt] + BD[0, vd]
        met = ctx.eval_valid(def_score)
        if met["primary"] > best:
            best = float(met["primary"]); state = tuple(x.copy() for x in (U,V,A,T,D,W,BV,BA,BT,BD)); stale = 0
        else: stale += 1
        if stale >= 3 or ctx.elapsed_seconds() > ctx.max_seconds - 6: break

    if state is not None:
        U,V,A,T,D,W,BV,BA,BT,BD = state
    out = []
    for sp in (va, te):
        su = np.asarray(sp.user_idx, np.int64); sv = np.asarray(sp.video_idx, np.int64); sa = np.asarray(sp.author_idx, np.int64); st = np.asarray(sp.tab_idx, np.int64)
        sd = np.searchsorted(edges, np.nan_to_num(np.asarray(sp.duration_ms, np.float64), nan=0), side="right")
        x = U[su] + V[sv] + A[sa] + T[st] + D[sd]
        h = .5 * (x*x - U[su]*U[su] - V[sv]*V[sv] - A[sa]*A[sa] - T[st]*T[st] - D[sd]*D[sd])
        out.append((h*W[0]).sum(1) + BV[0,sv] + BA[0,sa] + BT[0,st] + BD[0,sd])
    return {"valid": out[0].astype(np.float32), "test": out[1].astype(np.float32)}