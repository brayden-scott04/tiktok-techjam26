import numpy as np


def fit_predict(ctx) -> dict:
    def get(s, k):
        return getattr(s, k) if hasattr(s, k) else s[k]

    tr = ctx.splits["train"]
    va = ctx.splits["valid"]
    te = ctx.splits["test"]
    y = np.asarray(get(tr, "long_view"), dtype=np.float64)
    ntr = y.size
    nva = len(get(va, "user_idx"))
    nte = len(get(te, "user_idx"))
    prior = float(np.clip(y.mean(), 1e-5, 1.0 - 1e-5))

    utr, uva, ute = [np.asarray(get(s, "user_idx"), dtype=np.int64) for s in (tr, va, te)]
    vtr, vva, vte = [np.asarray(get(s, "video_idx"), dtype=np.int64) for s in (tr, va, te)]
    atr, ava, ate = [np.asarray(get(s, "author_idx"), dtype=np.int64) for s in (tr, va, te)]
    ttr, tva, tte = [np.asarray(get(s, "tab_idx"), dtype=np.int64) for s in (tr, va, te)]
    dtr, dva, dte = [
        np.maximum(np.asarray(get(s, "duration_ms"), dtype=np.float64), 0.0)
        for s in (tr, va, te)
    ]

    dq = np.unique(np.quantile(dtr, np.linspace(0.0, 1.0, 21)[1:-1]))
    dbtr = np.searchsorted(dq, dtr, side="right").astype(np.int64)
    dbva = np.searchsorted(dq, dva, side="right").astype(np.int64)
    dbte = np.searchsorted(dq, dte, side="right").astype(np.int64)
    ndur = len(dq) + 1

    try:
        otr = np.asarray(get(tr, "time_ms"), dtype=np.float64)
    except Exception:
        otr = np.arange(ntr, dtype=np.float64)
    otr = np.nan_to_num(otr, nan=0.0, posinf=0.0, neginf=0.0)

    bt, bv, be = [], [], []
    raw = [(np.log1p(dtr), np.log1p(dva), np.log1p(dte))]
    for name in ("date", "time_ms"):
        try:
            raw.append(tuple(np.asarray(get(s, name), dtype=np.float64) for s in (tr, va, te)))
        except Exception:
            pass

    try:
        hm = [np.asarray(get(s, "hourmin"), dtype=np.float64) for s in (tr, va, te)]
        if np.nanmax(hm[0]) > 48:
            hm = [np.floor(x / 100.0) + np.mod(x, 100.0) / 60.0 for x in hm]
        raw.append((
            np.sin(2.0 * np.pi * hm[0] / 24.0),
            np.sin(2.0 * np.pi * hm[1] / 24.0),
            np.sin(2.0 * np.pi * hm[2] / 24.0),
        ))
        raw.append((
            np.cos(2.0 * np.pi * hm[0] / 24.0),
            np.cos(2.0 * np.pi * hm[1] / 24.0),
            np.cos(2.0 * np.pi * hm[2] / 24.0),
        ))
    except Exception:
        pass
    raw.append((ttr.astype(np.float64), tva.astype(np.float64), tte.astype(np.float64)))

    qgrid = np.linspace(0.0, 1.0, 33)[1:-1]
    for xa, xb, xc in raw:
        xa = np.nan_to_num(np.asarray(xa, dtype=np.float64))
        xb = np.nan_to_num(np.asarray(xb, dtype=np.float64))
        xc = np.nan_to_num(np.asarray(xc, dtype=np.float64))
        lo, hi = float(xa.min()), float(xa.max())
        cuts = np.unique(np.quantile(xa, qgrid)) if hi > lo else np.empty(0)
        cuts = cuts[(cuts > lo) & (cuts < hi)] if cuts.size else cuts
        bt.append(np.searchsorted(cuts, xa, side="right"))
        bv.append(np.searchsorted(cuts, xb, side="right"))
        be.append(np.searchsorted(cuts, xc, side="right"))

    specs = [
        (utr, uva, ute, 25.0),
        (vtr, vva, vte, 18.0),
        (atr, ava, ate, 35.0),
        (ttr, tva, tte, 100.0),
        (utr * ndur + dbtr, uva * ndur + dbva, ute * ndur + dbte, 12.0),
        (atr * ndur + dbtr, ava * ndur + dbva, ate * ndur + dbte, 20.0),
        (utr * int(ctx.n_tabs) + ttr, uva * int(ctx.n_tabs) + tva,
         ute * int(ctx.n_tabs) + tte, 15.0),
        (utr * int(ctx.n_videos) + vtr, uva * int(ctx.n_videos) + vva,
         ute * int(ctx.n_videos) + vte, 5.0),
        (utr * int(ctx.n_authors) + atr, uva * int(ctx.n_authors) + ava,
         ute * int(ctx.n_authors) + ate, 8.0),
    ]

    for ka, kb, kc, alpha in specs:
        if ctx.elapsed_seconds() > 0.45 * ctx.max_seconds:
            break
        ka = np.asarray(ka, dtype=np.int64)
        kb = np.asarray(kb, dtype=np.int64)
        kc = np.asarray(kc, dtype=np.int64)
        uniq, inv = np.unique(ka, return_inverse=True)
        cnt = np.bincount(inv).astype(np.float64)
        sm = np.bincount(inv, weights=y).astype(np.float64)

        order = np.lexsort((otr, inv))
        si = inv[order]
        sy = y[order]
        starts = np.r_[True, si[1:] != si[:-1]]
        base = np.maximum.accumulate(np.where(starts, np.arange(ntr), 0))
        pc = np.arange(ntr) - base
        cs = np.cumsum(sy)
        before = np.r_[0.0, cs[:-1]]
        gb = np.maximum.accumulate(np.where(starts, before, 0.0))
        rt = (before - gb + alpha * prior) / (pc + alpha)
        rtr = np.empty(ntr, dtype=np.float64)
        ctr = np.empty(ntr, dtype=np.float64)
        rtr[order] = rt
        ctr[order] = pc

        def lookup(keys):
            pos = np.searchsorted(uniq, keys)
            clipped = np.minimum(pos, len(uniq) - 1)
            ok = (pos < len(uniq)) & (uniq[clipped] == keys)
            rate = (sm[clipped] + alpha * prior) / (cnt[clipped] + alpha)
            count = cnt[clipped]
            return np.where(ok, rate, prior), np.where(ok, count, 0.0)

        rva, cva = lookup(kb)
        rte, cte = lookup(kc)
        for xa, xb, xc in ((rtr, rva, rte), (np.log1p(ctr), np.log1p(cva), np.log1p(cte))):
            lo, hi = float(xa.min()), float(xa.max())
            cuts = np.unique(np.quantile(xa, qgrid)) if hi > lo else np.empty(0)
            cuts = cuts[(cuts > lo) & (cuts < hi)] if cuts.size else cuts
            bt.append(np.searchsorted(cuts, xa, side="right"))
            bv.append(np.searchsorted(cuts, xb, side="right"))
            be.append(np.searchsorted(cuts, xc, side="right"))

    Btr = np.column_stack(bt).astype(np.int64)
    Bva = np.column_stack(bv).astype(np.int64)
    Bte = np.column_stack(be).astype(np.int64)
    nf = Btr.shape[1]
    nb = Btr.max(axis=0) + 1

    # Build train-only, within-user positive-negative pairs. Small groups use
    # every pair. For large groups, two pairs per impression are sufficient
    # to keep runtime linear while ensuring every row participates.
    uorder = np.argsort(utr, kind="stable")
    usorted = utr[uorder]
    ustarts = np.r_[0, np.flatnonzero(usorted[1:] != usorted[:-1]) + 1]
    uends = np.r_[ustarts[1:], ntr]
    usizes = uends - ustarts
    ysorted = y[uorder]
    upos = np.add.reduceat(ysorted, ustarts).astype(np.int64)
    active = (upos > 0) & (upos < usizes)
    mean_pos = float(np.mean(upos[active])) if np.any(active) else 1.0

    rng = np.random.default_rng(int(ctx.seed) + 731)
    pair_pos_parts = []
    pair_neg_parts = []
    pair_weight_parts = []

    active_ids = np.flatnonzero(active)
    for jj, gi in enumerate(active_ids):
        a = int(ustarts[gi])
        b = int(uends[gi])
        rows = uorder[a:b]
        yy = y[rows]
        pos_rows = rows[yy > 0.5]
        neg_rows = rows[yy <= 0.5]
        np_ = pos_rows.size
        nn_ = neg_rows.size
        total = np_ * nn_
        max_pairs = (b - a) if ctx.smoke else 2 * (b - a)

        if total <= max_pairs:
            pp = np.repeat(pos_rows, nn_)
            pn = np.tile(neg_rows, np_)
        else:
            # First pass anchors every positive and every negative at least once.
            pp0 = np.concatenate((
                pos_rows,
                pos_rows[rng.integers(0, np_, size=nn_)],
            ))
            pn0 = np.concatenate((
                neg_rows[rng.integers(0, nn_, size=np_)],
                neg_rows,
            ))
            k = min(total, max_pairs)
            if pp0.size > k:
                take = rng.choice(pp0.size, size=k, replace=False)
                pp = pp0[take]
                pn = pn0[take]
            elif pp0.size < k:
                extra = k - pp0.size
                pp = np.concatenate((pp0, pos_rows[rng.integers(0, np_, size=extra)]))
                pn = np.concatenate((pn0, neg_rows[rng.integers(0, nn_, size=extra)]))
            else:
                pp, pn = pp0, pn0

        # P approximates GAUC's user weight; mean_pos is the equal-user nDCG
        # component put on the same aggregate scale.
        strength = float(np_ + mean_pos)
        pw = np.full(pp.size, strength / max(1, pp.size), dtype=np.float64)
        pair_pos_parts.append(pp.astype(np.int64, copy=False))
        pair_neg_parts.append(pn.astype(np.int64, copy=False))
        pair_weight_parts.append(pw)

        if (jj + 1) % 2048 == 0 and ctx.elapsed_seconds() > 0.72 * ctx.max_seconds:
            break

    if pair_pos_parts:
        pair_pos = np.concatenate(pair_pos_parts)
        pair_neg = np.concatenate(pair_neg_parts)
        pair_weight = np.concatenate(pair_weight_parts)
    else:
        pair_pos = np.empty(0, dtype=np.int64)
        pair_neg = np.empty(0, dtype=np.int64)
        pair_weight = np.empty(0, dtype=np.float64)

    init = np.log(prior / (1.0 - prior))
    st = np.full(ntr, init)
    sv = np.full(nva, init)
    se = np.full(nte, init)
    best = -1.0
    bvbest = sv.copy()
    btest = se.copy()
    stale = 0
    trees = 20 if ctx.smoke else 90

    for it in range(trees):
        if ctx.elapsed_seconds() > 0.91 * ctx.max_seconds:
            break
        if pair_pos.size == 0:
            break

        # Negative derivatives and diagonal Hessians for RankNet's
        # log(1 + exp(-(score_pos-score_neg))) pair loss.
        margin = np.clip(st[pair_pos] - st[pair_neg], -30.0, 30.0)
        prob_wrong = 1.0 / (1.0 + np.exp(margin))
        lam = pair_weight * prob_wrong
        hp = pair_weight * prob_wrong * (1.0 - prob_wrong)

        g = (
            np.bincount(pair_pos, weights=lam, minlength=ntr)
            - np.bincount(pair_neg, weights=lam, minlength=ntr)
        )
        h = (
            np.bincount(pair_pos, weights=hp, minlength=ntr)
            + np.bincount(pair_neg, weights=hp, minlength=ntr)
        )
        h = np.maximum(h, 1e-5)

        leaf = np.zeros(ntr, dtype=np.int64)
        lv = np.zeros(nva, dtype=np.int64)
        le = np.zeros(nte, dtype=np.int64)
        depth = 0

        for dep in range(3 if ctx.smoke else 5):
            L = 1 << dep
            gainbest = -1e99
            ff = -1
            cc = -1
            for f in range(nf):
                ix = leaf * nb[f] + Btr[:, f]
                gs = np.bincount(ix, weights=g, minlength=L * nb[f]).reshape(L, nb[f])
                hs = np.bincount(ix, weights=h, minlength=L * nb[f]).reshape(L, nb[f])
                G = np.cumsum(gs, axis=1)
                H = np.cumsum(hs, axis=1)
                gl, hl = G[:, :-1], H[:, :-1]
                gt, ht = G[:, -1:], H[:, -1:]
                gain = np.sum(
                    gl * gl / (hl + 12.0)
                    + (gt - gl) ** 2 / (ht - hl + 12.0)
                    - gt * gt / (ht + 12.0),
                    axis=0,
                )
                j = int(np.argmax(gain))
                if gain[j] > gainbest:
                    gainbest, ff, cc = float(gain[j]), f, j
            if ff < 0 or gainbest <= 0:
                break
            leaf = leaf * 2 + (Btr[:, ff] > cc)
            lv = lv * 2 + (Bva[:, ff] > cc)
            le = le * 2 + (Bte[:, ff] > cc)
            depth += 1

        if not depth:
            break
        L = 1 << depth
        val = np.clip(
            np.bincount(leaf, weights=g, minlength=L)
            / (np.bincount(leaf, weights=h, minlength=L) + 12.0),
            -2.5,
            2.5,
        )
        st += 0.065 * val[leaf]
        sv += 0.065 * val[lv]
        se += 0.065 * val[le]

        if (it + 1) % (5 if ctx.smoke else 10) == 0:
            metrics = ctx.eval_valid(sv)
            cur = float(metrics["primary"])
            if cur > best + 1e-6:
                best = cur
                bvbest = sv.copy()
                btest = se.copy()
                stale = 0
            else:
                stale += 1
            if stale >= 3:
                break

    return {"valid": bvbest, "test": btest}
