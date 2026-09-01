import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = lr
        self.l2 = l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(axis=1)
        inter = 0.5 * ((S * S).sum(axis=1) - (E * E).sum(axis=(1, 2)))
        return self.b + self.W[X].sum(axis=1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self._adam_update(gV, gW)
        self.b -= self.lr * g.sum()

    def pair_step(self, Xp, Xn):
        B = len(Xp)
        zp, Ep, Sp = self.logits(Xp)
        zn, En, Sn = self.logits(Xn)

        q = ((sigmoid(zp - zn) - 1.0) / B).astype(np.float32)
        gp = q
        gn = -q

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, Xp, gp[:, None])
        np.add.at(gW, Xn, gn[:, None])
        np.add.at(gV, Xp, gp[:, None, None] * (Sp[:, None, :] - Ep))
        np.add.at(gV, Xn, gn[:, None, None] * (Sn[:, None, :] - En))
        self._adam_update(gV, gW)

    def _adam_update(self, gV, gW):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in (
            (self.V, gV, self.mV, self.vV),
            (self.W, gW, self.mW, self.vW),
        ):
            M *= b1
            M += (1.0 - b1) * G
            Vv *= b2
            Vv += (1.0 - b2) * (G * G)
            mh = M / (1.0 - b1 ** self.t)
            vh = Vv / (1.0 - b2 ** self.t)
            P -= self.lr * mh / (np.sqrt(vh) + eps)

    def reset_optimizer(self, lr):
        self.lr = lr
        self.l2 = 0.0
        self.mV.fill(0.0)
        self.vV.fill(0.0)
        self.mW.fill(0.0)
        self.vW.fill(0.0)
        self.t = 0

    def predict(self, X, bs=200000):
        parts = []
        for i in range(0, len(X), bs):
            parts.append(self.logits(X[i:i + bs])[0])
        if not parts:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(parts)


def _bucket_edges(durations, n=10):
    values = np.asarray(durations, dtype=np.float64)
    return np.quantile(values, np.linspace(0.0, 1.0, n + 1)[1:-1])


def _build_fields(ctx):
    train = ctx.splits["train"]
    valid = ctx.splits["valid"]
    test = ctx.splits["test"]
    edges = _bucket_edges(train.duration_ms)

    field_dims = [ctx.n_users, ctx.n_videos, ctx.n_authors, ctx.n_tabs, 10]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    dim = int(sum(field_dims))

    def stack(split):
        db = np.searchsorted(edges, split.duration_ms).astype(np.int32)
        return np.stack(
            [
                split.user_idx + offsets[0],
                split.video_idx + offsets[1],
                split.author_idx + offsets[2],
                split.tab_idx + offsets[3],
                db + offsets[4],
            ],
            axis=1,
        ).astype(np.int32)

    return stack(train), stack(valid), stack(test), dim


def _user_segments(users):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if len(order) == 0:
        return order, np.array([0], dtype=np.int64)
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    bounds = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            cuts.astype(np.int64),
            np.array([len(order)], dtype=np.int64),
        ]
    )
    return order, bounds


def _mine_top_pairs(order, bounds, y, scores, recency, min_impressions,
                    top_k=5, recent_window=64):
    pos_parts = []
    neg_parts = []
    cycle = np.arange(top_k, dtype=np.int64)

    for j in range(len(bounds) - 1):
        lo = int(bounds[j])
        hi = int(bounds[j + 1])
        if hi - lo < min_impressions:
            continue

        rows = order[lo:hi]
        if len(rows) > recent_window:
            recent_take = np.argpartition(
                recency[rows], -recent_window
            )[-recent_window:]
            rows = rows[recent_take]

        labels = y[rows]
        pos = rows[labels > 0.5]
        neg = rows[labels <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue

        kp = min(top_k, len(pos))
        kn = min(top_k, len(neg))
        if len(pos) > kp:
            take = np.argpartition(scores[pos], -kp)[-kp:]
            pos = pos[take]
        if len(neg) > kn:
            take = np.argpartition(scores[neg], -kn)[-kn:]
            neg = neg[take]

        pos = pos[np.argsort(scores[pos])[::-1]]
        neg = neg[np.argsort(scores[neg])[::-1]]
        pos_parts.append(pos[cycle % len(pos)])
        neg_parts.append(neg[cycle % len(neg)])

    if not pos_parts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


def _row_cohorts(users, n_users):
    users = np.asarray(users, dtype=np.int64)
    if len(users) == 0:
        return np.empty(0, dtype=np.int8)
    counts = np.bincount(users, minlength=n_users)
    row_counts = counts[users]
    cohorts = np.full(len(users), 2, dtype=np.int8)
    cohorts[row_counts <= 20] = 1
    cohorts[row_counts <= 5] = 0
    return cohorts


def _select_cohort_blends(ctx, base, tuned, users, n_users):
    cohorts = _row_cohorts(users, n_users)
    residual = tuned - base
    alphas = np.full(3, 0.5, dtype=np.float32)
    current = base + np.float32(0.5) * residual
    current_metrics = ctx.eval_valid(current)
    ctx.log(
        f"initial 0.50 blend | valid GAUC {current_metrics['GAUC']:.4f} "
        f"nDCG@5 {current_metrics['nDCG@5']:.4f} "
        f"primary {current_metrics['primary']:.4f}"
    )

    search = (0.25, 0.75, 0.0, 1.0)
    names = ("1-5", "6-20", "21+")
    stop_time = float(ctx.max_seconds) - 2.0

    for cohort in range(3):
        mask = cohorts == cohort
        if not np.any(mask):
            continue
        best_alpha = 0.5
        best_metrics = current_metrics

        for alpha in search:
            if ctx.elapsed_seconds() >= stop_time:
                ctx.log("stopping cohort blend search near the wall-clock deadline")
                break
            candidate = current.copy()
            candidate[mask] = base[mask] + np.float32(alpha) * residual[mask]
            metrics = ctx.eval_valid(candidate)
            if metrics["primary"] > best_metrics["primary"] + 1e-6:
                best_alpha = alpha
                best_metrics = metrics

        alphas[cohort] = np.float32(best_alpha)
        current[mask] = base[mask] + np.float32(best_alpha) * residual[mask]
        current_metrics = best_metrics
        ctx.log(
            f"cohort {names[cohort]} selected blend {best_alpha:.2f} | "
            f"valid primary {current_metrics['primary']:.4f}"
        )

    return current, alphas


def _apply_cohort_blends(base, tuned, users, n_users, alphas):
    cohorts = _row_cohorts(users, n_users)
    result = base.copy()
    residual = tuned - base
    for cohort in range(3):
        mask = cohorts == cohort
        if np.any(mask):
            result[mask] = base[mask] + alphas[cohort] * residual[mask]
    return result


def fit_predict(ctx):
    Xtr, Xva, Xte, dim = _build_fields(ctx)
    full_train = ctx.splits["train"]
    full_valid = ctx.splits["valid"]
    full_test = ctx.splits["test"]
    ytr = full_train.long_view.astype(np.float32)
    utr = full_train.user_idx.astype(np.int32)
    ttr = np.asarray(full_train.time_ms)

    if ctx.smoke:
        n = min(len(ytr), 50000)
        Xtr = Xtr[:n]
        ytr = ytr[:n]
        utr = utr[:n]
        ttr = ttr[:n]
        epochs, patience, bs = 3, 1, 4096
    else:
        epochs, patience, bs = 40, 4, 8192

    model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    best = -1.0
    best_state = None
    bad = 0
    deadline = max(0.0, float(ctx.max_seconds) - 8.0)

    for ep in range(1, epochs + 1):
        if ctx.elapsed_seconds() >= deadline:
            ctx.log("stopping BCE training near the wall-clock deadline")
            break
        idx = rng.permutation(len(ytr))
        interrupted = False
        for batch_no, i in enumerate(range(0, len(idx), bs)):
            model.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
            if batch_no % 10 == 9 and ctx.elapsed_seconds() >= deadline:
                interrupted = True
                break

        metrics = ctx.eval_valid(model.predict(Xva))
        ctx.log(
            f"BCE epoch {ep:2d} | valid GAUC {metrics['GAUC']:.4f} "
            f"nDCG@5 {metrics['nDCG@5']:.4f} primary {metrics['primary']:.4f}"
        )
        if metrics["primary"] > best + 1e-5:
            best = metrics["primary"]
            bad = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
        if interrupted or bad >= patience:
            if bad >= patience:
                ctx.log(f"BCE early stop at epoch {ep}")
            break

    if best_state is not None:
        model.V, model.W, model.b = best_state

    base_V = model.V.copy()
    base_W = model.W.copy()
    base_b = np.float32(model.b)
    base_valid = model.predict(Xva)

    pair_best = best
    pair_best_state = None
    blend = 0.5

    if ctx.elapsed_seconds() < float(ctx.max_seconds) - 8.0 and len(ytr) > 0:
        order, bounds = _user_segments(utr)
        model.reset_optimizer(lr=3e-4)
        pair_epochs = 2 if ctx.smoke else 5
        pair_bs = 4096 if ctx.smoke else 8192
        min_impressions = 8 if ctx.smoke else 20
        recent_window = 32 if ctx.smoke else 64
        pair_bad = 0

        for ep in range(1, pair_epochs + 1):
            if ctx.elapsed_seconds() >= deadline:
                break
            train_scores = model.predict(Xtr)
            pidx, nidx = _mine_top_pairs(
                order,
                bounds,
                ytr,
                train_scores,
                ttr,
                min_impressions=min_impressions,
                top_k=5,
                recent_window=recent_window,
            )
            if len(pidx) == 0:
                ctx.log("no eligible mixed-label active users for recent pairwise fine-tuning")
                break

            shuffle = rng.permutation(len(pidx))
            interrupted = False
            for batch_no, i in enumerate(range(0, len(shuffle), pair_bs)):
                take = shuffle[i:i + pair_bs]
                model.pair_step(Xtr[pidx[take]], Xtr[nidx[take]])
                if batch_no % 8 == 7 and ctx.elapsed_seconds() >= deadline:
                    interrupted = True
                    break

            tuned_valid = model.predict(Xva)
            candidate = (1.0 - blend) * base_valid + blend * tuned_valid
            metrics = ctx.eval_valid(candidate)
            ctx.log(
                f"recent top-pair epoch {ep:2d} ({len(pidx)} pairs) | "
                f"valid GAUC {metrics['GAUC']:.4f} "
                f"nDCG@5 {metrics['nDCG@5']:.4f} "
                f"primary {metrics['primary']:.4f}"
            )
            if metrics["primary"] > pair_best + 1e-5:
                pair_best = metrics["primary"]
                pair_bad = 0
                pair_best_state = (
                    model.V.copy(), model.W.copy(), np.float32(model.b)
                )
            else:
                pair_bad += 1
            if interrupted or pair_bad >= 2:
                break

    if pair_best_state is None:
        model.V, model.W, model.b = base_V, base_W, base_b
        return {"valid": base_valid, "test": model.predict(Xte)}

    model.V, model.W, model.b = base_V, base_W, base_b
    base_test = model.predict(Xte)

    tuned_V, tuned_W, tuned_b = pair_best_state
    model.V, model.W, model.b = tuned_V, tuned_W, tuned_b
    tuned_valid = model.predict(Xva)
    tuned_test = model.predict(Xte)

    valid_scores, cohort_alphas = _select_cohort_blends(
        ctx,
        base_valid,
        tuned_valid,
        full_valid.user_idx,
        ctx.n_users,
    )
    test_scores = _apply_cohort_blends(
        base_test,
        tuned_test,
        full_test.user_idx,
        ctx.n_users,
        cohort_alphas,
    )
    return {"valid": valid_scores, "test": test_scores}
