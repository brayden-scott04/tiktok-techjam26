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

    def step(self, X, y, weight):
        z, E, S = self.logits(X)
        denom = max(float(weight.sum()), 1e-8)
        g = ((sigmoid(z) - y) * weight / denom).astype(np.float32)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gW += self.l2 * self.W

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
            mhat = M / (1.0 - b1 ** self.t)
            vhat = Vv / (1.0 - b2 ** self.t)
            P -= self.lr * mhat / (np.sqrt(vhat) + eps)

        self.b -= self.lr * g.sum()

    def predict(self, X, bs=200_000):
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


def _recency_weights(dates):
    dates = np.asarray(dates)
    unique_dates = np.unique(dates)
    if len(unique_dates) <= 1:
        return np.ones(len(dates), dtype=np.float32)
    rank = np.searchsorted(unique_dates, dates).astype(np.float32)
    rank /= np.float32(len(unique_dates) - 1)
    weights = np.float32(0.65) + np.float32(0.70) * rank
    weights /= np.float32(weights.mean())
    return weights.astype(np.float32)


def fit_predict(ctx):
    Xtr, Xva, Xte, dim = _build_fields(ctx)
    train = ctx.splits["train"]
    ytr = train.long_view.astype(np.float32)
    wtr = _recency_weights(train.date)

    if ctx.smoke:
        n = min(len(ytr), 50_000)
        Xtr = Xtr[:n]
        ytr = ytr[:n]
        wtr = wtr[:n]
        wtr = wtr / max(float(wtr.mean()), 1e-8)
        epochs, patience, bs = 3, 1, 4096
    else:
        epochs, patience, bs = 40, 4, 8192

    ctx.log(
        "train date recency weights: "
        f"min={float(wtr.min()):.3f} max={float(wtr.max()):.3f} "
        f"mean={float(wtr.mean()):.3f}"
    )

    model = FM(dim, k=16, lr=0.001, seed=ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    best = -1.0
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        if ctx.elapsed_seconds() >= ctx.max_seconds - 3.0:
            ctx.log(f"stopping before epoch {ep}: wall-clock budget nearly reached")
            break

        idx = rng.permutation(len(ytr))
        completed = True
        for batch_no, i in enumerate(range(0, len(idx), bs)):
            batch = idx[i:i + bs]
            model.step(Xtr[batch], ytr[batch], wtr[batch])
            if batch_no % 16 == 15 and ctx.elapsed_seconds() >= ctx.max_seconds - 3.0:
                ctx.log(f"stopping during epoch {ep}: wall-clock budget nearly reached")
                completed = False
                break

        if not completed:
            break

        scores = model.predict(Xva)
        metrics = ctx.eval_valid(scores)
        ctx.log(
            f"epoch {ep:2d} | valid GAUC {metrics['GAUC']:.4f} "
            f"nDCG@5 {metrics['nDCG@5']:.4f} primary {metrics['primary']:.4f}"
        )

        if metrics["primary"] > best + 1e-5:
            best = metrics["primary"]
            bad = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                ctx.log(f"early stop at epoch {ep}")
                break

    if best_state is not None:
        model.V, model.W, model.b = best_state

    return {"valid": model.predict(Xva), "test": model.predict(Xte)}
