import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

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
            P -= self.lr * (M / (1.0 - b1 ** self.t)) / (
                np.sqrt(Vv / (1.0 - b2 ** self.t)) + eps
            )
        self.b -= self.lr * g.sum()

    def predict(self, X, bs=200_000):
        return np.concatenate(
            [self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)]
        )


def _bucket_edges(durations, n=10):
    values = np.asarray(durations, dtype=np.float64)
    return np.quantile(values, np.linspace(0, 1, n + 1)[1:-1])


def _infer_hour_encoding(hourmin):
    values = np.asarray(hourmin, dtype=np.int64)
    if len(values) == 0:
        return True
    sample = values[:min(len(values), 200000)]
    valid_hhmm = (
        (sample >= 0)
        & (sample <= 2359)
        & ((sample % 100) < 60)
        & ((sample // 100) < 24)
    )
    return float(valid_hhmm.mean()) >= 0.98


def _build_fields(ctx):
    train = ctx.splits["train"]
    valid = ctx.splits["valid"]
    test = ctx.splits["test"]

    edges = _bucket_edges(train.duration_ms)
    hour_is_hhmm = _infer_hour_encoding(train.hourmin)

    def dur_bucket(split):
        return np.searchsorted(edges, split.duration_ms).astype(np.int32)

    def hour_bucket(split):
        raw = np.asarray(split.hourmin, dtype=np.int64)
        if hour_is_hhmm:
            hour = raw // 100
        else:
            hour = raw // 60
        return np.clip(hour, 0, 23).astype(np.int32)

    field_dims = [ctx.n_users, ctx.n_videos, ctx.n_authors, ctx.n_tabs, 10, 24]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    dim = int(sum(field_dims))

    def stack(split):
        return np.stack(
            [
                split.user_idx + offsets[0],
                split.video_idx + offsets[1],
                split.author_idx + offsets[2],
                split.tab_idx + offsets[3],
                dur_bucket(split) + offsets[4],
                hour_bucket(split) + offsets[5],
            ],
            axis=1,
        ).astype(np.int32)

    return stack(train), stack(valid), stack(test), dim


def fit_predict(ctx) -> dict:
    Xtr, Xva, Xte, dim = _build_fields(ctx)
    ytr = ctx.splits["train"].long_view.astype(np.float32)

    if ctx.smoke:
        n = min(len(ytr), 50000)
        Xtr, ytr = Xtr[:n], ytr[:n]
        epochs, patience, bs = 3, 1, 4096
    else:
        epochs, patience, bs = 40, 4, 8192

    model = FM(dim, k=16, lr=0.001, seed=ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    best = -1.0
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        if ctx.elapsed_seconds() > ctx.max_seconds:
            ctx.log(f"stopping early at epoch {ep}: max_seconds budget reached")
            break

        idx = rng.permutation(len(ytr))
        completed_epoch = True
        for i in range(0, len(idx), bs):
            if i > 0 and ctx.elapsed_seconds() > ctx.max_seconds:
                completed_epoch = False
                break
            batch = idx[i:i + bs]
            model.step(Xtr[batch], ytr[batch])

        if not completed_epoch:
            ctx.log(f"stopping during epoch {ep}: max_seconds budget reached")
            break

        metrics = ctx.eval_valid(model.predict(Xva))
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
