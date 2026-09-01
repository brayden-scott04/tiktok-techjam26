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
        batch_size = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / batch_size).astype(np.float32)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gW += self.l2 * self.W

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for param, grad, first, second in (
            (self.V, gV, self.mV, self.vV),
            (self.W, gW, self.mW, self.vW),
        ):
            first *= b1
            first += (1.0 - b1) * grad
            second *= b2
            second += (1.0 - b2) * (grad * grad)
            first_hat = first / (1.0 - b1 ** self.t)
            second_hat = second / (1.0 - b2 ** self.t)
            param -= self.lr * first_hat / (np.sqrt(second_hat) + eps)

        self.b -= self.lr * g.sum()

    def predict(self, X, bs=200_000):
        outputs = []
        for start in range(0, len(X), bs):
            outputs.append(self.logits(X[start:start + bs])[0])
        return np.concatenate(outputs)


def _bucket_edges(values, n_buckets):
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
    return np.quantile(values, quantiles)


def _build_fields(ctx):
    train = ctx.splits["train"]
    valid = ctx.splits["valid"]
    test = ctx.splits["test"]

    coarse_edges = _bucket_edges(train.duration_ms, 10)
    fine_edges = _bucket_edges(train.duration_ms, 50)

    field_dims = [
        ctx.n_users,
        ctx.n_videos,
        ctx.n_authors,
        ctx.n_tabs,
        10,
        50,
    ]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    dim = int(sum(field_dims))

    def stack(split):
        coarse = np.searchsorted(coarse_edges, split.duration_ms).astype(np.int32)
        fine = np.searchsorted(fine_edges, split.duration_ms).astype(np.int32)
        return np.stack(
            [
                split.user_idx + offsets[0],
                split.video_idx + offsets[1],
                split.author_idx + offsets[2],
                split.tab_idx + offsets[3],
                coarse + offsets[4],
                fine + offsets[5],
            ],
            axis=1,
        ).astype(np.int32)

    return stack(train), stack(valid), stack(test), dim


def fit_predict(ctx) -> dict:
    Xtr, Xva, Xte, dim = _build_fields(ctx)
    ytr = ctx.splits["train"].long_view.astype(np.float32)

    if ctx.smoke:
        n = min(len(ytr), 50_000)
        Xtr = Xtr[:n]
        ytr = ytr[:n]
        epochs, patience, batch_size = 3, 1, 4096
    else:
        epochs, patience, batch_size = 40, 4, 8192

    model = FM(dim, k=16, lr=0.001, seed=ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    best = -1.0
    best_state = None
    bad_epochs = 0
    budget_stop = False

    for epoch in range(1, epochs + 1):
        if ctx.elapsed_seconds() >= ctx.max_seconds - 2.0:
            ctx.log(f"stopping before epoch {epoch}: wall-clock budget nearly reached")
            break

        order = rng.permutation(len(ytr))
        for batch_no, start in enumerate(range(0, len(order), batch_size)):
            ids = order[start:start + batch_size]
            model.step(Xtr[ids], ytr[ids])
            if batch_no % 16 == 15 and ctx.elapsed_seconds() >= ctx.max_seconds - 2.0:
                budget_stop = True
                break

        valid_scores = model.predict(Xva)
        metrics = ctx.eval_valid(valid_scores)
        ctx.log(
            f"epoch {epoch:2d} | valid GAUC {metrics['GAUC']:.4f} "
            f"nDCG@5 {metrics['nDCG@5']:.4f} primary {metrics['primary']:.4f}"
        )

        if metrics["primary"] > best + 1e-5:
            best = metrics["primary"]
            bad_epochs = 0
            best_state = (
                model.V.copy(),
                model.W.copy(),
                np.float32(model.b),
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                ctx.log(f"early stop at epoch {epoch}")
                break

        if budget_stop:
            ctx.log(f"stopping after epoch {epoch}: wall-clock budget nearly reached")
            break

    if best_state is not None:
        model.V, model.W, model.b = best_state

    return {
        "valid": model.predict(Xva),
        "test": model.predict(Xte),
    }
