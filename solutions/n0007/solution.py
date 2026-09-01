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
        for param, grad, moment, variance in (
            (self.V, gV, self.mV, self.vV),
            (self.W, gW, self.mW, self.vW),
        ):
            moment *= b1
            moment += (1.0 - b1) * grad
            variance *= b2
            variance += (1.0 - b2) * (grad * grad)
            mhat = moment / (1.0 - b1 ** self.t)
            vhat = variance / (1.0 - b2 ** self.t)
            param -= self.lr * mhat / (np.sqrt(vhat) + eps)

        self.b -= self.lr * g.sum()

    def predict(self, X, bs=200_000):
        return np.concatenate(
            [self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)]
        )


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
        duration_bucket = np.searchsorted(edges, split.duration_ms).astype(np.int32)
        return np.stack(
            [
                split.user_idx + offsets[0],
                split.video_idx + offsets[1],
                split.author_idx + offsets[2],
                split.tab_idx + offsets[3],
                duration_bucket + offsets[4],
            ],
            axis=1,
        ).astype(np.int32)

    return stack(train), stack(valid), stack(test), dim


def _build_user_author_residuals(ctx, history_n):
    train = ctx.splits["train"]
    users = np.asarray(train.user_idx[:history_n], dtype=np.int64)
    authors = np.asarray(train.author_idx[:history_n], dtype=np.int64)
    labels = np.asarray(train.long_view[:history_n], dtype=np.float64)

    global_rate = float(labels.mean()) if len(labels) else 0.5
    author_count = np.bincount(authors, minlength=ctx.n_authors).astype(np.float64)
    author_positive = np.bincount(
        authors, weights=labels, minlength=ctx.n_authors
    ).astype(np.float64)
    author_strength = 25.0
    author_rate = (
        author_positive + author_strength * global_rate
    ) / (author_count + author_strength)
    author_rate = np.clip(author_rate, 1e-4, 1.0 - 1e-4)

    keys = users * np.int64(ctx.n_authors) + authors
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    pair_count = np.bincount(inverse).astype(np.float64)
    pair_positive = np.bincount(inverse, weights=labels).astype(np.float64)
    pair_authors = (unique_keys % np.int64(ctx.n_authors)).astype(np.int64)
    prior = author_rate[pair_authors]

    pair_strength = 8.0
    pair_rate = (
        pair_positive + pair_strength * prior
    ) / (pair_count + pair_strength)
    pair_rate = np.clip(pair_rate, 1e-4, 1.0 - 1e-4)

    pair_logit = np.log(pair_rate) - np.log1p(-pair_rate)
    prior_logit = np.log(prior) - np.log1p(-prior)
    residual = np.clip(pair_logit - prior_logit, -2.5, 2.5).astype(np.float32)
    return unique_keys, residual


def _lookup_user_author_residual(split, n_authors, unique_keys, residual):
    query = (
        np.asarray(split.user_idx, dtype=np.int64) * np.int64(n_authors)
        + np.asarray(split.author_idx, dtype=np.int64)
    )
    positions = np.searchsorted(unique_keys, query)
    output = np.zeros(len(query), dtype=np.float32)
    in_range = positions < len(unique_keys)
    rows = np.nonzero(in_range)[0]
    if len(rows):
        matched = unique_keys[positions[rows]] == query[rows]
        matched_rows = rows[matched]
        output[matched_rows] = residual[positions[matched_rows]]
    return output


def fit_predict(ctx) -> dict:
    Xtr, Xva, Xte, dim = _build_fields(ctx)
    train = ctx.splits["train"]
    valid = ctx.splits["valid"]
    test = ctx.splits["test"]
    ytr = train.long_view.astype(np.float32)

    if ctx.smoke:
        history_n = min(len(ytr), 50_000)
        Xtr = Xtr[:history_n]
        ytr = ytr[:history_n]
        epochs, patience, bs = 3, 1, 4096
    else:
        history_n = len(ytr)
        epochs, patience, bs = 40, 4, 8192

    model = FM(dim, k=16, lr=0.001, seed=ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    best = -1.0
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        if ctx.elapsed_seconds() > ctx.max_seconds:
            ctx.log(f"stopping early at epoch {epoch}: max_seconds budget reached")
            break

        order = rng.permutation(len(ytr))
        for start in range(0, len(order), bs):
            model.step(Xtr[order[start:start + bs]], ytr[order[start:start + bs]])

        valid_scores = model.predict(Xva)
        metrics = ctx.eval_valid(valid_scores)
        ctx.log(
            f"epoch {epoch:2d} | valid GAUC {metrics['GAUC']:.4f} "
            f"nDCG@5 {metrics['nDCG@5']:.4f} primary {metrics['primary']:.4f}"
        )
        if metrics["primary"] > best + 1e-5:
            best = metrics["primary"]
            bad = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                ctx.log(f"early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.V, model.W, model.b = best_state

    base_valid = model.predict(Xva)
    base_test = model.predict(Xte)

    unique_keys, residual = _build_user_author_residuals(ctx, history_n)
    affinity_valid = _lookup_user_author_residual(
        valid, ctx.n_authors, unique_keys, residual
    )
    affinity_test = _lookup_user_author_residual(
        test, ctx.n_authors, unique_keys, residual
    )

    if ctx.smoke:
        alphas = (0.0, 0.5, 1.0)
    else:
        alphas = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

    best_alpha = 0.0
    best_blend = -1.0
    for alpha in alphas:
        scores = base_valid + np.float32(alpha) * affinity_valid
        metrics = ctx.eval_valid(scores)
        ctx.log(
            f"author-affinity alpha {alpha:.3f} | GAUC {metrics['GAUC']:.4f} "
            f"nDCG@5 {metrics['nDCG@5']:.4f} primary {metrics['primary']:.4f}"
        )
        if metrics["primary"] > best_blend + 1e-6:
            best_blend = metrics["primary"]
            best_alpha = alpha

    ctx.log(f"selected author-affinity alpha {best_alpha:.3f}")
    final_valid = base_valid + np.float32(best_alpha) * affinity_valid
    final_test = base_test + np.float32(best_alpha) * affinity_test
    return {"valid": final_valid, "test": final_test}
