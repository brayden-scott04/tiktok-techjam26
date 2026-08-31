"""Node 0: a faithful port of kit.baseline.FM / run_fm into the fit_predict(ctx)
contract. Not LLM-generated -- this is the tree root every agent iteration
grows from, and its whole purpose is to reproduce the official baseline's
validation primary (0.6016) under our harness so that every later delta is
measured against a verified-correct number (see scripts/verify_node0.py).

Fields: [user_idx, video_idx, author_idx, tab_idx, dur_bucket] -- the same five
categorical fields as kit.baseline's FIELDS, joint-embedded in one flat table
with per-field offsets. dur_bucket is not in ctx directly (kit.data.encode
computes it from a train-quantile duration bucketing), so it is rebuilt here
exactly as kit/data.py's _bucket_edges does.

Only numpy is used, matching the AST allowlist every solution runs under.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """Verbatim port of kit.baseline.FM (Adam-optimized factorization machine)."""

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
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations, dtype=np.float64), np.linspace(0, 1, n + 1)[1:-1])


def _build_fields(ctx):
    """Returns (X_train, X_valid, X_test, dim) with the 5 fields
    joint-embedded via cumulative offsets, matching kit.data.encode's scheme."""
    train, valid, test = ctx.splits["train"], ctx.splits["valid"], ctx.splits["test"]

    edges = _bucket_edges(train.duration_ms)

    def dur_bucket(split):
        return np.searchsorted(edges, split.duration_ms).astype(np.int32)  # 0..9

    field_dims = [ctx.n_users, ctx.n_videos, ctx.n_authors, ctx.n_tabs, 10]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    dim = int(sum(field_dims))

    def stack(split):
        db = dur_bucket(split)
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


def fit_predict(ctx):
    Xtr, Xva, Xte, dim = _build_fields(ctx)
    ytr = ctx.splits["train"].long_view.astype(np.float32)

    if ctx.smoke:
        n = min(len(ytr), 50_000)
        Xtr, ytr = Xtr[:n], ytr[:n]
        epochs, patience, bs = 3, 1, 4096
    else:
        epochs, patience, bs = 40, 4, 8192

    m = FM(dim, k=16, lr=0.001, seed=ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        if ctx.elapsed_seconds() > ctx.max_seconds:
            ctx.log(f"stopping early at epoch {ep}: max_seconds budget reached")
            break
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
        va = ctx.eval_valid(m.predict(Xva))
        ctx.log(f"epoch {ep:2d} | valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f}")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                ctx.log(f"early stop at epoch {ep}")
                break

    if best_state is not None:
        m.V, m.W, m.b = best_state

    return {"valid": m.predict(Xva), "test": m.predict(Xte)}
