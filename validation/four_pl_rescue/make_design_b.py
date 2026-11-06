"""Design B (identifiable-design candidate): fixed truth and data, seed 20261106. Run once; SHA256 in PROTOCOL.json."""
import hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
SEED, N, J = 20261106, 10000, 30


def truth(rng):
    b = np.linspace(-1.5, 1.5, J); rng.shuffle(b)
    return pd.DataFrame(dict(a=rng.uniform(1.3, 2.3, J), b=b, c=rng.uniform(.10, .20, J), d=rng.uniform(.80, .90, J)),
                        index=[f"I{j + 1:02d}" for j in range(J)])


def simulate(p, rng, n=N):
    th = rng.standard_normal(n)[:, None]
    P = p.c.values + (p.d.values - p.c.values) / (1 + np.exp(-p.a.values * (th - p.b.values)))
    return pd.DataFrame((rng.uniform(size=P.shape) < P).astype(int), columns=p.index)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED); p = truth(rng); x = simulate(p, rng)
    p.to_csv(HERE / "design_b_truth.csv", index_label="item"); x.to_csv(HERE / "design_b_data.csv", index=False)
    for f in ("design_b_truth.csv", "design_b_data.csv"):
        print(f, hashlib.sha256((HERE / f).read_bytes()).hexdigest())
    print(x.mean().round(3).min(), x.mean().round(3).max())
