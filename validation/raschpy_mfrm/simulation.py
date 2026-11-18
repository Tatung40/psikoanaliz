"""V9 Monte Carlo recovery and V10 person-cluster bootstrap uncertainty (PROTOCOL.json amendment_2026_11_18).
PYTHONPATH=. python -m validation.raschpy_mfrm.simulation [--reps 30] [--boot 50]   -> simulation.json
"""
import argparse, json, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
ITEMS = {"K1": -.5, "K2": -.1, "K3": .2, "K4": .4}; TAU = np.array([-1.2, 0., 1.2]); SEED = 20261118
DESIGNS = {"dense": {"P1": -.8, "P2": 0., "P3": .8}, "linked": {"P1": -.8, "P2": -.3, "P3": 0., "P4": .3, "P5": .8}}
MODELS = ("global", "items", "thresholds", "bivector", "matrix", "centrality", "pseudo_halo", "bistretch")
FOCUS = ("bivector", "centrality", "bistretch")


def simulate(design, N, rng):
    """Null data from the global rating-scale MFRM (no rater x item / threshold / stretch effects)."""
    raters = DESIGNS[design]; names = list(raters); th = rng.standard_normal(N); rows = []
    for n in range(N):
        who = names if design == "dense" else [names[n % 5], names[(n + 1) % 5]]
        for r in who:
            for i, dl in ITEMS.items():
                k = np.arange(4); ln = k * (th[n] - dl - raters[r]) - np.concatenate([[0], np.cumsum(TAU)])
                p = np.exp(ln - ln.max()); p /= p.sum()
                rows.append((f"S{n:04d}", i, r, int(rng.choice(4, p=p))))
    return pd.DataFrame(rows, columns=["person", "item", "rater", "score"])


def fit(d, model, se=True):
    import raschpy
    w = d.pivot(index=["rater", "person"], columns="item", values="score").astype(float)
    w.index.names = ["Rater", "Person"]; w.columns.name = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = raschpy.MFRM(w, max_score=3); m.calibrate(model=model)
        if se: m.std_errors(model=model, se_method="sandwich")
    return m


def cells(m, model):
    """Rater-side cells (full vector) and stretch parameters as {key: value}."""
    Rv = getattr(m, f"raters_{model}"); out = {}
    if isinstance(Rv, pd.Series):
        out.update({f"cell:{r}": float(v) for r, v in Rv.items()})
    else:
        for idx, row in Rv.iterrows():
            for c, v in row.items(): out[f"cell:{idx}:{c}"] = float(v)
    for p in {"centrality": ("omega",), "pseudo_halo": ("omega",), "bistretch": ("omega_items", "omega_thresholds")}.get(model, ()):
        out.update({f"{p}:{r}": float(v) for r, v in getattr(m, f"{p}_{model}").items()})
    return out


def sandwich(m, model):
    """RaschPy sandwich SE keyed like cells(), only where RaschPy reports an SE for exactly that parameter."""
    Rv, se, out = getattr(m, f"raters_{model}"), getattr(m, f"rater_se_{model}", None), {}
    if isinstance(Rv, pd.Series) and isinstance(se, pd.Series):
        out.update({f"cell:{r}": float(se[r]) for r in Rv.index})
    elif isinstance(se, pd.DataFrame) and se.shape == Rv.shape:
        for idx, row in se.iterrows():
            for c, v in row.items(): out[f"cell:{idx}:{c}"] = float(v)
    for p in {"centrality": ("omega",), "pseudo_halo": ("omega",), "bistretch": ("omega_items", "omega_thresholds")}.get(model, ()):
        out.update({f"{p}:{r}": float(v) for r, v in getattr(m, f"{p}_se_{model}").items()})
    return out


def truth(key, design):
    if key.startswith("omega"): return 1.0
    r = key.split(":")[1].strip("()'").split("'")[0]
    return DESIGNS[design][r]


def v9_job(args):
    design, N, rep = args; rng = np.random.default_rng([SEED, N, rep, len(design)]); d = simulate(design, N, rng); out = []
    for model in MODELS:
        try: m = fit(d, model)
        except Exception as e:
            out.append(dict(design=design, N=N, rep=rep, model=model, key="FIT_ERROR", value=np.nan, err=str(e)[:80])); continue
        se = sandwich(m, model)
        for k, v in cells(m, model).items():
            out.append(dict(design=design, N=N, rep=rep, model=model, key=k, value=v, truth=truth(k, design), se=se.get(k, np.nan)))
    return out


def v10_job(args):
    rep, B, N = args; rng = np.random.default_rng([SEED, 10, rep]); d = simulate("dense", N, rng)
    persons = d.person.unique(); ix = d.groupby("person").indices; est = {m: cells(fit(d, m, se=False), m) for m in FOCUS}; boots = {m: [] for m in FOCUS}
    for b in range(B):
        pick = rng.choice(persons, len(persons), replace=True)
        rows = np.concatenate([ix[p] for p in pick]); lab = np.repeat(np.arange(len(pick)), [len(ix[p]) for p in pick])
        db = d.iloc[rows].assign(person=[f"B{j:05d}" for j in lab]).reset_index(drop=True)
        for m in FOCUS:
            try: boots[m].append(cells(fit(db, m, se=False), m))
            except Exception: pass
    out = []
    for m in FOCUS:
        bt = pd.DataFrame(boots[m])
        for k, v in est[m].items():
            sd = float(bt[k].std(ddof=1)); tr = truth(k, "dense")
            out.append(dict(rep=rep, model=m, key=k, value=v, truth=tr, boot_se=sd, n_boot=len(bt),
                            covered=bool(abs(v - tr) <= 1.959964 * sd)))
    return out


def summarise_v9(df):
    g = df[df.key != "FIT_ERROR"].assign(dev=lambda x: x.value - x.truth).groupby(["model", "design", "N", "key"])
    s = g.agg(bias=("dev", "mean"), sd=("dev", "std"), n=("dev", "size"), se_mean=("se", "mean")).reset_index()
    s["mcse"] = s.sd / np.sqrt(s.n); s["rmse"] = np.sqrt(s.bias ** 2 + s.sd ** 2); res = {}
    for (model, design), x in s.groupby(["model", "design"]):
        big, small = x[x.N == x.N.max()].set_index("key"), x[x.N == x.N.min()].set_index("key")
        ratio = float((big.sd / small.sd.reindex(big.index)).median())
        worst = float((big.bias.abs() - np.maximum(3 * big.mcse, .05)).max())
        res[f"{model}:{design}"] = dict(max_abs_bias_Nmax=float(big.bias.abs().max()), max_sd_Nmin=float(small.sd.max()),
            max_sd_Nmax=float(big.sd.max()), max_rmse_Nmax=float(big.rmse.max()), median_sd_ratio=ratio,
            bias_gate=bool(worst <= 0), consistency_gate=bool(ratio <= .65),
            sandwich_se_over_sd=float((big.se_mean / big.sd).median()) if big.se_mean.notna().any() else None,
            sandwich_se_over_sd_range=[float((big.se_mean / big.sd).min()), float((big.se_mean / big.sd).max())]
                if big.se_mean.notna().any() else None,
            fit_errors=int((df[(df.model == model) & (df.design == design)].key == "FIT_ERROR").sum()))
    return res, s


def summarise_v10(df):
    res = {}
    for m, x in df.groupby("model"):
        per = x.assign(dev=x.value - x.truth).groupby("key").agg(cov=("covered", "mean"), se=("boot_se", "mean"), sd=("dev", "std"))
        cov = float(x.covered.mean()); ratio = float((per.se / per.sd).mean())
        res[m] = dict(coverage=cov, min_cell_coverage=float(per["cov"].min()), se_over_empirical_sd=ratio,
                      n_reps=int(x.rep.nunique()), boot_per_rep=int(x.n_boot.min()),
                      gate=bool(.90 <= cov <= .99 and .80 <= ratio <= 1.25))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=30); ap.add_argument("--boot", type=int, default=50)
    ap.add_argument("--n10", type=int, default=500); a = ap.parse_args()
    jobs = [(dsg, N, r) for dsg in DESIGNS for N in (250, 1000) for r in range(a.reps)]
    with Pool(2) as pool:
        v9 = pd.DataFrame([row for out in pool.map(v9_job, jobs) for row in out])
        v10 = pd.DataFrame([row for out in pool.map(v10_job, [(r, a.boot, a.n10) for r in range(a.reps)]) for row in out])
    r9, s9 = summarise_v9(v9); r10 = summarise_v10(v10)
    s9.to_csv(H / "simulation_v9_cells.csv", index=False)
    (H / "simulation.json").write_text(json.dumps(dict(protocol="PROTOCOL.json amendment_2026_11_18", seed=SEED, reps=a.reps,
        boot=a.boot, n_v10=a.n10, V9=r9, V10=r10), indent=1) + "\n")
    print(json.dumps(dict(V9={k: (v["bias_gate"], v["consistency_gate"], round(v["max_abs_bias_Nmax"], 3), round(v["median_sd_ratio"], 2))
                              for k, v in r9.items()}, V10=r10), indent=0))
