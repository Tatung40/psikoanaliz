"""Executes CERT_PROTOCOL.json gates through the production engine (core.rater_reliability.compute)."""
import json, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R_ = 1000; Z = 1.959963984540054


def W(x):
    import core.rater_reliability as R
    df = pd.DataFrame(x, columns=[f"R{j}" for j in range(x.shape[1])]); df.insert(0, "id", range(len(df)))
    return R.from_wide(df, "id", list(df.columns[1:]))


def C(m, x):
    import core.rater_reliability as R
    return R.compute(m, W(x))


def ci(r):
    v = r.get("ci95"); return (v[0], v[1]) if v is not None else (np.nan, np.nan)


def fam_a(_):
    rng = np.random.default_rng(1); out = {}
    two = {"cohen_kappa", "cohen_kappa_linear", "cohen_kappa_quadratic", "scott_pi", "pabak", "bangdiwala_b"}
    for m in json.loads(Path(__file__).with_name("CERT_PROTOCOL.json").read_text())["families"]["A_chance_corrected_and_agreement_estimates"]["methods"]:
        q = 2 if m == "pabak" else 4; r = 2 if m in two else 3
        x = rng.integers(1, q + 1, size=(20000, r)); est = C(m, x)["estimate"]
        target = 1 / q if m in ("percent_agreement", "bangdiwala_b") else 0.0
        y = np.repeat(rng.integers(1, q + 1, size=(500, 1)), r, 1); perfect = C(m, y)["estimate"]
        out[m] = dict(estimate_independent=float(est), target=target, perfect=float(perfect),
                      passed=bool(abs(est - target) <= .02 and abs(perfect - 1) <= 1e-12))
    return out


def fam_b(m):
    rng = np.random.default_rng(2); cov = rej = 0
    for _ in range(R_):
        a = rng.integers(1, 5, 200); b = np.where(rng.random(200) < .5, a, rng.integers(1, 5, 200)); lo, hi = ci(C(m, np.c_[a, b])); cov += lo <= .5 <= hi
        rej += C(m, np.c_[rng.integers(1, 5, 200), rng.integers(1, 5, 200)])["p"] < .05
    c, t = cov / R_, rej / R_
    return m, dict(coverage=c, type1=t, passed=bool(abs(c - .95) <= .02 and .036 <= t <= .064))


def icc_data(rng, model, n, k, s2):
    s = rng.normal(0, np.sqrt(s2), (n, 1))
    if model == 1: return s + rng.normal(0, np.sqrt(.75), (n, k))
    if model == 2: return s + rng.normal(0, .5, (1, k)) + rng.normal(0, np.sqrt(.5), (n, k))
    return s + np.array([[0, .3, -.2, .5]]) + rng.normal(0, np.sqrt(.5), (n, k))


def target(m, s2):
    k = 4; r1 = {1: s2 / (s2 + .75), 2: s2 / (s2 + .25 + .5), 3: s2 / (s2 + .5)}[int(m[4])]
    return r1 if m.endswith("_1") else k * r1 / (1 + (k - 1) * r1)


KIND = {"icc_1_1": "ICC(1,1)", "icc_1_k": "ICC(1,k)", "icc_2_1": "ICC(A,1)", "icc_2_k": "ICC(A,k)", "icc_3_1": "ICC(C,1)", "icc_3_k": "ICC(C,k)"}


def icc_fast(m, x):
    """Same estimators as production: closed-form ICC/F/p (pingouin definitions) + production _icc_ci."""
    from scipy.stats import f as F_
    from core.rater_reliability import _icc_ci
    n, k = x.shape; g = x.mean(); ssr = k * ((x.mean(1) - g) ** 2).sum(); ssc = n * ((x.mean(0) - g) ** 2).sum()
    sse = ((x - g) ** 2).sum() - ssr - ssc; msb, msj, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    msw = (ssc + sse) / (n * (k - 1))
    est = {"icc_1_1": (msb - msw) / (msb + (k - 1) * msw), "icc_1_k": (msb - msw) / msb,
           "icc_2_1": (msb - mse) / (msb + (k - 1) * mse + k * (msj - mse) / n), "icc_2_k": (msb - mse) / (msb + (msj - mse) / n),
           "icc_3_1": (msb - mse) / (msb + (k - 1) * mse), "icc_3_k": (msb - mse) / msb}[m]
    Fv, d2 = (msb / msw, n * (k - 1)) if m.startswith("icc_1") else (msb / mse, (n - 1) * (k - 1))
    return {"estimate": est, "F": Fv, "p": F_.sf(Fv, n - 1, d2), "ci95": _icc_ci(x, KIND[m])}


def fam_c(m):
    rng = np.random.default_rng(3); model = int(m[4]); cov = rej = 0; t = target(m, 1.0)
    ident = 0.0
    for _ in range(20):                                                   # identity with the production engine
        x = icc_data(rng, model, 30, 4, 1.0); a, b = icc_fast(m, x), C(m, x)
        ident = max(ident, abs(a["estimate"] - b["estimate"]), abs(a["F"] - b["F"]), abs(a["p"] - b["p"]),
                    *(abs(u - v) for u, v in zip(a["ci95"], b["ci95"])))
    big = icc_fast(m, icc_data(rng, model, 2000, 4, 1.0))["estimate"]
    for _ in range(R_):
        lo, hi = icc_fast(m, icc_data(rng, model, 30, 4, 1.0))["ci95"]; cov += lo <= t <= hi
        rej += icc_fast(m, icc_data(rng, model, 30, 4, 0.0))["p"] < .05
    c, ty = cov / R_, rej / R_
    return m, dict(target=t, estimate_n2000=float(big), coverage=c, type1=ty, production_identity_max_abs_diff=ident,
                   passed=bool(ident <= 1e-8 and abs(big - t) <= .03 and abs(c - .95) <= .02 and .036 <= ty <= .064))


def fam_d(m):
    rng = np.random.default_rng(4); rej = cov = 0
    for _ in range(R_):
        if m == "kendall_w":
            x = rng.normal(size=(10, 5))
        else:
            x = rng.normal(size=(50, 2))
        rej += C(m, x)["p"] < .05
        if m == "pearson":
            u = rng.normal(size=50); v = .5 * u + np.sqrt(.75) * rng.normal(size=50); lo, hi = ci(C(m, np.c_[u, v])); cov += lo <= .5 <= hi
    out = dict(type1=rej / R_, passed=bool(.036 <= rej / R_ <= .064))
    if m == "pearson":
        out.update(coverage=cov / R_, passed=out["passed"] and abs(cov / R_ - .95) <= .02)
    return m, out


def fam_e(m):
    rng = np.random.default_rng(5); q = 2 if m == "mcnemar" else 3; rej = 0
    for _ in range(R_):
        a = rng.integers(1, q + 1, 200); b = np.where(rng.random(200) < .6, a, rng.integers(1, q + 1, 200)); sw = rng.random(200) < .5
        x = np.c_[np.where(sw, a, b), np.where(sw, b, a)]          # exchangeable pair -> marginal homogeneity holds
        rej += C(m, x)["p"] < .05
    return m, dict(type1=rej / R_, passed=bool(rej / R_ <= .064))


def fam_f(m):
    rng = np.random.default_rng(6); cb = cl = cu = 0
    for _ in range(R_):
        u = rng.normal(size=50); x = np.c_[u + rng.normal(0, .5, 50) + .3, u]
        r = C(m, x); b = r["ci95_bias"]; lo, hi = r["ci95_loa"]
        cb += b[0] <= .3 <= b[1]; cl += lo[0] <= .3 - Z * .5 <= lo[1]; cu += hi[0] <= .3 + Z * .5 <= hi[1]
    c = dict(bias=cb / R_, loa_low=cl / R_, loa_high=cu / R_)
    return m, dict(coverage=c, passed=bool(all(abs(v - .95) <= .02 for v in c.values())))


if __name__ == "__main__":
    jobs = [(fam_c, m) for m in ("icc_1_1", "icc_1_k", "icc_2_1", "icc_2_k", "icc_3_1", "icc_3_k")] + [(fam_a, None)] + [(fam_b, m) for m in ("cohen_kappa", "cohen_kappa_linear", "cohen_kappa_quadratic")] + \
           [(fam_d, m) for m in ("pearson", "spearman", "kendall_w")] + [(fam_e, m) for m in ("mcnemar", "stuart_maxwell", "bhapkar")] + [(fam_f, "bland_altman")]
    with ProcessPoolExecutor(2) as ex:
        futs = [ex.submit(f, a) for f, a in jobs]; res = {}
        for f in futs:
            r = f.result(); res.update(r if isinstance(r, dict) else {r[0]: r[1]})
            print("done", list(r)[0] if isinstance(r, dict) else r[0], flush=True)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float) + "\n")
    for m, v in res.items(): print(m, v["passed"], {k: v[k] for k in v if k != "passed"})
