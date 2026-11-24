"""Fresh runs for amendment R-A1 (icc_2_1, bangdiwala_b, kendall_w)."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, "validation/rater_reliability")
from cert_run import C, icc_fast, icc_data, target


def icc():
    rng = np.random.default_rng(31); dev = []
    for _ in range(50):
        s = rng.normal(0, 1, (2000, 1)); b = rng.normal(0, .5, (1, 4)); x = s + b + rng.normal(0, np.sqrt(.5), (2000, 4))
        dev.append(icc_fast("icc_2_1", x)["estimate"] - 1 / (1 + b.var(ddof=1) + .5))
    ident = max(abs(icc_fast("icc_2_1", x := icc_data(rng, 2, 30, 4, 1.0))["estimate"] - C("icc_2_1", x)["estimate"]) for _ in range(20))
    t = target("icc_2_1", 1.0); cov = rej = 0
    for _ in range(1000):
        lo, hi = icc_fast("icc_2_1", icc_data(rng, 2, 30, 4, 1.0))["ci95"]; cov += lo <= t <= hi
        rej += icc_fast("icc_2_1", icc_data(rng, 2, 30, 4, 0.0))["p"] < .05
    md = float(np.mean(dev)); c, ty = cov / 1000, rej / 1000
    return "icc_2_1", dict(mean_dev_conditional=md, max_abs_dev=float(np.max(np.abs(dev))), coverage=c, type1=ty, identity=ident,
                           passed=bool(ident <= 1e-8 and abs(md) <= .03 and abs(c - .95) <= .02 and .036 <= ty <= .064))


def bang():
    rng = np.random.default_rng(11); est = C("bangdiwala_b", rng.integers(1, 5, (20000, 2)))["estimate"]
    perf = C("bangdiwala_b", np.repeat(rng.integers(1, 5, (500, 1)), 2, 1))["estimate"]
    return "bangdiwala_b", dict(estimate_independent=float(est), target=1 / 16, perfect=float(perf), passed=bool(abs(est - 1 / 16) <= .02 and abs(perf - 1) <= 1e-12))


def kw(seed):
    rng = np.random.default_rng(seed); return sum(C("kendall_w", rng.normal(size=(10, 5)))["p"] < .05 for _ in range(2500))


if __name__ == "__main__":
    with ProcessPoolExecutor(10) as ex:
        fi, fb = ex.submit(icc), ex.submit(bang)
        k = sum(ex.map(kw, np.random.SeedSequence(41).generate_state(8)))
        out = dict([fi.result(), fb.result()]); t1 = k / 20000
        out["kendall_w"] = dict(type1=t1, R=20000, mc_se=float(np.sqrt(t1 * (1 - t1) / 20000)), passed=bool(.036 <= t1 <= .064))
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=float); print(json.dumps(out, default=float))
