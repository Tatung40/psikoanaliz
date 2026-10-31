"""Pre-registered DIF Type I / power / FDR / FWER / coverage study (binary and ordinal).

DESIGN (fixed before results; copied into the output JSON):
 Binary generator (independent NumPy 2PL, D = 1), k = 20 items:
   a_ref = 0.7 + 1.1 * ((7 j) mod 20) / 19 (spread, not monotone in b), b_ref = linspace(-1.5, 1.5, 20).
   DIF items (0-based): 2, 7 uniform (b_foc = b_ref + 0.5); 12, 17 nonuniform (a_foc = a_ref - 0.6).
   Conditions: sizes {(500, 500), (1000, 200)} x impact mu_foc {0, -0.5} (sd 1) x {DIF, NULL} where NULL
   has no DIF item. R = 100 replications (fixed seeds).
   Methods: MH (purified, total-score matching), LR (purified, omnibus 2 df; uniform/nonuniform components),
   IRT-LR 2PL (core.irt_dif, purified) with R = 40 only in the (500, 500) conditions (cost).
 Ordinal generator (independent GRM, 4 categories), k = 10:
   a = linspace(.9, 1.8), thresholds b_j + (-1, 0, 1), b_j = linspace(-1, 1).
   DIF items 1, 6: all thresholds shifted +0.4 in focal group. Sizes (500, 500), impact {0, -0.5}, {DIF, NULL}.
   Methods: Mantel + SMD (purified), ordinal LR (purified). R = 100.
 Metrics: per-item rejection with unadjusted p < .05, BH q < .05 and Holm < .05; Type I = mean over
   non-DIF items; power per DIF type; FDR = mean V / max(R, 1); FWER = P(V >= 1); IRT-LR Wald 95% CI
   coverage of the true delta-a, delta-b (truth 0 for non-DIF items, planted value otherwise).
 Pre-registered criteria: unadjusted Type I in [.025, .075] (Bradley liberal); BH FDR <= .05 and Holm
   FWER <= .05, each judged with its exact 95% Clopper-Pearson / normal MC interval reported; CI
   coverage in [.925, .975]. Power is descriptive (no threshold). Failed analyses are recorded.
Run: PYTHONPATH=. python validation/dif_validation_study.py OUTDIR [--quick]

Precision amendment (2026-11-01): thresholds and historical evidence unchanged.
Pooled item-binomial intervals are replaced by dataset-cluster MCSE and marginal
Hoeffding-ratio intervals; only dataset-level FWER retains Clopper-Pearson.
Replay without fitting: python -m validation.dif_validation_study OUTDIR --reuse SOURCE_JSON
"""
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta as beta_dist
from validation.coverage_precision import coverage_precision

K_BIN = 20
A_REF = 0.7 + 1.1 * ((7 * np.arange(K_BIN)) % 20) / 19
B_REF = np.linspace(-1.5, 1.5, K_BIN)
UNIFORM, NONUNIFORM = (2, 7), (12, 17)
K_ORD = 10; ORD_DIF = (1, 6)
CRITERIA = {"type1": (.025, .075), "fdr": .05, "fwer": .05, "coverage": (.925, .975)}


def gen_binary(n_ref, n_foc, mu, dif, seed):
    rng = np.random.default_rng(seed)
    th = np.r_[rng.standard_normal(n_ref), mu + rng.standard_normal(n_foc)]
    g = np.r_[np.zeros(n_ref, int), np.ones(n_foc, int)]
    a = np.tile(A_REF, (len(th), 1)); b = np.tile(B_REF, (len(th), 1))
    if dif:
        b[np.ix_(g == 1, list(UNIFORM))] += .5
        a[np.ix_(g == 1, list(NONUNIFORM))] -= .6
    x = (rng.random(a.shape) < expit(a * (th[:, None] - b))).astype(int)
    return pd.DataFrame(x, columns=[f"M{j + 1:02d}" for j in range(K_BIN)]), g


def gen_ordinal(n_ref, n_foc, mu, dif, seed):
    rng = np.random.default_rng(seed)
    th = np.r_[rng.standard_normal(n_ref), mu + rng.standard_normal(n_foc)]
    g = np.r_[np.zeros(n_ref, int), np.ones(n_foc, int)]
    a = np.linspace(.9, 1.8, K_ORD); loc = np.linspace(-1, 1, K_ORD)
    x = np.zeros((len(th), K_ORD), int)
    for j in range(K_ORD):
        thr = loc[j] + np.array([-1., 0., 1.])
        shift = np.where((g == 1) & dif & (j in ORD_DIF), .4, 0.)
        pstar = expit(a[j] * (th[:, None] - (thr[None, :] + shift[:, None])))  # P(X >= c)
        u = rng.random(len(th))
        x[:, j] = (u[:, None] < pstar).sum(1)
    return pd.DataFrame(x, columns=[f"P{j + 1:02d}" for j in range(K_ORD)]), g


def _decisions(p, padj_bh, padj_holm):
    return {"raw": (np.asarray(p) < .05), "bh": (np.asarray(padj_bh) < .05), "holm": (np.asarray(padj_holm) < .05)}


def run_job(job):
    from core.dif_classical import adjust_pvalues, dif_screen
    kind, method, n_ref, n_foc, mu, dif, seed = job
    rec = {"kind": kind, "method": method, "n_ref": n_ref, "n_foc": n_foc, "mu": mu, "dif": dif, "seed": seed,
           "status": "FAILED"}
    t0 = time.perf_counter()
    try:
        if kind == "binary":
            data, g = gen_binary(n_ref, n_foc, mu, dif, seed)
        else:
            data, g = gen_ordinal(n_ref, n_foc, mu, dif, seed)
        if method in ("MH", "LR"):
            frame, meta = dif_screen(data, g, 0, 1, method=method, ordinal=(kind == "ordinal"), adjust="BH")
            p = frame["p"].to_numpy(float)
            rec["p"] = p.tolist()
            rec["p_bh"] = adjust_pvalues(p, "BH").tolist(); rec["p_holm"] = adjust_pvalues(p, "Holm").tolist()
            if method == "LR":
                rec["p_uniform"] = frame["p_uniform"].tolist(); rec["p_nonuniform"] = frame["p_nonuniform"].tolist()
            rec["n_untestable"] = int(np.isnan(p).sum())
        else:  # IRT-LR 2PL, purified (existing module, unchanged)
            from core.irt_dif import dif_analysis
            out = dif_analysis(data, pd.Series(g), 0, 1, model="2PL", purify=True)
            t = out["table"]
            p = t["p"].to_numpy(float)
            rec["p"] = p.tolist(); rec["p_bh"] = adjust_pvalues(p, "BH").tolist()
            rec["p_holm"] = adjust_pvalues(p, "Holm").tolist()
            rec["stable"] = bool(out["stable"])
            true_da = np.zeros(K_BIN); true_db = np.zeros(K_BIN)
            if dif:
                true_db[list(UNIFORM)] = .5; true_da[list(NONUNIFORM)] = -.6
            cov = {}
            for par, truth in (("a", true_da), ("b", true_db)):
                lo = t[f"Δ{par} %95 GA alt"].to_numpy(float); hi = t[f"Δ{par} %95 GA üst"].to_numpy(float)
                ok = np.isfinite(lo) & np.isfinite(hi)
                cov[par] = {"covered": ((lo <= truth) & (truth <= hi))[ok].astype(int).tolist(),
                            "items": np.flatnonzero(ok).tolist()}
            rec["coverage"] = cov
        rec["status"] = "OK"
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    rec["seconds"] = time.perf_counter() - t0
    return rec


def jobs(quick=False):
    R = 5 if quick else 100
    out = []
    for dif in (True, False):
        for (nr, nf) in ((500, 500), (1000, 200)):
            for mu in (0., -.5):
                for r in range(R):
                    seed = 1_000_000 + 10_000 * int(dif) + 1000 * (nr == 1000) + 100 * int(mu != 0) + r
                    for m in ("MH", "LR"):
                        out.append(("binary", m, nr, nf, mu, dif, seed))
                    if nr == 500 and r < (3 if quick else 40):
                        out.append(("binary", "IRTLR", nr, nf, mu, dif, seed))
        for mu in (0., -.5):
            for r in range(R):
                seed = 2_000_000 + 10_000 * int(dif) + 100 * int(mu != 0) + r
                for m in ("MH", "LR"):
                    out.append(("ordinal", m, 500, 500, mu, dif, seed))
    return out


def cp(x, n):
    lo = beta_dist.ppf(.025, x, n - x + 1) if x > 0 else 0.
    hi = beta_dist.ppf(.975, x + 1, n - x) if x < n else 1.
    return [float(lo), float(hi)]


def clustered_rate(parts, capacity):
    """Independent datasets, not dependent item intervals, are sampling units.

    95% distribution-free ratio interval: Hoeffding bounds for E(C/k),
    E(M/k), each in [0,1], with a union bound. Targets coverage/rejection
    conditional on eligible successful analyses; not a certification rule.
    """
    precision = coverage_precision(parts, len(parts))
    counts = precision['coverage_cluster_counts']
    c, m = np.asarray(counts['covered']), np.asarray(counts['eligible'])
    if capacity <= 0 or np.any(m > capacity):
        raise ValueError('Invalid per-replication capacity')
    eps = np.sqrt(np.log(4 / .05) / (2 * len(parts)))
    cm, mm = c.mean() / capacity, m.mean() / capacity
    ci = [max(0., (cm-eps)/(mm+eps)),
          min(1., (cm+eps)/(mm-eps)) if mm > eps else 1.]
    return precision, ci


def summarize(records):
    groups = {}
    for r in records:
        groups.setdefault((r["kind"], r["method"], r["n_ref"], r["n_foc"], r["mu"], r["dif"]), []).append(r)
    out = []
    for (kind, method, nr, nf, mu, dif), rs in sorted(groups.items(), key=str):
        ok = [r for r in rs if r["status"] == "OK"]
        k = K_BIN if kind == "binary" else K_ORD
        dif_items = (set(UNIFORM) | set(NONUNIFORM)) if kind == "binary" else set(ORD_DIF)
        null_items = [j for j in range(k) if not (dif and j in dif_items)]
        row = {"kind": kind, "method": method, "n_ref": nr, "n_foc": nf, "mu_focal": mu,
               "condition": "DIF" if dif else "NULL", "attempted": len(rs), "ok": len(ok),
               "failed": len(rs) - len(ok)}
        if ok:
            for lab in ("raw", "bh", "holm"):
                key = {"raw": "p", "bh": "p_bh", "holm": "p_holm"}[lab]
                rej = np.array([[np.isfinite(v) and v < .05 for v in r[key]] for r in ok])
                nn = rej[:, null_items]
                x = int(nn.sum()); n = nn.size
                precision, ci = clustered_rate([v[:, None] for v in nn], len(null_items))
                row[f"type1_{lab}"] = x / n; row[f"type1_{lab}_ci"] = ci
                row[f"type1_{lab}_precision"] = precision
                V = nn.sum(1); Rn = rej.sum(1)
                fdr = V / np.maximum(Rn, 1)
                row[f"fdr_{lab}"] = float(fdr.mean())
                row[f"fdr_{lab}_mcse"] = float(fdr.std(ddof=1) / np.sqrt(len(fdr))) if len(fdr) > 1 else np.nan
                eps = np.sqrt(np.log(2 / .05) / (2 * len(fdr)))
                row[f"fdr_{lab}_ci"] = [max(0., float(fdr.mean())-eps), min(1., float(fdr.mean())+eps)]
                # Failed analyses have unknown FDP, not FDP=0. This is a
                # sensitivity range, not a sampling confidence interval.
                row[f"fdr_{lab}_failure_range"] = [float(fdr.sum()/len(rs)),
                    float((fdr.sum()+len(rs)-len(ok))/len(rs))]
                fw = int((V > 0).sum()); row[f"fwer_{lab}"] = fw / len(ok); row[f"fwer_{lab}_ci"] = cp(fw, len(ok))
                if dif:
                    for name, items in ((("uniform", UNIFORM), ("nonuniform", NONUNIFORM)) if kind == "binary"
                                        else (("threshold_shift", ORD_DIF),)):
                        pw = rej[:, list(items)]
                        precision, ci = clustered_rate([v[:, None] for v in pw], len(items))
                        row[f"power_{name}_{lab}"] = float(pw.mean()); row[f"power_{name}_{lab}_ci"] = ci
                        row[f"power_{name}_{lab}_precision"] = precision
            row["untestable_mean"] = float(np.mean([r.get("n_untestable", 0) for r in ok]))
            if method == "IRTLR":
                row["unstable_purification"] = int(sum(not r["stable"] for r in ok))
                for par in ("a", "b"):
                    c = [v for r in ok for v in r["coverage"][par]["covered"]]
                    row[f"coverage_{par}"] = float(np.mean(c)) if c else np.nan
                    row[f"coverage_{par}_n"] = len(c)
                    if c:
                        precision, ci = clustered_rate([np.array(r['coverage'][par]['covered'])[:, None] for r in ok], k)
                        row[f"coverage_{par}_ci"] = ci
                        row[f"coverage_{par}_precision"] = precision
                    else:
                        row[f"coverage_{par}_ci"] = [None, None]
            row["seconds_mean"] = float(np.mean([r["seconds"] for r in ok]))
            row["meets_type1_raw"] = bool(CRITERIA["type1"][0] <= row["type1_raw"] <= CRITERIA["type1"][1])
            row["meets_fdr_bh"] = bool(row["fdr_bh"] <= CRITERIA["fdr"])
            row["meets_fwer_holm"] = bool(row["fwer_holm"] <= CRITERIA["fwer"])
            row['precision_note'] = ('Dataset-cluster MCSE; 95% Hoeffding-ratio intervals for item rates, '
                'Hoeffding for mean FDP; Clopper-Pearson only for replication-level FWER. '
                'meets_* are historical point-estimate checks, not precision-adjusted certification. '
                'All metrics conditional on successful analyses; untestable items are not rejections.')
        out.append(row)
    return out


def main(outdir, quick=False, reuse=None):
    os.makedirs(outdir, exist_ok=True)
    if reuse is not None:
        import hashlib
        import platform
        from importlib.metadata import version
        from pathlib import Path
        raw = Path(reuse).read_bytes(); original = json.loads(raw)
        if Path(reuse).resolve() == (Path(outdir)/'dif_validation_study.json').resolve():
            raise ValueError('Never overwrite source evidence')
        records = original['runs']
        keys = [(r['kind'],r['method'],r['n_ref'],r['n_foc'],r['mu'],r['dif'],r['seed']) for r in records]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate independent replication in a design cell')
        if original['criteria'] != json.loads(json.dumps(CRITERIA)):
            raise ValueError('Source preregistered criteria differ')
        payload = dict(source=str(reuse), source_sha256=hashlib.sha256(raw).hexdigest(),
            execution='SAVED RECORD REANALYSIS — NO FITS OR SIMULATION',
            scientific_status='NOT CERTIFIED', criteria=original['criteria'], n_jobs=len(records),
            source_versions=original.get('versions','UNKNOWN'),
            analysis_versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=version('scipy')),
            code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__),Path(__file__).with_name('coverage_precision.py'))}, summary=summarize(records))
        (Path(outdir)/'dif_validation_study.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')))
        return payload
    js = jobs(quick); t0 = time.time()
    with Pool(int(os.environ.get("P2_PROCS", "2"))) as pool:
        recs = pool.map(run_job, js, chunksize=1)
    summary = summarize(recs)
    payload = {"design": __doc__, "criteria": CRITERIA, "n_jobs": len(js), "wall_seconds": time.time() - t0,
               "summary": summary, "runs": recs}
    with open(os.path.join(outdir, "dif_validation_study.json"), "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=float)
    pd.DataFrame(summary).to_csv(os.path.join(outdir, "dif_validation_summary.csv"), index=False)
    return payload


if __name__ == "__main__":
    reuse = sys.argv[sys.argv.index('--reuse')+1] if '--reuse' in sys.argv else None
    main(sys.argv[1], quick="--quick" in sys.argv, reuse=reuse)
