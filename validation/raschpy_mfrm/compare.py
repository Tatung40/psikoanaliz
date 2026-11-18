"""RaschPy MFRM validation (PROTOCOL.json): identities V1-V3, recovery V4/V5, TAM differences V6, edge cases V7.
PYTHONPATH=. python -m validation.raschpy_mfrm.compare [--write]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

import core.raschpy_mfrm as R
from core.mfrm import MFRMError

H = Path(__file__).resolve().parent; EX = H.parents[1] / "examples/mfrm"
FIX = ("balanced_3facet", "unbalanced_linked", "interaction_bias"); RECOVERY = ("balanced_3facet", "unbalanced_linked")
ID_TOL = {"V1_expected_score_max_diff": 1e-12, "V1b_stretch_reconstruction_max_diff": 1e-12, "V2_wle_max_diff": 1e-4,
          "V3_fit_max_diff": 1e-6}


def run(fx, model):
    return R.fit(pd.read_csv(EX / f"{fx}.csv"), "person_id", "item_id", "rater_id", "score", model=model)


def rows_for(fx, model, res, truth):
    out = []; add = lambda check, key, est, ref, tol: out.append(dict(fixture=fx, model=model, check=check, key=key, value=est,
                                                                       reference=ref, diff=abs(est - ref) if ref is not None else None,
                                                                       tol=tol, ok=None if tol is None else bool(abs(est - ref) <= tol)))
    for k, v in res["identity"].items():
        if k in ID_TOL and v is not None:
            add(k.split("_")[0], k, v, 0., ID_TOL[k])
    if "error" in res["identity"]:
        out.append(dict(fixture=fx, model=model, check="identity_error", key=res["identity"]["error"], ok=False))
    if fx not in RECOVERY:
        return out
    m = res["_m"]; t = truth[fx]["truth"]
    it = res["items"]; th = res["thresholds"]
    for i, v in t["items"].items():                                    # V4/V5 item and threshold recovery
        add("V4" if model == "global" else "V5", f"item:{i}", float(it.loc[i, "Estimate"]), v, max(3 * float(it.loc[i, "SE"]), .15))
    for k, v in enumerate(t["thresholds"]):
        add("V4" if model == "global" else "V5", f"tau{k + 1}", float(th["Estimate"].iloc[k]), v, max(3 * float(th["SE"].iloc[k]), .15))
    Rv = getattr(m, f"raters_{model}"); Se = getattr(m, f"rater_se_{model}", None)
    if model == "global":
        for r, v in t["raters"].items():
            add("V4", f"rater:{r}", float(Rv[r]), v, max(3 * float(Se[r]), .15))
        return out
    se_ok = isinstance(Se, pd.DataFrame) and Se.shape == Rv.shape
    for idx, row in Rv.iterrows():                                     # V5: every rater-side cell vs the generating λ_r
        r = idx[0] if isinstance(idx, tuple) else idx
        for col, est in row.items():
            se = float(Se.loc[idx, col]) if se_ok else np.nan
            add("V5", f"rater:{idx}:{col}", float(est), t["raters"][r], float(np.nanmax([3 * se, .30])))
    for p in R.STRETCH.get(model, ())[1:]:                             # stretch parameters: null ω = 1
        est, se = getattr(m, f"{p}_{model}"), getattr(m, f"{p}_se_{model}")
        for r in est.index:
            add("V5", f"{p}:{r}", float(est[r]), 1., float(np.nanmax([3 * float(se[r]), .30])))
    return out


def tam_differences():
    """V6: descriptive only — RaschPy global (RCL) vs the repository's TAM (MML) values in validation/mfrm/comparison.csv."""
    tam = pd.read_csv(H.parent / "mfrm/comparison.csv"); rows = []
    for fx in FIX:
        res = run(fx, "global"); m = res["_m"]
        t = tam[(tam.fixture == fx) & (tam.output.isin(["facet_measures", "thresholds"]))]
        for r in t.itertuples():
            if r.key.startswith("item:"): ours = float(m.items_global[r.key[5:]])
            elif r.key.startswith("rater:"): ours = float(m.raters_global[r.key[6:]])
            elif r.key.startswith("tau"): ours = float(np.asarray(m.thresholds_global)[-3:][int(r.key[3:]) - 1])
            else: continue
            rows.append(dict(fixture=fx, model="global", check="V6", key=r.key, value=ours, reference=r.python_tam,
                             diff=abs(ours - r.python_tam), tol=None, ok=None))
    return rows


def interaction_descriptive():
    """Not a preregistered gate: does the items model place P2 × K3 above P2's other items (generated +1.0)?"""
    m = run("interaction_bias", "items")["_m"]; Rv = m.raters_items
    gap = float(Rv.loc["P2", "K3"] - Rv.loc["P2"].drop("K3").mean())
    return [dict(fixture="interaction_bias", model="items", check="descriptive", key="λ(P2,K3) − mean λ(P2,other)",
                 value=gap, reference=1.0, diff=abs(gap - 1.0), tol=None, ok=None)]


def edge_cases():
    out = []
    def expect_fail(name, df):
        try:
            R.to_wide(df, "person_id", "item_id", "rater_id", "score"); out.append(dict(case=name, failed_closed=False))
        except MFRMError as e:
            out.append(dict(case=name, failed_closed=True, message=str(e)[:160]))
    expect_fail("disconnected", pd.read_csv(EX / "disconnected.csv"))
    expect_fail("invalid_duplicate_missing", pd.read_csv(EX / "invalid_duplicate_missing.csv"))
    expect_fail("four_facet_without_task (duplicates)", pd.read_csv(EX / "four_facet.csv"))
    b = pd.read_csv(EX / "balanced_3facet.csv")
    expect_fail("single_rater", b[b.rater_id == "P1"])
    expect_fail("empty_middle_category", b.assign(score=b.score.replace({1: 2})))
    _, _, rep = R.to_wide(pd.read_csv(EX / "sparse_category.csv"), "person_id", "item_id", "rater_id", "score")
    out.append(dict(case="sparse_category", failed_closed=False, warned=bool(rep["sparse_categories"]),
                    message=f"sparse categories {rep['sparse_categories']}"))
    miss = b.copy(); miss.loc[miss.index[:7], "score"] = np.nan
    _, _, rep = R.to_wide(miss, "person_id", "item_id", "rater_id", "score")
    out.append(dict(case="missing_scores", failed_closed=False, reported=rep["missing_scores_excluded"] == 7,
                    message=f"excluded {rep['missing_scores_excluded']}"))
    return out


def evidence(df, edges):
    models = {}
    for model in R.MODELS:
        s = df[df.model == model]; ident = s[s.check.isin(["V1", "V1b", "V2", "V3", "identity_error"])]
        rec = s[s.check == ("V4" if model == "global" else "V5")]
        id_ok = bool(len(ident) and ident.ok.all()); rec_ok = bool(len(rec) and rec.ok.all() and set(rec.fixture) >= set(RECOVERY))
        part = "PARTIAL" if id_ok and rec_ok else "NOT CERTIFIED"
        fail = rec[rec.ok == False][["fixture", "key", "value", "reference", "tol"]].to_dict("records")
        v1 = bool(ident[ident.check.isin(["V1", "V1b"])].ok.all()); v2 = bool(ident[ident.check == "V2"].ok.all())
        v3 = bool(ident[ident.check == "V3"].ok.all())
        models[model] = {
            "parameters": dict(status=part, basis="identity V1-V3 + recovery " + ("V4" if model == "global" else "V5") +
                               " on balanced_3facet and unbalanced_linked; no independent RCL implementation (CERTIFIED unattainable)",
                               recovery_max_diff=float(rec["diff"].max()) if len(rec) else None, recovery_failures=fail),
            "se": dict(status="NOT CERTIFIED", basis="RaschPy sandwich/bootstrap SE; used only inside the recovery criterion, "
                       "no independent SE reference"),
            "persons": dict(status="PARTIAL" if v2 and rec_ok else "NOT CERTIFIED", basis="V2 independent Warm WLE at RaschPy parameters",
                            max_diff=float(ident[ident.check == "V2"]["value"].max()) if len(ident) else None),
            "fit": dict(status="PARTIAL" if v3 and rec_ok else "NOT CERTIFIED", basis="V3 independent infit/outfit recomputation (items, raters)",
                        max_diff=float(ident[ident.check == "V3"]["value"].max()) if len(ident) else None),
            "probabilities": dict(status="PARTIAL" if v1 and rec_ok else "NOT CERTIFIED",
                                  basis="V1 expected scores (and V1b stretch substitution) from the documented log-numerator"),
            "category_counts": dict(status="PARTIAL", basis="descriptive counts; equal to the validated input frame"),
            "category_stats": dict(status="NOT CERTIFIED", basis="RaschPy bootstrap category widths; no independent reference"),
            "test_stats": dict(status="NOT CERTIFIED", basis="RaschPy separation/reliability summaries; not independently recomputed"),
            "model_selection": dict(status="NOT CERTIFIED", basis="AIC/BIC from the response log-likelihood at RCL estimates, which are "
                                    "not ML estimates; RaschPy-reported, unvalidated"),
            "dif": dict(status="NOT CERTIFIED", basis="RaschPy dif_test; no independent reference run"),
            "drf": dict(status="NOT CERTIFIED", basis="RaschPy drf_test; no independent reference run"),
        }
    return models


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    truth = json.loads((EX / "fixtures.json").read_text())["fixtures"]; rows = []; timing = {}
    for fx in FIX:
        for model in R.MODELS:
            res = run(fx, model); rows += rows_for(fx, model, res, truth); timing[f"{fx}:{model}"] = round(res["seconds"], 3)
    rows += tam_differences() + interaction_descriptive()
    df = pd.DataFrame(rows); edges = edge_cases(); ev = evidence(df, edges)
    print(df[df.ok.notna()].groupby(["model", "check"]).agg(n=("ok", "size"), ok=("ok", "all"), max_diff=("diff", "max")).to_string())
    print(df[df.ok == False].to_string() if (df.ok == False).any() else "no gate failures")
    v6 = df[df.check == "V6"]; print("V6 max |RaschPy - TAM|:", v6.groupby("fixture")["diff"].max().round(4).to_dict())
    print(edges); print({m: v["parameters"]["status"] for m, v in ev.items()})
    if a.write:
        df.to_csv(H / "comparison.csv", index=False)
        (H / "evidence.json").write_text(json.dumps(dict(protocol="validation/raschpy_mfrm/PROTOCOL.json", engine=f"raschpy {R.RASCHPY_VERSION}",
            scope="examples/mfrm fixtures only; experimental; not equivalent to TAM (MML) or FACETS (JMLE)", models=ev,
            tam_difference=v6.groupby("fixture")["diff"].max().to_dict(), edge_cases=edges, unsupported=R.UNSUPPORTED,
            seconds_per_fit=timing), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
