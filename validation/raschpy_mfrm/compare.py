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
SIM = json.loads((H / "simulation.json").read_text())
V18 = json.loads((H / "familywise_v18.json").read_text())["results"] if (H / "familywise_v18.json").exists() else {}
HOLD = json.loads((H / "holdout_v17.json").read_text())["results"] if (H / "holdout_v17.json").exists() else {}
V15 = json.loads((H / "simulation_v15.json").read_text())["results"] if (H / "simulation_v15.json").exists() else {}


def V15S(test, model):
    """PARTIAL only for the global model (the configuration V15 tested) when identity and null-size gates pass."""
    v = V15.get(test, {})
    return "PARTIAL" if model == "global" and v.get("gate_identity") and v.get("gate_size") else "NOT CERTIFIED"
FIX = ("balanced_3facet", "unbalanced_linked", "interaction_bias"); RECOVERY = ("balanced_3facet", "unbalanced_linked")
ID_TOL = {"V1_expected_score_max_diff": 1e-12, "V1b_stretch_reconstruction_max_diff": 1e-12, "V2_wle_max_diff": 1e-4, "V2b_csem_max_diff": 1e-4,
          "V3_fit_max_diff": 1e-6, "V13_separation_max_diff": 1e-6}


WM = {}                                            # descriptive Wright-Masters model reliability (CSEM), per model/fixture


def run(fx, model):
    return R.fit(pd.read_csv(EX / f"{fx}.csv"), "person_id", "item_id", "rater_id", "score", model=model)


def rows_for(fx, model, res, truth):
    out = []; add = lambda check, key, est, ref, tol: out.append(dict(fixture=fx, model=model, check=check, key=key, value=est,
                                                                       reference=ref, diff=abs(est - ref) if ref is not None else None,
                                                                       tol=tol, ok=None if tol is None else bool(abs(est - ref) <= tol)))
    for k, v in res["identity"].items():
        if k in ID_TOL and v is not None:
            add(k.split("_")[0], k, v, 0., ID_TOL[k])
    if "V13_model_reliability_wright_masters" in res["identity"]:
        WM.setdefault(model, {})[fx] = res["identity"]["V13_model_reliability_wright_masters"]
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


def bootstrap_recovery(truth, B=200):
    """V11 (amendment_2026_11_22): fixture recovery of bivector/centrality/bistretch with person-cluster bootstrap SE."""
    rows = []
    for fx in RECOVERY:
        for model in R.BOOT_MODELS:
            res = run(fx, model); bt = R.cell_bootstrap(res, B=B, seed=20261118); t = truth[fx]["truth"]["raters"]
            n_ok = int(bt["B (başarılı)"].iloc[0])
            rows.append(dict(fixture=fx, model=model, check="V11_convergence", key="bootstrap refits succeeded", value=n_ok / B,
                             reference=0.95, diff=None, tol=None, ok=bool(n_ok / B >= .95)))
            for r in bt.itertuples():
                key = r.Parametre; ref = 1.0 if key.startswith("omega") else t[key.split(":")[0].strip("()'").split("'")[0]]
                tol = max(3 * r._3, .30)
                rows.append(dict(fixture=fx, model=model, check="V11", key=key, value=r.Kestirim, reference=ref,
                                 diff=abs(r.Kestirim - ref), tol=tol, ok=bool(abs(r.Kestirim - ref) <= tol), se_boot=r._3))
    return rows


def interaction_descriptive():
    """Not a preregistered gate: does the items model place P2 × K3 above P2's other items (generated +1.0)?"""
    m = run("interaction_bias", "items")["_m"]; Rv = m.raters_items
    gap = float(Rv.loc["P2", "K3"] - Rv.loc["P2"].drop("K3").mean())
    return [dict(fixture="interaction_bias", model="items", check="descriptive", key="λ(P2,K3) − mean λ(P2,other)",
                 value=gap, reference=1.0, diff=abs(gap - 1.0), tol=None, ok=None)]


def recovery_diagnosis(truth):
    """V5 failures: API/model error vs identification/precision. Not a gate; tolerances unchanged."""
    import warnings, raschpy
    out = {}
    for model in ("bivector", "centrality", "bistretch"):
        rec = {}
        for fx in RECOVERY:
            res = run(fx, model); m = res["_m"]; tr = pd.Series(truth[fx]["truth"]["raters"])
            dev = lambda Rv: Rv.sub(tr.reindex(Rv.index.get_level_values(0)).to_numpy(), axis=0)
            d0 = dev(getattr(m, f"raters_{model}")); m2 = R._model(R._rewide(res), res["K"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore"); m2.calibrate(model=model, ridge=0, interaction_ridge=0, stretch_ridge=0)
            rec[fx] = dict(max_abs_cell_dev=float(d0.abs().to_numpy().max()), max_abs_cell_dev_no_ridge=float(dev(getattr(m2, f"raters_{model}")).abs().to_numpy().max()),
                           max_abs_threshold_column_shift=float(d0.mean().abs().max()), identities_exact=bool(max(
                               v for k, v in res["identity"].items() if k in ID_TOL and v is not None) < 1e-10))
        out[model] = dict(per_fixture=rec, classification="precision of threshold-level rater cells, not an API/model error",
                          basis="V1-V3/V1b identities exact (API consistent); no common per-threshold shift (no identification offset); "
                                "ridge=0 leaves deviations unchanged (not regularisation bias); RaschPy reports no per-cell SE for this "
                                "model, so the preregistered criterion falls to its 0.30 floor (matrix passes only through 3 SE)")
    return out


def independent_estimator():
    """V8: own re-implementation of the composite likelihood (validation/raschpy_mfrm/independent.py), unregularised
    (preregistered) and RaschPy's default regularisation (the configuration the app shows); tolerance 1e-4 for both."""
    from core.mfrm import prepare
    import validation.raschpy_mfrm.independent as V
    rows = []
    for fx in RECOVERY:
        d, _ = prepare(pd.read_csv(EX / f"{fx}.csv"), "person_id", "item_id", "rater_id", "score")
        assert V.thin_widths(d, 3) == []                               # no thin-width ridge in these fixtures
        for model in R.MODELS:
            for cfg, reg in (("unregularised", False), ("default", True)):
                diff, ok, _ = V.compare(d, model, 3, reg)
                rows.append(dict(fixture=fx, model=model, check=f"V8_{cfg}", key="max |Δ step grid|", value=diff, reference=0.,
                                 diff=diff, tol=1e-4, ok=bool(diff <= 1e-4)))
            g0 = V.fit(d, model, 3, False)[0]; g1 = V.fit(d, model, 3, True)[0]
            dd = float(np.abs((g0 - g0.mean()) - (g1 - g1.mean())).max())
            rows.append(dict(fixture=fx, model=model, check="V8_discrimination", key="max |default − unregularised grid|",
                             value=dd, reference=None, diff=None, tol=None, ok=None))
    return rows


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
        s = df[df.model == model]; ident = s[s.check.isin(["V1", "V1b", "V2", "V2b", "V3", "identity_error"])]
        rec = s[s.check == ("V4" if model == "global" else "V5")]
        id_ok = bool(len(ident) and ident.ok.all()); rec_ok = bool(len(rec) and rec.ok.all() and set(rec.fixture) >= set(RECOVERY))
        v11 = s[s.check.isin(["V11", "V11_convergence"])]
        v11_ok = bool(len(v11) and v11.ok.all() and set(v11.fixture) >= set(RECOVERY))
        if model in R.BOOT_MODELS and not rec_ok and v11_ok:               # amendment_2026_11_22: V11 supplies the missing SE
            rec, rec_ok = s[s.check == "V11"], True
        v8 = s[s.check.isin(["V8_unregularised", "V8_default"])]
        v8_ok = bool(len(v8) == 2 * len(RECOVERY) and v8.ok.all())          # both configurations x both fixtures
        cert_hist = id_ok and rec_ok and v8_ok                              # historical rule (V1-V8), unchanged
        hist_status = "CERTIFIED" if cert_hist else "PARTIAL" if id_ok and rec_ok else "NOT CERTIFIED"
        v18 = V18.get(model) if V18 else None                                  # V18 prospective family-wise protocol
        v18_pass = bool(v18 and v18.get("pass_both_designs"))
        cert = bool(id_ok and v8_ok and v18_pass) if v18 is not None else cert_hist
        rec_ok = rec_ok or (v18 is not None and v18_pass)                       # V18 recovery replaces fixture recovery prospectively
        part = "CERTIFIED" if cert else "PARTIAL" if id_ok and (rec_ok or cert_hist) else "NOT CERTIFIED"
        up = lambda ok: "CERTIFIED" if cert and ok else "PARTIAL" if ok and (rec_ok or cert_hist) else "NOT CERTIFIED"
        fail = rec[rec.ok == False][["fixture", "key", "value", "reference", "tol"]].to_dict("records")
        v1 = bool(ident[ident.check.isin(["V1", "V1b"])].ok.all()); v2 = bool(ident[ident.check.isin(["V2", "V2b"])].ok.all())
        v3 = bool(ident[ident.check == "V3"].ok.all())
        models[model] = {
            "parameters": dict(status=part, historical_rule_status=hist_status,
                               v18=dict(pass_both_designs=v18_pass, **{d: {k: v18[d][k] for k in ("max_abs_bias", "se_ratio",
                                        "pooled_coverage", "min_param_coverage", "familywise_recovery", "gate_bias", "gate_se_ratio",
                                        "gate_coverage", "gate_familywise")} for d in ("balanced", "linked")}) if v18 else None, basis="identity V1-V3 + recovery " + ("V4" if model == "global" else
                               "V11 (V5 with person-cluster bootstrap SE)" if model in R.BOOT_MODELS and v11_ok and len(rec) and rec.check.iloc[0] == "V11" else "V5") +
                               " on balanced_3facet and unbalanced_linked + V8 independent estimator re-implementation "
                               "(unregularised and default regularisation, <= 1e-4); CERTIFIED requires all three",
                               v8_ok=v8_ok, v8_max_diff=float(v8["diff"].max()) if len(v8) else None,
                               recovery_max_diff=float(rec["diff"].max()) if len(rec) else None, recovery_failures=fail),
            "se": dict(status="PARTIAL" if any(R.se_reliable(model, g) for g in ("rater", "items", "thresholds")) else "NOT CERTIFIED",
                       shown=[g for g in ("rater", "items", "thresholds") if R.se_reliable(model, g)] or "hidden",
                       basis="V12 (100 replications, dense/linked, N 250/1000): Wald 95% coverage and SE/SD gates per parameter "
                             "group; a group's sandwich SE are shown only if all four cells pass. PARTIAL: simulation evidence under "
                             "the null generating model only. Category-width SE: no evidence, hidden",
                       v12={g: R.simulation_v12().get("results", {}).get(model, {}).get(g, {}).get("all_cells_pass")
                            for g in ("rater", "items", "thresholds")},
                       v12_convergence=R.simulation_v12().get("results", {}).get(model, {}).get("convergence")),
            "cell_bootstrap_se": dict(status="PARTIAL" if R.cell_bootstrap_validated(model) else "UNSUPPORTED",
                       basis="V10 person-cluster bootstrap (B=50), dense N=500, 30 replications: coverage and SE ratio gate"
                             if model in R.BOOT_MODELS else "not offered for this model", v10=SIM["V10"].get(model)),
            "recovery_simulation": dict(status="CHECK", v9={dz: {k: SIM["V9"][f"{model}:{dz}"][k] for k in
                       ("bias_gate", "consistency_gate", "max_abs_bias_Nmax", "max_sd_Nmax", "median_sd_ratio")} for dz in ("dense", "linked")}),
            "persons": dict(status=up(v2), basis="V2 independent Warm WLE at RaschPy parameters",
                            max_diff=float(ident[ident.check == "V2"]["value"].max()) if len(ident) else None),
            "fit": dict(status=up(v3), basis="V3 independent infit/outfit recomputation (items, raters)",
                        max_diff=float(ident[ident.check == "V3"]["value"].max()) if len(ident) else None),
            "probabilities": dict(status=up(v1),
                                  basis="V1 expected scores (and V1b stretch substitution) from the documented log-numerator"),
            "category_counts": dict(status="PARTIAL", basis="descriptive counts; equal to the validated input frame"),
            "category_stats": dict(status=part, basis="widths tau_k - tau_(k-1) computed by the app from the threshold estimates (derived "
                                   "quantity; status follows the threshold parameters). RaschPy bootstrap width SE / 'Prop disordered' "
                                   "not validated and not shown"),
            "test_stats": dict(status="PARTIAL" if bool(s[s.check == "V13"].ok.all()) and len(s[s.check == "V13"]) else "NOT CERTIFIED",
                               basis="V13: RaschPy's documented separation/strata/reliability recomputed from own WLE, residuals and "
                                     "information (<= 1e-6). PARTIAL at most: person rows use RaschPy's RSEM (sqrt(infit)-scaled, no "
                                     "floor at 1, mean over all persons while the variance excludes extremes) and item rows depend on "
                                     "the item sandwich SE",
                               max_diff=float(s[s.check == "V13"]["value"].max()) if len(s[s.check == "V13"]) else None,
                               wright_masters_model_reliability=WM.get(model)),
            "model_selection": dict(status="UNSUPPORTED", basis="AIC/BIC require a maximised likelihood; RCL estimates do not "
                                    "maximise the response likelihood (V16). Removed from the app; no replacement"),
            "dif": dict(status=V15S("dif", model), basis="V15: independent z/p/BH identity + empirical size under the null "
                        "(global model, dense N=500, 60 replications) + power (descriptive); evidence for the global model only",
                        v15=V15.get("dif")),
            "drf": dict(status=V15S("drf", model), basis="V15: as DIF, rater x group; omnibus LR follows its own null gate",
                        v15=V15.get("drf")),
            "dif_omnibus_lr": dict(status="PARTIAL" if model == "global" and V15.get("dif", {}).get("gate_omnibus") else "NOT CERTIFIED",
                                   basis="RaschPy omnibus LR at RCL estimates with plug-in persons: no chi-square reference "
                                         "(amendment_2026_11_23 root cause); V15 null size gate", omnibus_size=V15.get("dif", {}).get("omnibus_size")),
            "global_bh": dict(status="PARTIAL" if model == "global" and V15.get("dif", {}).get("gate_global_bh")
                              and V15.get("drf", {}).get("gate_global_bh") else "NOT CERTIFIED",
                              basis="V15c: global DIF/DRF decision = min BH-adjusted Wald p < .05; null size <= .075, DIF power >= .80",
                              dif=dict((k, V15.get("dif", {}).get(k)) for k in ("global_bh_size", "global_bh_power", "gate_global_bh")),
                              drf=dict((k, V15.get("drf", {}).get(k)) for k in ("global_bh_size", "gate_global_bh"))),
            "holdout_v17": dict(status="CHECK", basis="V17 diagnostic: 200 independent holdout replications of each fixture design; "
                                "bias gate per cell, probability that a correct-by-construction replicate fails V5, fixture tail rank",
                                results={fx: HOLD.get(f"{model}:{fx}") for fx in RECOVERY} if f"{model}:{RECOVERY[0]}" in HOLD else None),
            "person_csem": dict(status="PARTIAL" if R.v14_pass(model, "csem") else "NOT CERTIFIED",
                                basis="V14 coverage of the generating theta by WLE +/- 1.96 CSEM (non-extreme persons)",
                                v14=R.simulation_v14().get("results", {}).get(model, {}).get("csem")),
        }
    return models


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    truth = json.loads((EX / "fixtures.json").read_text())["fixtures"]; rows = []; timing = {}
    for fx in FIX:
        for model in R.MODELS:
            res = run(fx, model); rows += rows_for(fx, model, res, truth); timing[f"{fx}:{model}"] = round(res["seconds"], 3)
    rows += tam_differences() + interaction_descriptive() + independent_estimator() + bootstrap_recovery(truth)
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
            seconds_per_fit=timing, recovery_diagnosis=recovery_diagnosis(truth)), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
