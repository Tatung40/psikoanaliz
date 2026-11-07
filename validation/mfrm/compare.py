"""Production pipeline (TAM) vs mfrmr MML and rasch PCML (PROTOCOL.json); writes comparison.csv and evidence.json.
PYTHONPATH=. python -m validation.mfrm.compare [--write]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

import core.mfrm as M

H = Path(__file__).resolve().parent; EX = H.parents[1] / "examples/mfrm"
FIX = {"balanced_3facet": (), "unbalanced_linked": (), "interaction_bias": ()}   # four_facet: see four_facet_attempt()
RECOVERY_FIXTURES = {"balanced_3facet", "unbalanced_linked", "four_facet"}
TOL = {"facet": 5e-3, "step": 5e-3, "sigma": 5e-3, "loglik": .05}


def run_fixture(name):
    df = pd.read_csv(EX / f"{name}.csv")
    return M.analyze(df, "person_id", "item_id", "rater_id", "score", FIX[name], interaction=name == "interaction_bias", show_ids=True)


def four_facet_attempt():
    try:
        M.analyze(pd.read_csv(EX / "four_facet.csv"), "person_id", "item_id", "rater_id", "score", ("task_id",), allow_extra=True)
        return {"status": "ran"}
    except M.MFRMError as e:
        return {"status": "FAILED under preregistered TAM controls", "message": str(e)}


def compare():
    ref = pd.read_csv(H / "reference.csv"); truth = json.loads((EX / "fixtures.json").read_text())["fixtures"]; rows = []
    add = lambda fx, out, key, ours, theirs, tol, ref_: rows.append(dict(fixture=fx, output=out, key=key, python_tam=ours,
                                                                       reference=theirs, diff=abs(ours - theirs), tol=tol, ref=ref_,
                                                                       ok=None if tol is None else bool(abs(ours - theirs) <= tol)))
    for fx in FIX:
        r = run_fixture(fx); el = r["elements"]; mm = ref[(ref.fixture == fx) & (ref.engine == "mfrmr")]
        name = {"item": "item_id", "rater": "rater_id", "facet1": "task_id"}
        for e in el.itertuples():
            q = mm[(mm.kind == "element") & (mm.facet == name[e.facet]) & (mm.level == e.Düzey)].set_index("key").value
            add(fx, "facet_measures", f"{e.facet}:{e.Düzey}", e._4, q["Estimate"], TOL["facet"], "mfrmr")
            add(fx, "facet_se", f"{e.facet}:{e.Düzey}", e.SE, q["ModelSE"], None, "mfrmr")
            add(fx, "fit_mnsq", f"infit {e.facet}:{e.Düzey}", e._8, q["Infit"], None, "mfrmr")
            add(fx, "fit_mnsq", f"outfit {e.facet}:{e.Düzey}", e._9, q["Outfit"], None, "mfrmr")
            add(fx, "observed_expected", f"expected {e.facet}:{e.Düzey}", e._11, q["ExpectedAverage"], None, "mfrmr")
            gen = truth[fx]["truth"]; block = {"item": "items", "rater": "raters", "facet1": "tasks"}[e.facet]
            if fx not in RECOVERY_FIXTURES: continue          # PROTOCOL: recovery on balanced/unbalanced/four-facet only
            add(fx, "recovery_tam", f"{e.facet}:{e.Düzey}", e._4, gen[block][e.Düzey], max(3 * e.SE, .15), "truth")
            rq = ref[(ref.fixture == fx) & (ref.engine == "rasch") & (ref.facet == name[e.facet]) & (ref.level == e.Düzey)]
            if len(rq):
                add(fx, "recovery_rasch", f"{e.facet}:{e.Düzey}", rq.value.iloc[0], gen[block][e.Düzey], .15, "truth (rasch PCML)")
        st = mm[mm.kind == "step"].sort_values("level").value.to_numpy()
        for k, (a, b) in enumerate(zip(r["params"]["tau"], st), 1):
            add(fx, "thresholds", f"tau{k}", a, b, TOL["step"], "mfrmr")
            add(fx, "threshold_se", f"tau{k}", r["thresholds"]["Eşik SE"].iloc[k], np.nan, None, "none")
        mo = mm[mm.kind == "model"].set_index("key").value
        add(fx, "model_fit", "logLik", r["model"]["logLik"], mo["loglik"], TOL["loglik"], "mfrmr")
        add(fx, "latent", "sigma", np.sqrt(r["model"]["Gizil varyans"]), np.sqrt(mo["variance"]), TOL["sigma"], "mfrmr")
        add(fx, "latent", "mean", r["model"]["Gizil ortalama"], mo["mean"], None, "mfrmr")
        pe = mm[(mm.kind == "element") & (mm.facet == "person") & (mm.key == "Estimate")].set_index("level").value
        add(fx, "person_eap", "max |EAP - mfrmr|", 0., float((r["persons"].EAP - pe.loc[r["persons"].index]).abs().max()), None, "mfrmr")
        for k, v in r["identity"].items():
            if k != "information_condition":
                add(fx, "identity", k, v, 0., {"loglik": 1e-5, "eap": 1e-6, "wle": 1e-4}[k], "Python identity")
        rl = mm[mm.kind == "reliability"]
        for f, s in r["separation"].iterrows():
            q = rl[rl.facet == {"item": "item_id", "rater": "rater_id", "facet1": "task_id", "person": "person"}[f]].set_index("key").value
            for k in ("Separation", "Reliability"):
                add(fx, "separation", f"{f} {k}", s[k], q[k], None, "mfrmr")
        if "interaction" in r and "LR" in r["interaction"]:
            add(fx, "interaction_lr", "LR", r["interaction"]["LR"], 2 * (mo["interaction_loglik"] - mo["loglik"]), None, "mfrmr")
    return pd.DataFrame(rows)


RULES = {  # status from PROTOCOL.json (tolerance must have been preregistered for CERTIFIED)
    "facet_measures": "tol", "thresholds": "tol", "model_fit": "tol", "latent": "tol", "rater_ranking": "follows facet_measures",
    "facet_se": "partial", "threshold_se": "partial", "facet_ci": "partial", "person_wle": "partial", "person_eap": "partial",
    "fit_mnsq": "partial", "observed_expected": "partial", "separation": "partial", "interaction_lr": "partial",
    "fair_average": "partial", "category_diagnostics": "partial", "fit_zstd": "none", "interaction_estimates": "none",
    "interaction_se": "none"}


FORMULA_ONLY = {"facet_ci", "person_wle", "fair_average", "category_diagnostics", "threshold_se"}


def evidence(df):
    out = {}
    for o, rule in RULES.items():
        s = df[df.output == o]; tested = s[s.ok.notna()]
        if rule == "tol":
            fx = set(tested.fixture); ok = bool(len(tested) and tested.ok.all() and {"balanced_3facet", "unbalanced_linked"} <= fx)
            out[o] = dict(status="CERTIFIED" if ok else "NOT CERTIFIED", max_diff=float(tested["diff"].max()) if len(tested) else None,
                          fixtures=sorted(fx), basis="mfrmr MML within preregistered tolerance + Python identity")
        elif rule == "partial" and s.empty and o not in FORMULA_ONLY:
            out[o] = dict(status="NOT CERTIFIED", basis="no comparison could be made (e.g. TAM did not converge); withheld")
        elif rule == "partial":
            out[o] = dict(status="PARTIAL", max_diff=float(s["diff"].max()) if len(s) and s.reference.notna().all() else None,
                          fixtures=sorted(set(s.fixture)), basis="formula/identity evidence or agreement without a preregistered tolerance; definition documented")
        elif rule == "none":
            out[o] = dict(status="NOT CERTIFIED", basis="no independent evidence; withheld")
        else:
            out[o] = dict(out["facet_measures"], basis="ranking of certified facet measures")
    out["four_facet_model"] = dict(status="NOT CERTIFIED", attempt=four_facet_attempt(),
                                   basis="TAM four-facet fit under the preregistered controls; production refuses 4+ facets")
    for k in ("identity", "recovery_tam", "recovery_rasch"):
        s = df[df.output == k]; out[k] = dict(status="CHECK", all_ok=bool(s.ok.all()), max_diff=float(s["diff"].max()), n=len(s),
                                              failures=s[s.ok == False][["fixture", "key", "diff", "tol"]].to_dict("records"))
    eap_fail = [r for r in out["identity"]["failures"] if r["key"] == "eap"]
    if eap_fail:
        out["person_eap"]["basis"] = ("PARTIAL: TAM-reported EAP differs from the Python posterior at the final parameters by up to "
                                      f"{max(r['diff'] for r in eap_fail):.1e} (> preregistered 1e-6 identity tolerance); the app shows the "
                                      "Python EAP recomputed from the final TAM parameters, which agrees with mfrmr within "
                                      f"{out['person_eap']['max_diff']:.1e} (no preregistered tolerance)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    df = compare(); ev = evidence(df)
    print(df.groupby(["output"]).agg(n=("diff", "size"), max_diff=("diff", "max"), ok=("ok", lambda x: x.dropna().all())).to_string())
    print(df[df.ok == False].to_string() if (df.ok == False).any() else "no tolerance failures")
    print({k: v["status"] for k, v in ev.items()})
    if a.write:
        df.to_csv(H / "comparison.csv", index=False)
        (H / "evidence.json").write_text(json.dumps(dict(protocol="validation/mfrm/PROTOCOL.json", outputs=ev,
                                                         scope="examples/mfrm fixtures; not FACETS-equivalent"), indent=1) + "\n")
