"""G1 of PROTOCOL.json: Python mirt vs R mirt 1.48 on the pre-registered fixtures, judged by an independent arbiter.

    python validation/binary_certification/r_equivalence.py prepare --out DIR      # fixtures + Python fits + manifest
    Rscript validation/binary_certification/reference.R DIR/manifest.json DIR      # R mirt fits
    python validation/binary_certification/r_equivalence.py compare --out DIR --evidence FILE
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from validation.binary_certification.common import Objective, simulate  # noqa: E402

PROTOCOL = json.loads((Path(__file__).with_name("PROTOCOL.json")).read_text())
G1 = PROTOCOL["gates"]["G1_R_equivalence"]
SIM = re.compile(r"sim\(N=(\d+), k=(\d+), seed=(\d+)\)")


def cases():
    out = []
    for model, fixtures in G1["fixtures"].items():
        variants = ("fixed_a", "common_a") if model == "1PL" else (None,)
        for variant in variants:
            for fx in fixtures:
                out.append(dict(model=model, variant=variant, fixture=fx,
                                estimation="MAP" if model in ("3PL", "4PL") else "ML",
                                id=f"{model}{'_' + variant if variant else ''}__{re.sub(r'[^A-Za-z0-9]+', '_', fx).strip('_')}"))
    return out


def load(case):
    m = SIM.fullmatch(case["fixture"])
    if m:
        n, k, seed = map(int, m.groups())
        return simulate(case["model"], n, k, seed, case["variant"])
    return pd.read_csv(ROOT / case["fixture"]).to_numpy(int)


def python_fit(x, case):
    from core.irt_python_mirt_adapter import fit_python_mirt
    fit = fit_python_mirt(x, model=case["model"], estimation=case["estimation"], n_quadpts=61, tol=1e-8, max_iter=5000,
                          common_discrimination=case["variant"] == "common_a").fit
    coef = fit.coef(); V = np.asarray(fit.vcov() if callable(fit.vcov) else fit.vcov)
    labels = list(fit.vcov_labels() if callable(fit.vcov_labels) else fit.vcov_labels)
    names = {"discrimination": "a", "difficulty": "b", "guessing": "c", "upper": "d"}
    par = {names[c]: coef[c].astype(float).tolist() for c in coef.columns if c in names}
    se = {q: [None] * x.shape[1] for q in par}
    items = [str(i) for i in coef.index]
    for i, label in enumerate(labels):
        p, _, item = label.partition("["); se[names[p]][items.index(item.rstrip("]"))] = float(np.sqrt(V[i, i]))
    return {"converged": bool(fit.converged), "par": par, "se": se, "loglik": float(fit.log_likelihood)}


def prepare(out):
    out.mkdir(parents=True, exist_ok=True); manifest = []
    for case in cases():
        x = load(case); path = out / f"{case['id']}.csv"
        pd.DataFrame(x, columns=[f"V{j + 1}" for j in range(x.shape[1])]).to_csv(path, index=False)
        case.update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), shape=list(x.shape))
        (out / f"{case['id']}.python.json").write_text(json.dumps(python_fit(x, case)))
        manifest.append(case); print("python", case["id"], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def _free(case, k):
    if case["model"] == "1PL":
        return ([("a", None)] if case["variant"] == "common_a" else []) + [("b", j) for j in range(k)]
    qs = {"2PL": "ab", "3PL": "abc", "4PL": "abcd"}[case["model"]]
    return [(q, j) for j in range(k) for q in qs]


def compare(out, evidence):
    c = G1["criteria"]; rows = []
    for case in json.loads((out / "manifest.json").read_text()):
        x = pd.read_csv(case["path"]).to_numpy(int); k = x.shape[1]
        py = json.loads((out / f"{case['id']}.python.json").read_text())
        rp = out / f"{case['id']}.R.json"
        row = {"id": case["id"], "model": case["model"], "variant": case["variant"], "fixture": case["fixture"],
               "sha256": case["sha256"], "shape": case["shape"]}
        if not rp.exists():
            row.update(pass_=False, error="R result missing"); rows.append(row); continue
        r = json.loads(rp.read_text())
        objective = Objective(x, case["model"], map_priors=case["estimation"] == "MAP")
        P = {q: np.asarray(v, float) for q, v in py["par"].items()}
        R = {q: np.asarray(r["par"][q], float) for q in P}
        diff = {q: float(np.max(np.abs(P[q] - R[q]))) for q in P}
        f_py, f_r = objective.value(P), objective.value(R)
        free = _free(case, k)
        py_se = np.array([py["se"][q][0 if j is None else j] for q, j in free], float)
        if case["estimation"] == "ML":
            ref_se = np.array([r["se"][q][0 if j is None else j] for q, j in free], float); ref = "R mirt Oakes"
        else:
            ref_se = objective.se(P, free); ref = "independent numerical inverse Hessian"
        se_rel = float(np.nanmax(np.abs(py_se / ref_se - 1)))
        checks = {"both_converged": bool(py["converged"] and r["converged"]),
                  "a_b": max(diff.get("a", 0), diff["b"]) <= c["max_abs_diff_a_b"],
                  "c_d": max([diff[q] for q in ("c", "d") if q in diff] or [0]) <= c["max_abs_diff_c_d"],
                  "objective": abs(f_py - f_r) <= c["abs_objective_diff"],
                  "python_not_worse": f_py >= f_r - c["python_not_worse_than_R_by_more_than"],
                  "se": se_rel <= c["se_relative_diff_max"] and np.isfinite(py_se).all()}
        row.update(max_abs_diff=diff, objective_python=f_py, objective_R=f_r, se_reference=ref,
                   se_max_relative_diff=se_rel, R_iterations=r.get("iterations"), R_se_type=r.get("se_type"),
                   R_se_max_relative_diff_descriptive=(float(np.nanmax(np.abs(py_se / np.array(
                       [r["se"][q][0 if j is None else j] if r["se"][q][0 if j is None else j] is not None else np.nan
                        for q, j in free], float) - 1))) if case["estimation"] == "MAP" else None),
                   checks={k_: bool(v) for k_, v in checks.items()}, pass_=bool(all(checks.values())))
        rows.append(row); print(case["id"], "PASS" if row["pass_"] else "FAIL", {k_: v for k_, v in checks.items() if not v})
    versions = json.loads((out / "versions.json").read_text()) if (out / "versions.json").exists() else {}
    import mirt
    result = {"protocol": "validation/binary_certification/PROTOCOL.json G1", "cases": rows,
              "all_pass": all(r["pass_"] for r in rows), "versions": {**versions, "python_mirt": mirt.__version__}}
    Path(evidence).parent.mkdir(parents=True, exist_ok=True)
    Path(evidence).write_text(json.dumps(result, indent=1, allow_nan=True) + "\n")
    return result["all_pass"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=("prepare", "compare"))
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--evidence")
    a = ap.parse_args()
    if a.step == "prepare":
        prepare(a.out)
    else:
        compare(a.out, a.evidence); sys.exit(0)
