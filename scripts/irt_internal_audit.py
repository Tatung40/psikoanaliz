"""Small, reproducible *internal* 3PL/4PL verification bundle.

This is a diagnostic smoke benchmark, not publication-grade Monte Carlo
evidence and not a reference-software comparison.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from core.irt_truth_scenarios import generating_items
from core.irt_simulation import simulate_irt
from core.irt_se_coverage import interval_coverage_study
from core.irt_quadrature_validation import quadrature_sensitivity
from core.irt_multistart import fit_irt_multistart
from core.irt_stationarity import stationarity_diagnostics

def main():
    destination=Path("artifacts/irt-internal-audit")
    destination.mkdir(parents=True,exist_ok=True)
    summary=[]
    coverage=[]
    quadrature=[]
    stationarity=[]
    for model in ("3PL","4PL"):
        truth=generating_items(model,k=5)
        x,_=simulate_irt(500,truth,seed=2026)
        audit=fit_irt_multistart(
            x,model=model,starts=(0.,),points=21,max_iter=70)
        kkt,kkt_summary=stationarity_diagnostics(
            x,audit["fit"]["items"],model,points=21)
        kkt.insert(0,"Model",model)
        stationarity.append(kkt)
        q=quadrature_sensitivity(
            x,model,fit_points=(15,21,31),
            reference_points=41,max_iter=70,starts=(0.,))
        quadrature.append(q)
        cov=interval_coverage_study(
            truth,model,sample_sizes=(500,),repetitions=2,
            seed=2026,starts=(0.,),points=21,max_iter=70)
        coverage.append(cov["summary"])
        summary.append({
            "Model":model,
            "EM yakınsaması":audit["fit"]["converged"],
            "Sınırdaki parametre":len(audit["boundary"]),
            "KKT toleransını aşan":kkt_summary["violations"],
            "Monte Carlo SH ret sayısı":int(
                (cov["runs"]["SH mümkün"]==False).sum()),
            "Monte Carlo tekrar":len(cov["runs"])
        })
    reports={"Durum":pd.DataFrame(summary),
             "SH_Kapsama":pd.concat(coverage,ignore_index=True),
             "Kuadratur":pd.concat(quadrature,ignore_index=True),
             "KKT":pd.concat(stationarity,ignore_index=True)}
    with pd.ExcelWriter(destination/"audit.xlsx",engine="xlsxwriter") as writer:
        for sheet,frame in reports.items():
            frame.to_excel(writer,sheet_name=sheet,index=False)
            frame.to_csv(destination/(sheet+".csv"),index=False)
    print(reports["Durum"].to_string(index=False))
    print(reports["SH_Kapsama"].to_string(index=False))

if __name__=="__main__":
    main()
