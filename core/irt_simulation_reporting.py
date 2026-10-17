"""Reproducible, opt-in IRT simulation benchmarking and export.

No Streamlit dependency, paid service, R installation or automatic heavy runs.
Clock readings are wall time for a given machine, not universal performance.
"""
from io import BytesIO
from time import perf_counter
import json
import numpy as np
import pandas as pd

from core.irt_simulation import repeated_recovery


def run_timed_recovery(true_items, models=("3PL", "4PL"),
                       sample_sizes=(200, 500, 1000, 2000),
                       repetitions=10, seed=2026, **fit_kwargs):
    """Time an explicitly requested simulation study; preserve all run outputs."""
    started=perf_counter()
    result=repeated_recovery(
        true_items, models=models, sample_sizes=sample_sizes,
        repetitions=repetitions, seed=seed, **fit_kwargs)
    elapsed=perf_counter()-started
    # Record settings for repeatability; do not claim a scientific finding.
    result["metadata"]={
        "seed":int(seed), "models":list(models),
        "sample_sizes":[int(n) for n in sample_sizes],
        "repetitions":int(repetitions),
        "fit_kwargs":dict(fit_kwargs),
        "elapsed_seconds":float(elapsed),
        "total_model_fits":int(len(result["runs"])),
        "successful_convergence":int(
            result["runs"]["Durum"].eq("Yakınsadı").sum()),
        "note":"Elapsed wall time depends on available CPU and memory. "
               "Recovery metrics use converged runs only."
    }
    return result


def export_recovery_excel(result):
    """Generate a standalone in-memory XLSX; no upload and no disk writes."""
    output=BytesIO()
    with pd.ExcelWriter(output,engine="xlsxwriter",
                        engine_kwargs={"options":{"in_memory":True}}) as writer:
        for name,key in (("Özet","summary"),("Koşular","runs"),
                         ("Madde_Parametre","item_recovery")):
            frame=result[key]
            (frame if not frame.empty else pd.DataFrame({"Bilgi":["Veri yok"]})
             ).to_excel(writer,sheet_name=name,index=False)
        meta=result.get("metadata",{})
        pd.DataFrame([{"Ayar":k,"Değer":json.dumps(v,ensure_ascii=False)
                       if isinstance(v,(list,dict)) else v}
                      for k,v in meta.items()]).to_excel(
            writer,sheet_name="Çalışma_Bilgileri",index=False)
    return output.getvalue()
