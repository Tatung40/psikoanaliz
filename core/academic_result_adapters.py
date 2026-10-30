"""Read-only pilot adapters for 2PL, RSM, DIF and G/D study outputs.

All numerical values come from existing engines; no estimation or significance testing.
"""
import math
import pandas as pd
from core.academic_result_schema import build_result


def _clean(value):
    """Explicitly mark missing diagnostics, never invent estimates."""
    if isinstance(value, pd.DataFrame):
        return [_clean(row) for row in value.to_dict(orient="records")]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _clean(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def irt_report_to_result(report):
    """Adapt a completed LibraryReport (2PL or RSM); no re-fit."""
    if report.model not in ("2PL", "RSM"):
        raise ValueError("Pilot supports only 2PL and RSM")
    m = report.metadata
    if report.errors:
        raise ValueError("Incomplete model report: " + "; ".join(map(str, report.errors)))
    if report.parameters.empty:
        raise ValueError("No calibrated item parameters")
    n, k = int(m["N"]), int(m["Madde"])
    if len(report.parameters["Madde"].unique()) != k:
        raise ValueError("Item count inconsistent with calibrated parameters")
    diagnostics = {key: m[key] for key in ("LogLik", "AIC", "BIC", "Yakınsama", "İterasyon", "Serbest_parametre") if key in m}
    if m.get("Yakınsama") is not True:
        raise ValueError("Convergence not confirmed")
    return build_result(family="IRT", model=report.model, n=n, k=k,
        parameters=_clean(report.parameters),
        standard_errors=[],  # No SE is reported by this adapter.
        fit_statistics=_clean(diagnostics),
        tables=_clean({"item_parameters": report.parameters,
                       "category_curves": report.curves, "information": report.information}),
        warnings=list(map(str, report.warnings)) + ["Standard errors not supplied by this report adapter."],
        provenance={"engine": m.get("Motor"), "estimation": m.get("Kestirim"),
                    "status": m.get("Kullanım durumu"), "source": "LibraryReport"},
        metadata={"analysis_type": "item_calibration", "model_metadata": _clean(m)})


def dif_report_to_result(result):
    """Adapt completed multigroup Python mirt DIF report; inference remains disabled."""
    m, tables = result["metadata"], result["tables"]
    if m.get("Inference") is not False or m.get("Correction_applied") is not False:
        raise ValueError("Unexpected DIF inference state")
    if "DIF" not in tables or "Parameters" not in tables:
        raise ValueError("Missing DIF tables")
    n = int(m["N_reference"]) + int(m["N_focal"])
    items = list(m["Items"])
    return build_result(family="IRT", model=str(m["Model"]), n=n, k=len(items),
        parameters=_clean(tables["Parameters"]),
        standard_errors=[],
        fit_statistics=_clean({"fit_models": tables.get("Fit", pd.DataFrame())}),
        tables=_clean({key: value for key, value in tables.items() if isinstance(value, pd.DataFrame)}),
        warnings=[str(m.get("Limitations", "DIF inference not certified")),
                  "p/q and DIF decisions are unavailable; do not infer significance from LR."],
        provenance={"engine": m.get("Engine"), "status": m.get("Status"),
                    "data_sha256": m.get("Data_SHA256"), "source": "python_mirt_dif"},
        metadata={"analysis_type": "DIF", "reference": m.get("Reference"),
                  "focal": m.get("Focal"), "anchors": _clean(m.get("Anchors", [])),
                  "inference_enabled": False})


def gt_report_to_result(result):
    """Adapt completed GeneralizIT G-study and D-study output without recomputation."""
    counts = result["counts"]
    if result.get("status") != "LIBRARY TESTED" or "p" not in counts:
        raise ValueError("Unverified generalizability result")
    required = ("anova", "variance", "g", "errors", "d")
    if any(not isinstance(result.get(key), pd.DataFrame) for key in required):
        raise ValueError("Missing G/D study table")
    settings = result.get("settings", {})
    return build_result(family="GT", model=str(settings.get("design", "G-study")), n=int(counts["p"]),
        k=int(counts["i"]) if "i" in counts else None,
        parameters=_clean(result["variance"].reset_index().rename(columns={"index": "component"})),
        standard_errors=[],
        fit_statistics={},
        tables=_clean({key: result[key].reset_index() for key in ("anova", "variance", "g", "errors", "d")}),
        warnings=[str(result.get("method", "")),
                  "Variance components and G/D coefficients are descriptive; no CI or random-effect p-values."],
        provenance={"engine": "GeneralizIT", "status": result["status"],
                    "data_sha256": settings.get("input_sha256"), "source": "generalizability.analyze"},
        metadata={"analysis_type": "G_and_D_study", "facet_counts": _clean(counts),
                  "settings": _clean(settings)})
