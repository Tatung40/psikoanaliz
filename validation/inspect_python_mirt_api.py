"""Inspect documented/actual Python mirt fitted-object API without assumptions.

Prints only structural metadata and safe small numeric shapes, never respondent
data. Use in CI to identify a verified parameter extraction route.
"""
import json
import numpy as np
import mirt


def describe(value):
    try:
        if isinstance(value, (np.ndarray, list, tuple)):
            a = np.asarray(value)
            return {"type": type(value).__name__, "shape": list(a.shape),
                    "dtype": str(a.dtype)}
        if isinstance(value, (str, bool, int, float, type(None))):
            return {"type": type(value).__name__}
        if hasattr(value, "shape"):
            return {"type": type(value).__name__, "shape": list(value.shape)}
    except Exception:
        pass
    return {"type": type(value).__name__}


def main():
    report = {"mirt_version": getattr(mirt, "__version__", "unknown"), "models": {}}
    for model in ("1PL", "2PL", "3PL", "4PL", "GRM", "PCM"):
        kw = {"model": model, "n_persons": 100, "n_items": 8, "seed": 2026}
        if model in ("GRM", "PCM"):
            kw["n_categories"] = 4
        x = mirt.simdata(**kw)
        fit_kw = {"model": model}
        if "n_categories" in kw:
            fit_kw["n_categories"] = kw["n_categories"]
        fit = mirt.fit_mirt(x, **fit_kw)
        attrs = {}
        for name in dir(fit):
            if name.startswith("_"):
                continue
            try:
                value = getattr(fit, name)
                if not callable(value):
                    attrs[name] = describe(value)
            except Exception:
                attrs[name] = {"type": "unavailable"}
        try:
            coef = fit.coef()
        except (AttributeError, TypeError, ValueError) as exc:
            coef = None
            coef_error = type(exc).__name__
        else:
            coef_error = None
        if hasattr(coef, "columns"):
            coef_info = {"type": type(coef).__name__,
                         "columns": [str(x) for x in coef.columns],
                         "shape": list(coef.shape)}
        elif isinstance(coef, dict):
            coef_info = {"type": "dict", "keys": list(map(str, coef.keys()))}
        else:
            coef_info = describe(coef)
        model_attrs = {}
        for name in dir(fit.model):
            if name.startswith("_"):
                continue
            try:
                value = getattr(fit.model, name)
                if not callable(value):
                    model_attrs[name] = describe(value)
            except Exception:
                pass
        report["models"][model] = {"fit_type": type(fit).__name__,
                                    "attributes": attrs,
                                    "coef": coef_info,
                                    "coef_error": coef_error,
                                    "model_attributes": model_attrs}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
