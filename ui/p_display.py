"""Display-only APA formatting for p/q columns of an on-screen table.

Numbers stay untouched in results, decisions, CSV/XLSX exports and multiplicity
corrections; only the frame handed to ``st.dataframe`` (and therefore the
captured academic Word table) is formatted. p < .001, including a computed
floating-point zero (underflow), is shown as "< .001"; never as 0.
"""
import numpy as np
import pandas as pd


def is_p_column(name):
    """p/q value columns by name only (e.g. p, p_adj, p_omnibus, q (BH), Bootstrap p, p_value)."""
    n = str(name).strip().casefold()
    return (n in {"p", "q", "p_value", "p_value_adjusted", "pvalue"} or "p_value" in n or "p-değer" in n
            or "p değer" in n or n.startswith(("p ", "p_", "p(", "p-", "q ", "q_", "q(")) or n.endswith((" p", "_p")))


def format_p(value, digits=4):
    if value is None or isinstance(value, (bool, np.bool_)):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(v):
        return None
    if v < .001:
        return "< .001"
    return f"{v:.{digits}f}"


def apa_p_frame(frame, digits=4):
    """Copy with p/q columns as display strings; other columns are left for the caller to round."""
    out = frame.copy()
    for column in out.columns:
        if is_p_column(column) and pd.api.types.is_numeric_dtype(out[column]) \
                and not pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].map(lambda v: format_p(v, digits)).astype(object)
    return out


def truncation_note(shown, total, unit="kişi", export="CSV/XLSX"):
    return (f"İlk {shown} / {total} {unit} gösteriliyor; tam tablo {export} dışa aktarımındadır."
            if shown < total else "")
