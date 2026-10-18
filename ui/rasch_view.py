"""Testable helpers for the Streamlit Rasch page (B8). No Streamlit import."""
from time import perf_counter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.rasch import rasch_analysis

MNSQ_OPTIONS = {"0,7–1,3 (çoktan seçmeli testler)": (0.7, 1.3),
                "0,5–1,5 (ölçme için üretken aralık)": (0.5, 1.5),
                "0,8–1,2 (yüksek riskli sınavlar)": (0.8, 1.2)}
METHOD_NOTE = ("Kestirim: marjinal maksimum olabilirlik (MML; doğrudan optimizasyon, Gauss–Hermite). "
               "Ölçek: örtük ortalama 0, örtük standart sapma serbest; tüm madde güçlükleri serbest. "
               "Kişi konumları Warm (1989) WLE; uyum istatistikleri WLE ile hesaplanır. "
               "Bu sayfa, a=1 ve SD=1 sabitlenen 1PL sayfasından farklıdır.")


def screen_items(frame, columns):
    """Columns usable for the dichotomous Rasch model (0/1 with missing allowed)."""
    rows, usable = [], []
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce")
        bad = int((frame[col].notna() & values.isna()).sum() + (values.notna() & ~values.isin([0, 1])).sum())
        observed = values.dropna()
        constant = observed.nunique() < 2
        reason = ("0/1 dışı değer" if bad else "tüm yanıtlar aynı" if constant else "")
        rows.append({"Madde": col, "Gözlenen": int(len(observed)), "Eksik": int(values.isna().sum() - bad),
                     "Doğru oranı": float(observed.mean()) if len(observed) else np.nan,
                     "Durum": reason or "Uygun"})
        if not reason:
            usable.append(col)
    return usable, pd.DataFrame(rows)


def rasch_settings(points=41, mnsq_label=next(iter(MNSQ_OPTIONS)), zstd_limit=2.0, include_extreme=False):
    return {"points": int(points), "mnsq_label": mnsq_label, "mnsq_range": MNSQ_OPTIONS[mnsq_label],
            "zstd_limit": float(zstd_limit), "include_extreme": bool(include_extreme)}


def run_rasch(frame, columns, settings):
    data = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    started = perf_counter()
    result = rasch_analysis(data, points=settings["points"], mnsq_range=settings["mnsq_range"],
                            zstd_limit=settings["zstd_limit"],
                            include_extreme_in_item_fit=settings["include_extreme"])
    result["ui_settings"] = dict(settings, seconds=perf_counter() - started, columns=list(columns))
    return result


def fingerprint(frame, columns):
    data = frame.loc[:, columns]
    return (tuple(columns), data.shape, int(pd.util.hash_pandas_object(data, index=True).sum()))


def summary_rows(result):
    f = result["fit"]
    return pd.DataFrame({"Gösterge": ["Kişi (N)", "Madde (k)", "Örtük SS (σ)", "SH(σ)", "logL", "AIC", "BIC",
                                      "Yakınsama", "Skor normu/kişi", "Eksik hücre oranı", "Uç puanlı kişi",
                                      "Marjinal (EAP) güvenirlik", "Gauss–Hermite düğümü", "Süre (sn)"],
                         "Değer": [f["n"], f["k"], round(f["sd"], 4),
                                   round(f["se_sd"], 4) if np.isfinite(f["se_sd"]) else "—",
                                   round(f["loglik"], 3), round(f["aic"], 2), round(f["bic"], 2),
                                   "Evet" if f["converged"] else "Hayır", f"{f['max_score_per_person']:.2g}",
                                   f"%{100 * f['missing_rate']:.1f}", int(result["extreme"].sum()),
                                   round(result["marginal_reliability"], 3), f["points"],
                                   round(result.get("ui_settings", {}).get("seconds", np.nan), 2)]})


def misfit_persons(result):
    persons = result["persons"]
    return persons[persons["Uyum"] == "İnceleme önerilir"]


def wright_map(result, bins=30):
    """Person WLE histogram (left) and item difficulties (right) on one logit axis."""
    persons = np.asarray(result["wright"]["persons"], float)
    items = result["wright"]["items"]
    lo = float(min(persons.min(), items.min())) - .5
    hi = float(max(persons.max(), items.max())) + .5
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(persons, edges)
    centres = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure()
    fig.add_bar(x=-counts, y=centres, orientation="h", name="Kişiler (WLE)",
                marker_color="#4C78A8", width=(edges[1] - edges[0]) * .9,
                customdata=counts, hovertemplate="θ ≈ %{y:.2f}<br>kişi: %{customdata}<extra></extra>")
    fig.add_scatter(x=np.full(len(items), counts.max() * .15 + 1), y=items.to_numpy(float),
                    mode="markers+text", text=list(items.index), textposition="middle right",
                    name="Maddeler (b)", marker=dict(symbol="diamond", size=10, color="#E45756"),
                    hovertemplate="%{text}<br>b = %{y:.2f}<extra></extra>")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(template="plotly_white", title="Wright Haritası (kişi–madde eşlemesi)",
                      xaxis=dict(title="← kişi sayısı | maddeler →", showticklabels=False, zeroline=True),
                      yaxis=dict(title="Logit (θ ve b)", range=[lo, hi]), barmode="overlay",
                      legend=dict(orientation="h", y=-.15), height=520, margin=dict(l=10, r=10, t=50, b=10))
    return fig
