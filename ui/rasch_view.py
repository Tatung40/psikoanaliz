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


# ----------------------------------------------------------------------------- B9 helpers
DIF_NOTE = ("Tahmin edilen nicelik: Δ = b_odak − b_referans (logit). Δ > 0, aynı θ düzeyinde maddenin odak "
            "grup için daha zor olduğunu gösterir. Ölçek, referans grubun gizil ortalaması 0 ve çapa "
            "maddelerinin gruplar arasında ortak güçlüğüyle tanımlanır; grupların yetenek farkı ayrıca "
            "kestirilir. p değerleri seçilen çapa kümesine koşulludur. Anlamlı DIF, maddenin yanlı olduğunun "
            "kanıtı değildir; içerik incelemesi gerektirir.")
Q3_NOTE = ("Q3, kişi kestirimi sabit tutulduğunda madde artıklarının korelasyonudur. θ aynı yanıtlardan "
           "kestirildiği için Q3 yerel bağımsızlık altında da negatif yanlıdır (yaklaşık −1/(k−1)); "
           "düzeltilmiş Q3 tüm çift ortalamasını çıkarır. Evrensel bir kesme değeri test olarak "
           "kullanılmaz; yüksek çiftler içerik açısından incelenmelidir.")


def group_candidates(frame, items, max_levels=20):
    """Non-item columns with 2..max_levels distinct non-missing values."""
    out = []
    for col in frame.columns:
        if col in items:
            continue
        n = frame[col].dropna().nunique()
        if 2 <= n <= max_levels:
            out.append(col)
    return out


def dif_figure(dif):
    t = dif["table"].dropna(subset=["Δ (b_odak − b_ref)"])
    colors = np.where(t["DIF işareti"], "#E45756", "#4C78A8")
    fig = go.Figure()
    fig.add_scatter(x=t["Δ (b_odak − b_ref)"], y=t["Madde"], mode="markers", marker=dict(color=colors, size=10),
                    error_x=dict(type="data", array=1.96 * t["SH(Δ)"].fillna(0)), name="Δ ± 1,96 SH",
                    hovertemplate="%{y}: Δ = %{x:.3f}<extra></extra>")
    for x0, dash in ((0, "solid"), (.426, "dot"), (-.426, "dot"), (.638, "dash"), (-.638, "dash")):
        fig.add_vline(x=x0, line_dash=dash, line_color="gray", opacity=.6)
    fig.update_layout(template="plotly_white", title=f"Tek biçimli DIF: {dif['focal']} − {dif['reference']} (logit)",
                      xaxis_title="Δ = b_odak − b_referans", yaxis=dict(autorange="reversed"), height=max(320, 28 * len(t)),
                      margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    return fig


def q3_heatmap(q3):
    adj = q3["adjusted"]
    fig = go.Figure(go.Heatmap(z=adj.to_numpy(), x=list(adj.columns), y=list(adj.index), colorscale="RdBu",
                               zmid=0, colorbar=dict(title="Düz. Q3"),
                               hovertemplate="%{y} – %{x}: %{z:.3f}<extra></extra>"))
    fig.update_layout(template="plotly_white", title="Düzeltilmiş Q3 artık korelasyonları", height=520,
                      yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=50, b=10))
    return fig


def raw_score_figure(gf):
    raw = gf["raw_scores"]
    fig = go.Figure()
    fig.add_bar(x=raw["Ham puan"], y=raw["Gözlenen"], name="Gözlenen", marker_color="#4C78A8")
    fig.add_scatter(x=raw["Ham puan"], y=raw["Beklenen"], name="Model beklenen", mode="lines+markers",
                    line=dict(color="#E45756"))
    fig.update_layout(template="plotly_white", title="Ham puan dağılımı: gözlenen ve beklenen (betimsel)",
                      xaxis_title="Ham puan", yaxis_title="Kişi sayısı", legend=dict(orientation="h", y=-.2),
                      height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def run_global_fit(result, strata=5):
    from core.rasch_globalfit import global_fit
    started = perf_counter()
    out = global_fit(result, points=result["ui_settings"]["points"], strata=int(strata))
    out["ui_settings"] = {"strata": int(strata), "seconds": perf_counter() - started}
    return out


def m2_rows(gf):
    m = gf["m2"]
    if not m["available"]:
        return pd.DataFrame({"Gösterge": ["M2"], "Değer": [m["reason"]]})
    return pd.DataFrame({"Gösterge": ["M2", "sd", "p (asimptotik χ²)", "RMSEA2", "SRMSR", "N", "k",
                                      "Serbest parametre (b + σ)"],
                         "Değer": [round(m["M2"], 3), m["df"], round(m["p"], 4), round(m["RMSEA2"], 4),
                                   round(m["SRMSR"], 4), m["n"], m["k"], m["k"] + 1]})


def run_q3(frame, columns, result, person_method="WLE", min_pairs=30, bootstrap=0, seed=2026):
    from core.rasch_q3 import q3_analysis
    data = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    started = perf_counter()
    f = result["fit"]
    out = q3_analysis(data, b=f["b"], sd=f["sd"], person_method=person_method, min_pairs=int(min_pairs),
                      bootstrap=int(bootstrap), seed=int(seed), points=result["ui_settings"]["points"])
    out["ui_settings"] = {"person_method": person_method, "min_pairs": int(min_pairs),
                          "bootstrap": int(bootstrap), "seed": int(seed), "seconds": perf_counter() - started}
    return out


def run_dif(frame, columns, group_col, reference, focal, alpha=.05, purify=True, max_rounds=10,
            min_anchors=None, points=41):
    from core.rasch_dif import dif_analysis
    if group_col in columns:
        raise ValueError("Grup değişkeni madde olarak kullanılamaz.")
    data = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    group = frame[group_col].reset_index(drop=True)
    started = perf_counter()
    out = dif_analysis(data, group, reference, focal, alpha=float(alpha), purify=bool(purify),
                       max_rounds=int(max_rounds), min_anchors=min_anchors, points=int(points))
    out["ui_settings"] = {"group_col": group_col, "reference": reference, "focal": focal, "alpha": float(alpha),
                          "purify": bool(purify), "max_rounds": int(max_rounds), "min_anchors": min_anchors,
                          "points": int(points), "seconds": perf_counter() - started}
    return out


def dif_summary_rows(dif):
    j = dif["joint"]
    return pd.DataFrame({"Gösterge": [f"N ({dif['reference']}, referans)", f"N ({dif['focal']}, odak)",
                                      "Çapa madde sayısı", "Arındırma", "Arındırma kararlı",
                                      "Odak grup gizil ortalaması (μ_odak)", "σ_referans", "σ_odak",
                                      "logL (ortak model)", "Yakınsama", "Anlamlılık (BH-FDR)"],
                         "Değer": [dif["sizes"][dif["reference"]], dif["sizes"][dif["focal"]], len(dif["anchors"]),
                                   "Evet" if dif["purify"] else "Hayır", "Evet" if dif["stable"] else "Hayır",
                                   round(j["mu_F"], 4), round(j["sd_R"], 4), round(j["sd_F"], 4),
                                   round(j["loglik"], 3), "Evet" if j["converged"] else "Hayır", dif["alpha"]]})
