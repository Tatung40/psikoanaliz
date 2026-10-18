"""Testable helpers for the B10 1PL/2PL diagnostic tabs (no Streamlit import)."""
from time import perf_counter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.special import expit
from core.irt_q3 import q3_analysis, wle, eap
from core.irt_mg import as_matrix
from core.irt_dif import dif_analysis
from core.irt_globalfit import global_fit
from ui.irt_optimization import run_irt_calibration
from ui.rasch_view import group_candidates, q3_heatmap, raw_score_figure   # shared, model-agnostic

B10_MODELS = ("1PL", "2PL")
Q3_NOTE = ("Q3, kişi kestirimi sabit tutulduğunda madde artıklarının korelasyonudur; artıklar seçili modelin "
           "(1PL: a = 1; 2PL: maddeye özgü a) tepki olasılığıyla hesaplanır. θ aynı yanıtlardan kestirildiği için "
           "Q3 yerel bağımsızlık altında da negatif yanlıdır (yaklaşık −1/(k−1)); düzeltilmiş Q3 tüm çift "
           "ortalamasını çıkarır. Evrensel bir kesme değeri karar kuralı olarak kullanılmaz.")
DIF_NOTE = {
    "1PL": ("Ortak ölçek: referans grupta θ ~ N(0, 1); odak grubun ortalaması ve standart sapması serbest; çapa "
            "maddeleri iki grupta aynı parametreye sahiptir; a = 1 sabittir. Δb = b_odak − b_referans; Δb > 0 "
            "maddenin aynı θ düzeyinde odak grup için daha zor olduğunu gösterir. 1PL'de maddeye özgü ayırt "
            "edicilik kestirilmediği için tek biçimli olmayan DIF bu model içinde tanımlanamaz; eğim farkları "
            "yalnızca 2PL sayfasında, farklı bir modelle incelenebilir."),
    "2PL": ("Ortak ölçek: referans grupta θ ~ N(0, 1); odak grubun ortalaması ve standart sapması serbest; çapa "
            "maddeleri iki grupta aynı a ve b'ye sahiptir. Genel LR testi (sd = 2) a ve b farkını birlikte sınar; "
            "tek biçimli (sd = 1) ve tek biçimli olmayan (sd = 1) LR testleri ayrıştırmayı verir. DIF türü grup "
            "tepki fonksiyonlarının kesişip kesişmediğine göre belirlenir; yalnızca b farkı tek biçimli DIF "
            "anlamına gelmez.")}
COMMON_NOTE = ("p değerleri verinin seçtiği çapa kümesine koşulludur (seçim belirsizliği SH'ye ve LR referans "
               "dağılımına yansımaz). Yakınsamayan veya sınırda kalan modellerde p değeri verilmez. Anlamlı DIF, "
               "maddenin yanlı olduğunun kanıtı değildir; içerik incelemesi gerektirir.")


def result_key(model, fingerprint, fit):
    """Ties B10 results to the model, the data and the exact calibration run."""
    s = fit.get("settings", {})
    return (model, fingerprint, tuple(sorted((k, str(v)) for k, v in s.items())))


def person_estimates(frame, fit, method="WLE"):
    x, obs = as_matrix(frame)
    a, b = fit["items"]["a"].to_numpy(float), fit["items"]["b"].to_numpy(float)
    if method == "WLE":
        return wle(x, obs, a, b)[0]
    if method == "EAP":
        return eap(x, obs, a, b, int(fit.get("settings", {}).get("points", 41)))[0]
    raise ValueError("Kişi kestirimi WLE veya EAP olmalıdır.")


def run_global_fit(frame, fit, model, person_method="WLE", strata=5):
    started = perf_counter()
    theta = person_estimates(frame, fit, person_method)
    points = int(fit.get("settings", {}).get("points", 41))
    out = global_fit(frame, fit["items"]["a"], fit["items"]["b"], model, theta, person_method,
                     converged=bool(fit["converged"]), points=points, strata=int(strata))
    out["ui_settings"] = {"person_method": person_method, "strata": int(strata), "seconds": perf_counter() - started}
    return out


def m2_rows(gf):
    m = gf["m2"]
    if not m["available"]:
        return pd.DataFrame({"Gösterge": ["M2"], "Değer": [m["reason"]]})
    return pd.DataFrame({"Gösterge": ["M2", "sd", "p (asimptotik χ²)", "RMSEA2", "SRMSR", "N", "k",
                                      "Serbest madde parametresi (q)"],
                         "Değer": [round(m["M2"], 3), m["df"], round(m["p"], 4), round(m["RMSEA2"], 4),
                                   round(m["SRMSR"], 4), m["n"], m["k"], m["free_parameters"]]})


def item_fit_figure(gf, fit, item):
    s = gf["strata"][gf["strata"]["Madde"] == item]
    row = fit["items"].set_index("Madde").loc[item]
    grid = np.linspace(-4, 4, 161)
    fig = go.Figure()
    fig.add_scatter(x=grid, y=expit(float(row["a"]) * (grid - float(row["b"]))), mode="lines", name="Model (ICC)",
                    line=dict(color="#E45756"))
    fig.add_scatter(x=s["Ortalama θ"], y=s["Gözlenen p"], mode="markers", name="Gözlenen oran (tabaka)",
                    marker=dict(size=10, color="#4C78A8"),
                    error_y=dict(type="data", array=1.96 * np.sqrt(s["Beklenen p"] * (1 - s["Beklenen p"]) / s["n"])),
                    customdata=s["n"], hovertemplate="θ̄ = %{x:.2f}<br>p = %{y:.3f}<br>n = %{customdata}<extra></extra>")
    fig.update_layout(template="plotly_white", title=f"{item}: gözlenen ve beklenen doğru oranı (betimsel)",
                      xaxis_title=f"{gf['theta_method']} θ", yaxis_title="P(X = 1)", yaxis=dict(range=[0, 1]),
                      legend=dict(orientation="h", y=-.2), height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def run_q3(frame, fit, model, person_method="WLE", min_pairs=30, bootstrap=0, seed=2026):
    started = perf_counter()
    settings = dict(fit.get("settings", {"model": model, "points": 41, "max_iter": 70, "a_upper": 4.0,
                                         "d_lower": .65, "starts": 1, "refine": False}))

    def refit(sim):                                     # same estimator and settings as the shown model
        f = run_irt_calibration(sim, settings)
        return f["items"]["a"].to_numpy(float), f["items"]["b"].to_numpy(float), bool(f["converged"])

    out = q3_analysis(frame, fit["items"]["a"], fit["items"]["b"], model, person_method=person_method,
                      min_pairs=int(min_pairs), bootstrap=int(bootstrap), seed=int(seed),
                      points=int(settings.get("points", 41)), refit=refit, converged=bool(fit["converged"]))
    out["ui_settings"] = {"person_method": person_method, "min_pairs": int(min_pairs), "bootstrap": int(bootstrap),
                          "seed": int(seed), "seconds": perf_counter() - started}
    return out


def run_dif(frame, columns, group_col, reference, focal, model, alpha=.05, purify=True, max_rounds=10, points=41):
    if group_col in columns:
        raise ValueError("Grup değişkeni madde olarak kullanılamaz.")
    data = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    started = perf_counter()
    out = dif_analysis(data, frame[group_col].reset_index(drop=True), reference, focal, model=model,
                       alpha=float(alpha), purify=bool(purify), max_rounds=int(max_rounds), points=int(points))
    out["ui_settings"] = {"group_col": group_col, "reference": reference, "focal": focal, "alpha": float(alpha),
                          "purify": bool(purify), "max_rounds": int(max_rounds), "points": int(points),
                          "seconds": perf_counter() - started}
    return out


def dif_summary_rows(dif):
    j = dif["joint"]
    return pd.DataFrame({"Gösterge": [f"N ({dif['reference']}, referans)", f"N ({dif['focal']}, odak)", "Model",
                                      "Çapa madde sayısı", "Arındırma", "Arındırma kararlı",
                                      "Odak grup gizil ortalaması (μ_odak)", "Odak grup gizil SS (σ_odak)",
                                      "logL (ortak model)", "Yakınsama", "Anlamlılık (BH-FDR)", "Süre (sn)"],
                         "Değer": [dif["sizes"][dif["reference"]], dif["sizes"][dif["focal"]], dif["model"],
                                   len(dif["anchors"]), "Evet" if dif["purify"] else "Hayır",
                                   "Evet" if dif["stable"] else "Hayır",
                                   round(j.get("mu_F", np.nan), 4), round(j.get("sd_F", np.nan), 4),
                                   round(j["loglik"], 3), "Evet" if j["converged"] else "Hayır", dif["alpha"],
                                   round(dif.get("ui_settings", {}).get("seconds", np.nan), 1)]})


def dif_effect_figure(dif):
    t = dif["table"].dropna(subset=["İşaretli fark E_F[P_F − P_R]"])
    colors = np.where(t["DIF işareti"], "#E45756", "#4C78A8")
    fig = go.Figure()
    fig.add_bar(x=t["İşaretli fark E_F[P_F − P_R]"], y=t["Madde"], orientation="h", marker_color=colors,
                name="İşaretli fark", hovertemplate="%{y}: %{x:.3f}<extra></extra>")
    fig.add_scatter(x=t["İşaretsiz fark E_F|P_F − P_R|"], y=t["Madde"], mode="markers", name="İşaretsiz fark",
                    marker=dict(symbol="diamond", color="#333", size=8))
    fig.add_vline(x=0, line_color="gray")
    fig.update_layout(template="plotly_white", barmode="overlay",
                      title=f"DIF etki büyüklüğü, olasılık ölçeğinde ({dif['focal']} − {dif['reference']})",
                      xaxis_title="E_odak[P_odak(θ) − P_ref(θ)] (negatif: odak grup için daha zor)",
                      yaxis=dict(autorange="reversed"), height=max(320, 28 * len(t)),
                      legend=dict(orientation="h", y=-.2), margin=dict(l=10, r=10, t=50, b=10))
    return fig


def group_irf_figure(dif, item):
    row = dif["table"].set_index("Madde").loc[item]
    j = dif["joint"]; grid = np.linspace(-4, 4, 161)
    fig = go.Figure()
    fig.add_scatter(x=grid, y=expit(row["a_ref"] * (grid - row["b_ref"])), mode="lines",
                    name=f"{dif['reference']} (referans)", line=dict(color="#4C78A8"))
    fig.add_scatter(x=grid, y=expit(row["a_odak"] * (grid - row["b_odak"])), mode="lines",
                    name=f"{dif['focal']} (odak)", line=dict(color="#E45756", dash="dash"))
    if dif["model"] == "2PL" and np.isfinite(row.get("Kesişim θ*", np.nan)) and abs(row["Kesişim θ*"]) <= 4:
        fig.add_vline(x=row["Kesişim θ*"], line_dash="dot", line_color="gray", annotation_text="θ*")
    mu, sd = j.get("mu_F", 0.), j.get("sd_F", 1.)
    fig.add_vrect(x0=mu - 1.96 * sd, x1=mu + 1.96 * sd, fillcolor="#E45756", opacity=.06, line_width=0)
    fig.update_layout(template="plotly_white", title=f"{item}: grup tepki fonksiyonları (ortak ölçek)",
                      xaxis_title="θ (referans: N(0,1))", yaxis_title="P(X = 1)", yaxis=dict(range=[0, 1]),
                      legend=dict(orientation="h", y=-.2), height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig
