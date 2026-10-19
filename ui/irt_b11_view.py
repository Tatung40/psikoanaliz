"""Testable helpers for the B11 3PL/4PL diagnostic tabs (no Streamlit import)."""
from time import perf_counter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.irt_mg import as_matrix
from core.irt_mg34 import default_bounds
from core.irt_q3 import q3_analysis, wle_general, eap, probabilities
from core.irt_dif34 import dif_analysis_34
from core.irt_globalfit import global_fit_34
from core.irt_se_diagnostics import hessian_boundary_diagnostics
from ui.irt_optimization import run_irt_calibration
from ui.irt_b10_view import result_key, dif_effect_figure                 # model-agnostic B10 helpers
from ui.rasch_view import group_candidates, q3_heatmap, raw_score_figure  # noqa: F401  (re-exported)

B11_MODELS = ("3PL", "4PL")
DIF_INFERENTIAL = {"3PL": True, "4PL": False}          # 4PL: descriptive only (see docs/IRT_B11_VALIDATION.md)
MODEL_NOTE = {
    "3PL": ("Model: P(θ) = c + (1 − c)·lojistik[a(θ − b)], θ ~ N(0, 1). Kestirim sınırlı (kutu kısıtlı) ML'dir; önsel "
            "veya ceza kullanılmaz: a ∈ [0,15; a üst sınırı], b ∈ [−4, 4], c ∈ [0; 0,35]. Bir parametre sınırdaysa "
            "olağan asimptotik çıkarım geçerli değildir; bu sekmeler o durumda p değeri vermez."),
    "4PL": ("Model: P(θ) = c + (d − c)·lojistik[a(θ − b)], θ ~ N(0, 1), 0 ≤ c < d ≤ 1. Kestirim sınırlı ML'dir; önsel "
            "kullanılmaz: a ∈ [0,15; a üst sınırı], b ∈ [−4, 4], c ∈ [0; 0,35], d ∈ [d alt sınırı; 1]. c ve d çoğu "
            "örneklemde zayıf tanımlanır; sınır kestirimleri ve kötü koşullu bilgi matrisleri yaygındır.")}
Q3_NOTE = ("Artıklar modelin tam tepki fonksiyonuyla (3PL'de c, 4PL'de c ve d dahil) hesaplanır; 2PL lojistiği "
           "kullanılmaz. "
           "Q3 yerel bağımsızlık altında da negatif yanlıdır (≈ −1/(k−1)); düzeltilmiş Q3 tüm çift ortalamasını çıkarır. "
           "Eğimler bağımlılığın bir kısmını soğurabildiğinden 3PL/4PL'de Q3'ün duyarlılığı sınırlıdır; evrensel bir "
           "kesme değeri karar kuralı olarak kullanılmaz.")
DIF_NOTE = {
    "3PL": ("Ortak ölçek: referans grupta θ ~ N(0, 1); odak grubun ortalaması ve SS'si serbest; çapa maddelerinin a, b, c "
            "parametreleri iki grupta eşittir. Genel LR testi (sd = 3) a, b ve c farkını birlikte sınar; yalnızca maddenin "
            "parametreleri iki modelde de iç noktadaysa asimptotik χ² p değeri verilir (diğer maddelerin sınır "
            "parametreleri sabit kabul edilir). Madde sınıra dokunuyorsa (örn. c = 0) sonuç betimseldir. Tek "
            "parametreli testler (Δa, Δb, Δc) bu parametreleştirmedeki hipotezlerdir; c > 0 iken yalnızca b farkı "
            "sabit bir logit kayması değildir, DIF'in biçimi grup tepki fonksiyonlarından okunur."),
    "4PL": ("4PL DIF için çıkarımsal test DOĞRULANMADI: c ve d'nin sınır davranışı ve zayıf tanımlanması nedeniyle χ² "
            "veya Wald referansı savunulabilir değildir. Bu sekme yalnızca betimsel karşılaştırma verir: her madde, "
            "diğer tüm maddeler çapa alınarak grup bazında kestirilir; Δa, Δb, Δc, Δd ve olasılık ölçeğinde farklar "
            "gösterilir; p değeri ve arındırma yoktur.")}
COMMON_NOTE = ("p değerleri (varsa) verinin seçtiği çapa kümesine koşulludur. Anlamlı DIF maddenin yanlı olduğunun kanıtı "
               "değildir; içerik incelemesi gerektirir.")


def _items(fit):
    t = fit["items"]
    return (t["a"].to_numpy(float), t["b"].to_numpy(float), t["c"].to_numpy(float), t["d"].to_numpy(float))


def boundary_mask(fit, model):
    """Parameters within the shared B1 tolerance of a bound (same rule as the Hessian SE gate)."""
    k = len(fit["items"])
    diag = hessian_boundary_diagnostics(fit["items"].assign(Madde=np.arange(k)), model,
                                        a_upper=fit.get("a_upper", 4.), d_lower=fit.get("d_lower", .65))
    mask = {p: np.zeros(k, bool) for p in "abcd"}
    for _, r in diag.iterrows():
        mask[r["Parametre"]][int(r["Madde"])] = True
    return mask, diag


def person_estimates(frame, fit, method="EAP"):
    x, obs = as_matrix(frame)
    a, b, c, d = _items(fit)
    if method == "EAP":
        return eap(x, obs, a, b, int(fit.get("settings", {}).get("points", 41)), c=c, d=d)[0]
    if method == "WLE":
        return wle_general(x, obs, a, b, c, d)[0]
    raise ValueError("Kişi kestirimi EAP veya WLE olmalıdır.")


def run_global_fit(frame, fit, model, person_method="EAP", strata=5):
    started = perf_counter()
    a, b, c, d = _items(fit)
    mask, diag = boundary_mask(fit, model)
    out = global_fit_34(frame, a, b, c, d, model, person_estimates(frame, fit, person_method), person_method,
                        converged=bool(fit["converged"]), fixed=mask,
                        points=int(fit.get("settings", {}).get("points", 41)), strata=int(strata))
    out["boundary"] = diag
    out["ui_settings"] = {"person_method": person_method, "strata": int(strata), "seconds": perf_counter() - started}
    return out


def m2_rows(gf):
    m = gf["m2"]
    if not m["available"]:
        return pd.DataFrame({"Gösterge": ["M2"], "Değer": [m["reason"]]})
    fmt = lambda v, n: "— (sınır: verilmez)" if not np.isfinite(v) else round(v, n)
    return pd.DataFrame({"Gösterge": ["M2", "sd", "p (asimptotik χ²)", "RMSEA2", "SRMSR", "N", "k",
                                      "Serbest parametre (q)", "Sınırda sabit kabul edilen", "Δ'Ξ⁻¹Δ koşul sayısı"],
                         "Değer": [round(m["M2"], 3), m["df"], fmt(m["p_reported"], 4), fmt(m["RMSEA2_reported"], 4),
                                   round(m["SRMSR"], 4), m["n"], m["k"], m["free_parameters"], m["fixed_parameters"],
                                   f"{m['condition']:.2g}"]})


def item_fit_figure(gf, fit, item):
    s = gf["strata"][gf["strata"]["Madde"] == item]
    row = fit["items"].set_index("Madde").loc[item]
    grid = np.linspace(-4, 4, 161)
    icc = probabilities(grid, [row["a"]], [row["b"]], [row["c"]], [row["d"]])[:, 0]
    fig = go.Figure()
    fig.add_scatter(x=grid, y=icc, mode="lines", name="Model (ICC)", line=dict(color="#E45756"))
    fig.add_scatter(x=s["Ortalama θ"], y=s["Gözlenen p"], mode="markers", name="Gözlenen oran (tabaka)",
                    marker=dict(size=10, color="#4C78A8"),
                    error_y=dict(type="data", array=1.96 * np.sqrt(s["Beklenen p"] * (1 - s["Beklenen p"]) / s["n"])),
                    customdata=s["n"], hovertemplate="θ̄ = %{x:.2f}<br>p = %{y:.3f}<br>n = %{customdata}<extra></extra>")
    fig.add_hline(y=float(row["c"]), line_dash="dot", line_color="gray", annotation_text="c")
    if float(row["d"]) < 1:
        fig.add_hline(y=float(row["d"]), line_dash="dot", line_color="gray", annotation_text="d")
    fig.update_layout(template="plotly_white", title=f"{item}: gözlenen ve beklenen doğru oranı (betimsel)",
                      xaxis_title=f"{gf['theta_method']} θ", yaxis_title="P(X = 1)", yaxis=dict(range=[0, 1]),
                      legend=dict(orientation="h", y=-.2), height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def run_q3(frame, fit, model, person_method="EAP", min_pairs=30, bootstrap=0, seed=2026):
    started = perf_counter()
    settings = dict(fit["settings"])
    a, b, c, d = _items(fit)

    def refit(sim):                                     # the production estimator with the shown settings
        f = run_irt_calibration(sim, settings)
        aa, bb, cc, dd = _items(f)
        return aa, bb, cc, dd, bool(f["converged"])

    out = q3_analysis(frame, a, b, model, person_method=person_method, min_pairs=int(min_pairs),
                      bootstrap=int(bootstrap), seed=int(seed), points=int(settings.get("points", 41)), refit=refit,
                      converged=bool(fit["converged"]), c=c, d=d)
    out["ui_settings"] = {"person_method": person_method, "min_pairs": int(min_pairs), "bootstrap": int(bootstrap),
                          "seed": int(seed), "seconds": perf_counter() - started}
    return out


def run_dif(frame, columns, group_col, reference, focal, model, fit, alpha=.05, purify=True, max_rounds=5,
            points=31):
    if group_col in columns:
        raise ValueError("Grup değişkeni madde olarak kullanılamaz.")
    data = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    bounds = default_bounds(fit.get("a_upper", 4.), fit.get("d_lower", .65))
    started = perf_counter()
    out = dif_analysis_34(data, frame[group_col].reset_index(drop=True), reference, focal, model=model,
                          alpha=float(alpha), purify=bool(purify), max_rounds=int(max_rounds), points=int(points),
                          bounds=bounds, inferential=DIF_INFERENTIAL[model])
    out["ui_settings"] = {"group_col": group_col, "reference": reference, "focal": focal, "alpha": float(alpha),
                          "purify": bool(purify), "max_rounds": int(max_rounds), "points": int(points),
                          "seconds": perf_counter() - started}
    return out


def dif_summary_rows(dif):
    j = dif["joint"]
    return pd.DataFrame({"Gösterge": [f"N ({dif['reference']}, referans)", f"N ({dif['focal']}, odak)", "Model",
                                      "Çıkarım", "Çapa madde sayısı", "Arındırma", "μ_odak", "σ_odak",
                                      "logL (ortak model)", "Yakınsama (KKT)", "Sınırdaki parametre", "Süre (sn)"],
                         "Değer": [dif["sizes"][dif["reference"]], dif["sizes"][dif["focal"]], dif["model"],
                                   "Asimptotik (iç nokta koşullu)" if dif["inferential"] else "Yalnızca betimsel",
                                   len(dif["anchors"]), "Evet" if dif["purify"] else "Hayır",
                                   round(j.get("mu_F", np.nan), 4), round(j.get("sd_F", np.nan), 4),
                                   round(j["loglik"], 3), "Evet" if j["converged"] else "Hayır",
                                   len(j["bound_labels"]), round(dif.get("ui_settings", {}).get("seconds", np.nan), 1)]})


def group_irf_figure(dif, item):
    row = dif["table"].set_index("Madde").loc[item]
    j = dif["joint"]; grid = np.linspace(-4, 4, 161)
    ref = [row["a_ref"], row["b_ref"], row["c_ref"], row["d_ref"]]
    foc = [row["a_odak"], row["b_odak"], row["c_odak"], row["d_odak"]]
    fig = go.Figure()
    fig.add_scatter(x=grid, y=probabilities(grid, *[[v] for v in ref])[:, 0], mode="lines",
                    name=f"{dif['reference']} (referans)", line=dict(color="#4C78A8"))
    fig.add_scatter(x=grid, y=probabilities(grid, *[[v] for v in foc])[:, 0], mode="lines",
                    name=f"{dif['focal']} (odak)", line=dict(color="#E45756", dash="dash"))
    mu, sd = j.get("mu_F", 0.), j.get("sd_F", 1.)
    fig.add_vrect(x0=mu - 1.96 * sd, x1=mu + 1.96 * sd, fillcolor="#E45756", opacity=.06, line_width=0)
    fig.update_layout(template="plotly_white", title=f"{item}: grup tepki fonksiyonları (ortak ölçek)",
                      xaxis_title="θ (referans: N(0,1))", yaxis_title="P(X = 1)", yaxis=dict(range=[0, 1]),
                      legend=dict(orientation="h", y=-.2), height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig
