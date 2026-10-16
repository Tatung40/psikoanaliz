"""
PsikoAnaliz — Bilimsel Görselleştirme Modülü (ui/charts.py)
Plotly tabanlı, yayın kalitesinde interaktif psikometrik grafikler.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Akademik Renk Paleti
COLOR_PRIMARY = "#1E3A8A"      # Koyu Mavi
COLOR_SECONDARY = "#0284C7"    # Canlı Mavi
COLOR_SUCCESS = "#10B981"      # Zümrüt Yeşili
COLOR_WARNING = "#F59E0B"      # Kehribar Sarısı
COLOR_DANGER = "#EF4444"       # Kırmızı
COLOR_SLATE = "#64748B"        # Arduvaz Grisi
COLOR_BACKGROUND = "#FFFFFF"   # Beyaz


def apply_academic_layout(fig: go.Figure, title: str, x_title: str = "", y_title: str = "") -> go.Figure:
    """
    Tüm Plotly grafiklerine tutarlı, sade, APA uyumlu akademik tema uygular.
    """
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "y": 0.95,
            "x": 0.05,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": "#0F172A", "family": "Inter, sans-serif"}
        },
        xaxis_title={"text": x_title, "font": {"size": 13, "color": "#334155"}},
        yaxis_title={"text": y_title, "font": {"size": 13, "color": "#334155"}},
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
        font={"family": "Inter, sans-serif", "color": "#1E293B"},
        margin=dict(l=50, r=40, t=60, b=50),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#E2E8F0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E2E8F0", zeroline=False)
    return fig


def plot_score_histogram(scores: pd.Series, bins: int = 20) -> go.Figure:
    """
    1. Toplam Puan Histogramı ve Teorik Normal Eğri Örtüşmesi
    """
    clean = scores.dropna()
    mean = clean.mean()
    sd = clean.std(ddof=1)
    n = len(clean)

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=clean,
        nbinsx=bins,
        name="Gözlenen Puanlar",
        marker_color=COLOR_SECONDARY,
        opacity=0.75,
        histnorm="probability density"
    ))

    # Normal Dağılım Çizgisi
    if sd > 1e-12:
        x_norm = np.linspace(clean.min() - 0.5 * sd, clean.max() + 0.5 * sd, 200)
        y_norm = stats.norm.pdf(x_norm, mean, sd)
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            mode="lines",
            name="Teorik Normal Eğri",
            line=dict(color=COLOR_PRIMARY, width=2.5)
        ))

    # Ortalama Çizgisi
    fig.add_vline(x=mean, line_dash="dash", line_color=COLOR_DANGER, annotation_text=f"Ort = {mean:.2f}")

    return apply_academic_layout(
        fig,
        title="Toplam Puan Dağılımı ve Normal Eğri Uyumu",
        x_title="Toplam Test Puanı",
        y_title="Olasılık Yoğunluğu"
    )


def plot_score_boxplot(scores: pd.Series) -> go.Figure:
    """
    2. Kutu (Boxplot) ve Keman (Violin) Puan Dağılımı
    """
    clean = scores.dropna()
    fig = go.Figure()

    fig.add_trace(go.Violin(
        y=clean,
        box_visible=True,
        meanline_visible=True,
        points="all",
        jitter=0.25,
        scalemode="count",
        marker_color=COLOR_PRIMARY,
        line_color=COLOR_PRIMARY,
        name="Test Puanları",
        fillcolor="rgba(30, 58, 138, 0.2)"
    ))

    return apply_academic_layout(
        fig,
        title="Toplam Puan Kutu ve Keman (Violin) Dağılımı",
        x_title="",
        y_title="Toplam Puan"
    )


def plot_qq_normality(scores: pd.Series) -> go.Figure:
    """
    3. Q-Q Normallik Grafiği
    """
    clean = scores.dropna()
    osm, osr = stats.probplot(clean, dist="norm")
    theo_q, obs_q = osm[0], osm[1]
    slope, intercept, r = osm[1], osr[0], osr[1]

    fig = go.Figure()

    # Saçılım Noktaları
    fig.add_trace(go.Scatter(
        x=theo_q,
        y=obs_q,
        mode="markers",
        name="Gözlenen Değerler",
        marker=dict(color=COLOR_SECONDARY, size=7, opacity=0.8)
    ))

    # Referans Doğrusu
    line_x = np.array([theo_q.min(), theo_q.max()])
    line_y = osr[1] + osr[0] * line_x
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="Normallik Referans Doğrusu",
        line=dict(color=COLOR_DANGER, width=2, dash="dash")
    ))

    return apply_academic_layout(
        fig,
        title=f"Normal Q-Q Grafiği (R² = {osr[2]**2:.4f})",
        x_title="Teorik Normal Yüzdelikler (Standard Normal)",
        y_title="Gözlenen Örneklem Puanları"
    )


def plot_item_difficulty_bar(summary_df: pd.DataFrame) -> go.Figure:
    """
    4. Madde Güçlük İndeksi (p) Çubuk Grafiği (Bölge renklendirmeli)
    """
    df = summary_df.copy().dropna(subset=["Güçlük (p)"])
    colors = []
    for p in df["Güçlük (p)"]:
        if p < 0.20:
            colors.append(COLOR_DANGER)      # Çok zor
        elif p > 0.80:
            colors.append(COLOR_WARNING)     # Çok kolay
        else:
            colors.append(COLOR_SUCCESS)     # İdeal / Orta güçlük

    fig = go.Figure(go.Bar(
        x=df["Madde"],
        y=df["Güçlük (p)"],
        marker_color=colors,
        text=df["Güçlük (p)"].round(2),
        textposition="outside",
        name="Madde Güçlüğü (p)"
    ))

    # Eşik Çizgileri
    fig.add_hline(y=0.20, line_dash="dot", line_color=COLOR_DANGER, annotation_text="Çok Zor (0.20)")
    fig.add_hline(y=0.50, line_dash="dash", line_color=COLOR_PRIMARY, annotation_text="İdeal (0.50)")
    fig.add_hline(y=0.80, line_dash="dot", line_color=COLOR_WARNING, annotation_text="Çok Kolay (0.80)")

    fig.update_yaxes(range=[0, 1.1])
    return apply_academic_layout(
        fig,
        title="Madde Güçlük İndeksleri (p) Dağılımı",
        x_title="Maddeler",
        y_title="Güçlük İndeksi (p)"
    )


def plot_item_discrimination_bar(summary_df: pd.DataFrame, metric: str = "r_ir (Düzeltilmiş)") -> go.Figure:
    """
    5. Madde Ayırt Edicilik Çubuk Grafiği
    """
    df = summary_df.copy().dropna(subset=[metric])
    colors = []
    for d in df[metric]:
        if d < 0.0:
            colors.append(COLOR_DANGER)       # Negatif (Kritik hata)
        elif d < 0.20:
            colors.append(COLOR_WARNING)      # Zayıf
        elif d < 0.30:
            colors.append("#38BDF8")          # Sınırda
        else:
            colors.append(COLOR_SUCCESS)      # Yüksek ayırt edici

    fig = go.Figure(go.Bar(
        x=df["Madde"],
        y=df[metric],
        marker_color=colors,
        text=df[metric].round(2),
        textposition="outside",
        name=metric
    ))

    fig.add_hline(y=0.0, line_color="#0F172A", line_width=1.5)
    fig.add_hline(y=0.20, line_dash="dot", line_color=COLOR_WARNING, annotation_text="Kritik Eşik (0.20)")
    fig.add_hline(y=0.30, line_dash="dash", line_color=COLOR_SUCCESS, annotation_text="İyi Eşik (0.30)")

    return apply_academic_layout(
        fig,
        title=f"Madde Ayırt Edicilik İndeksleri ({metric})",
        x_title="Maddeler",
        y_title=metric
    )


def plot_difficulty_discrimination_scatter(summary_df: pd.DataFrame) -> go.Figure:
    """
    6. Güçlük - Ayırt Edicilik Saçılım (Kuadrant) Grafiği
    """
    df = summary_df.copy().dropna(subset=["Güçlük (p)", "r_ir (Düzeltilmiş)"])

    fig = px.scatter(
        df,
        x="Güçlük (p)",
        y="r_ir (Düzeltilmiş)",
        text="Madde",
        color="Tanısal Bayrak",
        hover_data=["Madde", "Güçlük (p)", "r_ir (Düzeltilmiş)", "D (%27)"],
        size_max=12
    )

    fig.update_traces(textposition="top center", marker=dict(size=10, opacity=0.85))

    # Kuadrant Eksenleri
    fig.add_vline(x=0.50, line_dash="dash", line_color="#94A3B8")
    fig.add_hline(y=0.30, line_dash="dash", line_color=COLOR_SUCCESS, annotation_text="İyi Ayırt Edicilik (0.30)")
    fig.add_hline(y=0.00, line_dash="solid", line_color=COLOR_DANGER)

    fig.update_xaxes(range=[0, 1.05])
    return apply_academic_layout(
        fig,
        title="Madde Güçlük (p) ve Ayırt Edicilik (r_ir) Kuadrant Dağılımı",
        x_title="Madde Güçlüğü (p)",
        y_title="Düzeltilmiş Madde-Toplam Korelasyonu (r_ir)"
    )


def plot_alpha_if_deleted(summary_df: pd.DataFrame, overall_alpha: float) -> go.Figure:
    """
    7. Madde Silindiğinde Alfa Değişim Grafiği
    """
    df = summary_df.copy().dropna(subset=["Silindiğinde Alfa"])

    colors = [COLOR_DANGER if a > (overall_alpha + 0.005) else COLOR_PRIMARY for a in df["Silindiğinde Alfa"]]

    fig = go.Figure(go.Scatter(
        x=df["Madde"],
        y=df["Silindiğinde Alfa"],
        mode="lines+markers",
        marker=dict(size=9, color=colors),
        line=dict(color=COLOR_PRIMARY, width=1.5),
        name="Silindiğinde Alfa"
    ))

    # Mevcut Genel Alfa Çizgisi
    fig.add_hline(
        y=overall_alpha,
        line_dash="dash",
        line_color=COLOR_DANGER,
        line_width=2,
        annotation_text=f"Mevcut Alfa = {overall_alpha:.4f}"
    )

    return apply_academic_layout(
        fig,
        title="Madde Silindiğinde Cronbach Alfa Katsayısı Değişimi",
        x_title="Maddeler",
        y_title="Madde Silindiğinde Alfa"
    )


def plot_distractor_bar(distractor_data: Dict[str, Any], item_name: str) -> go.Figure:
    """
    8. Tek Bir Madde İçin Seçenek Seçim Oranları Çubuk Grafiği
    """
    df_opts = distractor_data.get("options_table", pd.DataFrame())
    if df_opts.empty:
        return go.Figure()

    key_option = distractor_data.get("item_key", "")
    colors = [COLOR_SUCCESS if str(opt) == key_option else COLOR_SECONDARY for opt in df_opts["Seçenek"]]

    fig = go.Figure(go.Bar(
        x=df_opts["Seçenek"],
        y=df_opts["Yüzde (%)"],
        marker_color=colors,
        text=df_opts["Yüzde (%)"].apply(lambda x: f"%{x:.1f}"),
        textposition="outside",
        name="Seçim Oranı"
    ))

    return apply_academic_layout(
        fig,
        title=f"{item_name} Seçenek Seçim Oranları Dağılımı (Doğru Cevap: {key_option})",
        x_title="Seçenekler",
        y_title="Seçilme Yüzdesi (%)"
    )


def plot_upper_lower_options_comparison(distractor_data: Dict[str, Any], item_name: str) -> go.Figure:
    """
    9. Üst %27 ve Alt %27 Grupların Seçenek Tercihleri Karşılaştırma Grafiği
    """
    df_opts = distractor_data.get("options_table", pd.DataFrame())
    if df_opts.empty:
        return go.Figure()

    key_option = distractor_data.get("item_key", "")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_opts["Seçenek"],
        y=df_opts["Üst Grup %"],
        name="Üst %27 Grup",
        marker_color=COLOR_PRIMARY
    ))

    fig.add_trace(go.Bar(
        x=df_opts["Seçenek"],
        y=df_opts["Alt Grup %"],
        name="Alt %27 Grup",
        marker_color=COLOR_WARNING
    ))

    fig.update_layout(barmode="group")
    return apply_academic_layout(
        fig,
        title=f"{item_name} Üst ve Alt %27 Grupların Seçenek Dağılımı (Doğru Cevap: {key_option})",
        x_title="Seçenekler",
        y_title="Grup İçi Seçilme Oranı (%)"
    )


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    10. Maddeler Arası Korelasyon Isı Haritası
    """
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-0.3,
        zmax=1.0,
        labels=dict(color="Korelasyon (r)")
    )

    return apply_academic_layout(
        fig,
        title="Maddeler Arası Pearson Korelasyon Matrisi Isı Haritası",
        x_title="Maddeler",
        y_title="Maddeler"
    )


def plot_group_density_comparison(df: pd.DataFrame, group_col: str, scores: pd.Series) -> go.Figure:
    """
    11. Gruplar Arası Puan Dağılımı Yoğunluk Eğrileri Karşılaştırması
    """
    clean_mask = scores.notna() & df[group_col].notna()
    plot_df = pd.DataFrame({
        "Grup": df.loc[clean_mask, group_col].astype(str),
        "Puan": scores[clean_mask]
    })

    fig = px.histogram(
        plot_df,
        x="Puan",
        color="Grup",
        marginal="box",
        barmode="overlay",
        opacity=0.6,
        histnorm="probability density"
    )

    return apply_academic_layout(
        fig,
        title=f"Gruplara Göre Puan Dağılımı ({group_col})",
        x_title="Toplam Puan",
        y_title="Yoğunluk"
    )


def plot_missing_data_overview(df: pd.DataFrame, item_columns: List[str]) -> go.Figure:
    """
    12. Madde Düzeyinde Kayıp Veri Oranları Grafiği
    """
    missing_pcts = (df[item_columns].isna().sum() / len(df) * 100.0).round(2)
    fig = go.Figure(go.Bar(
        x=item_columns,
        y=missing_pcts,
        marker_color=COLOR_DANGER,
        text=missing_pcts.apply(lambda x: f"%{x:.1f}"),
        textposition="outside"
    ))

    return apply_academic_layout(
        fig,
        title="Maddelere Göre Kayıp (Boş) Yanıt Oranları",
        x_title="Maddeler",
        y_title="Kayıp Veri Oranı (%)"
    )
