"""
PsikoAnaliz — Kullanıcı Arayüzü Bileşenleri (ui/components.py)
Akademik başlık, metrik kartları, bildirim kutuları, özel CSS ve görsel düzenleyiciler.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

import streamlit as st
from typing import Dict, List, Optional, Any


def inject_custom_css():
    """
    PsikoAnaliz için modern, akademik, göze hitap eden özel CSS stillerini enjekte eder.
    """
    st.markdown("""
    <style>
    /* Ana Font ve Arka Plan İyileştirmeleri */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Üst Banner Kartı */
    .academic-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 50%, #2563EB 100%);
        color: white;
        padding: 1.8rem 2.2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 14px rgba(30, 58, 138, 0.15);
    }
    .academic-header h1 {
        color: #FFFFFF !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .academic-header h3 {
        color: #E0E7FF !important;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 0.8rem;
    }
    .academic-header .badge-container {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 0.8rem;
    }
    .academic-badge {
        background-color: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    
    /* Kurumsal marka kimliği: başlık ve sol navigasyon */
    .academic-header {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.14);
        background: linear-gradient(115deg, #142C58 0%, #1E3A8A 58%, #2556A8 100%);
        box-shadow: 0 8px 25px rgba(22,46,91,0.16);
    }
    .academic-header h1 {
        font-size: clamp(2.65rem, 4.4vw, 3.65rem);
        font-weight: 800;
        letter-spacing: -1.6px;
        line-height: 1.12;
        margin-bottom: 0.55rem;
        text-shadow: 0 2px 12px rgba(0,0,0,0.12);
    }
    .psiko-sidebar-brand {
        padding: 1rem 0.35rem 1.15rem;
        margin-bottom: 0.2rem;
        border-bottom: 2px solid #D8E4F7;
        text-align: center;
    }
    .psiko-sidebar-brand-name {
        color: #18356B;
        font-size: clamp(1.9rem, 2.6vw, 2.35rem);
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -1.05px;
    }
    .psiko-sidebar-brand-sub {
        color: #42669C;
        font-size: 0.63rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        margin-top: 0.38rem;
    }
    .psiko-sidebar-brand-author {
        color: #64748B;
        font-size: 0.81rem;
        margin-top: 0.7rem;
    }
    /* Metrik Kartları */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
        border-color: #CBD5E1;
    }
    .metric-card .metric-label {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .metric-card .metric-value {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .metric-card .metric-sub {
        color: #94A3B8;
        font-size: 0.78rem;
    }
    
    /* Akademik Bilgi Kutuları */
    .info-card {
        background-color: #F0F9FF;
        border-left: 4px solid #0284C7;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.2rem;
        color: #0369A1;
        font-size: 0.92rem;
    }
    .warning-card {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.2rem;
        color: #B45309;
        font-size: 0.92rem;
    }
    .success-card {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.2rem;
        color: #047857;
        font-size: 0.92rem;
    }
    
    /* Tablo İyileştirmeleri */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)


def render_academic_header():
    """
    Sayfa üstü kurumsal akademik başlık kartını çizer.
    """
    st.markdown("""
    <div class="academic-header">
        <h1>PsikoAnaliz</h1>
        <h3>Psikometrik Analiz Platformu</h3>
        <p style="margin: 0; font-size: 0.95rem; opacity: 0.92;">
            Klasik Test Kuramı (KTK) ve Madde Tepki Kuramı (MTK) Temelli Ölçme ve Değerlendirme Uygulamaları
        </p>
        <div class="badge-container">
            <span class="academic-badge">👨‍🏫 Doç. Dr. Mehmet Taha Eser</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: Any, sub_text: str = "", delta: Optional[str] = None):
    """
    Stillendirilmiş tek bir metrik kartı çizer.
    """
    delta_html = f'<div style="color: #10B981; font-size: 0.8rem; font-weight: 600;">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub_text}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_apa_table_header(table_number: int, title: str, note: Optional[str] = None):
    """
    APA 7 formatında tablo numarası ve başlığı oluşturur.
    """
    st.markdown(f"""
    <div style="margin-top: 1.5rem; margin-bottom: 0.5rem;">
        <div style="font-weight: 700; font-size: 1rem; color: #0F172A;">Tablo {table_number}</div>
        <div style="font-style: italic; font-size: 0.95rem; color: #334155;">{title}</div>
    </div>
    """, unsafe_allow_html=True)


def render_apa_table_footer(note: str):
    """
    APA 7 formatında tablo altı açıklama notu (Not.) oluşturur.
    """
    st.markdown(f"""
    <div style="font-size: 0.82rem; color: #64748B; margin-top: 0.4rem; margin-bottom: 1.5rem; line-height: 1.4;">
        <em>Not.</em> {note}
    </div>
    """, unsafe_allow_html=True)
