"""
PsikoAnaliz — Klasik Test Kuramı Analiz Platformu
Web Tabanlı Psikometrik Analiz, Çeldirici İnceleme ve Raporlama Sistemi
Geliştirici: Doç. Dr. Mehmet Taha Eser
Lisans: MIT (Açık Kaynak, 0 TL Bütçe)
"""

import io
import os
import numpy as np
import pandas as pd
import streamlit as st

# Core Modülleri
from core.data_processing import (
    load_dataset,
    inspect_dataset,
    handle_missing_data,
    check_duplicate_participants
)
from core.scoring import (
    score_dichotomous,
    reverse_score_items,
    calculate_total_scores
)
from core.test_statistics import (
    compute_test_summary,
    compute_standard_scores,
    compute_frequency_distribution
)
from core.item_analysis import compute_item_statistics
from core.distractor_analysis import (
    analyze_distractors_for_item,
    compute_all_items_distractors
)
from core.reliability import compute_reliability_indices
from core.group_analysis import (
    compare_test_scores_by_group,
    compute_group_item_fairness
)
from core.validation import validate_analysis_inputs

# UI ve Raporlama Modülleri
from ui.components import (
    inject_custom_css,
    render_academic_header,
    render_metric_card,
    render_apa_table_header,
    render_apa_table_footer
)
from ui.charts import (
    plot_score_histogram,
    plot_score_boxplot,
    plot_qq_normality,
    plot_item_difficulty_bar,
    plot_item_discrimination_bar,
    plot_difficulty_discrimination_scatter,
    plot_alpha_if_deleted,
    plot_distractor_bar,
    plot_upper_lower_options_comparison,
    plot_correlation_heatmap,
    plot_group_density_comparison,
    plot_missing_data_overview
)
from reports.excel_export import create_excel_report
from reports.word_export import create_word_report
from core.irt import (fit_irt, estimate_ability, approximate_q3,
                      approximate_parameter_uncertainty, item_fit_by_restscore,
                      sx2_pilot)
import plotly.graph_objects as go
from core.irt_itemfit import sx2_mirt_compatible
from core.irt_hessian import observed_information_se
from core.irt_se_diagnostics import hessian_boundary_diagnostics
from core.irt_multistart import fit_irt_multistart
from core.irt_model_reliability import model_reliability
from core.irt_stationarity import stationarity_diagnostics
from core.warning_explanations import boundary_report, opg_explanation, GUIDE
from core.descriptives import describe_items, describe_total
from ui.irt_optimization import (REFINE_LABEL, REFINE_HELP, calibration_settings,
                                 run_irt_calibration, estimation_method, result_points,
                                 settings_mismatch, optimization_summary, convergence_messages)


# =============================================================================
# SAYFA VE DURUM YAPILANDIRMASI
# =============================================================================
st.set_page_config(
    page_title="PsikoAnaliz — Psikometrik Analiz Platformu",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)


def download_table_pair(frame, basename, label, key):
    """Present consistent CSV and Excel exports for a tabular result."""
    table = pd.DataFrame(frame)
    left, right = st.columns(2)
    with left:
        st.download_button(
            f"⬇️ {label} — CSV",
            data=table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{basename}.csv", mime="text/csv",
            key=f"{key}_csv", use_container_width=True)
    with right:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            table.to_excel(writer, index=False, sheet_name="Sonuclar")
        st.download_button(
            f"⬇️ {label} — Excel (.xlsx)",
            data=buffer.getvalue(),
            file_name=f"{basename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key}_xlsx", use_container_width=True)


inject_custom_css()

# Session State Başlatma
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "scored_df" not in st.session_state:
    st.session_state.scored_df = None
if "item_cols" not in st.session_state:
    st.session_state.item_cols = []
if "id_col" not in st.session_state:
    st.session_state.id_col = None
if "group_col" not in st.session_state:
    st.session_state.group_col = None
if "answer_key" not in st.session_state:
    st.session_state.answer_key = {}
if "is_dichotomous" not in st.session_state:
    st.session_state.is_dichotomous = True
if "test_summary" not in st.session_state:
    st.session_state.test_summary = {}
if "item_summary" not in st.session_state:
    st.session_state.item_summary = None
if "reliability_data" not in st.session_state:
    st.session_state.reliability_data = {}
if "distractor_data" not in st.session_state:
    st.session_state.distractor_data = {}


# =============================================================================
# YAN MENÜ (SIDEBAR NAVIGATION)
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div class="psiko-sidebar-brand">
        <div class="psiko-sidebar-brand-name">PsikoAnaliz</div>
        <div class="psiko-sidebar-brand-sub">PSİKOMETRİK ANALİZ PLATFORMU</div>
        <div class="psiko-sidebar-brand-author">Doç. Dr. Mehmet Taha Eser</div>
    </div>
    """, unsafe_allow_html=True)

    # Kategori bazlı gezinme: seçili sayfa oturum boyunca korunur.
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "1. Ana Sayfa"

    def navigate_to(page):
        st.session_state.nav_page = page

    st.markdown("### Genel")
    st.button("🏠 Ana Sayfa", use_container_width=True,
              on_click=navigate_to, args=("1. Ana Sayfa",))
    st.button("📁 Veri Yükleme", use_container_width=True,
              on_click=navigate_to, args=("2. Veri Yükleme",))
    st.button("⚙️ Puanlama ve Veri Hazırlama", use_container_width=True,
              on_click=navigate_to, args=("3. Puanlama & Veri Hazırlama",))

    st.button("📋 Betimsel İstatistikler", use_container_width=True,
              on_click=navigate_to, args=("3B. Betimsel İstatistikler",))

    st.markdown("---")
    st.markdown("### I. Klasik Test Kuramı (KTK)")
    for label, page in [
        ("📊 Test İstatistikleri", "4. Test İstatistikleri"),
        ("🔍 Madde Analizi", "5. Madde Analizi"),
        ("🎯 Çeldirici Analizi", "6. Çeldirici Analizi"),
        ("📐 Güvenirlik ve Ölçme Hatası", "7. Güvenirlik & Ölçme Hatası"),
        ("👥 Grup Bazlı Analizler", "8. Grup Bazlı Analizler"),
        ("📈 Bilimsel Grafikler", "9. Bilimsel Grafikler"),
    ]:
        st.button(label, key=f"nav_{page}", use_container_width=True,
                  on_click=navigate_to, args=(page,))

    st.markdown("---")
    st.markdown("### II. Madde Tepki Kuramı (MTK)")
    st.markdown("**İki Kategorili Modeller**")
    for m in ("1PL", "2PL", "3PL", "4PL"):
        st.button(f"📐 {m} Modeli", key=f"nav_irt_{m}",
                  use_container_width=True, on_click=navigate_to,
                  args=(f"MTK: {m}",))
    st.button("📊 Model Karşılaştırması", key="nav_irt_comparison",
              use_container_width=True, on_click=navigate_to,
              args=("MTK: Karşılaştırma",))
    st.markdown("**Çok Kategorili Modeller**")
    st.button("🔒 GRM / GPCM / NRM — Yakında", disabled=True,
              use_container_width=True)

    st.markdown("---")
    st.markdown("### Ortak Araçlar")
    for label, page in [
        ("📄 Raporlama", "10. Raporlama"),
        ("📚 Formüller ve Yöntemler", "11. Formüller ve Yöntemler"),
        ("❔ Kullanım Kılavuzu", "12. Kullanım Kılavuzu"),
    ]:
        st.button(label, key=f"nav_{page}", use_container_width=True,
                  on_click=navigate_to, args=(page,))

    selected_page = st.session_state.nav_page

    st.markdown("---")
    st.markdown("### 📊 Mevcut Veri Durumu")
    if st.session_state.scored_df is not None:
        n_rows = len(st.session_state.scored_df)
        k_items = len(st.session_state.item_cols)
        st.success(f"Aktif Veri: N={n_rows}, k={k_items}")
        st.caption(f"Veri Tipi: {'İkili (0/1)' if st.session_state.is_dichotomous else 'Çok Kategorili (Likert/Rubrik)'}")
    else:
        st.info("Henüz analiz için veri yüklenmedi.")


# =============================================================================
# SAYFA 1: ANA SAYFA
# =============================================================================
if selected_page == "1. Ana Sayfa":
    render_academic_header()

    st.markdown("""
    ### 🎯 Platform Hakkında ve Amaç

    **PsikoAnaliz**, eğitim ve psikoloji alanlarında kullanılan ölçme araçlarının
    geliştirilmesi, analizi ve psikometrik özelliklerinin değerlendirilmesi amacıyla
    tasarlanmış web tabanlı bir psikometrik analiz platformudur.

    Platform, **Klasik Test Kuramı (KTK)** kapsamında madde ve test istatistikleri,
    madde güçlüğü ve ayırt ediciliği, çeldirici analizi, güvenirlik ve ölçme hatası
    kestirimleri ile grup bazlı karşılaştırmaları sunmaktadır. Analiz sonuçları,
    bilimsel görselleştirmeler ve akademik raporlarla desteklenmektedir.

    **Madde Tepki Kuramı (MTK)** kapsamındaki 1PL/Rasch, 2PL ve 3PL modelleri,
    madde karakteristik eğrileri, test bilgi fonksiyonları, yetenek kestirimi ve
    model uyum analizlerinin ilerleyen sürümlerde geliştirilmesi hedeflenmektedir.
    Bu analizler mevcut sürümde henüz kullanıma sunulmamıştır.

    PsikoAnaliz, psikometrik analiz süreçlerinin anlaşılır, bütünleşik ve
    yeniden üretilebilir bir ortamda yürütülmesini amaçlamaktadır.

    **Doç. Dr. Mehmet Taha Eser**  
    Aydın Adnan Menderes Üniversitesi  
    Eğitim Fakültesi  
    *Eğitimde Ölçme ve Değerlendirme Anabilim Dalı*
    """)


# =============================================================================
# SAYFA 2: VERİ YÜKLEME VE ÖNİZLEME
# =============================================================================
elif selected_page == "2. Veri Yükleme":
    render_academic_header()
    st.subheader("📁 Veri Yükleme ve Örnek Veri Setleri")

    tab_upload, tab_sample = st.tabs(["📤 Kendi Verinizi Yükleyin", "🧪 Örnek Veri Setleri ile Deneyin"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Veri dosyanızı seçin (CSV, Excel .xlsx/.xls veya SPSS .sav)",
            type=["csv", "xlsx", "xls", "sav"]
        )

        col_enc, col_sep = st.columns(2)
        with col_enc:
            enc_opt = st.selectbox("Karakter Kodlaması (Encoding)", ["Otomatik Algıla", "utf-8", "windows-1254", "iso-8859-9", "latin1"])
        with col_sep:
            sep_opt = st.selectbox("CSV Ayırıcı (Delimiter)", ["Otomatik Algıla", ",", ";", "\\t"])

        if uploaded_file is not None:
            enc_param = None if enc_opt == "Otomatik Algıla" else enc_opt
            sep_param = None if sep_opt == "Otomatik Algıla" else sep_opt
            file_ext = uploaded_file.name.split(".")[-1].lower()

            try:
                df_loaded, meta = load_dataset(uploaded_file, file_type=file_ext, encoding=enc_param, delimiter=sep_param)
                st.session_state.raw_df = df_loaded
                st.success(f"✅ '{uploaded_file.name}' başarıyla yüklendi! ({len(df_loaded)} satır, {len(df_loaded.columns)} sütun)")
            except Exception as e:
                st.error(f"Dosya okunurken hata oluştu: {str(e)}")

    with tab_sample:
        st.markdown("Uygulamanın yeteneklerini test etmek için hazır sentetik psikometrik veri setlerini yükleyebilirsiniz:")
        col_s1, col_s2, col_s3 = st.columns(3)

        sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")

        with col_s1:
            if st.button("📘 1. Çoktan Seçmeli Ham Veri (30 Madde, N=350)"):
                csv_path = os.path.join(sample_dir, "coktan_secmeli_ham.csv")
                key_path = os.path.join(sample_dir, "cevap_anahtari.csv")
                if os.path.exists(csv_path):
                    df_mc = pd.read_csv(csv_path)
                    st.session_state.raw_df = df_mc
                    st.session_state.id_col = "Ogrenci_ID"
                    st.session_state.group_col = "Cinsiyet"
                    st.session_state.item_cols = [c for c in df_mc.columns if c.startswith("M_")]
                    if os.path.exists(key_path):
                        df_k = pd.read_csv(key_path)
                        st.session_state.answer_key = dict(zip(df_k["Madde"], df_k["Dogru_Cevap"]))
                    st.success("Çoktan seçmeli ham veri seti ve cevap anahtarı yüklendi!")

        with col_s2:
            if st.button("📗 2. Önceden Puanlanmış İkili Veri (25 Madde, N=300)"):
                csv_path = os.path.join(sample_dir, "ikili_puanlanmis.csv")
                if os.path.exists(csv_path):
                    df_bin = pd.read_csv(csv_path)
                    st.session_state.raw_df = df_bin
                    st.session_state.scored_df = df_bin.copy()
                    st.session_state.id_col = "Katilimci_ID"
                    st.session_state.group_col = "Grup"
                    st.session_state.item_cols = [c for c in df_bin.columns if c.startswith("Item_")]
                    st.session_state.is_dichotomous = True
                    st.success("0/1 puanlanmış ikili veri seti yüklendi!")

        with col_s3:
            if st.button("📙 3. Likert Tutum Ölçeği (20 Madde, N=250)"):
                csv_path = os.path.join(sample_dir, "likert_olcek.csv")
                if os.path.exists(csv_path):
                    df_lik = pd.read_csv(csv_path)
                    st.session_state.raw_df = df_lik
                    st.session_state.scored_df = df_lik.copy()
                    st.session_state.id_col = "ID"
                    st.session_state.group_col = "Fakulte"
                    st.session_state.item_cols = [c for c in df_lik.columns if c.startswith("Likert_")]
                    st.session_state.is_dichotomous = False
                    st.success("Likert tipi ölçek veri seti yüklendi!")

    # Veri Önizleme ve Değişken Seçimi
    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        st.markdown("---")
        st.subheader("🔍 Veri Seti Önizleme ve Değişken Tanımlama")

        col_id, col_grp, col_type = st.columns(3)
        cols_all = list(df.columns)

        with col_id:
            id_opts = ["(Yok)"] + cols_all
            default_id_idx = id_opts.index(st.session_state.id_col) if st.session_state.id_col in id_opts else 0
            sel_id = st.selectbox("Katılımcı ID Sütunu (Opsiyonel)", id_opts, index=default_id_idx)
            st.session_state.id_col = None if sel_id == "(Yok)" else sel_id

        with col_grp:
            grp_opts = ["(Yok)"] + cols_all
            default_grp_idx = grp_opts.index(st.session_state.group_col) if st.session_state.group_col in grp_opts else 0
            sel_grp = st.selectbox("Grup / Alt Grup Değişkeni (Cinsiyet, Şube vb.)", grp_opts, index=default_grp_idx)
            st.session_state.group_col = None if sel_grp == "(Yok)" else sel_grp

        with col_type:
            data_nature = st.radio(
                "Veri Formatı ve Puanlama Durumu",
                ["Çoktan Seçmeli Ham Yanıtlar (Cevap anahtarı ile puanlanacak)", 
                 "Önceden Puanlanmış İkili Veri (0 ve 1)", 
                 "Çok Kategorili Derecelendirme (Likert 1-5, Rubrik vb.)"],
                index=0 if len(st.session_state.answer_key) > 0 else (0 if not st.session_state.is_dichotomous else 1)
            )

        # Madde Sütunlarını Seçme
        excluded = [c for c in [st.session_state.id_col, st.session_state.group_col] if c is not None]
        suggested_items = [c for c in cols_all if c not in excluded]

        sel_items = st.multiselect(
            "Analiz Edilecek Madde Sütunlarını Seçin",
            options=cols_all,
            default=st.session_state.item_cols if st.session_state.item_cols else suggested_items
        )
        st.session_state.item_cols = sel_items

        st.dataframe(df.head(10), use_container_width=True)

        # Temel Bilgi
        info = inspect_dataset(df)
        c_i1, c_i2, c_i3 = st.columns(3)
        c_i1.metric("Toplam Katılımcı (Satır)", info["total_rows"])
        c_i2.metric("Toplam Değişken (Sütun)", info["total_cols"])
        c_i3.metric("Kayıp Hücre Oranı", f"%{info['missing_pct']:.2f}")


# =============================================================================
# SAYFA 3: PUANLAMA VE VERİ HAZIRLAMA
# =============================================================================
elif selected_page == "3. Puanlama & Veri Hazırlama":
    render_academic_header()
    st.subheader("⚙️ Puanlama Motoru ve Veri Hazırlama")

    if st.session_state.raw_df is None or not st.session_state.item_cols:
        st.warning("Lütfen önce 'Veri Yükleme' sekmesinden bir veri seti yükleyin ve maddeleri seçin.")
    else:
        df = st.session_state.raw_df
        item_cols = st.session_state.item_cols

        tab_key, tab_rev, tab_missing = st.tabs([
            "🔑 Çoktan Seçmeli Cevap Anahtarı", 
            "🔄 Ters Kodlama (Reverse Scoring)", 
            "❓ Kayıp Veri Protokolü"
        ])

        with tab_key:
            st.markdown("#### Cevap Anahtarı Girişi (0/1 İkili Puanlama)")
            st.write("Her bir madde için doğru seçeneği belirleyin veya toplu olarak girin:")

            # Hızlı anahtar doldurma
            col_k1, col_k2 = st.columns([2, 1])
            with col_k1:
                batch_key_input = st.text_input("Toplu Anahtar Girişi (Virgülle ayrılmış, örn: A,B,C,D,A,...)", "")
                if st.button("Toplu Anahtarı Uygula") and batch_key_input:
                    tokens = [t.strip().upper() for t in batch_key_input.split(",") if t.strip()]
                    if len(tokens) == len(item_cols):
                        st.session_state.answer_key = dict(zip(item_cols, tokens))
                        st.success("Tüm maddelerin cevap anahtarı başarıyla güncellendi!")
                    else:
                        st.error(f"Girdiğiniz anahtar sayısı ({len(tokens)}) madde sayısına ({len(item_cols)}) eşit değil.")

            # Madde bazında anahtarlar
            st.markdown("##### Madde Bazında Doğru Cevaplar:")
            k_cols_ui = st.columns(min(6, len(item_cols)))
            updated_key = dict(st.session_state.answer_key)

            for idx, item in enumerate(item_cols):
                with k_cols_ui[idx % len(k_cols_ui)]:
                    cur_val = str(updated_key.get(item, "A"))
                    new_val = st.text_input(f"{item}", value=cur_val, key=f"key_{item}", max_chars=5).strip().upper()
                    updated_key[item] = new_val

            st.session_state.answer_key = updated_key

            col_sc1, col_sc2 = st.columns(2)
            with col_sc1:
                missing_zero = st.checkbox("Boş bırakılan / cevapsız maddeleri 0 olarak puanla (Başarı testleri için standart)", value=True)
            with col_sc2:
                if st.button("🎯 Veriyi Cevap Anahtarına Göre Puanla", type="primary"):
                    scored, diag = score_dichotomous(
                        df,
                        item_columns=item_cols,
                        answer_key=st.session_state.answer_key,
                        missing_as_zero=missing_zero,
                        id_column=st.session_state.id_col
                    )
                    st.session_state.scored_df = scored
                    st.session_state.is_dichotomous = True
                    st.success(f"Puanlama başarıyla tamamlandı! ({len(item_cols)} madde puanlandı)")

        with tab_rev:
            st.markdown("#### Kuramsal Sınırlara Dayalı Ters Kodlama")
            st.markdown("""
            > [!IMPORTANT]
            > PsikoAnaliz, ölçek güvenliğini korumak için ampirik min/max değerlerini **kullanmaz**. 
            > Ters kodlama daima kuramsal formüle dayanır: $X_{yeni} = Min_{teorik} + Max_{teorik} - X_{eski}$.
            """)
            col_rmin, col_rmax = st.columns(2)
            with col_rmin:
                t_min = st.number_input("Kuramsal Minimum Puan (Örn: 1)", value=1.0)
            with col_rmax:
                t_max = st.number_input("Kuramsal Maksimum Puan (Örn: 5)", value=5.0)

            rev_items = st.multiselect("Ters Kodlanacak Maddeleri Seçin", options=item_cols)

            if st.button("🔄 Seçili Maddeleri Ters Kodla") and rev_items:
                base_df = st.session_state.scored_df if st.session_state.scored_df is not None else df
                rev_res, rev_report = reverse_score_items(base_df, rev_items, t_min, t_max)
                st.session_state.scored_df = rev_res
                st.session_state.is_dichotomous = False
                st.success(f"Seçilen {len(rev_items)} madde kuramsal sınırlara dayalı olarak ters kodlandı!")
                if rev_report["out_of_bounds_warnings"]:
                    st.warning(f"Sınır dışı değer uyarısı: {rev_report['out_of_bounds_warnings']}")

        with tab_missing:
            st.markdown("#### Kayıp Veri Yönetimi")
            m_method = st.selectbox(
                "Kayıp Veri Protokolü",
                ["keep (Olduğu gibi bırak / NaN)", "listwise (Kayıp verisi olan katılımcıyı tamamen sil)", "zero (Boş maddeleri 0 yap)", "mean (Madde ortalaması ata)"]
            )
            clean_btn = st.button("Uygula")
            if clean_btn:
                base_df = st.session_state.scored_df if st.session_state.scored_df is not None else df
                method_code = m_method.split()[0]
                cleaned, m_info = handle_missing_data(base_df, method=method_code, item_columns=item_cols)
                st.session_state.scored_df = cleaned
                st.success(f"Kayıp veri kuralı ({method_code}) uygulandı. Kalan katılımcı: {m_info['final_rows']}")

        # Puanlanmış Veri Tablosu ve İndirme
        if st.session_state.scored_df is not None:
            st.markdown("---")
            st.subheader("📋 Puanlanmış Veri Matrisi")
            st.dataframe(st.session_state.scored_df.head(10), use_container_width=True)

            download_table_pair(st.session_state.scored_df,
                                "psikoanaliz_puanlanmis_veri",
                                "Puanlanmış Veri Seti", "scored")


# =============================================================================
# BETİMSEL İSTATİSTİKLER — ANALİZDEN ÖNCE VERİ KONTROLÜ
# =============================================================================
elif selected_page == "3B. Betimsel İstatistikler":
    render_academic_header()
    st.subheader("📋 Betimsel İstatistikler")
    st.caption("KTK veya MTK analizi seçmeden önce veriyi inceleyebilirsiniz. "
               "Bu ekran model kestirimi yapmaz.")
    raw=st.session_state.raw_df
    scored=st.session_state.scored_df
    if raw is None and scored is None:
        st.warning("Önce Veri Yükleme bölümünden veri seti yükleyin.")
    else:
        choices={}
        if scored is not None:
            choices["Puanlanmış / hazırlanmış veri"]=scored
        if raw is not None:
            choices["Ham veri"]=raw
        selected=st.radio("İncelenecek veri",list(choices),horizontal=True)
        df=choices[selected]
        defaults=[c for c in st.session_state.item_cols if c in df.columns]
        if not defaults:
            defaults=[c for c in df.columns
                      if c not in (st.session_state.id_col,st.session_state.group_col)]
        selected_cols=st.multiselect("İncelenecek madde/değişken sütunları",
                                     list(df.columns),default=defaults)
        if not selected_cols:
            st.info("İstatistikleri hesaplamak için en az bir sütun seçin.")
        else:
            subset=df.loc[:,selected_cols]
            numeric=subset.apply(pd.to_numeric,errors="coerce")
            invalid=[c for c in selected_cols
                     if numeric[c].notna().sum()<subset[c].notna().sum()]
            if invalid:
                st.warning("Sayısal olmayan yanıtlar istatistik hesabına alınmaz: "
                           +", ".join(str(c) for c in invalid[:20])
                           +". Ham kategorik cevaplara ait frekansları "
                           "aşağıdaki dağılım sekmesinden inceleyin.")
            total_nonempty=int(subset.notna().sum().sum())
            a,b,c,d=st.columns(4)
            a.metric("Katılımcı",len(df))
            b.metric("Seçili sütun",len(selected_cols))
            c.metric("Eksik hücre",int(subset.isna().sum().sum()))
            d.metric("Doluluk %",f"{100*total_nonempty/subset.size:.1f}" if subset.size else "—")
            tab1,tab2,tab3,tab4=st.tabs([
                "Madde İstatistikleri","Toplam Puan","Dağılımlar","Eksik Veri"])
            with tab1:
                item_summary=describe_items(numeric)
                st.dataframe(item_summary.round(3),hide_index=True,
                             use_container_width=True)
                st.caption("SS ve varyans örneklem esaslıdır (ddof=1). "
                           "0/1 sütunlarında ortalama madde doğru yanıt oranıdır. "
                           "Çarpıklık ve basıklık tanımlayıcıdır.")
                download_table_pair(item_summary,"psikoanaliz_betimsel_maddeler",
                                    "Madde Betimsel İstatistikleri","desc_item")
            with tab2:
                complete,stats=describe_total(numeric)
                if stats is None:
                    st.info("Eksiksiz sayısal satır bulunamadı; toplam puan üretilemedi.")
                else:
                    complete_binary=bool(numeric.notna().all().all() and
                                          numeric.isin([0,1]).all().all())
                    if not complete_binary:
                        st.warning("Toplam, seçili sayısal sütunların aritmetik toplamıdır; "
                                   "ham seçenek kodları, farklı ölçekler veya henüz "
                                   "puanlanmamış maddeler için psikometrik test puanı "
                                   "olarak yorumlanmamalıdır.")
                    st.caption("Herhangi bir seçili maddesi eksik/sayısal olmayan "
                               "katılımcı toplam puandan dışlanır; eksikler sıfır yapılmaz.")
                    total_table=pd.DataFrame([stats])
                    st.dataframe(total_table.round(3),hide_index=True,
                                 use_container_width=True)
                    st.bar_chart(complete.value_counts().sort_index().rename("Frekans"))
                    download_table_pair(total_table,"psikoanaliz_betimsel_toplam",
                                        "Toplam Puan Özeti","desc_total")
            with tab3:
                variable=st.selectbox("Dağılımı görüntülenecek değişken",
                                      selected_cols,key="desc_variable")
                counts=subset[variable].astype("string").fillna("(Eksik)").value_counts(
                    dropna=False).rename_axis("Değer").reset_index(name="Frekans")
                counts["Yüzde"]=100*counts["Frekans"]/len(df)
                st.dataframe(counts,hide_index=True,use_container_width=True)
                st.bar_chart(counts.set_index("Değer")["Frekans"])
                download_table_pair(counts,"psikoanaliz_betimsel_frekans",
                                    "Frekans Dağılımı","desc_freq")
            with tab4:
                missing=pd.DataFrame({"Değişken":selected_cols,
                                      "Eksik N":subset.isna().sum().values,
                                      "Eksik %":(100*subset.isna().mean()).values,
                                      "Geçerli N":subset.notna().sum().values})
                st.dataframe(missing.round(2),hide_index=True,use_container_width=True)
                st.caption("Eksik değerler ham veri üzerinden sayılmıştır. "
                           "Sayısal olmayan yanıtların sayısala dönüştürülmesi "
                           "sırasında oluşan eksikler madde istatistiğinde ayrıca dışlanır.")
                download_table_pair(missing,"psikoanaliz_betimsel_eksik",
                                    "Eksik Veri Özeti","desc_missing")


# =============================================================================
# SAYFA 4: TEST İSTATİSTİKLERİ
# =============================================================================
elif selected_page == "4. Test İstatistikleri":
    render_academic_header()
    st.subheader("📊 Test Düzeyi Betimsel İstatistikler ve Normlar")

    target_df = st.session_state.scored_df if st.session_state.scored_df is not None else st.session_state.raw_df
    item_cols = st.session_state.item_cols

    if target_df is None or not item_cols:
        st.warning("Lütfen önce veri yükleyin ve puanlama işlemini tamamlayın.")
    else:
        # Toplam puan hesaplama
        total_scores = calculate_total_scores(target_df, item_cols)
        k_items = len(item_cols)
        summary = compute_test_summary(total_scores, k_items)
        st.session_state.test_summary = summary

        # Metrik Kartları
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Katılımcı (N)", summary["n_examinees"])
        c2.metric("Madde Sayısı (k)", summary["k_items"])
        c3.metric("Puan Ortalaması", f"{summary['mean']:.2f}")
        c4.metric("Standart Sapma", f"{summary['std_dev']:.2f}")
        c5.metric("Medyan", f"{summary['median']:.2f}")
        c6.metric("Ortalama Güçlük", f"{summary['mean_difficulty']:.2f}")

        st.markdown("---")
        render_apa_table_header(1, "Test Toplam Puanlarına İlişkin Kapsamlı Betimsel İstatistikler")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            t1_df = pd.DataFrame([
                {"İstatistik": "Katılımcı Sayısı (N)", "Değer": summary["n_examinees"]},
                {"İstatistik": "Madde Sayısı (k)", "Değer": summary["k_items"]},
                {"İstatistik": "Ortalama (Mean)", "Değer": summary["mean"]},
                {"İstatistik": "Medyan (Ortanca)", "Değer": summary["median"]},
                {"İstatistik": "Mod (Tepe Değer)", "Değer": summary["mode"]},
                {"İstatistik": "Örneklem Varyansı (s²)", "Değer": summary["variance"]},
                {"İstatistik": "Standart Sapma (s)", "Değer": summary["std_dev"]},
                {"İstatistik": "Ortalamanın Standart Hatası (SE_M)", "Değer": summary["sem_mean"]}
            ])
            st.dataframe(t1_df, hide_index=True, use_container_width=True)

        with col_t2:
            t2_df = pd.DataFrame([
                {"İstatistik": "Gözlenen Minimum", "Değer": summary["min"]},
                {"İstatistik": "Gözlenen Maksimum", "Değer": summary["max"]},
                {"İstatistik": "Ranj (Aralık)", "Değer": summary["range"]},
                {"İstatistik": "Çarpıklık (Skewness)", "Değer": summary["skewness"]},
                {"İstatistik": "Basıklık (Kurtosis)", "Değer": summary["kurtosis"]},
                {"İstatistik": "Çeyrekler Arası Ranj (IQR)", "Değer": summary["iqr"]},
                {"İstatistik": "Taban Etkisi (Floor %)", "Değer": f"%{summary['floor_pct']:.2f}"},
                {"İstatistik": "Tavan Etkisi (Ceiling %)", "Değer": f"%{summary['ceiling_pct']:.2f}"}
            ])
            st.dataframe(t2_df, hide_index=True, use_container_width=True)

        render_apa_table_footer("Örneklem varyansı serbestlik derecesi düzeltmesi (ddof = 1) ile hesaplanmıştır. Basıklık değeri normal dağılım için 0 referanslıdır.")

        # Frekans Tablosu
        with st.expander("📑 Detaylı Puan Frekans Dağılımı ve Yüzdelik Sıralar"):
            freq_df = compute_frequency_distribution(total_scores)
            st.dataframe(freq_df, hide_index=True, use_container_width=True)


# =============================================================================
# SAYFA 5: MADDE ANALİZİ
# =============================================================================
elif selected_page == "5. Madde Analizi":
    render_academic_header()
    st.subheader("🔍 Klasik Madde Analizi (Item Analysis)")

    target_df = st.session_state.scored_df if st.session_state.scored_df is not None else st.session_state.raw_df
    item_cols = st.session_state.item_cols

    if target_df is None or not item_cols:
        st.warning("Lütfen önce veri yükleyin ve puanlama işlemini tamamlayın.")
    else:
        total_scores = calculate_total_scores(target_df, item_cols)
        summary_df, meta = compute_item_statistics(
            target_df,
            item_columns=item_cols,
            total_scores=total_scores,
            is_dichotomous=st.session_state.is_dichotomous
        )
        st.session_state.item_summary = summary_df

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Ortalama Güçlük (p)", f"{meta['mean_difficulty']:.2f}")
        col_m2.metric("Ortalama Ayırt Edicilik (r_ir)", f"{meta['mean_discrimination_rir']:.2f}")
        col_m3.metric("Ortalama D (%27)", f"{meta['mean_discrimination_D']:.2f}")
        col_m4.metric("İncelenmesi Gereken Madde", meta["problematic_items_count"])

        render_apa_table_header(2, "Klasik Madde İstatistikleri, Ayırt Edicilik ve Tanısal Bayraklar")

        # Filtreleme
        flag_filter = st.selectbox(
            "Madde Filtrele",
            ["Tüm Maddeler", "Yalnızca Sorunlu Maddeler (İnceleme Gerektirenler)", "İdeal / Kabul Edilebilir Maddeler"]
        )

        display_df = summary_df.copy()
        if flag_filter == "Yalnızca Sorunlu Maddeler (İnceleme Gerektirenler)":
            display_df = display_df[display_df["Tanısal Bayrak"] != "İdeal / Kabul Edilebilir"]
        elif flag_filter == "İdeal / Kabul Edilebilir Maddeler":
            display_df = display_df[display_df["Tanısal Bayrak"] == "İdeal / Kabul Edilebilir"]

        st.dataframe(display_df, use_container_width=True, hide_index=True)
        render_apa_table_footer(
            "r_it: Madde-toplam korelasyonu; r_ir: Düzeltilmiş (madde-kalan) korelasyon; "
            "r_pbis: Nokta-çift serili; r_bis: Çift-serili (normal ordinat bazlı); "
            "D (%27): Kelley üst-alt %27 grup ayırt ediciliği."
        )

        download_table_pair(summary_df, "madde_istatistikleri",
                            "Madde Analizi Tablosu", "item_stats")


# =============================================================================
# SAYFA 6: ÇELDİRİCİ ANALİZİ
# =============================================================================
elif selected_page == "6. Çeldirici Analizi":
    render_academic_header()
    st.subheader("🎯 Çoktan Seçmeli Çeldirici ve Seçenek Analizi")

    if st.session_state.raw_df is None or not st.session_state.answer_key:
        st.warning("Çeldirici analizi için ham yanıtlar ve bir cevap anahtarı gereklidir. Lütfen 'Puanlama' sekmesinden cevap anahtarını giriniz.")
    else:
        raw_df = st.session_state.raw_df
        item_cols = st.session_state.item_cols
        total_scores = calculate_total_scores(st.session_state.scored_df if st.session_state.scored_df is not None else raw_df, item_cols)

        # Eşik belirleme
        col_th, col_sel = st.columns([1, 2])
        with col_th:
            thresh = st.slider("Çalışmayan Çeldirici Eşiği (%)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
        with col_sel:
            selected_item = st.selectbox("İncelenecek Maddeyi Seçin", item_cols)

        all_distractors = compute_all_items_distractors(
            raw_df=raw_df,
            item_columns=item_cols,
            answer_key=st.session_state.answer_key,
            total_scores=total_scores,
            scored_df=st.session_state.scored_df,
            threshold_pct=thresh
        )
        st.session_state.distractor_data = all_distractors

        item_dist = all_distractors.get(selected_item, {})
        if item_dist and "options_table" in item_dist:
            render_apa_table_header(3, f"{selected_item} Maddesi Seçenek Seçim Dağılımı ve Ayırt Edicilik İndeksleri")
            st.dataframe(item_dist["options_table"], hide_index=True, use_container_width=True)

            # Değerlendirme uyarısı
            eval_text = item_dist.get("overall_evaluation", "")
            if "KRİTİK SORUN" in eval_text:
                st.error(eval_text)
            elif "Zayıf" in eval_text:
                st.warning(eval_text)
            else:
                st.success(f"✅ {selected_item} maddesinde çeldiriciler beklenen şekilde çalışmaktadır.")

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig1 = plot_distractor_bar(item_dist, selected_item)
                st.plotly_chart(fig1, use_container_width=True)
            with col_g2:
                fig2 = plot_upper_lower_options_comparison(item_dist, selected_item)
                st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# SAYFA 7: GÜVENİRLİK VE ÖLÇME HATASI
# =============================================================================
elif selected_page == "7. Güvenirlik & Ölçme Hatası":
    render_academic_header()
    st.subheader("📐 Güvenirlik Kestirimi ve Ölçme Hatası")

    target_df = st.session_state.scored_df if st.session_state.scored_df is not None else st.session_state.raw_df
    item_cols = st.session_state.item_cols

    if target_df is None or not item_cols:
        st.warning("Lütfen önce veri yükleyin ve puanlama işlemini tamamlayın.")
    else:
        rel = compute_reliability_indices(
            target_df,
            item_columns=item_cols,
            is_dichotomous=st.session_state.is_dichotomous
        )
        st.session_state.reliability_data = rel

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cronbach Alfa (α)", f"{rel.get('cronbach_alpha', 0.0):.4f}")
        if rel.get("kr20") is not None:
            c2.metric("KR-20 Katsayısı", f"{rel.get('kr20', 0.0):.4f}")
        else:
            c2.metric("McDonald Omega (ω_t)", f"{rel.get('mcdonald_omega', 0.0):.4f}" if rel.get('mcdonald_omega') else "-")
        c3.metric("Ölçmenin Standart Hatası (SEM)", f"{rel.get('sem', 0.0):.2f}")
        c4.metric("%95 GA Hata Marjı", f"± {rel.get('ci_95_margin', 0.0):.2f}")

        # KR-20 ve Alfa Eşdeğerlik Kanıtı Bannerı
        if rel.get("alpha_kr20_difference") is not None:
            diff = rel["alpha_kr20_difference"]
            st.markdown(f"""
            <div class="success-card">
                <b>🧪 Matematiksel Doğrulama ve Eşdeğerlik Kanıtı:</b><br>
                İkili veri setinde örneklem varyansı (ddof=1) tutarlılığı sağlandığından:
                <br><b>Cronbach Alfa (α)</b> = {rel['cronbach_alpha']:.8f} | <b>KR-20</b> = {rel['kr20']:.8f}
                <br>Mutlak Fark: <b>{diff:.2e}</b> (Virgülden sonra 12+ basamak birebir özdeşlik sağlanmıştır).
            </div>
            """, unsafe_allow_html=True)

        render_apa_table_header(4, "Test Güvenirlik Katsayıları ve Standart Hata Kestirimleri")

        t4_data = [
            {"Metrik": "Cronbach Alfa (α)", "Değer": rel.get("cronbach_alpha"), "Açıklama": "İç tutarlık güvenirlik katsayısı"},
            {"Metrik": "Standartlaştırılmış Alfa", "Değer": rel.get("standardized_alpha"), "Açıklama": "Maddeler arası ortalama korelasyona dayalı katsayı"},
            {"Metrik": "KR-20 (Kuder-Richardson)", "Değer": rel.get("kr20") if rel.get("kr20") is not None else "Çok Kategorili Veri", "Açıklama": "İkili veriler için Cronbach Alfa ile tam özdeş"},
            {"Metrik": "KR-21 Katsayısı", "Değer": rel.get("kr21") if rel.get("kr21") is not None else "-", "Açıklama": "Maddelerin eşit güçlükte olduğu varsayımı altında alt sınır"},
            {"Metrik": "İki-Yarı (Tek-Çift) Spearman-Brown", "Değer": rel.get("spearman_brown_odd_even"), "Açıklama": "Test yarısı eşdeğerlik kestirimi"},
            {"Metrik": "Guttman Split-Half (λ4)", "Değer": rel.get("guttman_lambda4"), "Açıklama": "Yarı-test varyans temelli güvenirlik"},
            {"Metrik": "McDonald Omega (ω_t)", "Değer": rel.get("mcdonald_omega") if rel.get("mcdonald_omega") else "-", "Açıklama": "Tek faktörlü model faktör yüklerine dayalı genel güvenirlik"},
            {"Metrik": "Maddeler Arası Ort. Korelasyon", "Değer": rel.get("mean_inter_item_r"), "Açıklama": "Tüm madde ikililerinin Pearson korelasyon ortalaması"},
            {"Metrik": "Ölçmenin Standart Hatası (SEM)", "Değer": rel.get("sem"), "Açıklama": "Bireysel ölçüm hatası standart sapması"},
            {"Metrik": "Feldt (1969) %95 Alfa Güven Aralığı", "Değer": f"[{rel.get('alpha_ci_lower')}, {rel.get('alpha_ci_upper')}]", "Açıklama": "F-dağılımı temelli kuramsal güven aralığı"}
        ]
        st.dataframe(pd.DataFrame(t4_data), hide_index=True, use_container_width=True)
        render_apa_table_footer(f"Genel Değerlendirme: {rel.get('reliability_evaluation', '')}")


# =============================================================================
# SAYFA 8: GRUP BAZLI ANALİZLER
# =============================================================================
elif selected_page == "8. Grup Bazlı Analizler":
    render_academic_header()
    st.subheader("👥 Gruplar Arası Karşılaştırmalar ve KTK Adillik Taraması")

    target_df = st.session_state.scored_df if st.session_state.scored_df is not None else st.session_state.raw_df
    item_cols = st.session_state.item_cols
    grp_col = st.session_state.group_col

    if target_df is None or not item_cols:
        st.warning("Lütfen önce veri yükleyiniz.")
    elif not grp_col or grp_col not in target_df.columns:
        st.info("Grup analizi için sol menüden 'Veri Yükleme' sekmesine gidip bir Grup Değişkeni (Cinsiyet, Şube vb.) seçiniz.")
    else:
        total_scores = calculate_total_scores(target_df, item_cols)
        grp_res = compare_test_scores_by_group(target_df, grp_col, total_scores)

        st.markdown(f"#### 📊 Grup Puan Karşılaştırması ({grp_col})")
        st.dataframe(grp_res["group_summary_table"], hide_index=True, use_container_width=True)

        if grp_res.get("comparison_type") == "2_groups_ttest":
            col_tt1, col_tt2, col_tt3 = st.columns(3)
            col_tt1.metric("Student t Değeri", f"{grp_res['student_t']:.3f} (p = {grp_res['student_p']:.4f})")
            col_tt2.metric("Welch t Değeri", f"{grp_res['welch_t']:.3f} (p = {grp_res['welch_p']:.4f})")
            col_tt3.metric("Cohen's d Etki Büyüklüğü", f"{grp_res['cohens_d']:.3f}", delta=grp_res["effect_size_interpretation"])
        else:
            col_an1, col_an2 = st.columns(2)
            col_an1.metric("ANOVA F Değeri", f"{grp_res.get('f_statistic', 0.0):.3f} (p = {grp_res.get('anova_p', 0.0):.4f})")
            col_an2.metric("Eta-kare (η²)", f"{grp_res.get('eta_squared', 0.0):.3f}")

        # KTK Madde Adillik Taraması
        st.markdown("---")
        st.markdown("#### ⚖️ KTK Düzeyinde Madde Güçlük Farkı (Δp) Taraması")
        st.markdown("""
        > [!NOTE]
        > Bu analiz Klasik Test Kuramı çerçevesinde tanımlayıcı bir ön taramadır ($\Delta p = p_{ref} - p_{focal}$). 
        > Yetenek düzeyini kontrol eden formel Değişen Madde Fonksiyonlaşması (DIF) analizlerinin yerine geçmez.
        """)
        fairness_df, f_meta = compute_group_item_fairness(target_df, grp_col, item_cols)
        if not fairness_df.empty:
            st.dataframe(fairness_df, hide_index=True, use_container_width=True)


# =============================================================================
# SAYFA 9: BİLİMSEL GRAFİKLER
# =============================================================================
elif selected_page == "9. Bilimsel Grafikler":
    render_academic_header()
    st.subheader("📈 İnteraktif Bilimsel Görselleştirme Paneli (Plotly)")

    target_df = st.session_state.scored_df if st.session_state.scored_df is not None else st.session_state.raw_df
    item_cols = st.session_state.item_cols

    if target_df is None or not item_cols:
        st.warning("Lütfen önce veri yükleyiniz.")
    else:
        total_scores = calculate_total_scores(target_df, item_cols)
        if st.session_state.item_summary is None:
            st.session_state.item_summary, _ = compute_item_statistics(target_df, item_cols, total_scores)

        tab_g1, tab_g2, tab_g3, tab_g4 = st.tabs([
            "📊 Test Puan Dağılımı", 
            "📐 Madde Güçlük ve Ayırt Edicilik", 
            "🔄 Güvenirlik & Alfa Silindiğinde", 
            "🗺️ Korelasyon Isı Haritası"
        ])

        with tab_g1:
            c_g1, c_g2 = st.columns(2)
            with c_g1:
                st.plotly_chart(plot_score_histogram(total_scores), use_container_width=True)
            with c_g2:
                st.plotly_chart(plot_score_boxplot(total_scores), use_container_width=True)
            st.plotly_chart(plot_qq_normality(total_scores), use_container_width=True)

        with tab_g2:
            st.plotly_chart(plot_item_difficulty_bar(st.session_state.item_summary), use_container_width=True)
            st.plotly_chart(plot_item_discrimination_bar(st.session_state.item_summary), use_container_width=True)
            st.plotly_chart(plot_difficulty_discrimination_scatter(st.session_state.item_summary), use_container_width=True)

        with tab_g3:
            overall_alpha = st.session_state.reliability_data.get("cronbach_alpha", 0.70)
            st.plotly_chart(plot_alpha_if_deleted(st.session_state.item_summary, overall_alpha), use_container_width=True)

        with tab_g4:
            corr_mat = target_df[item_cols].apply(pd.to_numeric, errors="coerce").corr()
            st.plotly_chart(plot_correlation_heatmap(corr_mat), use_container_width=True)


# =============================================================================
# SAYFA 10: RAPORLAMA
# =============================================================================
elif selected_page == "10. Raporlama":
    render_academic_header()
    st.subheader("📄 Akademik Raporlama ve Dışa Aktarma")

    target_df = st.session_state.scored_df if st.session_state.scored_df is not None else st.session_state.raw_df
    item_cols = st.session_state.item_cols

    if target_df is None or not item_cols:
        st.warning("Lütfen önce veri yükleyin ve analizleri tamamlayın.")
    else:
        st.markdown("""
        Tüm istatistiksel sonuçları, APA 7 standartlarında tablolar ve metodolojik açıklamalar içeren 
        **Excel (.xlsx)** ve **Word (.docx)** belgeleri olarak anında indirebilirsiniz:
        """)

        # Gerekli verileri hazırla
        total_scores = calculate_total_scores(target_df, item_cols)
        if not st.session_state.test_summary:
            st.session_state.test_summary = compute_test_summary(total_scores, len(item_cols))
        if st.session_state.item_summary is None:
            st.session_state.item_summary, _ = compute_item_statistics(target_df, item_cols, total_scores)
        if not st.session_state.reliability_data:
            st.session_state.reliability_data = compute_reliability_indices(target_df, item_cols)

        freq_df = compute_frequency_distribution(total_scores)

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown("### 📗 Excel Çalışma Kitabı (.xlsx)")
            st.markdown("- Çok sayfalı: Test Özeti, Madde Analizi, Çeldirici Analizi, Frekans Dağılımı")
            st.markdown("- Formül enjeksiyonuna karşı tam korumalı ve renkli stillendirilmiş")
            excel_bytes = create_excel_report(
                test_summary=st.session_state.test_summary,
                item_summary_df=st.session_state.item_summary,
                reliability_data=st.session_state.reliability_data,
                distractor_data=st.session_state.distractor_data if st.session_state.distractor_data else None,
                frequency_df=freq_df
            )
            st.download_button(
                "📥 Excel Raporunu İndir (.xlsx)",
                data=excel_bytes.getvalue(),
                file_name="PsikoAnaliz_Raporu.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

        with col_r2:
            st.markdown("### 📘 Word Akademik Araştırma Raporu (.docx)")
            st.markdown("- APA 7 uyumlu Tablo 1, Tablo 2 ve Tablo 3")
            st.markdown("- Metodolojik açıklamalar, psikometrik yorumlar ve APA kaynakça")
            word_bytes = create_word_report(
                test_summary=st.session_state.test_summary,
                item_summary_df=st.session_state.item_summary,
                reliability_data=st.session_state.reliability_data,
                distractor_data=st.session_state.distractor_data if st.session_state.distractor_data else None
            )
            st.download_button(
                "📥 Word Raporunu İndir (.docx)",
                data=word_bytes.getvalue(),
                file_name="PsikoAnaliz_Akademik_Rapor.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary"
            )


# =============================================================================
# SAYFA 11: FORMÜLLER VE YÖNTEMLER
# =============================================================================
elif selected_page == "11. Formüller ve Yöntemler":
    render_academic_header()
    st.subheader("📐 Matematiksel Formüller, Varsayımlar ve Yöntemler")

    st.markdown(r"""
    ### 1. Klasik Güvenirlik Formülleri ve KR-20 Eşdeğerliği

    #### Cronbach Alfa ($\alpha$)
    $$\alpha = \frac{k}{k - 1} \left( 1 - \frac{\sum_{i=1}^k s_i^2}{s_X^2} \right)$$
    - $k$: Madde sayısı.
    - $s_i^2$: $i$. maddenin örneklem varyansı ($ddof=1$).
    - $s_X^2$: Test toplam puanının örneklem varyansı ($ddof=1$).

    #### Kuder-Richardson 20 (KR-20)
    $$KR_{20} = \frac{k}{k - 1} \left( 1 - \frac{\frac{N}{N-1} \sum_{i=1}^k p_i q_i}{s_X^2} \right)$$
    **Önemli Kanıt:** İkili (0/1) yanıtlarda bir maddenin örneklem varyansı $s_i^2 = \frac{N}{N-1} p_i q_i$ olduğundan,
    örneklem varyansı esasına göre hesaplanan KR-20 formülü ile Cronbach Alfa formülü **matematiksel olarak birebir özdeştir**.

    #### Kuder-Richardson 21 (KR-21)
    $$KR_{21} = \frac{k}{k - 1} \left( 1 - \frac{\bar{X} (k - \bar{X})}{k \cdot s_X^2} \right)$$

    ---

    ### 2. Madde İstatistikleri Formülleri

    #### Düzeltilmiş Madde-Toplam Korelasyonu ($r_{ir}$ / Madde-Kalan)
    $$X_{kalan} = X_{toplam} - X_i$$
    $$r_{ir} = \frac{\text{Cov}(X_i, X_{kalan})}{s_i \cdot s_{kalan}}$$

    #### Çift-Serili Korelasyon ($r_{bis}$)
    $$r_{bis} = \frac{\bar{X}_1 - \bar{X}_0}{s_X} \cdot \frac{p \cdot q}{y}$$
    - $y$: Standart normal eğrinin $p$ kümülatif olasılığına karşılık gelen ordinat değeri.

    #### Üst-Alt %27 Grup Ayırt Ediciliği ($D$)
    $$D = p_{üst} - p_{alt}$$
    """)


# =============================================================================
# SAYFA 12: KULLANIM KILAVUZU
# =============================================================================
elif selected_page == "12. Kullanım Kılavuzu":
    render_academic_header()
    st.subheader("📖 Kullanım Kılavuzu ve Sıkça Sorulan Sorular")

    st.markdown("""
    ### Adım Adım Analiz Rehberi
    1. **Verinizi Hazırlayın:** Veriniz CSV veya Excel formatında olmalıdır. Her satır bir katılımcıyı, her sütun bir maddeyi temsil etmelidir.
    2. **Veri Yükleme:** Sol menüden 'Veri Yükleme' sayfasına giderek dosyanızı yükleyin ve madde sütunlarını işaretleyin.
    3. **Puanlama:** Çoktan seçmeli bir test ise 'Puanlama' sekmesinden cevap anahtarını tanımlayın ve puanlama butonuna basın.
    4. **Sonuçları İnceleyin:** 'Madde Analizi' ve 'Güvenirlik' sekmelerinde tabloları ve pedagojik tanı bayraklarını değerlendirin.
    5. **Raporu İndirin:** 'Raporlama' sekmesinden APA 7 uyumlu Word belgenizi veya renkli Excel tablonuzu tek tıkla alın.

    ---
    ### 👨‍🏫 Hazırlayan ve İletişim
    **Doç. Dr. Mehmet Taha Eser**  
    *Eğitimde Ölçme ve Değerlendirme Anabilim Dalı*  
    *PsikoAnaliz — Psikometrik Analiz Platformu*
    """)

    st.markdown(r"""
    ---
    ### 💻 Teknik Altyapı, Hesaplama Yöntemleri ve Yazılım Kaynakçası

    **PsikoAnaliz'in hesaplama altyapısı Python tabanlıdır.**
    Uygulamanın mevcut sürümünde arka planda R yorumlayıcısı, R oturumu
    veya `mirt`, `ltm`, `ShinyItemAnalysis` gibi R paketleri
    çalıştırılmamaktadır. Bu paketler yöntemsel karşılaştırma için
    incelenebilir; fakat mevcut uygulamanın yürütme bağımlılıkları değildir.

    #### Kullanılan Python bileşenleri

    | Bileşen | Uygulamadaki temel görevi |
    |---|---|
    | **Python** | Uygulama ve istatistiksel hesaplama dili |
    | **Streamlit** | Web arayüzü, seçimler ve etkileşimli analiz sayfaları |
    | **pandas** | CSV/Excel veri okuma, veri çerçeveleri, tablo ve dışa aktarma |
    | **NumPy** | Matris işlemleri, varyans, korelasyon ve sayısal hesaplamalar |
    | **SciPy** | İstatistiksel işlevler, optimizasyon ve MTK sayısal integrasyonu |
    | **statsmodels** | Projenin bağımlılık listesinde yer alan istatistik paketi |
    | **Plotly** | Etkileşimli psikometrik grafikler |
    | **openpyxl** | Excel (.xlsx) çalışma kitapları ve biçimlendirme |
    | **python-docx** | Word (.docx) akademik raporlar |
    | **pytest** | Otomatik birim ve regresyon testleri |

    **KTK hesaplamaları:** Madde güçlüğü ve ayırt ediciliği,
    test betimsel istatistikleri, Cronbach alfa, KR-20/KR-21,
    diğer güvenirlik göstergeleri ve çeldirici analizleri proje içindeki
    Python modülleriyle hesaplanır.

    **MTK hesaplamaları:** İki kategorili 1PL, 2PL, 3PL ve 4PL modelleri için
    Python'daki özel `core/irt.py` modülü kullanılır. Madde parametreleri
    standart normal örtük özellik varsayımı altında marjinal maksimum
    olabilirlik (MML), EM yinelemesi, Gauss–Hermite quadrature ve
    SciPy optimizasyonu ile kestirilir. Kişi yeteneği için EAP, MAP
    ve ML kestirimleri sunulur. ICC, IIF, TIF, koşullu standart hata,
    test karakteristik eğrisi ve kişi–madde görselleri bu çıktılardan üretilir.

    **Bilimsel sınırlar:** 1PL uygulamasında ortak ayırt edicilik
    `a = 1` olarak sabittir. Çok kategorili MTK modelleri henüz
    kullanıma sunulmamıştır. MTK parametrelerinin standart hataları ve
    S-X² madde uyum testi mevcut uygulamada henüz hesaplanmaz.
    Otomatik yazılım testleri, bağımsız psikometrik doğrulamanın yerini tutmaz.

    #### Kaynakça (APA 7)

    Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R.,
    Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S.,
    Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H.,
    Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P.,
    ... Oliphant, T. E. (2020). Array programming with NumPy.
    *Nature, 585*, 357–362. https://doi.org/10.1038/s41586-020-2649-2

    McKinney, W. (2010). Data structures for statistical computing in Python.
    In S. van der Walt & J. Millman (Eds.),
    *Proceedings of the 9th Python in Science Conference* (pp. 56–61).
    https://doi.org/10.25080/Majora-92bf1922-00a

    Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and
    statistical modeling with Python. In S. van der Walt & J. Millman (Eds.),
    *Proceedings of the 9th Python in Science Conference* (pp. 92–96).
    https://doi.org/10.25080/Majora-92bf1922-011

    Sievert, C. (2020). *Interactive web-based data visualization with R,
    plotly, and shiny*. Chapman & Hall/CRC.
    https://doi.org/10.1201/9780429447273

    Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T.,
    Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W.,
    Bright, J., van der Walt, S. J., Brett, M., Wilson, J.,
    Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E.,
    Kern, R., Larson, E., ... SciPy 1.0 Contributors. (2020).
    SciPy 1.0: Fundamental algorithms for scientific computing in Python.
    *Nature Methods, 17*, 261–272.
    https://doi.org/10.1038/s41592-019-0686-2

    **Çevrim içi teknik belgeler:** [Streamlit](https://docs.streamlit.io/),
    [pandas](https://pandas.pydata.org/docs/),
    [Plotly Python](https://plotly.com/python/),
    [openpyxl](https://openpyxl.readthedocs.io/),
    [python-docx](https://python-docx.readthedocs.io/) ve
    [pytest](https://docs.pytest.org/) resmî belgeleridir.

    **Kaynak kodu:** https://github.com/Tatung40/psikoanaliz
    """)


# =============================================================================
# SAYFA 13: MADDE TEPKİ KURAMI (MTK / IRT)
# =============================================================================
elif selected_page in ("MTK: 1PL", "MTK: 2PL", "MTK: 3PL", "MTK: 4PL", "MTK: Karşılaştırma"):
    render_academic_header()
    st.subheader("🧪 Madde Tepki Kuramı — 1PL, 2PL, 3PL, 4PL")
    st.caption("Tek boyutlu ikili MTK | Marjinal maksimum olabilirlik (EM) | "
               "N(0,1) örtük özellik dağılımı | EAP yetenek kestirimi")
    st.info("1PL: Sabit a=1 (Rasch benzeri). 2PL: a,b; 3PL: a,b,c; "
            "4PL: a,b,c,d. MTK analizi için yalnızca 0/1 puanlı, "
            "eksiksiz madde yanıtları kullanılmalıdır.")
    # Hazır 0/1 veride puanlama adımını zorunlu tutma.
    scored = st.session_state.scored_df
    raw = st.session_state.raw_df
    target = scored if scored is not None else raw
    columns = [c for c in st.session_state.item_cols
               if c not in (st.session_state.id_col, st.session_state.group_col)]
    if target is not None and not columns:
        columns = [c for c in target.columns if c not in
                   (st.session_state.id_col, st.session_state.group_col)
                   and pd.to_numeric(target[c], errors="coerce").notna().all()
                   and pd.to_numeric(target[c], errors="coerce").isin([0, 1]).all()]
    if target is None:
        st.warning("Önce Veri Yükleme bölümünden bir veri dosyası yükleyin. "
                   "0/1 olarak puanlanmış veriler için ayrıca puanlama gerekmez.")
    elif not columns:
        st.warning("MTK için madde sütunu seçilmemiş. Veri Yükleme bölümünden maddeleri seçin.")
    else:
        # Tüm ikili MTK modelleri aynı veri ön kontrolünü kullanır.
        # ID/grup sütunları ve uç güçlükteki maddeler, kullanıcıya gösterilerek ayıklanır.
        candidate_columns = [col for col in columns
                             if col not in (st.session_state.id_col, st.session_state.group_col)]
        raw_irt = target.loc[:, candidate_columns].apply(pd.to_numeric, errors="coerce")
        diagnostics = pd.DataFrame({
            "Madde": candidate_columns,
            "Ortalama (p)": raw_irt.mean(axis=0).to_numpy(),
            "Eksik / sayısal olmayan": raw_irt.isna().sum(axis=0).to_numpy(),
            "0/1 dışında": (~raw_irt.isin([0, 1]) & raw_irt.notna()).sum(axis=0).to_numpy()
        })
        problematic = diagnostics[
            (diagnostics["Ortalama (p)"] <= .01) |
            (diagnostics["Ortalama (p)"] >= .99) |
            (diagnostics["Eksik / sayısal olmayan"] > 0) |
            (diagnostics["0/1 dışında"] > 0)
        ]["Madde"].tolist()
        if problematic:
            st.warning(f"MTK için uygun olmayan {len(problematic)} sütun tespit edildi. "
                       "Aşağıdaki tabloda gerekçeleri gösterilmektedir.")
            st.dataframe(diagnostics[diagnostics["Madde"].isin(problematic)].round(4),
                         hide_index=True, use_container_width=True)
            with st.expander("Neden dışlanıyor? Her uyarı ne demek ve nasıl düzeltilir?"):
                st.markdown(
                    "**Ortalama (p) ≤ 0,01 veya ≥ 0,99:** Maddeyi neredeyse "
                    "herkes yanlış ya da doğru yanıtlamış. Bu uç oranlar "
                    "özellikle ayırt edicilik ve asimptot kestirimlerini "
                    "kararsızlaştırabilir. **Öneri:** Kodlamayı, cevap anahtarını, "
                    "örneklem düzeyini ve maddenin içeriğini kontrol edin. "
                    "0,01–0,99 aralığı burada kullanılan yazılım filtresidir; "
                    "evrensel madde silme kuralı değildir.\n\n"
                    "**Eksik / sayısal olmayan:** Seçili bir sütunda boş hücre "
                    "veya sayıya dönüştürülemeyen yanıt vardır. MTK modeli "
                    "eksiksiz 0/1 matrisi bekler. **Öneri:** Veri Hazırlama'da "
                    "eksik veri politikasını bilinçli biçimde seçin; ham A/B/C/D "
                    "yanıtları için önce cevap anahtarıyla puanlama yapın.\n\n"
                    "**0/1 dışında:** Sayısal ama ikili olmayan değerler "
                    "(örneğin 2, 3, 4, 5) bulunuyor. **Öneri:** Likert verisine "
                    "ikili model uygulamayın; uygun veri türünü ve modelini "
                    "seçin.\n\n"
                    "**Dışlama neyi değiştirir?** Analizde madde sayısı ve "
                    "toplam puan tanımı değişir; diğer modellerle karşılaştırma "
                    "yaparken aynı madde matrisini kullanın. Kaynak dosyanız "
                    "değiştirilmez.")
        if len(candidate_columns) != len(columns):
            st.info("Kimlik ve grup değişkenleri MTK madde matrisinden çıkarıldı.")
        use_valid_only = st.checkbox(
            "MTK için uygun olmayan maddeleri yalnızca bu analizde otomatik dışla",
            value=True, key="irt_filter_bad_items")
        irt_columns = [col for col in candidate_columns if col not in problematic
                       ] if use_valid_only else candidate_columns
        if len(irt_columns) < 3:
            st.error("MTK analizi için en az 3 uygun ikili madde gerekli. "
                     "Veri Yükleme bölümünde madde sütunlarını kontrol edin.")
            st.stop()
        if problematic and use_valid_only:
            st.info(f"MTK analizine {len(irt_columns)} madde alınacak; "
                    f"{len(problematic)} sütun dışarıda bırakılacak. "
                    "Asıl veri dosyanız değiştirilmez.")
        model = selected_page.replace("MTK: ", "")
        if model == "Karşılaştırma":
            st.subheader("📊 MTK Model Karşılaştırması")
            st.markdown(
                "Aynı veri üzerinde kestirilen **1PL, 2PL, 3PL ve 4PL** modelleri "
                "**AIC**, **BIC** ve **logLik** ölçütleriyle karşılaştırılır. "
                "AIC ve BIC için küçük değer daha iyi uyumu–karmaşıklık dengesini gösterir. "
                "LogLik için büyük değer daha yüksek olabilirlik anlamına gelir."
            )
            history = st.session_state.get("irt_model_results", {})
            data = target.loc[:, irt_columns]
            fingerprint = (tuple(columns), data.shape, int(pd.util.hash_pandas_object(data, index=True).sum()))
            usable = {m: entry["result"] for m,entry in history.items()
                      if entry.get("fingerprint") == fingerprint}
            if not usable:
                st.info("Bu veri setiyle henüz bir MTK modeli çalıştırılmadı. "
                        "Önce ilgili 1PL–4PL sayfalarında kalibrasyon yapın.")
            else:
                model_order = ["1PL", "2PL", "3PL", "4PL"]
                comparison = pd.DataFrame([
                    {"Model":m, "AIC":float(usable[m]["aic"]),
                     "BIC":float(usable[m]["bic"]),
                     "logLik":float(usable[m]["loglik"]),
                     "Parametre sayısı":int(usable[m]["param_count"]),
                     "Yakınsama":bool(usable[m]["converged"]),
                     "Yöntem":estimation_method(usable[m]),
                     "GH düğümü":(usable[m].get("settings") or {}).get("points")}
                    for m in model_order if m in usable
                ])
                valid = comparison[comparison["Yakınsama"]]
                if valid.empty:
                    st.error("Hiçbir model yakınsama sağlamadı. Güvenilir bir 'EN İYİ' seçimi yapılamaz.")
                else:
                    best_aic = valid.loc[valid["AIC"].idxmin(), "Model"]
                    best_bic = valid.loc[valid["BIC"].idxmin(), "Model"]
                    st.markdown("#### Model karşılaştırma istatistikleri")
                    table_rows = []
                    for m in model_order:
                        if m in usable:
                            rr = comparison.loc[comparison["Model"] == m].iloc[0]
                            table_rows.append({"Model":m, "AIC":f'{rr["AIC"]:.3f}',
                                               "BIC":f'{rr["BIC"]:.3f}',
                                               "logLik":f'{rr["logLik"]:.3f}',
                                               "Yakınsama":"Evet" if rr["Yakınsama"] else "Hayır",
                                               "Yöntem":rr["Yöntem"]})
                        else:
                            table_rows.append({"Model":m, "AIC":"—", "BIC":"—",
                                               "logLik":"—", "Yakınsama":"Çalıştırılmadı",
                                               "Yöntem":"—"})
                    table_rows.append({"Model":"EN İYİ", "AIC":best_aic, "BIC":best_bic,
                                       "logLik":"—", "Yakınsama":"Yalnızca yakınsayanlar",
                                       "Yöntem":"—"})
                    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
                    ca, cb = st.columns(2)
                    ca.metric("AIC'ye göre en iyi model", best_aic)
                    cb.metric("BIC'ye göre en iyi model", best_bic)
                    fig = go.Figure()
                    for metric in ("AIC", "BIC"):
                        fig.add_bar(name=metric, x=comparison["Model"], y=comparison[metric])
                    fig.update_layout(barmode="group", template="plotly_white",
                        title="Model Karşılaştırması (AIC / BIC)",
                        xaxis_title="MTK modeli", yaxis_title="Bilgi ölçütü")
                    st.plotly_chart(fig, use_container_width=True)
                    download_table_pair(pd.DataFrame(table_rows),
                                        "psikoanaliz_mtk_model_karsilastirma",
                                        "Model Karşılaştırması", "irt_comparison")
                if comparison["GH düğümü"].nunique(dropna=False)>1:
                    st.warning("Modeller farklı Gauss–Hermite düğüm sayılarıyla kestirildi; "
                               "logLik/AIC/BIC değerleri tam olarak aynı olabilirlik "
                               "yaklaşımına dayanmaz. Karşılaştırma için aynı düğüm sayısıyla "
                               "yeniden kestirin.")
                if (comparison["Yöntem"]=="EM + L-BFGS-B").any():
                    st.caption("Yöntem sütunu her modelin hangi optimizasyonla kestirildiğini gösterir. "
                               "İyileştirme yalnızca aynı olabilirlik içinde daha yüksek bir noktaya "
                               "ulaşır; model seçiminin gerekçesi değildir.")
                if (~comparison["Yakınsama"]).any():
                    st.warning("Yakınsamamış model(ler) bulunuyor. Bu modeller 'EN İYİ' "
                               "seçiminde değerlendirilmedi.")
                st.caption("AIC ve BIC farklı modelleri seçebilir; bu normaldir. "
                           "Karşılaştırma yalnızca aynı veri ve aynı olabilirlik tanımıyla "
                           "kestirilen modeller için geçerlidir. "
                           "Standart olabilirlik oranı testi ancak düzenli iç içe modeller için "
                           "uygundur; özellikle 3PL/4PL sınır parametreleri için dikkat gerekir.")
            st.stop()
        selected_model = st.radio("İki kategorili MTK modeli",
            ["1PL", "2PL", "3PL", "4PL"],
            index=("1PL", "2PL", "3PL", "4PL").index(model),
            key=f"model_radio_{model}", horizontal=True)
        if selected_model != model:
            st.session_state.nav_page = f"MTK: {selected_model}"
            st.rerun()
        mc1, mc2 = st.columns(2)
        with mc1:
            qpoints = st.select_slider("Gauss-Hermite düğüm sayısı",
                         options=[15, 21, 31, 41], value=31)
        with mc2:
            iterations = st.slider("Azami EM yinelemesi", 20, 150, 70, 10)
        st.caption("Madde parametreleri MML/EM ile kalibre edilir; kişi yeteneğinde EAP, MAP veya ML seçilebilir.")
        st.caption("3PL ve 4PL için yeterli örneklem ve çok sayıda madde önerilir. "
                   "Kestirim uzun sürebilir; model uyumunu ve sınır parametreleri kontrol edin.")
        if model in ("3PL", "4PL"):
            with st.expander("⚙️ İleri kalibrasyon: sınırlar ve çoklu başlangıç"):
                a_upper = st.select_slider("a üst sınırı", options=[4.0, 6.0, 10.0, 20.0, 30.0],
                                           value=4.0, key=f"a_upper_{model}")
                d_lower = (st.select_slider("d alt sınırı",options=[0.05,0.20,0.40,0.65],
                            value=0.65,key="d_lower_4PL") if model=="4PL" else 0.65)
                starts = st.select_slider("Bağımsız başlangıç sayısı", options=[1,2,3],
                                           value=1,key=f"starts_{model}")
                st.caption("Başlangıçlar farklı ilk parametre değerlerinden yürütülür; "
                           "en yüksek logLik değerli çözüm sunulur. Geniş sınırlar "
                           "kararsız veya gerçekçi olmayan kestirimlere yol açabilir.")
                refine = st.checkbox(REFINE_LABEL, value=False, key=f"refine_{model}")
                st.caption(REFINE_HELP + " Seçilirse her başlangıçta uygulanır ve "
                           "kestirim süresini birkaç kat artırabilir.")
        else:
            a_upper,d_lower,starts,refine=4.0,0.65,1,False
        current_settings = calibration_settings(model, qpoints, iterations, a_upper,
                                                d_lower, starts, refine)
        if st.button("▶️ MTK Analizini Başlat", type="primary"):
            try:
                data = target.loc[:, irt_columns].copy()
                spinner_text=(f"{model} kalibrasyonu hesaplanıyor "
                              f"({'EM + L-BFGS-B' if current_settings['refine'] else 'EM'})…")
                with st.spinner(spinner_text):
                    result=run_irt_calibration(data,current_settings)
                fingerprint = (tuple(columns), data.shape, int(pd.util.hash_pandas_object(data, index=True).sum()))
                if "irt_model_results" not in st.session_state:
                    st.session_state.irt_model_results = {}
                st.session_state.irt_model_results[model] = {"fingerprint": fingerprint, "result": result}
                st.success(f"{model} kestirimi tamamlandı "
                           f"({result['settings']['method']}, "
                           f"{result['settings']['seconds']:.1f} sn).")
            except Exception as exc:
                st.session_state.get("irt_model_results", {}).pop(model, None)
                st.error(f"MTK analizi tamamlanamadı: {exc}")
        data_for_check = target.loc[:, irt_columns]
        current_fingerprint = (tuple(columns), data_for_check.shape,
            int(pd.util.hash_pandas_object(data_for_check, index=True).sum()))
        saved = st.session_state.get("irt_model_results", {}).get(model, {})
        r = saved.get("result") if saved.get("fingerprint") == current_fingerprint else None
        if r is not None:
            st.markdown(f"### {r['model']} — Kestirim Sonuçları")
            rpoints = result_points(r, qpoints)
            st.caption(f"Yöntem: {estimation_method(r)} | Gauss–Hermite düğümü: {rpoints}"
                       + (f" | Hesaplama süresi: {r['settings']['seconds']:.1f} sn"
                          if "settings" in r else ""))
            changed = settings_mismatch(r, current_settings)
            if changed:
                st.info("Gösterilen sonuç farklı ayarlarla üretildi ("
                        + ", ".join(changed) + "). Tanılar kayıtlı ayarlarla hesaplanır; "
                        "yeni ayarlar için analizi yeniden başlatın.")
            for level, text in convergence_messages(r):
                getattr(st, level)(text)
            if not r["converged"]:
                with st.expander("Yakınsama uyarısı: açıklama ve öneriler"):
                    st.markdown(GUIDE["yakinsama"])
            if "multistart_audit" in r:
                with st.expander("🔎 Çoklu başlangıç ve optimizasyon denetimi"):
                    st.dataframe(optimization_summary(r), hide_index=True,
                                 use_container_width=True)
                    st.markdown("**Başlangıçlar**")
                    st.dataframe(r["multistart_audit"].round(5),
                                 hide_index=True,use_container_width=True)
                    st.caption(r["multistart_summary"]["Not"])
            if model in ("3PL","4PL"):
                evaluation=model_reliability(r)
                st.markdown("#### Bilimsel güvenilirlik ön değerlendirmesi")
                st.write("**"+evaluation["Karar"]+"**")
                st.caption(f"Kestirim yöntemi: {evaluation['Kestirim yöntemi']} | "
                           f"Yakınsama ölçütü: "
                           f"{'iyileştirme sonrası KKT' if evaluation['Yakınsama ölçütü']=='kkt_after_refinement' else 'EM logL değişimi'}")
                st.caption(evaluation["Sınırlılık"])
                for note in evaluation["Açıklamalar"]:
                    st.caption("• "+note)
            if r["warning"]:
                st.warning(r["warning"])
                with st.expander("4PL modelinde bu uyarı ne demek?"):
                    st.markdown(GUIDE["dortpl"])
            boundary = boundary_report(
                r["items"], model, a_upper=r.get("a_upper", 4.0),
                d_lower=r.get("d_lower", 0.65))
            if not boundary.empty:
                st.warning(f"{len(boundary)} madde-parametre kestirimi optimizasyon sınırına yakın.")
                with st.expander("Hangi madde, hangi parametre, hangi sınır? Ne yapmalıyım?",
                                 expanded=True):
                    st.dataframe(boundary.round(4), hide_index=True,
                                 use_container_width=True)
                    st.markdown("**Ne anlama gelir?** Parametre kestirimi, programın "
                                "izin verdiği alt/üst sınıra yaklaşmıştır. "
                                "Bu, maddenin mutlaka hatalı olduğu anlamına gelmez. "
                                "Kestirim kararsız olabilir; farklı başlangıçları, "
                                "örneklem çeşitliliğini ve daha yalın modelleri "
                                "inceleyin. Sınırı yalnızca uyarıyı kaldırmak "
                                "amacıyla genişletmeyin.")
                    download_table_pair(boundary, f"psikoanaliz_{model}_sinir_tanilari",
                                        "Sınır Parametreleri", f"irt_bounds_{model}")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Katılımcı (N)", r["n"])
            c2.metric("Madde (k)", r["k"])
            c3.metric("AIC", f"{r['aic']:,.1f}")
            c4.metric("BIC", f"{r['bic']:,.1f}")
            st.caption(f"Log-olabilirlik: {r['loglik']:.3f} | "
                       f"EM yineleme: {r['iterations']} | "
                       f"Yakınsama: {'Evet' if r['converged'] else 'Hayır'}"
                       + (" (iyileştirme sonrası KKT ölçütü)"
                          if (r.get("convergence") or {}).get("criterion")=="kkt_after_refinement"
                          else ""))
            st.caption(f"Başlangıç sonuçları (logLik): {', '.join(f'{x:.3f}' for x in r.get('start_logliks', [r['loglik']]))} | "
                       f"a üst sınırı: {r.get('a_upper', 4.0)} | d alt sınırı: {r.get('d_lower', 0.65)}")
            if model in ("3PL","4PL") and "multistart_audit" in r:
                all_attempts=r["multistart_audit"]
                converged_runs=all_attempts[all_attempts["Yakınsama"]]
                st.caption(
                    f"Başlangıç sayısı: {len(all_attempts)} | "
                    f"Yakınsayan: {len(converged_runs)} | "
                    f"Başarısız başlangıç: "
                    f"{int((~all_attempts['Başarılı']).sum())}")
                if len(converged_runs)>1:
                    spread=converged_runs["logLik"].max()-converged_runs["logLik"].min()
                    if spread>0.1:
                        st.warning("Başlangıçlar farklı logLik çözümlerine ulaştı; "
                                   "parametre kararlılığını inceleyin.")
            if model in ("3PL","4PL"):
                with st.expander("🧪 Birinci mertebe KKT/durağanlık tanısı"):
                    st.caption("Türev tabanlı sınır optimizasyonu tanısıdır; "
                               "yakınsama ve model geçerliğini tek başına kanıtlamaz.")
                    if st.button("KKT tanısını hesapla",key=f"kkt_{model}"):
                        kkt_table,kkt_summary=stationarity_diagnostics(
                            target.loc[:,irt_columns],r["items"],model,
                            points=rpoints,a_upper=r.get("a_upper",4.),
                            d_lower=r.get("d_lower",.65))
                        st.metric("KKT toleransını aşan parametre",
                                  kkt_summary["violations"])
                        st.dataframe(kkt_table.round(6),hide_index=True,
                                     use_container_width=True)
                        download_table_pair(
                            kkt_table,f"psikoanaliz_{model}_kkt_tanisi",
                            "KKT Tanı Tablosu",f"kkt_export_{model}")
            tabs = st.tabs(["📋 Parametreler", "📉 Madde Karakteristik Eğrileri",
                            "📈 Madde Bilgi Fonksiyonları", "📊 Test Bilgi Fonksiyonu",
                            "👤 Yetenek Kestirimi", "🔎 Model Tanıları"])
            with tabs[0]:
                param_names = {
                    "a": "a (Ayırt edicilik)",
                    "b": "b (Güçlük / konum)",
                    "c": "c (Alt asimptot / şans)",
                    "d": "d (Üst asimptot)"
                }
                param_view = st.radio("Parametre gösterimi",
                    ["MTK (a, b, c, d)", "Eğim–sabit (a, intercept, c, d)"],
                    horizontal=True, key=f"irt_param_view_{model}")
                display_params = r["items"].copy()
                if param_view.startswith("Eğim"):
                    display_params["intercept"] = -display_params["a"] * display_params["b"]
                    display_params = display_params.drop(columns=["b"])
                    param_names["intercept"] = "intercept (−a × b)"
                display_params = display_params.rename(columns=param_names)
                if st.checkbox("Yaklaşık parametre standart hataları ve %95 güven aralıkları (OPG)",
                               value=False, key=f"irt_opg_{model}"):
                    with st.spinner("Yaklaşık belirsizlik hesaplanıyor…"):
                        uncertainty, unavailable = approximate_parameter_uncertainty(
                            target.loc[:, irt_columns], r["items"], model,
                            points=rpoints, a_upper=r.get("a_upper",4.0),
                            d_lower=r.get("d_lower",0.65))
                    if unavailable:
                        st.warning(f"Yaklaşık SH hesaplanamadı: {unavailable}")
                        with st.expander("Bu uyarı neden çıktı? Sonuçlara etkisi nedir?",
                                         expanded=True):
                            st.markdown(opg_explanation(
                                unavailable, r["n"], r["k"], model,
                                a_upper=r.get("a_upper",4.0),
                                d_lower=r.get("d_lower",0.65)))
                    else:
                        extra = uncertainty.drop(columns=["Madde"])
                        extra.index = display_params.index
                        display_params = pd.concat([display_params, extra], axis=1)
                        st.info("OPG tabanlı yaklaşık standart hatalar ve normal %95 aralıklarıdır. "
                                "Gözlenen Hessian/Louis bilgi matrisi veya bootstrap değildir. "
                                "Özellikle 3PL/4PL için bağımsız doğrulama olmadan "
                                "yayın amaçlı kullanılmamalıdır.")
                with st.expander("🧪 Deneysel gözlenen bilgi matrisi SH (bağımsız doğrulanmadı)"):
                    st.caption("OPG'den farklı olarak marjinal log-olabilirliğin "
                               "gözlenen bilgi matrisi üzerinden sayısal türevini kullanır. "
                               "1PL/2PL'de R mirt ile bağımsız kestirimlere dayalı "
                               "ön karşılaştırma yapılmıştır; aynı sabit parametrelerle "
                               "birebir eşdeğerlik ve 3PL/4PL doğrulaması henüz yoktur. "
                               "Özellikle 3PL/4PL için yayınlarda kullanılmamalıdır.")
                    if st.button("Deneysel Hessian SH hesapla",
                                 key=f"irt_hessian_{model}"):
                        if not r["converged"]:
                            hessian_result,hessian_reason=(None,
                                "EM yakınsamadığı için standart hata raporlanmaz.")
                        else:
                            with st.spinner("Bilgi matrisi hesaplanıyor…"):
                                hessian_result,hessian_reason=observed_information_se(
                                    target.loc[:,irt_columns],r["items"],model,
                                    points=rpoints,a_upper=r.get("a_upper",4.0),
                                    d_lower=r.get("d_lower",0.65))
                        if hessian_reason:
                            st.warning("Hessian SH hesaplanamadı: "+hessian_reason)
                            blocked=hessian_boundary_diagnostics(
                                r["items"],model,
                                a_upper=r.get("a_upper",4.0),
                                d_lower=r.get("d_lower",0.65))
                            if not blocked.empty:
                                st.markdown("**Sınır nedeniyle sorunlu madde-parametreler**")
                                st.dataframe(blocked.round(5),hide_index=True,
                                             use_container_width=True)
                                download_table_pair(
                                    blocked,f"psikoanaliz_{model}_sh_sinir_tanisi",
                                    "SH Sınır Tanıları",f"irt_se_diag_{model}")
                            else:
                                st.info("Doğrudan sınıra dayanan madde-parametre "
                                        "bulunamadı. Bu uyarı, tekil veya kötü "
                                        "koşullanmış **model düzeyi bilgi matrisi** "
                                        "gibi toplu bir nedenden kaynaklanıyor olabilir; "
                                        "tek bir maddeye bağlanamaz.")
                            st.caption("Sınırdaki parametreler, bilgi matrisi "
                                       "tekilliği veya sayısal kararsızlık "
                                       "nedeniyle güvenilir SH hesaplanamayabilir. "
                                       "Bu durum tek başına maddeyi geçersiz kılmaz.")
                        else:
                            st.warning("Bu SH değerleri deneysel olup henüz "
                                       "R mirt ile bağımsız doğrulanmamıştır.")
                            st.dataframe(hessian_result.round(4),
                                         hide_index=True,use_container_width=True)
                            download_table_pair(hessian_result,
                                                f"psikoanaliz_{model}_hessian_deneysel",
                                                "Deneysel Hessian SH",
                                                f"irt_hessian_dl_{model}")
                st.dataframe(display_params.round(4), hide_index=True,
                             use_container_width=True)
                st.caption("a: Madde ayırt ediciliği; b: güçlük (konum) parametresi; "
                           "c: alt asimptot (tahmin/şans); d: üst asimptot (azami doğru yanıt olasılığı). "
                           "Modelde kestirilmeyen parametreler sabit tutulur.")
                download_table_pair(display_params,
                                    f"psikoanaliz_{r['model']}_parametreler",
                                    "Madde Parametreleri", f"irt_params_{model}")
            with tabs[1]:
                chosen = st.multiselect("ICC için maddeler",
                    options=r["items"]["Madde"].tolist(),
                    default=r["items"]["Madde"].head(8).tolist(), key="irt_icc_items")
                fig = go.Figure()
                for label in chosen:
                    i = r["items"].index[r["items"]["Madde"]==label][0]
                    fig.add_scatter(x=r["grid"], y=r["icc"][:,i],
                                    name=label, mode="lines")
                fig.update_layout(title="Madde Karakteristik Eğrileri (ICC)",
                    xaxis_title="Yetenek (θ)", yaxis_title="P(X=1 | θ)",
                    yaxis_range=[0,1], template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            with tabs[2]:
                chosen_iif = st.multiselect("IIF için maddeler",
                    options=r["items"]["Madde"].tolist(),
                    default=r["items"]["Madde"].head(8).tolist(),key="irt_iif_items")
                fig = go.Figure()
                for label in chosen_iif:
                    i = r["items"].index[r["items"]["Madde"]==label][0]
                    fig.add_scatter(x=r["grid"],y=r["iif"][:,i],
                                    mode="lines",name=label)
                fig.update_layout(title="Madde Bilgi Fonksiyonları (IIF)",
                    xaxis_title="Yetenek (θ)", yaxis_title="Bilgi",template="plotly_white")
                st.plotly_chart(fig,use_container_width=True)
            with tabs[3]:
                fig=go.Figure()
                fig.add_scatter(x=r["grid"],y=r["tif"],name="Test bilgisi")
                fig.add_scatter(x=r["grid"],
                    y=1/np.sqrt(np.maximum(r["tif"],1e-12)),
                    name="Koşullu SEM",yaxis="y2")
                fig.update_layout(title="Test Bilgi Fonksiyonu ve Koşullu SEM",
                    xaxis_title="Yetenek (θ)",yaxis_title="Test bilgisi",
                    yaxis2=dict(title="SEM",overlaying="y",side="right"),
                    template="plotly_white")
                st.plotly_chart(fig,use_container_width=True)
            with tabs[4]:
                ability_method = st.radio("Yetenek kestirim yöntemi",
                    ["EAP", "MAP", "ML"], horizontal=True, key=f"ability_method_{model}")
                prior_sd = st.number_input("Önsel dağılım SS (EAP/MAP)",
                    min_value=0.2,max_value=3.0,value=1.0,step=0.1,
                    key=f"prior_sd_{model}", disabled=ability_method=="ML")
                ability_table = estimate_ability(target.loc[:, irt_columns],
                    r["items"],method=ability_method,points=161,prior_sd=prior_sd)
                if ability_table["Sınır Kestirimi"].any():
                    st.warning("Bazı ML/MAP yetenek kestirimleri hesaplama aralığının sınırına ulaştı.")
                    with st.expander("Kişi yeteneğinin sınırda olması ne demek?"):
                        st.markdown(GUIDE["yetenek"])
                fig=go.Figure()
                fig.add_histogram(x=ability_table["Yetenek (theta)"],nbinsx=35,
                    name=f"{ability_method} θ")
                fig.update_layout(title=f"{ability_method} Yetenek Dağılımı",
                    xaxis_title=f"{ability_method} θ",yaxis_title="Katılımcı",
                    template="plotly_white")
                st.plotly_chart(fig,use_container_width=True)
                scores=ability_table.copy()
                scores["Sınır Kestirimi"] = scores["Sınır Kestirimi"].map(
                    {True: "Sınırda", False: "Sınırda değil"})
                scores.insert(0, "Kişi", [f"{i}. Kişi" for i in range(1, len(scores)+1)])
                st.dataframe(scores.head(20).round(3),hide_index=True,
                             use_container_width=True)
                download_table_pair(scores, f"psikoanaliz_{model}_yetenek",
                                    "Kişi Yetenekleri", f"irt_ability_{model}")
            with tabs[5]:
                st.markdown("#### Test Karakteristik Eğrisi (TCC)")
                tcc_fig = go.Figure()
                tcc_fig.add_scatter(x=r["grid"], y=r["icc"].sum(axis=1),
                                    mode="lines", name="Beklenen test puanı")
                tcc_fig.update_layout(
                    title="Test Karakteristik Eğrisi — Beklenen Toplam Puan",
                    xaxis_title="Yetenek (θ)", yaxis_title="Beklenen doğru sayısı",
                    template="plotly_white")
                st.plotly_chart(tcc_fig, use_container_width=True)

                st.markdown("#### Wright Haritası (Kişi–Madde Haritası)")
                map_fig = go.Figure()
                map_fig.add_histogram(x=ability_table["Yetenek (theta)"],
                                      name="Kişiler", nbinsx=35, opacity=0.65)
                for item_row in r["items"].itertuples(index=False):
                    map_fig.add_scatter(
                        x=[item_row.b], y=[0], mode="markers",
                        marker=dict(symbol="diamond", size=9),
                        name=str(item_row[0]), showlegend=False,
                        hovertemplate=f"{item_row[0]}: b=%{{x:.2f}}<extra></extra>")
                map_fig.update_layout(
                    title="Kişi Yetenekleri ve Madde Güçlükleri (Ortak θ Ölçeği)",
                    xaxis_title="Yetenek / madde güçlüğü (θ, b)",
                    yaxis_title="Kişi frekansı (elmaslar: madde güçlükleri)",
                    template="plotly_white")
                st.plotly_chart(map_fig, use_container_width=True)

                st.markdown("#### Toplam Puan ve MTK Yetenek Puanı İlişkisi")
                total = target.loc[:, irt_columns].sum(axis=1).to_numpy(dtype=float)
                z = (total-total.mean())/total.std(ddof=1) if total.std(ddof=1)>0 else np.zeros(len(total))
                ability_values = ability_table["Yetenek (theta)"].to_numpy()
                corr = float(np.corrcoef(z, ability_values)[0,1]) if np.std(z)>0 and np.std(ability_values)>0 else np.nan
                scatter = go.Figure()
                scatter.add_scatter(x=z,y=ability_values,mode="markers",
                    marker=dict(size=5,opacity=.35),name="Katılımcılar")
                scatter.update_layout(title="Standartlaştırılmış Test Puanı ve Yetenek Kestirimi",
                    xaxis_title="Toplam puan Z",yaxis_title=f"{ability_method} θ",
                    template="plotly_white")
                st.plotly_chart(scatter,use_container_width=True)
                st.caption(f"Pearson r = {corr:.4f}" if np.isfinite(corr) else
                           "Korelasyon hesaplanamadı (sabit puanlar).")

                st.markdown("#### Standart Puanlar ve Kişi Yetenekleri")
                ability_report = pd.DataFrame({
                    "Kişi":[f"{i}. Kişi" for i in range(1,len(total)+1)],
                    "Toplam puan":total, "Z puanı":z, "T puanı":50+10*z,
                    f"{ability_method} θ":ability_values,
                    "Yetenek SH":ability_table["Standart Hata"].to_numpy()})
                st.dataframe(ability_report.head(20).round(3), hide_index=True,
                             use_container_width=True)
                download_table_pair(ability_report,
                                    f"psikoanaliz_{model}_kisi_puanlari",
                                    "Tüm Kişi Puanları", f"irt_full_scores_{model}")

                fig=go.Figure()
                fig.add_scatter(x=list(range(1,len(r["history"])+1)),
                                y=r["history"],mode="lines+markers")
                fig.update_layout(title="EM Log-Olabilirlik İzleği",
                    xaxis_title="Yineleme",yaxis_title="Log-olabilirlik",
                    template="plotly_white")
                st.plotly_chart(fig,use_container_width=True)
                st.markdown("#### Yerel Bağımsızlık — Yaklaşık Q3")
                st.caption("Yetenek kestirimlerine koşullu madde artıkları arasındaki "
                           "korelasyonlar. Bu, keşifsel bir Yen Q3 benzeri tanıdır; "
                           "kritik değer veya anlamlılık testi değildir. Kişi puanlarının "
                           "aynı yanıtlarla kestirilmesi yanlılığa neden olabilir.")
                q3 = approximate_q3(target.loc[:, irt_columns],
                                    r["items"], ability_table["Yetenek (theta)"])
                if q3.shape[0] > 1:
                    fig_q3 = go.Figure(go.Heatmap(
                        z=q3.to_numpy(), x=q3.columns.tolist(),
                        y=q3.index.tolist(), zmid=0, colorscale="RdBu",
                        colorbar=dict(title="Artık r")))
                    fig_q3.update_layout(title="Madde Artıkları Korelasyon Matrisi (Q3 benzeri)",
                                         template="plotly_white")
                    st.plotly_chart(fig_q3, use_container_width=True)
                    upper = q3.to_numpy()[np.triu_indices(len(q3), k=1)]
                    finite = upper[np.isfinite(upper)]
                    if len(finite):
                        st.metric("En yüksek mutlak artık korelasyonu",
                                  f"{np.max(np.abs(finite)):.3f}")
                    download_table_pair(q3.reset_index().rename(columns={"index":"Madde"}),
                                        f"psikoanaliz_{model}_q3",
                                        "Yaklaşık Q3 Matrisi", f"irt_q3_{model}")
                st.markdown("#### Madde–Kalan Puan Uyum Tanısı (Keşifsel)")
                st.caption("Her madde için, diğer maddelerden elde edilen toplam "
                           "puana göre gözlenen ve beklenen doğru yanıt sayıları "
                           "karşılaştırılır. Bu Pearson tipi keşifsel bir ölçüdür; "
                           "Orlando–Thissen S-X² testi değildir. Serbestlik derecesi "
                           "ve p değeri hesaplanmaz.")
                if st.checkbox("Madde–kalan puan uyum tablosunu hesapla",
                               key=f"irt_restscore_{model}", value=False):
                    with st.spinner("Madde uyum tanıları hesaplanıyor…"):
                        fit_table=item_fit_by_restscore(
                            target.loc[:, irt_columns],r["items"])
                    st.dataframe(fit_table.round(3),hide_index=True,
                                 use_container_width=True)
                    download_table_pair(fit_table,
                                        f"psikoanaliz_{model}_rest_puan_uyum",
                                        "Madde–Kalan Puan Tanıları",
                                        f"irt_restscore_download_{model}")
                st.markdown("#### S-X² Madde Uyum Testi (mirt algoritması)")
                st.caption("Toplam puana koşullu madde uyumu; 61 eşit aralıklı "
                           "[-6,6] theta noktası, beklenen hücre eşiği 1 ve "
                           "komşu grup birleştirme kullanılır. Sabit N(0,1) "
                           "gizil dağılımı ve model başına madde parametre "
                           "sayısı varsayılır. Parametre kestirimleri ayrıca "
                           "bağımsız doğrulama gerektirir.")
                if not r["converged"]:
                    st.warning("Model yakınsamamış; S-X² p değerleri kesin "
                               "uyum kanıtı olarak yorumlanmamalıdır.")
                    with st.expander("S-X² ve yakınsama uyarısı hakkında"):
                        st.markdown(GUIDE["sx2"])
                if st.checkbox("S-X² madde uyumunu hesapla",
                               value=False,key=f"irt_sx2_{model}"):
                    try:
                        sx2_table=sx2_mirt_compatible(
                            target.loc[:,irt_columns],r["items"],model)
                        st.dataframe(sx2_table.round(5),hide_index=True,
                                     use_container_width=True)
                        st.caption("p < .05 olası uyumsuzluğa işaret eder; "
                                   "25 madde için çoklu test düzeltmesi "
                                   "uygulanmamıştır. Sınır kestirimlerini "
                                   "ve örneklem yeterliğini ayrıca inceleyin.")
                        download_table_pair(sx2_table,
                                            f"psikoanaliz_{model}_sx2",
                                            "S-X² Madde Uyumu",f"irt_sx2_dl_{model}")
                    except (ValueError,RuntimeError) as sx_error:
                        st.warning(f"S-X² hesaplanamadı: {sx_error}")
                diagnostic=r["items"][["Madde","Gözlenen p","Model p",
                                       "Artık RMS","Sınır uyarısı"]]
                st.dataframe(diagnostic.round(4),hide_index=True,
                             use_container_width=True)
                st.caption("Artık RMS betimsel bir tanıdır; formal madde uyum "
                           "testi değildir. AIC/BIC yalnızca aynı veri "
                           "üzerindeki modeller arasında karşılaştırılmalıdır.")
