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
from core.descriptives import describe_items, describe_total
from ui.rasch_view import (MNSQ_OPTIONS, METHOD_NOTE, screen_items, rasch_settings, run_rasch,
                           fingerprint as rasch_fingerprint, summary_rows as rasch_summary,
                           misfit_persons, wright_map, DIF_NOTE, Q3_NOTE, group_candidates,
                           dif_figure, q3_heatmap, raw_score_figure, run_global_fit, m2_rows, run_q3,
                           run_dif, dif_summary_rows)


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

PRIVACY_DISCLOSURE = (
    "Yüklediğiniz veriler etkin oturumda analiz için işlenir. Uygulama kalıcı bir kullanıcı veri arşivi tutmayı amaçlamaz. "
    "Verilerinizi oturumdan temizlemek için Oturum Verilerini Temizle düğmesini kullanın. "
    "Barındırma sağlayıcısının teknik kayıtları, tarayıcı önbelleği ve geçici sunucu kopyaları "
    "bu işlemle garanti olarak silinemez. Kimlik belirleyici bilgileri yüklemeden önce anonimleştirin." 
)

def clear_active_analysis_session():
    next_nonce = int(st.session_state.get('privacy_upload_nonce', 0)) + 1
    for name in list(st.session_state.keys()):
        if name not in ('nav_page', 'privacy_upload_nonce'):
            del st.session_state[name]
    st.session_state['privacy_upload_nonce'] = next_nonce

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
    with st.expander("🧮 EserPsychometrics Yerel KTK Motoru", expanded=True):
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
    with st.expander("Python Kütüphaneleri / Bağımsız KTK Motoru"):
        st.button("📄 Alfa ve Madde Karşılaştırması (Pingouin)",
                  key="nav_ctt_dual",use_container_width=True,
                  on_click=navigate_to,args=("KTK: Çift Motor Rapor",))
        st.caption("Bu sayfa yerel KTK ile bağımsız Python hesaplamalarını karşılaştırır.")

    st.markdown("---")
    st.markdown("### II. Madde Tepki Kuramı (MTK)")
    with st.expander("🧮 EserPsychometrics Yerel MTK Motoru", expanded=True):
        st.caption("İkili modeller: yerel analiz ve model karşılaştırmaları")
        binary_quick_model = st.selectbox(
            "Hızlı ikili MTK model seçimi",
            ("Rasch", "1PL", "2PL", "3PL", "4PL"),
            key="binary_irt_quick_model",
        )
        st.button("Seçili MTK analizine git",key="binary_irt_quick_go",
                  use_container_width=True,on_click=navigate_to,
                  args=(f"MTK: {binary_quick_model}",))
        for m in ("Rasch","1PL","2PL","3PL","4PL"):
            st.button(f"📐 {m} Modeli",key=f"native_irt_{m}",
                      use_container_width=True,on_click=navigate_to,args=(f"MTK: {m}",))
        for label,page in [
            ("📊 Model Karşılaştırması","MTK: Karşılaştırma"),
            ("📋 İkili MTK B13 Raporu","MTK: İkili Rapor"),
            ("🔍 MTK Varsayım Denetimi","MTK: Varsayım Denetimi"),
            ("⚖️ MTK DIF Ön Taraması","MTK: Çok Modelli DIF Taraması"),
        ]:
            st.button(label,key=f"native_{page}",use_container_width=True,
                      on_click=navigate_to,args=(page,))
        st.caption("Formel DIF sayfalarında doğrulama durumu ayrıca belirtilir.")
        st.button("⚖️ GRM Formel DIF (Deneysel)",key="nav_grm_dif",
                  use_container_width=True,on_click=navigate_to,args=("MTK: GRM Formel DIF",))
        st.button("⚖️ PCM/GPCM/RSM/NRM DIF (Deneysel)",key="nav_poly_dif",
                  use_container_width=True,on_click=navigate_to,args=("MTK: Çok Kategorili Formel DIF",))
    with st.expander("Python mirt Kütüphanesi — MTK"):
        st.caption("Python mirt 1.2.0: ayrı kestirim motoru; Rasch ve çok kategorili "
                   "DIF'in yerel uygulamalarıyla karıştırmayın.")
        for m in ("1PL","2PL","3PL","4PL","GRM","PCM","GPCM","NRM"):
            st.button(f"Python mirt — {m}",key=f"python_irt_{m}",
                      use_container_width=True,on_click=navigate_to,
                      args=(f"MTK: Python mirt {m}",))
        st.caption("RSM: Python mirt genel API'sinde desteklenmiyor; yerel B14 motorunu kullanın.")
        st.button("🧬 B14 Python mirt Çok Kategorili Rapor",
                  key="nav_irt_b14",use_container_width=True,
                  on_click=navigate_to,args=("MTK: Çok Kategorili B14",))

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
    st.markdown("### 🔐 Veri Gizliliği")
    st.caption(PRIVACY_DISCLOSURE)
    if st.button("🗑️ Oturum Verilerini Temizle", key="clear_privacy_session", use_container_width=True):
        clear_active_analysis_session()
        st.rerun()
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

    **Madde Tepki Kuramı (MTK)** bölümünde Rasch, 1PL, 2PL, 3PL, 4PL ve
    çok kategorili modeller için yerel ve Python kütüphane tabanlı analiz yolları
    bulunmaktadır. Modellerin destek kapsamı, yakınsama durumu ve bilimsel
    doğrulama düzeyi farklıdır. Bazı DIF yöntemleri deneysel olup 4PL formel
    DIF çıkarımı güvenilir koşullar sağlanana kadar kapalıdır.

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
    st.info(PRIVACY_DISCLOSURE)

    tab_upload, tab_sample = st.tabs(["📤 Kendi Verinizi Yükleyin", "🧪 Örnek Veri Setleri ile Deneyin"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Veri dosyanızı seçin (CSV, Excel .xlsx/.xls veya SPSS .sav)",
            type=["csv", "xlsx", "xls", "sav"],
            key=f"private_file_upload_{st.session_state.get('privacy_upload_nonce', 0)}"
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
# SAYFA 14: RASCH ANALİZİ (İKİLİ, MML)
# =============================================================================
elif selected_page.startswith("MTK: Python mirt "):
    from ui.irt_python_library_view import render as render_python_library
    py_model = selected_page.replace("MTK: Python mirt ", "", 1)
    py_data = st.session_state.scored_df if st.session_state.scored_df is not None else st.session_state.raw_df
    if py_data is None:
        st.warning("Önce veri yükleyin.")
    else:
        render_python_library(py_data,st.session_state.item_cols,
                              st.session_state.id_col,st.session_state.group_col,py_model)

elif selected_page == "MTK: Rasch":
    render_academic_header()
    st.subheader("📏 Rasch Analizi — ikili (dikotom) Rasch modeli")
    st.caption(METHOD_NOTE)
    scored = st.session_state.scored_df
    raw = st.session_state.raw_df
    target = scored if scored is not None else raw
    columns = [c for c in st.session_state.item_cols
               if c not in (st.session_state.id_col, st.session_state.group_col)]
    if target is not None and not columns:
        columns = [c for c in target.columns if c not in (st.session_state.id_col, st.session_state.group_col)]
    if target is None:
        st.warning("Önce Veri Yükleme bölümünden 0/1 puanlı bir veri dosyası yükleyin.")
    else:
        usable, screen = screen_items(target, columns)
        excluded = screen[screen["Durum"] != "Uygun"]
        if not excluded.empty:
            st.warning(f"{len(excluded)} sütun Rasch analizine alınmadı (0/1 dışı değer veya sabit yanıt).")
            with st.expander("Madde ön taraması"):
                st.dataframe(screen.round(3), hide_index=True, use_container_width=True)
        if len(usable) < 3:
            st.error("Rasch analizi için en az 3 uygun ikili madde gereklidir.")
        else:
            if screen.loc[screen["Durum"] == "Uygun", "Eksik"].sum():
                st.info("Eksik yanıtlar silinmez; olabilirlikten çıkarılır (MAR varsayımı). "
                        "Eksikliğin nedeni sistematikse sonuçlar yanlı olabilir.")
            c1, c2 = st.columns(2)
            with c1:
                points = st.select_slider("Gauss–Hermite düğüm sayısı", options=[21, 31, 41, 61],
                                          value=41, key="rasch_points")
                zlim = st.select_slider("|ZSTD| eşiği", options=[1.96, 2.0, 2.5, 3.0], value=2.0,
                                        key="rasch_zstd")
            with c2:
                label = st.selectbox("Kabul edilebilir MNSQ aralığı", list(MNSQ_OPTIONS),
                                     key="rasch_mnsq")
                include_ext = st.checkbox("Uç puanlı kişileri madde uyumuna kat", value=False,
                                          key="rasch_ext")
            st.caption("Bir madde veya kişi yalnızca MNSQ aralık dışında **ve** |ZSTD| eşiği "
                       "aşıldığında 'İnceleme önerilir' olarak işaretlenir (Wright & Linacre, 1994). "
                       "Bu bir karar kuralı değil, inceleme önerisidir.")
            settings = rasch_settings(points, label, zlim, include_ext)
            fp = rasch_fingerprint(target, usable)
            if st.button("▶️ Rasch Analizini Başlat", type="primary", key="rasch_run"):
                try:
                    with st.spinner("Rasch modeli kestiriliyor (MML)…"):
                        result = run_rasch(target, usable, settings)
                    st.session_state.rasch_result = {"fingerprint": fp, "result": result}
                    st.success(f"Rasch analizi tamamlandı ({result['ui_settings']['seconds']:.1f} sn).")
                except Exception as exc:
                    st.session_state.pop("rasch_result", None)
                    st.error(f"Rasch analizi tamamlanamadı: {exc}")
            saved = st.session_state.get("rasch_result") or {}
            rr = saved.get("result") if saved.get("fingerprint") == fp else None
            if rr is not None:
                stored = {k: rr["ui_settings"][k] for k in ("points", "mnsq_label", "zstd_limit", "include_extreme")}
                current = {k: settings[k] for k in stored}
                if stored != current:
                    st.info("Gösterilen sonuç farklı ayarlarla üretildi; yeni ayarlar için analizi yeniden başlatın.")
                if not rr["fit"]["converged"]:
                    st.warning("MML kestirimi yakınsama ölçütünü sağlamadı; sonuçlar geçicidir.")
                for note in rr["warnings"]:
                    st.caption("• " + note)
                rel = rr["reliability"].set_index("Gösterge")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Örtük SS (σ)", f"{rr['fit']['sd']:.3f}")
                m2.metric("Kişi güvenirliği", f"{rel['Güvenirlik'].iloc[0]:.3f}")
                m3.metric("Madde güvenirliği", f"{rel['Güvenirlik'].iloc[1]:.3f}"
                          if np.isfinite(rel['Güvenirlik'].iloc[1]) else "—")
                m4.metric("İncelenecek madde", int((rr["items"]["Uyum"] == "İnceleme önerilir").sum()))
                tabs = st.tabs(["📋 Madde parametreleri ve uyum", "👤 Kişi uyumu", "🗺️ Wright haritası",
                                "📐 Güvenirlik ve model özeti", "🧮 Model Uyumu", "🔗 Yerel Bağımsızlık",
                                "⚖️ DIF / Ölçme Değişmezliği"])
                with tabs[0]:
                    st.dataframe(rr["items"].round(3), hide_index=True, use_container_width=True)
                    st.caption("b: madde güçlüğü (logit; örtük ortalama 0). Infit bilgiye ağırlıklı, Outfit "
                               "uç yanıtlara duyarlı ortalama kare uyumdur; beklenen değer 1'dir. "
                               ">1 beklenenden fazla gürültü, <1 aşırı tahmin edilebilirlik gösterir.")
                    download_table_pair(rr["items"], "psikoanaliz_rasch_maddeler", "Rasch Maddeler", "rasch_items_dl")
                with tabs[1]:
                    flagged = misfit_persons(rr)
                    st.write(f"İnceleme önerilen kişi: **{len(flagged)}** / {len(rr['persons'])}")
                    only = st.checkbox("Yalnızca inceleme önerilenleri göster", value=True, key="rasch_only_flagged")
                    st.dataframe((flagged if only else rr["persons"]).round(3), hide_index=True,
                                 use_container_width=True)
                    st.caption("Kişi uyumu WLE ile hesaplanır; uç puanlı kişiler için uyum tanımsızdır. "
                               "Az sayıda maddede ZSTD değerleri kararsızdır.")
                    download_table_pair(rr["persons"], "psikoanaliz_rasch_kisiler", "Rasch Kişiler", "rasch_persons_dl")
                with tabs[2]:
                    st.plotly_chart(wright_map(rr), use_container_width=True)
                    st.caption("Sol: kişi WLE dağılımı; sağ: madde güçlükleri. Aynı logit ölçeğindedir; "
                               "kişi dağılımının madde güçlükleriyle örtüşmesi ölçme duyarlılığını gösterir.")
                with tabs[3]:
                    st.dataframe(rr["reliability"].round(4), hide_index=True, use_container_width=True)
                    st.caption("Kişi güvenirliği model SH ile ve uç puanlılar hariç hesaplanır; "
                               "uyumsuzlukla şişirilmiş (RSEM) hata kullanan yazılımlar (ör. RaschPy) daha "
                               "farklı değer verebilir. Marjinal güvenirlik EAP tabanlıdır.")
                    st.dataframe(rasch_summary(rr), hide_index=True, use_container_width=True)
                # ---- B9: genel uyum, yerel bağımsızlık, DIF (B8 sonucuna bağlı; ayrı oturum anahtarları)
                b8_key = (fp, tuple(sorted(stored.items())))
                with tabs[4]:
                    st.caption("Genel uyum B8 kestirimleri (b, σ, WLE θ) üzerinden yeniden kestirim yapmadan "
                               "hesaplanır. Her istatistiğin sınıfı (çıkarımsal/betimsel) tabloda belirtilir.")
                    strata = st.select_slider("Yetenek tabakası sayısı (WLE çeyreklik dilimleri)",
                                              options=[3, 4, 5, 6, 8, 10], value=5, key="rasch_gf_strata")
                    if st.button("▶️ Model uyumunu hesapla", key="rasch_gf_run"):
                        try:
                            with st.spinner("M2 ve betimsel uyum özetleri hesaplanıyor…"):
                                st.session_state.rasch_gf = {"key": b8_key, "result": run_global_fit(rr, strata)}
                        except Exception as exc:
                            st.session_state.pop("rasch_gf", None)
                            st.error(f"Model uyumu hesaplanamadı: {exc}")
                    saved_gf = st.session_state.get("rasch_gf") or {}
                    gf = saved_gf.get("result") if saved_gf.get("key") == b8_key else None
                    if gf is not None:
                        if gf["ui_settings"]["strata"] != strata:
                            st.info("Gösterilen sonuç farklı tabaka sayısıyla üretildi; yeniden hesaplayın.")
                        st.markdown("**Sınırlı bilgi uyumu (M2)**")
                        st.dataframe(m2_rows(gf), hide_index=True, use_container_width=True)
                        if gf["m2"]["available"]:
                            st.caption("M2 tek ve iki değişkenli marjinallerden hesaplanır (Maydeu-Olivares ve Joe, "
                                       "2006); sd = s − q, s = k + k(k−1)/2, q = k + 1. p değeri asimptotik χ² "
                                       "yaklaşımına dayanır ve küçük örneklemde kalibrasyonu sınırlıdır. RMSEA2 "
                                       "N ile hesaplanır (mirt N−1 kullanır).")
                        st.markdown("**İstatistik sınıflandırması**")
                        st.dataframe(gf["classification"], hide_index=True, use_container_width=True)
                        st.plotly_chart(raw_score_figure(gf), use_container_width=True)
                        st.caption(f"Ham puan karşılaştırması yalnızca eksiksiz yanıtlı {gf['raw_complete_n']} kişiyi "
                                   f"içerir ({gf['raw_incomplete_n']} kişi eksik yanıt nedeniyle dışarıda).")
                        download_table_pair(gf["raw_scores"].round(4), "psikoanaliz_rasch_hampuan",
                                            "Ham puan dağılımı", "rasch_gf_raw_dl")
                        st.markdown("**Madde × yetenek tabakası: gözlenen ve beklenen doğru oranı (betimsel)**")
                        st.dataframe(gf["strata"].round(3), hide_index=True, use_container_width=True)
                        download_table_pair(gf["strata"].round(4), "psikoanaliz_rasch_tabaka",
                                            "Tabaka tablosu", "rasch_gf_strata_dl")
                        st.markdown("**Standart artık özeti (betimsel)**")
                        st.dataframe(pd.DataFrame([gf["residuals"]]).round(4), hide_index=True,
                                     use_container_width=True)
                with tabs[5]:
                    st.caption(Q3_NOTE)
                    q1, q2, q3c = st.columns(3)
                    with q1:
                        q3_person = st.selectbox("Kişi kestirimi", ["WLE", "EAP"], key="rasch_q3_person")
                    with q2:
                        q3_min = st.select_slider("Çift başına en az ortak yanıt", options=[10, 20, 30, 50, 100],
                                                  value=30, key="rasch_q3_min")
                    with q3c:
                        q3_boot = st.select_slider("Parametrik bootstrap tekrar sayısı", options=[0, 50, 100, 200],
                                                   value=0, key="rasch_q3_boot",
                                                   help="Her tekrarda model yeniden kestirilir; 200 tekrar "
                                                        "orta boyutlu veride birkaç dakika sürebilir.")
                    q3_seed = st.number_input("Bootstrap tohum değeri", min_value=0, max_value=2**31 - 1,
                                              value=2026, step=1, key="rasch_q3_seed")
                    q3_settings = {"person_method": q3_person, "min_pairs": int(q3_min),
                                   "bootstrap": int(q3_boot), "seed": int(q3_seed)}
                    if st.button("▶️ Q3 analizini başlat", key="rasch_q3_run"):
                        try:
                            with st.spinner("Q3 artık korelasyonları hesaplanıyor…"):
                                st.session_state.rasch_q3 = {"key": b8_key, "result": run_q3(
                                    target, usable, rr, q3_person, q3_min, q3_boot, q3_seed)}
                        except Exception as exc:
                            st.session_state.pop("rasch_q3", None)
                            st.error(f"Q3 analizi tamamlanamadı: {exc}")
                    saved_q3 = st.session_state.get("rasch_q3") or {}
                    q3r = saved_q3.get("result") if saved_q3.get("key") == b8_key else None
                    if q3r is not None:
                        shown = {k: q3r["ui_settings"][k] for k in q3_settings}
                        if shown != q3_settings:
                            st.info("Gösterilen Q3 sonucu farklı ayarlarla üretildi; yeniden başlatın.")
                        a1, a2, a3 = st.columns(3)
                        a1.metric("Ortalama Q3", f"{q3r['mean_q3']:.3f}")
                        a2.metric("Bağımsızlık altında beklenen ≈ −1/(k−1)", f"{q3r['expected_under_li']:.3f}")
                        a3.metric("Tanımsız çift", q3r["n_undefined"])
                        bs = q3r["bootstrap"]
                        if bs:
                            st.markdown(
                                f"**Parametrik bootstrap** ({bs['used']} geçerli / {bs['replications']} tekrar, "
                                f"tohum {bs['seed']}): gözlenen en büyük düzeltilmiş Q3 = "
                                f"{bs['observed_max_adjusted_q3']:.3f}; yerel bağımsızlık altında %95 / %99 "
                                f"nicelikleri {bs['max_quantile_95']:.3f} / {bs['max_quantile_99']:.3f}; "
                                f"Monte Carlo p (en büyük değer) = {bs['p_max']:.3f}.")
                            st.caption("Bootstrap p değeri tüm çiftlerin en büyüğüne aittir (çoklu karşılaştırmayı "
                                       "hesaba katar); sınırlı tekrar sayısı p değerinin çözünürlüğünü 1/(B+1) ile "
                                       "sınırlar. Çift bazındaki %95 nicelikleri yalnızca betimseldir. Bu bootstrap "
                                       "referansı B9'da yalnızca sınırlı bir simülasyonla incelenmiştir (deneysel).")
                            if bs["failed"]:
                                st.warning(f"{bs['failed']} bootstrap tekrarı yakınsamadı veya kestirilemedi ve "
                                           "dışarıda bırakıldı.")
                        st.plotly_chart(q3_heatmap(q3r), use_container_width=True)
                        st.markdown("**Düzeltilmiş Q3'e göre sıralı madde çiftleri**")
                        st.dataframe(q3r["pairs"].round(4), hide_index=True, use_container_width=True)
                        download_table_pair(q3r["pairs"].round(5), "psikoanaliz_rasch_q3_ciftler", "Q3 çiftleri",
                                            "rasch_q3_dl")
                        if not q3r["fit_converged"]:
                            st.warning("Q3 için yeniden kestirim yakınsamadı; sonuçlar geçicidir.")
                with tabs[6]:
                    st.caption(DIF_NOTE)
                    excluded_cols = set(usable) | {st.session_state.id_col}
                    candidates = group_candidates(target.drop(columns=[c for c in excluded_cols
                                                                       if c in target.columns]), usable)
                    if not candidates:
                        st.info("DIF için 2–20 düzeyli, madde ve kimlik dışı bir grup sütunu bulunamadı.")
                    else:
                        default_g = (candidates.index(st.session_state.group_col)
                                     if st.session_state.group_col in candidates else 0)
                        gcol = st.selectbox("Grup değişkeni", candidates, index=default_g, key="rasch_dif_group")
                        levels = sorted(target[gcol].dropna().unique().tolist(), key=str)
                        d1, d2 = st.columns(2)
                        with d1:
                            ref = st.selectbox("Referans grup", levels, key="rasch_dif_ref")
                        with d2:
                            foc_options = [v for v in levels if v != ref]
                            foc = st.selectbox("Odak grup", foc_options, key="rasch_dif_focal")
                        if len(levels) > 2:
                            st.caption(f"Grup değişkeninde {len(levels)} düzey var; yalnızca seçilen iki grup "
                                       "karşılaştırılır, diğer kişiler bu analizden çıkarılır.")
                        e1, e2, e3 = st.columns(3)
                        with e1:
                            alpha = st.select_slider("FDR düzeyi (BH)", options=[.01, .05, .10], value=.05,
                                                     key="rasch_dif_alpha")
                        with e2:
                            purify = st.checkbox("Yinelemeli çapa arındırması", value=True, key="rasch_dif_purify")
                        with e3:
                            rounds = st.select_slider("En fazla arındırma turu", options=[1, 3, 5, 10], value=10,
                                                      key="rasch_dif_rounds", disabled=not purify)
                        dif_settings = {"group_col": gcol, "reference": ref, "focal": foc, "alpha": float(alpha),
                                        "purify": bool(purify), "max_rounds": int(rounds)}
                        if st.button("▶️ DIF analizini başlat", key="rasch_dif_run"):
                            try:
                                with st.spinner("İki gruplu ortak Rasch modeli kestiriliyor…"):
                                    st.session_state.rasch_dif = {"key": b8_key, "result": run_dif(
                                        target, usable, gcol, ref, foc, alpha, purify, rounds,
                                        points=settings["points"])}
                            except Exception as exc:
                                st.session_state.pop("rasch_dif", None)
                                st.error(f"DIF analizi tamamlanamadı: {exc}")
                        saved_dif = st.session_state.get("rasch_dif") or {}
                        dr = saved_dif.get("result") if saved_dif.get("key") == b8_key else None
                        if dr is not None:
                            shown = {k: dr["ui_settings"][k] for k in dif_settings}
                            if shown != dif_settings:
                                st.info("Gösterilen DIF sonucu farklı grup veya ayarlarla üretildi; yeniden başlatın.")
                            st.markdown(f"**Karşılaştırma:** {dr['focal']} (odak) − {dr['reference']} (referans); "
                                        f"grup değişkeni `{dr['ui_settings']['group_col']}`.")
                            for note in dr["warnings"]:
                                st.warning(note)
                            if not dr["joint"]["converged"]:
                                st.error("Ortak model yakınsamadı; DIF sonuçları yorumlanmamalıdır.")
                            st.dataframe(dif_summary_rows(dr), hide_index=True, use_container_width=True)
                            st.caption("μ_odak, aynı ölçekte grupların ortalama yetenek farkıdır (etki/impact); "
                                       "DIF değildir. Ölçek referans grup ortalaması 0 ile tanımlıdır.")
                            st.plotly_chart(dif_figure(dr), use_container_width=True)
                            st.dataframe(dr["table"].round(4), hide_index=True, use_container_width=True)
                            st.caption("Δ ve SH(Δ) ortak modelin gözlenen bilgi matrisinden; z Wald testi; q "
                                       "Benjamini–Hochberg düzeltilmiş değerdir. Etki sınıfı ETS A/B/C ölçütlerinin "
                                       "logit karşılığıdır (|Δ| 0,426 ve 0,638; D = 2,35·logit) ve yaklaşık bir "
                                       "rehberdir. Her madde, kendisi dışındaki çapa maddeleriyle bağlanmış ayrı bir "
                                       "modelde test edilir; 'Çapa' sütunu son arındırılmış çapa kümesini gösterir. Bu "
                                       "analiz CML tabanlı Andersen olabilirlik oranı testi değildir.")
                            st.markdown("**Arındırma geçmişi**")
                            st.dataframe(dr["history"], hide_index=True, use_container_width=True)
                            download_table_pair(dr["table"].round(5), "psikoanaliz_rasch_dif", "DIF tablosu",
                                                "rasch_dif_dl")

# =============================================================================
# SAYFA 13: MADDE TEPKİ KURAMI (MTK / IRT)
# =============================================================================
elif selected_page == "MTK: Çok Kategorili Formel DIF":
    from ui.polytomous_dif_view import render as render_poly_dif
    src = st.session_state.raw_df if st.session_state.raw_df is not None else st.session_state.scored_df
    if src is None: st.warning("Önce yanıt verisi yükleyin.")
    else: render_poly_dif(src, st.session_state.item_cols, st.session_state.id_col, st.session_state.group_col)

elif selected_page == "MTK: GRM Formel DIF":
    from ui.grm_dif_view import render as render_grm_dif
    src = st.session_state.raw_df if st.session_state.raw_df is not None else st.session_state.scored_df
    if src is None: st.warning("Önce sıralı puan verisi yükleyin.")
    else: render_grm_dif(src, list(st.session_state.item_cols), st.session_state.id_col, st.session_state.group_col)
elif selected_page == "MTK: Çok Kategorili B14":
    render_academic_header()
    from ui.irt_b14_view import render as render_b14
    src = st.session_state.raw_df if st.session_state.raw_df is not None else st.session_state.scored_df
    if src is None:
        st.warning("Önce Veri Yükleme bölümünden kategorili yanıt dosyası yükleyin.")
    else:
        render_b14(src, list(st.session_state.item_cols), st.session_state.id_col, st.session_state.group_col)
elif selected_page == "MTK: Çok Modelli DIF Taraması":
    from ui.irt_dif_multimodel_view import render as render_multi_dif
    dif_source=st.session_state.scored_df if st.session_state.scored_df is not None else st.session_state.raw_df
    if dif_source is None:st.warning("Önce veri yükleyin.")
    else:render_multi_dif(dif_source,list(st.session_state.item_cols),st.session_state.id_col,st.session_state.group_col)
elif selected_page == "KTK: Çift Motor Rapor":
    render_academic_header()
    from ui.ctt_dual_report_view import render as render_ctt_dual
    src=st.session_state.scored_df if st.session_state.scored_df is not None else st.session_state.raw_df
    if src is None:st.warning("Önce puanlanmış veri yükleyin.")
    else:render_ctt_dual(src,list(st.session_state.item_cols),st.session_state.id_col,st.session_state.group_col)
elif selected_page == "MTK: Varsayım Denetimi":
    render_academic_header()
    from ui.irt_assumptions_view import render as render_irt_assumptions
    src = st.session_state.raw_df if st.session_state.raw_df is not None else st.session_state.scored_df
    if src is None:
        st.warning("Önce yanıt verilerini yükleyin.")
    else:
        render_irt_assumptions(src, list(st.session_state.item_cols), st.session_state.id_col, st.session_state.group_col)
elif selected_page == "MTK: İkili Rapor":
    render_academic_header()
    from ui.irt_b13_view import render as render_b13
    src = st.session_state.scored_df if st.session_state.scored_df is not None else st.session_state.raw_df
    if src is None:
        st.warning("Önce veri dosyası yükleyin.")
    else:
        render_b13(src, list(st.session_state.item_cols), st.session_state.id_col, st.session_state.group_col)
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
        # SPSS/Xcalibre-style left settings, right output surface.
        # Selection changes only presentation, never silently recalibrates.
        with st.sidebar.expander("⚙️ MTK rapor ayarları", expanded=True):
            st.caption(f"Etkin model: {model} | Motor: Python mirt")
            report_params = st.checkbox("Madde parametreleri", value=True,
                                        key=f"py_mirt_show_params_{model}")
            report_icc = st.checkbox("ICC — karakteristik eğriler", value=True,
                                     key=f"py_mirt_show_icc_{model}")
            report_iif = st.checkbox("IIF — madde bilgi eğrileri", value=True,
                                     key=f"py_mirt_show_iif_{model}")
            report_tif = st.checkbox("TIF — test bilgi fonksiyonu", value=True,
                                     key=f"py_mirt_show_tif_{model}")
            report_diagnostics = st.checkbox("Kestirim tanıları (AIC/BIC)", value=True,
                                             key=f"py_mirt_show_diagnostics_{model}")
            st.caption("Bu kutucuklar yalnızca sağdaki rapor görünümünü değiştirir; "
                       "model yeniden kestirilmez.")
        with st.sidebar.expander("⚙️ MTK model seçimi", expanded=True):
            selected_model = st.radio(
                "İki kategorili MTK modeli", ["1PL", "2PL", "3PL", "4PL"],
                index=("1PL", "2PL", "3PL", "4PL").index(model),
                key=f"model_radio_{model}", horizontal=False)
            if selected_model != model:
                st.session_state.nav_page = f"MTK: {selected_model}"
                st.rerun()
            st.caption("Etkin kestirim motoru: Python mirt 1.2.0. "
                       "Desteklenmeyen optimizasyon ayarları gösterilmez.")
        st.warning("Etkin iki kategorili MTK motoru: Python mirt. "
                   "Parametre kurtarma ve bağımsız karşılaştırma tamamlanmadığından "
                   "bu çıktılar araştırma amaçlıdır.")
        # Optional external engine: isolated experimental output, never mixed with
        # the native model history, AIC/BIC or inferential diagnostics.
        # Library calibration controls live in the left settings panel; output stays right.
        with st.sidebar.expander("▶ MTK analizini çalıştır", expanded=True):
            st.caption("Python mirt hazır bir MTK kütüphanesidir. EserPsychometrics "
                       "arayüzündeki parametre eşleştirmeleri ve çıkarımsal "
                       "raporlama henüz bağımsız karşılaştırmadan geçmediği için "
                       "bu motorun çıktıları ayrı tutulur.")
            if st.button("▶️ Python mirt ile analiz", key=f"py_mirt_run_{model}"):
                try:
                    from core.irt_python_mirt_adapter import fit_python_mirt
                    external_data = target.loc[:, irt_columns].apply(
                        pd.to_numeric, errors="raise")
                    # Transparent calibration preflight: never hide dropped responses.
                    st.caption(
                        f"Ön kontrol: {external_data.shape[0]} kişi × "
                        f"{external_data.shape[1]} madde; "
                        f"eksik yanıt: {int(external_data.isna().sum().sum())}."
                    )
                    fingerprint_external = (
                        tuple(irt_columns), external_data.shape,
                        int(pd.util.hash_pandas_object(external_data, index=True).sum()))
                    with st.spinner("Deneysel Python mirt kestirimi hesaplanıyor…"):
                        missing_count = int(external_data.isna().sum().sum())
                        if missing_count:
                            raise ValueError(
                                f"{missing_count} eksik yanıt bulundu. Python mirt adaptörü "
                                "eksik yanıtları henüz desteklemiyor; otomatik sıfıra "
                                "dönüştürme veya örtük atama yapılmadı. Eksik veri "
                                "analizi sekmesinde uygun yöntemi değerlendirin."
                            )
                        external_fit = fit_python_mirt(external_data.to_numpy(), model=model)
                    if "python_mirt_experimental_results" not in st.session_state:
                        st.session_state.python_mirt_experimental_results = {}
                    st.session_state.python_mirt_experimental_results[model] = {
                        "fingerprint": fingerprint_external,
                        "fit": external_fit,
                    }
                    st.success("Python mirt analizi tamamlandı. Çıktılar ayrı kaydedildi.")
                except Exception as exc:
                    st.session_state.get("python_mirt_experimental_results", {}).pop(model, None)
                    st.error(f"Python mirt çalıştırılamadı: {exc}")
        previous_external = st.session_state.get("python_mirt_experimental_results", {}).get(model)
        if previous_external is not None:
            current_external_data = target.loc[:, irt_columns]
            current_external_fingerprint = (
                tuple(irt_columns), current_external_data.shape,
                int(pd.util.hash_pandas_object(current_external_data, index=True).sum()))
            if previous_external["fingerprint"] == current_external_fingerprint:
                external_fit = previous_external["fit"]
                st.caption(f"Motor: {external_fit.engine} | Model: {external_fit.model} "
                           f"| N: {external_fit.n_persons} | Madde: {external_fit.n_items}")
                st.info("Aşağıdaki özet, Python mirt kütüphanesinin doğrudan "
                        "ürettiği deneysel çıktıdır. Yerel EserPsychometrics "
                        "raporlarıyla eşdeğerliği henüz doğrulanmadı.")
                # Fit diagnostics are read directly from the external library,
                # never recomputed from the retired native engine.
                from core.irt_library_diagnostics import library_fit_diagnostics
                library_fit = external_fit.fit
                diag = library_fit_diagnostics(library_fit)
                diagnostics = {
                    "Yakınsama": diag["converged"],
                    "Yineleme": diag["iterations"],
                    "Log-olabilirlik": diag["log_likelihood"],
                    "AIC": diag["aic"],
                    "BIC": diag["bic"],
                    "Serbest parametre": diag["n_parameters"],
                    "Gözlem": diag["n_observations"],
                }
                if report_diagnostics:
                    st.markdown("#### Python mirt — kestirim tanıları")
                if report_diagnostics:
                    st.dataframe(
                    pd.DataFrame([diagnostics]),
                    hide_index=True,
                    use_container_width=True,
                )
                if not library_fit.converged:
                    st.error(
                        "Python mirt yakınsamadı. Parametreler ve model "
                        "karşılaştırmaları kesin sonuç olarak yorumlanmamalıdır."
                    )
                if model in ("1PL", "2PL", "3PL", "4PL"):
                    try:
                        from core.irt_python_mirt_parameters import python_mirt_binary_parameters
                        from core.irt_library_parameter_mapping import curves_from_normalized_table
                        import plotly.graph_objects as go
                        item_table = python_mirt_binary_parameters(
                            library_fit, model=model, item_names=irt_columns)
                        if report_params:
                            st.markdown("#### Python mirt — madde parametreleri")
                            st.dataframe(item_table, hide_index=True, use_container_width=True)
                        st.download_button(
                            "Madde parametrelerini CSV indir",
                            data=item_table.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"python_mirt_{model}_parameters.csv",
                            mime="text/csv", key=f"py_mirt_params_csv_{model}")
                        theta_grid = np.linspace(-4, 4, 161)
                        curve_data = curves_from_normalized_table(item_table, theta_grid)
                        if report_icc or report_iif:
                            with st.sidebar.expander("📈 Eğri madde seçimi", expanded=False):
                                selected_item = st.selectbox(
                                    "ICC / IIF için madde", list(irt_columns),
                                    key=f"py_mirt_curve_item_{model}")
                            idx = list(irt_columns).index(selected_item)
                        else:
                            idx = 0
                            selected_item = list(irt_columns)[0]
                        for title, key, ylabel in (
                            *((("ICC — Madde Karakteristik Eğrisi", "probability", "P(θ)"),) if report_icc else ()),
                            *((("IIF — Madde Bilgi Fonksiyonu", "information", "I(θ)"),) if report_iif else ())):
                            fig = go.Figure()
                            fig.add_scatter(x=theta_grid, y=curve_data[key][:, idx],
                                            mode="lines", name=str(selected_item))
                            fig.update_layout(title=title, xaxis_title="θ",
                                              yaxis_title=ylabel)
                            st.plotly_chart(fig, use_container_width=True)
                        if report_tif:
                            tif = go.Figure()
                        if report_tif:
                            tif.add_scatter(x=theta_grid, y=curve_data["test_information"],
                                        mode="lines", name="TIF")
                        if report_tif:
                            tif.update_layout(title="Test Bilgi Fonksiyonu",
                                          xaxis_title="θ", yaxis_title="I(θ)")
                        if report_tif:
                            st.plotly_chart(tif, use_container_width=True)
                        st.caption("ICC/IIF/TIF: Python mirt katsayılarından "
                                   "bağımsız analitik eğriler. Parametre kurtarma "
                                   "ve R mirt eşdeğerliği henüz sertifikalı değildir.")
                    except (ValueError, TypeError, AttributeError) as mapping_error:
                        st.error(f"Python mirt madde parametresi/eğri eşleştirme hatası: {mapping_error}")
                with st.expander("Python mirt — kütüphane kestirim özeti"):
                    try:
                        raw_summary = external_fit.fit.summary()
                        st.text(str(raw_summary)[:15000])
                        import importlib.metadata
                        try:
                            mirt_version = importlib.metadata.version("mirt")
                        except importlib.metadata.PackageNotFoundError:
                            mirt_version = "unknown"
                        st.caption(f"Kaynak: Python mirt v{mirt_version}; model: {model}; "
                                   "kestirim: kütüphane ham özeti. Parametre "
                                   "metriği veya çıkarımsal doğrulama iddiası yok.")
                        st.download_button(
                            "Ham kestirim özetini TXT olarak indir",
                            data=(f"Engine: Python mirt {mirt_version}\\n"
                                  f"Model: {model}\\n"
                                  "Status: library output; no cross-engine equivalence certified\\n\\n"
                                  +str(raw_summary)),
                            file_name=f"mirt_{model}_raw_summary.txt",
                            mime="text/plain",
                            key=f"mirt_raw_download_{model}",
                        )
                    except Exception as summary_error:
                        st.warning(f"Bu motor için okunabilir özet alınamadı: {summary_error}")
                st.caption("Parametre dönüşümü, SH ve AIC/BIC karşılaştırması "
                           "henüz bağımsız eşleştirmeden geçmediği için "
                           "resmî analiz sonuçlarına eklenmez.")
            else:
                st.info("Veri veya madde seçimi değişti; eski deneysel sonuç gösterilmiyor.")
        # Retired native IRT results and diagnostics are never rendered.
        # All current binary IRT reports above originate from Python mirt.
        if "irt_model_results" in st.session_state:
            st.session_state.pop("irt_model_results", None)
        if "irt_b10" in st.session_state:
            st.session_state.pop("irt_b10", None)
        if "irt_b11" in st.session_state:
            st.session_state.pop("irt_b11", None)
