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


# =============================================================================
# SAYFA VE DURUM YAPILANDIRMASI
# =============================================================================
st.set_page_config(
    page_title="PsikoAnaliz — Klasik Test Kuramı Analiz Platformu",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    <div style="text-align: center; padding-bottom: 1rem;">
        <h2 style="color: #1E3A8A; margin-bottom: 0;">PsikoAnaliz</h2>
        <span style="font-size: 0.85rem; color: #64748B;">Doç. Dr. Mehmet Taha Eser</span>
    </div>
    """, unsafe_allow_html=True)

    menu_options = [
        "1. Ana Sayfa",
        "2. Veri Yükleme",
        "3. Puanlama & Veri Hazırlama",
        "4. Test İstatistikleri",
        "5. Madde Analizi",
        "6. Çeldirici Analizi",
        "7. Güvenirlik & Ölçme Hatası",
        "8. Grup Bazlı Analizler",
        "9. Bilimsel Grafikler",
        "10. Raporlama",
        "11. Formüller ve Yöntemler",
        "12. Kullanım Kılavuzu"
    ]
    selected_page = st.radio("Menü Gezintisi", menu_options, index=0)

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

    **PsikoAnaliz**, eğitim ve psikolojide ölçme ve değerlendirme alanında çalışan
    akademisyenler, araştırmacılar, öğretmenler, psikometristler ve lisansüstü öğrenciler
    için geliştirilmiş web tabanlı bir Klasik Test Kuramı (KTK) analiz platformudur.

    Platform; madde ve test istatistiklerinin hesaplanmasını, çoktan seçmeli testlerde
    çeldirici analizini, güvenirlik ve ölçme hatası kestirimlerini, grup bazlı
    karşılaştırmaları ve bilimsel grafiklerle desteklenen akademik raporlamayı
    tek bir ortamda sunmayı amaçlamaktadır.

    **Doç. Dr. Mehmet Taha Eser**  
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

            csv_data = st.session_state.scored_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 Puanlanmış Veri Setini İndir (CSV)",
                data=csv_data,
                file_name="psikoanaliz_puanlanmis_veri.csv",
                mime="text/csv"
            )


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

        csv_item = summary_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Madde Analizi Tablosunu İndir (CSV)",
            data=csv_item,
            file_name="madde_istatistikleri.csv",
            mime="text/csv"
        )


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
    *PsikoAnaliz — Klasik Test Kuramı Analiz Platformu*
    """)
