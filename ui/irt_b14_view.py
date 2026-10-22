"""B14 polytomous pilot: ordinal GRM/PCM/GPCM/RSM or nominal NRM.

No scoring key is inferred from raw nominal responses. Results from
Python mirt are reported as experimental, never mixed with native B13.
"""
import io
import pandas as pd
import streamlit as st
from core.irt_b14 import MODELS,ORDINAL,prepare_polytomous,fit_b14
from reports.irt_b14_narrative import guided_report
from reports.irt_b14_audit import interpretation_audit
from core.irt_b14_profiles import empirical_profiles
from core.irt_b14_model_curves import fitted_category_curves
from core.irt_b14_diagnostics import model_diagnostics
from reports.irt_b14_report import report_tables,export_xlsx,export_zip,category_frequencies,quality_report,safe_frame

def render(data,existing_item_cols=(),id_col=None,group_col=None):
    st.subheader("B14 — Çok Kategorili MTK (Deneysel Pilot)")
    st.caption("Sıralı kategoriler: GRM, PCM, GPCM, RSM | Sırasız A–E seçenekleri: NRM. "
               "Python mirt ayrı bir kestirim motorudur; henüz bağımsız sayısal doğrulaması tamamlanmadı.")
    model=st.radio("Model",MODELS,horizontal=True,key="b14_model")
    available=[c for c in data.columns if c not in (id_col,group_col)]
    default=[c for c in existing_item_cols if c in available]
    cols=st.multiselect("Madde sütunları",available,default=default or available[:min(10,len(available))],key="b14_items")
    if model=="NRM":
        labels=st.text_input("Nominal seçenekler (virgülle)",value="A,B,C,D,E",key="b14_categories")
        categories=tuple(v.strip().upper() for v in labels.split(","))
        st.info("NRM doğru cevabı zorunlu kılmaz: A–E seçeneklerinin her biri ayrı kategoridir. "
                "Doğru anahtarına göre 0/1 yapmak istiyorsanız B13 ikili MTK kullanın.")
    else:
        categories=None
        st.info("GRM/PCM/GPCM/RSM için 0,1,2,3… gibi sıralı kısmi puanlar gerekli. "
                "A–E seçenek etiketleri bu modellerde başarı düzeyi olarak varsayılmaz.")
    result=None
    if not cols or len(cols)<3:
        st.warning("En az üç madde seçin.")
    else:
        subset=data.loc[:,cols]
        try:
            numeric,category_labels=prepare_polytomous(subset,model,categories=categories)
            st.success(f"Veri uygunluğu: {len(numeric)} kişi, {numeric.shape[1]} madde, "
                       f"{len(category_labels)} kategori.")
            counts=numeric.stack().value_counts().sort_index().rename_axis("Kategori kodu").reset_index(name="Frekans")
            counts["Etiket"]=counts["Kategori kodu"].map(dict(enumerate(category_labels)))
            st.dataframe(counts,hide_index=True,use_container_width=True)
            st.markdown("**Madde × kategori frekansları**")
            st.dataframe(category_frequencies(numeric,category_labels),hide_index=True,use_container_width=True)
            st.markdown("**Gözlenen kategori profilleri (betimsel)**")
            st.caption("Bunlar MTK modeliyle kestirilmiş kategori tepki eğrileri değildir. "
                       "Sıralı modellerde diğer maddelerin toplamına göre oluşan grupları; "
                       "NRM'de ise seçeneklerin ham tercih oranlarını gösterir.")
            profile=empirical_profiles(numeric,category_labels,model)
            if not profile.empty:
                selected_item=st.selectbox("Grafiği çizilecek madde",list(numeric.columns),key="b14_empirical_item")
                part=profile.loc[profile["Madde"]==str(selected_item)]
                if model=="NRM":
                    st.bar_chart(part.set_index("Kategori")["Oran"],y_label="Seçim oranı")
                else:
                    pivot=part.pivot(index="Grup",columns="Kategori",values="Oran")
                    st.line_chart(pivot,y_label="Gözlenen kategori oranı",x_label="Diğer maddelerin ortalama toplam puanı")
                st.download_button("Betimsel kategori profillerini CSV indir",
                    profile.to_csv(index=False).encode("utf-8-sig"),
                    "B14_betimsel_kategori_profilleri.csv",mime="text/csv",key="b14_empirical_csv")
            quality=quality_report(numeric,category_labels,model)
            st.markdown("**Kategori seyrekliği ve kalite denetimleri**")
            st.dataframe(quality,hide_index=True,use_container_width=True)
            if (quality["Uyarı"]!="—").any():
                st.warning("Bazı maddelerde kategoriler seyrek (<5). Parametre tahminleri kararsız olabilir.")
        except ValueError as e:
            numeric=None
            st.error(f"Veri denetimi: {e}")
        if st.button("Python mirt ile B14 pilot analizi",key="b14_run",type="primary",disabled=numeric is None):
            try:
                with st.spinner("Çok kategorili model kalibre ediliyor"):
                    result=fit_b14(subset,model,categories=categories)
                st.session_state["b14_current"]={"model":model,"columns":tuple(cols),
                    "datahash":int(pd.util.hash_pandas_object(subset,index=True).sum()),
                    "result":result}
            except Exception as e:
                st.error(f"Kestirim başarısız: {e}")
        saved=st.session_state.get("b14_current")
        if saved and saved["model"]==model and saved["columns"]==tuple(cols) and saved["datahash"]==int(pd.util.hash_pandas_object(subset,index=True).sum()):
            result=saved["result"]
    if result is not None:
        st.warning(result["status"])
        fit=result["result"]
        st.write("Model:",model,"| Motor:",result["engine"],"| N:",result["N"],"| Madde:",result["k"])
        try:
            summary=fit.coef_with_se()
            st.dataframe(summary,hide_index=False,use_container_width=True)
            st.download_button("Madde parametrelerini CSV indir",safe_frame(summary.reset_index()).to_csv(index=False).encode("utf-8-sig"),
                               "B14_pilot_parametreler.csv",mime="text/csv",key="b14_csv")
        except Exception as exc:
            st.info(f"Standartlaştırılmış parametre tablosu alınamadı: {exc}")
        st.markdown("### Model ve Madde Uyum Tanıları (Deneysel)")
        if st.button("Model sonrası tanıları çalıştır",key="b14_diagnostics_run"):
            status,tables=model_diagnostics(result)
            st.session_state["b14_diagnostics"]={"model":model,"status":status,"tables":tables}
        diagnostic=st.session_state.get("b14_diagnostics")
        if diagnostic and diagnostic["model"]==model:
            st.dataframe(diagnostic["status"],hide_index=True,use_container_width=True)
            for diagnostic_name,diagnostic_table in diagnostic["tables"].items():
                st.markdown("**"+diagnostic_name+"**")
                st.dataframe(diagnostic_table,hide_index=True,use_container_width=True)
                st.download_button(diagnostic_name+" CSV indir",
                    safe_frame(diagnostic_table).to_csv(index=False).encode("utf-8-sig"),
                    "B14_"+diagnostic_name.replace(" ","_")+".csv",
                    mime="text/csv",key="b14_diag_"+diagnostic_name)
            st.caption("Motor çıktılarının varlığı tek başına bilimsel geçerliği kanıtlamaz; "
                       "model, veri ve bağımsız referans incelemesi gerekir.")
        audit=interpretation_audit(result)
        st.markdown("### Model Tabanlı Kategori Tepki Eğrileri (CRC)")
        try:
            curve_table=fitted_category_curves(result)
            curve_items=list(result["items"])
            curve_selected=st.selectbox("CRC maddesi",curve_items,key="b14_crc_item")
            shown_curve=curve_table.loc[curve_table["Madde"]==str(curve_selected)]
            st.line_chart(shown_curve.pivot(index="Theta",columns="Kategori",values="Olasılık"),
                          y_label="Kestirilen yanıt olasılığı",x_label="Theta")
            st.download_button("Model CRC verisi CSV",curve_table.to_csv(index=False).encode("utf-8-sig"),
                               "B14_model_CRC.csv",mime="text/csv",key="b14_crc_csv")
            st.caption("Bu eğriler doğrudan kestirilen modelin olasılık fonksiyonundan hesaplanır; "
                       "modelin bağımsız bilimsel geçerliği ayrıca doğrulanmalıdır.")
        except Exception as exc:
            st.info(f"Model CRC grafiği gösterilemiyor: {exc}. Betimsel profiller kullanılabilir.")
        st.markdown("### Bilimsel Yorumlama ve Kanıt Denetimi")
        st.write("**Karar:** "+audit["decision"])
        st.info(audit["explanation"])
        st.dataframe(audit["table"],hide_index=True,use_container_width=True)
        st.download_button("Kanıt denetimini CSV indir",
            audit["table"].to_csv(index=False).encode("utf-8-sig"),
            "B14_bilimsel_denetim.csv",mime="text/csv",key="b14_audit_csv")
        st.markdown("### Akademik Türkçe Açıklamalı Rapor")
        narrative=guided_report(result)
        st.markdown(narrative)
        st.download_button("Açıklamalı raporu Markdown indir",narrative.encode("utf-8"),
                           "B14_Aciklamali_Rapor.md",mime="text/markdown",key="b14_narrative_dl")
        tables=report_tables(result)
        st.markdown("**Model ve kestirim bilgileri**")
        st.dataframe(tables["Meta"],hide_index=True,use_container_width=True)
        left,right=st.columns(2)
        left.download_button("B14 tüm tablolar Excel",export_xlsx(tables),
            "B14_rapor.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key="b14_xlsx")
        if right.button("B14 ZIP paketini hazırla",key="b14_zip_make"):
            st.session_state["b14_zip"]=export_zip(result)
        if "b14_zip" in st.session_state:
            right.download_button("B14 ZIP indir",st.session_state["b14_zip"],
                "B14_rapor.zip",mime="application/zip",key="b14_zip_dl")
        st.caption("Parametre SH, model uyumu ve karşılaştırma yorumları bağımsız doğrulama tamamlanmadan "
                   "kesin bilimsel kanıt olarak sunulmaz.")
