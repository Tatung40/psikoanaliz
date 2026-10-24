"""B13 integrated native dichotomous IRT reporting page.
The native estimators remain authoritative; external-engine scientific mappings
are not silently inferred. CSV/XLSX/ZIP exports exclude IDs by default.
"""
import io
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.irt_b13_curves import compute_curves, check_curves
from core.irt_b13_diagnostics import calculate_diagnostics
from reports.irt_b13_png import png_bytes
from core.irt_b13_compare import compare_engines
from core.irt import fit_irt
from core.rasch import rasch_analysis

MODELS=("Rasch","1PL","2PL","3PL","4PL")
def prepare(data, model):
    x=data.apply(pd.to_numeric,errors="coerce")
    if len(x)<30 or x.shape[1]<3: raise ValueError("En az 30 kişi ve 3 madde gereklidir.")
    if x.isna().any().any() or not x.isin([0,1]).all().all():
        raise ValueError("Eksiksiz, 0/1 kodlu madde yanıt matrisi gereklidir.")
    if (x.nunique()!=2).any():raise ValueError("Sabit yanıtlı maddeleri inceleyin veya dışlayın.")
    return x.astype(int)

def native_fit(data,model):
    x=prepare(data,model)
    if model=="Rasch":
        r=rasch_analysis(x)
        f=r["fit"]
        it=pd.DataFrame({"Madde":x.columns,"a":1.0,"b":f["b"],"c":0.0,"d":1.0,
                         "SH(b)":f["items"]["SH(b)"].to_numpy()})
        p=r["persons"]
        th=np.asarray(p["WLE θ"],float)
        se=np.asarray(p["SH(θ)"],float)
        fit=f
        diag=r["items"]
        scale=f"Rasch logit, sigma={f['sd']:.5g}"
    else:
        f=fit_irt(x.to_numpy(),model=model)
        it=f["items"].copy()
        it["Madde"]=list(map(str,x.columns))
        th=np.asarray(f["theta"],float)
        se=np.asarray(f["theta_sd"],float)
        fit=f
        diag=None
        scale="theta ~ N(0,1); D=1"
    if len(th)!=len(x):raise ValueError("Kişi yetenekleri ve yanıt satırları eşleşmiyor.")
    persons=pd.DataFrame({"Sıra":np.arange(1,len(x)+1),"Theta":th,"KSH":se,
                          "Ham puan":x.sum(axis=1).to_numpy()})
    overview=pd.DataFrame([{"Model":model,"Motor":"EserPsychometrics",
      "N":len(x),"Madde":x.shape[1],"logLik":fit["loglik"],
      "AIC":fit["aic"],"BIC":fit["bic"],"Yineleme":fit["iterations"],
      "Yakınsama":bool(fit["converged"]),"Ölçek":scale}])
    d=calculate_diagnostics(x,fit,it,model)
    if d["se"] is not None:
        it=it.merge(d["se"],on="Madde",how="left",validate="one_to_one")
    return {"items":it,"persons":persons,"overview":overview,"diagnostics":diag,
            "item_fit":d["item_fit"],"q3":d["q3"],"diagnostic_notes":d["messages"],
            "model":model,"scale":scale,"fit":fit,"checks":check_curves(it,model)}

def safe_export(frame):
    """Protect spreadsheet cell evaluation in Excel-compatible exports."""
    result=frame.copy()
    for col in result.columns:
        if pd.api.types.is_object_dtype(result[col]) or pd.api.types.is_string_dtype(result[col]):
            result[col]=result[col].map(lambda v: "'" + v if isinstance(v,str) and
                                           v.lstrip().startswith(("=","+","-","@")) else v)
    return result

def excel_bytes(tables):
    out=io.BytesIO()
    with pd.ExcelWriter(out,engine="openpyxl") as writer:
        for name,df in tables.items():
            safe_export(df).to_excel(writer,sheet_name=name[:31],index=False)
    return out.getvalue()

def render(target,columns,id_col=None,group_col=None):
    st.subheader("İkili MTK — Bütünleşik Analiz ve Rapor")
    st.caption("Rasch, 1PL, 2PL, 3PL ve 4PL | EserPsychometrics yerel motor")
    st.info("Bu sayfa, etkin 1PL–4PL sayfalarındaki Python mirt motorundan AYRI olarak emekliye ayrılmış yerel "
            "1PL–4PL kestiricisini (core.irt) araştırma ve motor karşılaştırması için kullanır. Sonuçları etkin "
            "Python mirt sonuçlarıyla birleştirilmez ve onların yerine geçmez; 3PL/4PL kullanım sınırları için "
            "validation/IRT_3PL4PL_ROOT_CAUSE.md belgesine bakın. Rasch yerel motoru etkindir.")
    tabs=st.tabs(["Model ve Ayarlar","Genel Sonuçlar","Madde Parametreleri",
                  "Kişi Yetenekleri","Bilimsel Grafikler","Uyum ve Tanılar",
                  "Motor Karşılaştırması","Raporlar ve İndirme"])
    data=target[[c for c in columns if c not in (id_col,group_col)]]
    with tabs[0]:
        model=st.radio("İkili MTK modeli",MODELS,horizontal=True,key="b13_model")
        if st.button("Analizi başlat",key="b13_run",type="primary"):
            try:
                with st.spinner("Parametre kestirimi ve tanılar hesaplanıyor"):
                    result=native_fit(data,model)
                st.session_state["b13_latest"]={"result":result,
                    "fingerprint":int(pd.util.hash_pandas_object(data,index=True).sum())}
                st.success("Analiz tamamlandı.")
            except Exception as exc:
                st.session_state.pop("b13_latest",None)
                st.error(str(exc))
    saved=st.session_state.get("b13_latest")
    fingerprint=int(pd.util.hash_pandas_object(data,index=True).sum())
    result=saved["result"] if saved and saved["fingerprint"]==fingerprint and saved["result"]["model"]==model else None
    if result is None:
        with tabs[1]:st.info("Önce Model ve Ayarlar sekmesinden analizi başlatın.")
        return
    overview=result["overview"];items=result["items"];persons=result["persons"]
    with tabs[1]:
        row=overview.iloc[0]
        cols=st.columns(4)
        for col,key in zip(cols,("N","Madde","AIC","BIC")):col.metric(key,round(float(row[key]),2))
        st.dataframe(overview,hide_index=True,use_container_width=True)
        if not bool(row["Yakınsama"]):st.warning("Yakınsama sağlanmadı; çıkarımsal yorumlardan kaçının.")
    with tabs[2]:
        st.dataframe(items,hide_index=True,use_container_width=True)
        st.download_button("Madde parametreleri CSV",safe_export(items).to_csv(index=False).encode("utf-8-sig"),
                           "b13_items.csv",mime="text/csv")
    with tabs[3]:
        ids=st.checkbox("Kişi kimliklerini rapora ekle",value=False,key="b13_ids",disabled=id_col is None)
        shown=persons.copy()
        if ids and id_col in target:
            shown.insert(0,"Kişi kimliği",target[id_col].astype(str).to_numpy())
        st.dataframe(shown.head(500),hide_index=True,use_container_width=True)
        st.download_button("Kişi yetenekleri CSV",safe_export(shown).to_csv(index=False).encode("utf-8-sig"),
                           "b13_persons.csv",mime="text/csv")
    with tabs[4]:
        kind=st.selectbox("Grafik",("ICC","IIC","TIF","SEM","Theta dağılımı","Wright haritası","Parametre dağılımı","Asimptotlar"),key="b13_figure")
        cv=compute_curves(items)
        fig=go.Figure()
        if kind in ("ICC","IIC"):
            names=items["Madde"].astype(str).tolist()
            selected=st.multiselect("Maddeler",names,default=names[:8],key="b13_fig_items")
            for item in selected:
                j=names.index(item)
                fig.add_scatter(x=cv["theta"],y=cv[kind.lower()][:,j],name=item)
        elif kind=="TIF":fig.add_scatter(x=cv["theta"],y=cv["tif"],name="Test bilgisi")
        elif kind=="SEM":fig.add_scatter(x=cv["theta"],y=cv["sem"],name="Koşullu SEM")
        elif kind=="Theta dağılımı":
            fig.add_histogram(x=persons["Theta"],name="Theta")
        elif kind=="Wright haritası":
            fig.add_histogram(y=persons["Theta"],orientation="h",name="Kişiler")
            fig.add_scatter(x=np.zeros(len(items)),y=items["b"],mode="markers+text",
                            text=items["Madde"],name="Maddeler (b)")
        elif kind=="Parametre dağılımı":
            for p in ("a","b","c","d"):
                fig.add_box(y=items[p],name=p)
        elif model in ("3PL","4PL"):
            fig.add_bar(x=items["Madde"],y=items["c"],name="c")
            if model=="4PL":fig.add_scatter(x=items["Madde"],y=items["d"],name="d")
        else:
            st.info("Asimptot grafiği yalnızca 3PL/4PL için anlamlıdır.")
        fig.update_layout(title=f"{model} — {kind}",xaxis_title="Theta",
                          template="plotly_white")
        st.plotly_chart(fig,use_container_width=True)
        st.write("Matematiksel kontroller:",result["checks"])
        if kind!="Asimptotlar" or model in ("3PL","4PL"):
            st.download_button("PNG indir (300 dpi)",
                png_bytes(result,kind,selected if kind in ("ICC","IIC") else None),
                file_name=f"B13_{model}_{kind}.png",mime="image/png",key="b13_png_dl")
    with tabs[5]:
        st.write("Yakınsama",bool(overview.iloc[0]["Yakınsama"]))
        if result["diagnostics"] is not None:
            st.dataframe(result["diagnostics"],hide_index=True,use_container_width=True)
        if result["item_fit"] is not None:
            st.markdown("**S-X² madde uyum istatistikleri**")
            st.dataframe(result["item_fit"],hide_index=True,use_container_width=True)
        if result["q3"] is not None:
            st.markdown("**Yerel bağımsızlık: Q3 madde çiftleri**")
            st.dataframe(result["q3"].head(30),hide_index=True,use_container_width=True)
        for note in result["diagnostic_notes"]:
            st.info(note)
        if result["diagnostics"] is None and result["item_fit"] is None:
            st.info("Model uyum tanılarını mevcut model sayfasında da inceleyin.")
    with tabs[6]:
        st.caption("Motor karşılaştırması yalnızca Python mirt IRF parametre eşleşmesi "
                   "doğrulanırsa sunulur.")
        if model=="Rasch":
            st.warning("Rasch örtük SD serbesttir; Python mirt 1PL ile doğrudan kıyaslanmaz.")
        elif st.button("Python mirt ile karşılaştırmayı başlat",key="b13_compare"):
            try:
                if not result["fit"]["converged"]:
                    raise ValueError("Yerel model yakınsamadı; karşılaştırma yapılamaz.")
                x=prepare(data,model)
                with st.spinner("İki motorun parametreleri karşılaştırılıyor"):
                    comp=compare_engines(result,x.to_numpy())
                st.session_state["b13_comparison"]={"model":model,"fp":fingerprint,"data":comp}
            except Exception as exc:
                st.session_state.pop("b13_comparison",None)
                st.error(f"Karşılaştırma engellendi: {exc}")
        comp_saved=st.session_state.get("b13_comparison")
        if comp_saved and comp_saved["model"]==model and comp_saved["fp"]==fingerprint:
            comp=comp_saved["data"]
            st.success(f"IRF doğrulandı (maksimum sapma {comp['irf_deviation']:.2e}).")
            st.dataframe(comp["summary"],hide_index=True,use_container_width=True)
            st.dataframe(comp["items"],hide_index=True,use_container_width=True)
            st.caption(f"logLik: Eser {comp['native_loglik']:.4f}; Python mirt "
                       f"{comp['library_loglik']:.4f}. Kuadratür ayarları farklı olabilir.")
    with tabs[7]:
        tablemap={"Model_Ozeti":overview,"Madde_Parametreleri":items,
                  "Kisi_Yetenekleri":shown,"Kontroller":pd.DataFrame([
                     {"Denetim":k,"Durum":str(v)} for k,v in result["checks"]["checks"].items()])}
        if result["item_fit"] is not None:
            tablemap["Madde_Uyumu"]=result["item_fit"]
        if result["q3"] is not None:
            tablemap["Yerel_Bagimsizlik_Q3"]=result["q3"]
        rawxlsx=excel_bytes(tablemap)
        txt=f"EserPsychometrics B13\nModel: {model}\nMotor: native\nOlcek: {result['scale']}\n"
        st.download_button("Excel — tüm tablolar",rawxlsx,"B13_rapor.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Yöntem TXT",txt,"B13_yontem.txt")
        mem=io.BytesIO()
        with zipfile.ZipFile(mem,"w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("B13_rapor.xlsx",rawxlsx)
            z.writestr("B13_items.csv",safe_export(items).to_csv(index=False))
            z.writestr("B13_persons.csv",safe_export(shown).to_csv(index=False))
            z.writestr("B13_yontem.txt",txt)
            for img_kind in ("ICC","IIC","TIF","SEM","Theta dağılımı",
                             "Wright haritası","Parametre dağılımı"):
                z.writestr(f"grafikler/{img_kind}.png",png_bytes(result,img_kind))
            if model in ("3PL","4PL"):
                z.writestr("grafikler/Asimptotlar.png",png_bytes(result,"Asimptotlar"))
        st.download_button("ZIP — rapor paketi",mem.getvalue(),"B13_rapor.zip",mime="application/zip")
