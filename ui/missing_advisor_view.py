"""Shared missing-data advisor panel, hand-off to the Kayıp Veri İşlemleri page and the 'applied method' banner."""
import pandas as pd
import streamlit as st

import core.missing_advisor as A

MISSING_PAGE_FILE = "pages/90_Kayip_Veri_Analizi_ve_Atama.py"
# Routes without item-matrix analyses or with their own long-format handling (MFRM) get no global panel.
SKIP = ("1. Ana Sayfa", "2. Veri Yükleme", "3. Puanlama", "12. Kullanım", "MTK: RaschPy MFRM", "MTK: Çok Yüzeyli Rasch",
        "MTK: 1PL", "MTK: 2PL", "MTK: 3PL", "MTK: 4PL", "MTK: Karşılaştırma", "MTK: Python mirt ")


def source_key():
    return "scored_df" if st.session_state.get("scored_df") is not None else "raw_df"


def _fp(df):
    return (df.shape, int(pd.util.hash_pandas_object(df, index=True).sum())) if df is not None else None


def applied():
    """The applied record if it still matches the active data; a stale record (new data loaded) is dropped."""
    rec = st.session_state.get("missing_applied")
    if rec and _fp(st.session_state.get(rec["source"])) != rec["fingerprint"]:
        st.session_state.pop("missing_applied", None); st.session_state.pop(f"_missing_original_{rec['source']}", None); rec = None
    return rec


def banner():
    rec = applied()
    if not rec:
        return
    st.info(f"Kayıp veri işlemi uygulandı: **{A.method_label(rec['method'])}** ({rec['columns_n']} madde; {rec['cells']} kayıp hücre; "
            f"satır {rec['n_before']} → {rec['n_after']}). Analizler işlenmiş veriyle çalışır; orijinal veri saklıdır. "
            "Tek atama standart hataları küçümseyebilir.")
    st.download_button("Kayıp veri işlem kaydı (.txt)", provenance_text(rec).encode("utf-8"), "kayip_veri_islemi.txt",
                       "text/plain", key="missing_provenance_download")
    if st.button("↩ Orijinal veriye dön", key="missing_revert"):
        st.session_state[rec["source"]] = st.session_state.pop(f"_missing_original_{rec['source']}")
        st.session_state.pop("missing_applied", None); st.rerun()


def panel(page, frame, columns, key="missing_advisor"):
    """Render the advice for `columns` of `frame` on `page`; returns the advice dict (None when nothing is missing)."""
    cols = [c for c in columns if c in frame.columns]
    if not cols or not frame[cols].isna().to_numpy().any():
        return None
    adv = A.advise(frame, cols, page); p = adv["pattern"]
    box = st.success if adv["status"] == "native" else st.warning
    box(f"**Kayıp veri danışmanı:** {p['missing_cells']} kayıp hücre (%{p['missing_pct']:.1f}); {p['rows_with_missing']} satır "
        f"(%{p['row_loss_pct']:.1f}). {adv['reason']}")
    with st.expander("Kayıp veri önerisinin gerekçesi", expanded=False):
        st.caption(adv["mechanism_note"] + " " + adv["multiple_imputation"])
        if adv["allowed"]:
            st.markdown("**Kullanılabilir seçenekler (varsayımlarıyla):**\n" + "\n".join(f"- {A.method_label(m)}: {r}"
                                                                                for m, r in adv["allowed"].items()))
        if adv["excluded"]:
            st.markdown("**Önerilmeyen yöntemler:**\n" + "\n".join(f"- {A.method_label(m)}: {r}" for m, r in adv["excluded"].items()))
        st.caption("Kaynaklar: " + "; ".join(adv["references"]))
    if adv["status"] == "action" and st.button("🧩 Kayıp Veri İşlemleri'ne git (birincil önerilen yöntem yok; seçenekleri değerlendirin)",
                                               key=f"{key}_go"):
        st.session_state["missing_advice"] = dict(page=page, source=source_key(), columns=cols, recommended=adv["recommended"],
                                                  allowed=list(adv["allowed"]), reason=adv["reason"])
        st.switch_page(MISSING_PAGE_FILE)
    return adv


def global_panel(page):
    """Show the advice; return False while an engine rule that changes scoring/sample (approval=True) awaits approval."""
    if any(str(page).startswith(s) for s in SKIP):
        return True
    frame = st.session_state.get(source_key()); cols = list(st.session_state.get("item_cols") or ())
    if frame is None or not cols:
        return True
    adv = panel(page, frame, cols, key="missing_advisor_global")
    if not adv or not adv.get("approval"):
        return True
    rule = A.engine(page)["note"]; token = (str(page), _fp(frame[[c for c in cols if c in frame.columns]]))
    approved = st.session_state.setdefault("missing_rule_approved", [])
    if token in approved:
        st.caption(f"Onaylanan motor kuralı: {rule}")
        return True
    if st.checkbox(f"Motorun kendi eksik veri kuralını bu analiz için onaylıyorum: {rule}", key=f"missing_rule_ok_{abs(hash(token))}"):
        approved.append(token); return True
    st.info("Analiz, eksik veri kuralını onaylamanızı veya Kayıp Veri İşlemleri'nde bir yöntem uygulamanızı bekliyor.")
    return False


def accept(result, method, n_before, cells):
    """Called on the Kayıp Veri page: keep the original, make the processed frame active, return to the analysis."""
    adv = st.session_state.get("missing_advice"); src = adv["source"]
    st.session_state.setdefault(f"_missing_original_{src}", st.session_state.get(src))
    st.session_state[src] = result
    st.session_state["missing_applied"] = dict(source=src, method=method, page=adv["page"], columns_n=len(adv["columns"]),
                                               cells=cells, n_before=n_before, n_after=len(result), fingerprint=_fp(result))
    st.session_state["nav_page"] = adv["page"]; st.session_state.pop("missing_advice", None)
    st.switch_page("app.py")


def provenance_text(rec):
    return (f"Kayıp veri işlemi: {A.method_label(rec['method'])}; {rec['columns_n']} madde; {rec['cells']} kayıp hücre; "
            f"satır {rec['n_before']} -> {rec['n_after']}; analiz: {rec['page']}. Orijinal veri değiştirilmeden saklandı. "
            "Kayıp mekanizması (MCAR/MAR/MNAR) doğrulanmadı; tek atama standart hataları küçümseyebilir.")


def stamp(data, file_name, rec):
    """Add missing-data provenance to an export: XLSX -> extra sheet, DOCX -> closing section, CSV -> unchanged (standard CSV; provenance is the separate banner download),
    ZIP -> provenance text file. Other types and non-bytes payloads are returned unchanged."""
    import io, zipfile
    name = str(file_name or "").lower(); text = provenance_text(rec)
    raw = data.encode("utf-8") if isinstance(data, str) else data
    if not isinstance(raw, (bytes, bytearray)):
        return data
    try:
        if name.endswith(".xlsx"):
            from openpyxl import load_workbook
            wb = load_workbook(io.BytesIO(raw)); ws = wb.create_sheet("Kayıp veri işlemi"); ws["A1"] = text
            out = io.BytesIO(); wb.save(out); return out.getvalue()
        if name.endswith(".docx"):
            from docx import Document
            doc = Document(io.BytesIO(raw)); doc.add_heading("Kayıp Veri İşlemi", 1); doc.add_paragraph(text)
            out = io.BytesIO(); doc.save(out); return out.getvalue()
        if name.endswith(".csv"):
            return data
        if name.endswith(".zip"):
            src = zipfile.ZipFile(io.BytesIO(raw)); out = io.BytesIO()
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
                for n in src.namelist(): z.writestr(n, src.read(n))
                z.writestr("kayip_veri_islemi.txt", text)
            return out.getvalue()
    except Exception:
        return data                                                     # never break an export; the banner still shows it
    return data
