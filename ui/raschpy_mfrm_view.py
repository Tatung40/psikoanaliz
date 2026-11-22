"""MTK → RaschPy MFRM (Deneysel): independent RaschPy 1.4.0 page. Does not call or replace the R/TAM page."""
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import core.raschpy_mfrm as RP
from core.mfrm import MFRMError

EX = Path(__file__).resolve().parents[1] / "examples/mfrm"
FIX = json.loads((EX / "fixtures.json").read_text())["fixtures"]
EXAMPLES = {"balanced_3facet": "Dengeli: 300 birey × 4 ölçüt × 3 puanlayıcı", "unbalanced_linked": "Dengesiz bağlantılı: 5 puanlayıcıdan 2'si",
            "interaction_bias": "Etkileşim: P2 puanlayıcısı K3 ölçütünde katı", "sparse_category": "Seyrek üst kategori",
            "disconnected": "Bağlantısız tasarım (hata örneği)", "invalid_duplicate_missing": "Tekrar ve eksik puan (hata örneği)"}
KEEP = 3                                                       # fitted results kept per session (memory bound)
BADGE = {"PARTIAL": "🟡 PARTIAL", "NOT CERTIFIED": "🔴 NOT CERTIFIED", "CERTIFIED": "🟢 CERTIFIED", "UNSUPPORTED": "⚪ UNSUPPORTED"}


def _cache():
    return st.session_state.setdefault("rpm_results", {})


def _store(fp, value):
    c = _cache(); c[fp] = value
    while len(c) > KEEP:
        c.pop(next(iter(c)))


def render():
    st.header("RaschPy Çok Yüzeyli Rasch (MFRM) — Deneysel")
    st.caption(f"Motor: RaschPy {RP.RASCHPY_VERSION} (Apache-2.0), düzenlileştirilmiş ikili bileşik olabilirlik (RCL). "
               "R/TAM sayfasından bağımsızdır; TAM (MML) ile eşdeğer kestirim varsayılmaz. Durumlar modele ve çıktıya göre kanıta dayanır; kapsama kanıtı olmayan standart hatalar gösterilmez.")
    with st.expander("1. Veri ve örnekler", expanded="rpm_source" not in st.session_state):
        st.markdown("Uzun biçim: her satır bir birey–madde–puanlayıcı puanı. RaschPy yalnız **bir** puanlayıcı yüzeyi destekler.")
        ex = st.selectbox("Örnek veri", list(EXAMPLES), format_func=EXAMPLES.get, key="rpm_example")
        if st.button("Örnek veriyi yükle", key="rpm_load"):
            st.session_state["rpm_source"] = dict(kind="example", id=ex, sha=FIX[ex]["sha256"])
    up = st.file_uploader("Uzun biçimli CSV/XLSX", type=["csv", "xlsx"], key="rpm_upload")
    if up is not None:
        raw = up.getvalue(); sha = hashlib.sha256(raw).hexdigest()
        try:
            df = pd.read_excel(io.BytesIO(raw)) if up.name.lower().endswith("xlsx") else pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
        except Exception as e:
            st.error(f"Dosya okunamadı: {e}"); return
        st.session_state["rpm_source"] = dict(kind="upload", name=up.name, sha=sha)
    elif st.session_state.get("rpm_source", {}).get("kind") == "example":
        df = pd.read_csv(EX / f"{st.session_state['rpm_source']['id']}.csv")
    else:
        st.info("Bir dosya yükleyin veya örnek veri seçin."); return
    with st.expander("2. Yüzeyler, model ve standart hata", expanded=True):
        cols = list(df.columns); pick = lambda lab, g, k: st.selectbox(lab, cols, index=cols.index(g) if g in cols else 0, key=k)
        person = pick("Birey", "person_id", "rpm_person"); item = pick("Madde / ölçüt", "item_id", "rpm_item")
        rater = pick("Puanlayıcı", "rater_id", "rpm_rater"); score = pick("Puan", "score", "rpm_score")
        model = st.selectbox("Puanlayıcı modeli", list(RP.MODELS), format_func=lambda m: f"{m} — {RP.MODELS[m][0]}", key="rpm_model")
        st.caption(f"{RP.EQUATION}; puanlayıcı terimi: {RP.MODELS[model][1]}")
        se = st.radio("Standart hata", ["sandwich", "bootstrap"], horizontal=True, key="rpm_se",
                      help="Bootstrap kategori genişliği istatistiklerini de üretir; daha yavaştır.")
        n_boot = st.number_input("Bootstrap örneklemi", 50, 1000, 200, 50, key="rpm_nboot", disabled=se != "bootstrap")
        show_ids = st.checkbox("Gerçek birey kimliklerini göster (varsayılan: takma ad)", key="rpm_ids")
    data_error = None
    with st.expander("3. Veri denetimi", expanded=True):
        try:
            _, _, rep = RP.to_wide(df, person, item, rater, score)
            c = st.columns(4)
            c[0].metric("Birey", rep["n_persons"]); c[1].metric("Puan", rep["n_rows"])
            c[2].metric("Dışlanan eksik puan", rep["missing_scores_excluded"]); c[3].metric("Kategori", len(rep["categories"]))
            st.caption("Bağlantılılık, tekrar eden puanlar ve kategori kullanımı denetlendi; tasarım analize uygun.")
            if rep["sparse_categories"]:
                st.warning(f"Seyrek kategori(ler): {rep['sparse_categories']} (< 10 gözlem).")
        except MFRMError as e:
            data_error = str(e); st.error(data_error)
    config = dict(person=person, item=item, rater=rater, score=score, model=model, se=se, n_boot=int(n_boot),
                  source=st.session_state.get("rpm_source"))
    fp = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    if st.button("RaschPy MFRM kestirimini çalıştır", type="primary", key="rpm_run", disabled=bool(data_error),
                 help=data_error or "Seçilen modeli kestirir"):
        with st.spinner("RaschPy kestirimi..."):
            try:
                _store(fp, {"result": RP.fit(df, person, item, rater, score, model, se, int(n_boot))})
            except MFRMError as e:
                _store(fp, {"error": str(e)})
    saved = _cache().get(fp)
    if not saved:
        return
    if "error" in saved:
        st.error(saved["error"]); return
    _results(saved, df, config, person, item, rater, score, model, show_ids)


def _show(title, table, note, status=None):
    st.markdown(f"**{title}**")
    st.caption(note + (f" Durum: {BADGE.get(status, status)}." if status else ""))
    st.dataframe(table.round(3) if hasattr(table, "round") else table, hide_index=True, use_container_width=True)


def _results(saved, df, config, person, item, rater, score, model, show_ids):
    res = saved["result"]; S = RP.summary(res)
    st.success(f"Kestirim tamamlandı ({res['seconds']:.2f} sn).")
    c = st.columns(5)
    c[0].metric("Birey (N)", S["n_persons"]); c[1].metric("Madde", S["n_items"]); c[2].metric("Puanlayıcı", S["n_raters"])
    c[3].metric("Model", model); c[4].metric("Parametre durumu", S["status"])
    if S["status"] == "NOT CERTIFIED":
        st.warning(f"Bu modelin ({model}) parametreleri NOT CERTIFIED: önceden kaydedilmiş geri kazanım ölçütü örnek veri "
                   "setlerinde karşılanmadı. Sonuçları doğrulanmış bulgu olarak yorumlamayın.")
    if not RP.identity_ok(res):
        st.error("Bağımsız özdeşlik denetimlerinden biri başarısız oldu; ayrıntılar Teknik Ayrıntılar bölümündedir.")
    tables = {k: (t, d, n) for k, t, d, n in RP.academic_tables(res, show_ids)}
    status_of = lambda k: RP.status(model, RP.TABLE_STATUS[k]) if k in RP.TABLE_STATUS else None
    tab = st.tabs(["Genel Bakış", "Parametreler", "Bireyler", "Grafikler", "Tanılar", "İndir"])
    with tab[0]:
        st.markdown(f"**{RP.MODELS[model][0]}.** Bireyler, maddeler ve puanlayıcılar aynı logit ölçeğinde kestirildi "
                    f"({S['n_obs']} puan; uç puanlı birey: {S['extreme']}; dışlanan eksik puan: {S['missing']}).")
        ev = RP.evidence_rows(model); ev["Durum"] = [BADGE.get(RP.status(model, k), "") for k in RP.OUTPUT_LABELS]
        st.dataframe(ev, hide_index=True, use_container_width=True)
        st.caption(RP.EVIDENCE_SCOPE + " " + RP.GENERATED_NOTE)
    with tab[1]:
        for k in ("items", "thresholds", "raters", "rater_params", "cell_bootstrap"):
            if k in tables: _show(*tables[k], status_of(k))
        if RP.cell_bootstrap_validated(model) and res.get("cell_bootstrap") is None:
            if st.button("Puanlayıcı hücreleri için bootstrap SE ve %95 GA hesapla (B = 50)", key="rpm_cellboot"):
                with st.spinner("Bootstrap yeniden kestirimleri..."):
                    res["cell_bootstrap"] = RP.cell_bootstrap(res, B=50); saved.pop("docx", None); st.rerun()
        if not RP.rater_se_reliable(model):
            st.caption("Standart hatalar: kapsama kanıtı olmayan madde, eşik ve puanlayıcı SE'leri gösterilmez.")
    with tab[2]:
        for k in ("persons_summary", "persons"): _show(*tables[k], status_of(k))
    with tab[3]:
        if "figs" not in saved:
            with st.spinner("Grafikler hazırlanıyor..."):
                saved["figs"] = RP.plots(res, fast=True)
        for name, png in saved["figs"].items():
            st.image(png, caption=RP.fig_title(name))
        if not saved.get("figs_full") and st.button("Ek grafikler: madde ve test eğrileri, artıklar", key="rpm_more_plots"):
            with st.spinner("Ek grafikler..."):
                saved["figs"] = RP.plots(res); saved["figs_full"] = True; saved.pop("docx", None); st.rerun()
    with tab[4]:
        for k in ("category_counts", "test"): _show(*tables[k], status_of(k))
        st.markdown("**Grup testleri: DIF ve DRF (isteğe bağlı)**")
        st.caption(f"DIF: {BADGE.get(RP.status(model, 'dif'))}; DRF: {BADGE.get(RP.status(model, 'drf'))}. "
                   "Model seçimi (AIC/BIC) sunulmaz: " + RP.UNSUPPORTED["model_selection_aic_bic"])
        cov = st.selectbox("Birey düzeyi grup değişkeni", ["—"] + [c for c in df.columns if c not in (person, item, rater, score)], key="rpm_cov")
        kind = st.radio("Test", ["drf", "dif"], horizontal=True, key="rpm_kind",
                        format_func={"drf": "DRF (puanlayıcı × grup)", "dif": "DIF (madde × grup)"}.get)
        if st.button("Grup testini çalıştır", key="rpm_group", disabled=cov == "—"):
            try: saved["group"] = {kind: RP.group_test(res, df, person, cov, kind)}; saved.pop("docx", None)
            except MFRMError as e: st.error(str(e))
        for kk, tabs in (saved.get("group") or {}).items():
            for name, t in tabs.items():
                st.markdown(f"**{kk.upper()} — {name}**")
                if not RP.group_part_validated(model, kk, name):
                    st.caption("Bu tablo için bağımsız doğrulama yoktur (NOT CERTIFIED); RaschPy çıktısı olarak betimsel yorumlayın.")
                st.dataframe(t, hide_index=True)
    extra = {}
    for kk, tabs in (saved.get("group") or {}).items():
        extra.update({f"{kk.upper()} {n}": t for n, t in tabs.items() if isinstance(t, pd.DataFrame)})
    with tab[5]:
        st.caption("Word raporu ekranda üretilmiş grafikleri ve tabloları içerir; tüm bireylerin tam tablosu Excel/CSV dosyasındadır.")
        if "docx" not in saved:
            saved["docx"] = RP.docx_report(res, saved.get("figs") or {}, config, show_ids, extra)
        c = st.columns(3)
        c[0].download_button("Akademik Word raporu (.docx)", saved["docx"], file_name="raschpy_mfrm_raporu.docx",
                             key="rpm_dl_docx", use_container_width=True)
        if c[1].button("Excel ve CSV paketini hazırla", key="rpm_exports", use_container_width=True):
            with st.spinner("Hazırlanıyor..."):
                saved["files"] = RP.exports(res, config, show_ids, extra, figs=saved.get("figs"))
        for i, name in enumerate(n for n in ("raschpy_mfrm.xlsx", "raschpy_mfrm.zip") if n in (saved.get("files") or {})):
            st.download_button(name, saved["files"][name], file_name=name, key=f"rpm_dl_{name}", use_container_width=True)
    with st.expander("İleri Düzey / Teknik Ayrıntılar", expanded=False):
        st.markdown(f"**Kestirim:** RaschPy {RP.RASCHPY_VERSION}, yöntem `{res['estimation']}` (düzenlileştirilmiş ikili bileşik "
                    f"olabilirlik); SE yöntemi {res['se_method']}; tohum {res['seed']}. Denklem: {RP.EQUATION}; puanlayıcı terimi "
                    f"{RP.MODELS[model][1]}.")
        st.markdown("**Bağımsız özdeşlik denetimleri** (RaschPy kodu kullanılmadan yeniden hesaplanır)")
        st.dataframe(RP.identity_table(res), hide_index=True, use_container_width=True)
        st.markdown("**Desteklenmeyen özellikler**")
        for k, v in RP.UNSUPPORTED.items(): st.caption(f"{k}: {v}")
        st.caption("Sürümler: " + "; ".join(res["versions"]))
