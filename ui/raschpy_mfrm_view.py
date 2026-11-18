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
            st.dataframe(pd.DataFrame([{k: str(v) for k, v in rep.items()}]).T.rename(columns={0: "Değer"}))
            if rep["sparse_categories"]:
                st.warning(f"Seyrek kategori(ler): {rep['sparse_categories']} (< 10 gözlem).")
        except MFRMError as e:
            data_error = str(e); st.error(data_error)
    config = dict(person=person, item=item, rater=rater, score=score, model=model, se=se, n_boot=int(n_boot),
                  source=st.session_state.get("rpm_source"))
    fp = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    if st.button("RaschPy MFRM kestirimini çalıştır", type="primary", key="rpm_run", disabled=bool(data_error),
                 help=data_error or "RaschPy calibrate + std_errors + fit_statistics"):
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
    res = saved["result"]
    st.success(f"Kestirim tamamlandı ({res['seconds']:.2f} sn; yöntem {res['estimation']}).")
    with st.expander("4. Kanıt durumu", expanded=True):
        stt = RP.status_table(model); stt["Durum"] = stt["Durum"].map(lambda s: BADGE.get(s, s))
        st.dataframe(stt, hide_index=True, use_container_width=True)
        st.caption("Durumlar yalnız depodaki fikstürler için geçerlidir (validation/raschpy_mfrm/evidence.json).")
        st.json(res["identity"], expanded=False)
    if not RP.rater_se_reliable(model):
        st.info("Bu modelde puanlayıcı SE'leri benzetimde güvenilir kapsama göstermedi; madde/eşik/kategori SE'leri için kapsama "
                "kanıtı yok. Bu standart hatalar gizlendi.")
    if RP.cell_bootstrap_validated(model):
        if st.button("Puanlayıcı hücreleri için bootstrap SE ve %95 GA (B = 50, birey kümeli)", key="rpm_cellboot"):
            with st.spinner("Bootstrap yeniden kestirimleri..."):
                res["cell_bootstrap"] = RP.cell_bootstrap(res, B=50); saved.pop("files", None)
    T = RP.tables(res, show_ids)
    with st.expander("5. Sonuç tabloları", expanded=True):
        for k, t in T.items():
            st.markdown(f"**{k}**"); st.dataframe(t, hide_index=True, use_container_width=True)
    with st.expander("6. Model seçimi ve grup testleri (isteğe bağlı)", expanded=False):
        st.caption("RaschPy tarafından raporlanır; bağımsız doğrulanmamıştır (NOT CERTIFIED). Andersen LR testi RaschPy'de devre dışıdır.")
        test = st.radio("Ölçüt", ["AIC", "BIC"], horizontal=True, key="rpm_ms_test")
        if st.button("8 modeli karşılaştır", key="rpm_ms"):
            try: saved["selection"] = RP.model_selection(res, test)
            except Exception as e: st.error(f"Model seçimi başarısız: {e}")
        if saved.get("selection") is not None:
            st.dataframe(saved["selection"], hide_index=True)
        cov = st.selectbox("Birey düzeyi grup değişkeni", ["—"] + [c for c in df.columns if c not in (person, item, rater, score)], key="rpm_cov")
        kind = st.radio("Test", ["drf", "dif"], horizontal=True, key="rpm_kind",
                        format_func={"drf": "DRF (puanlayıcı × grup)", "dif": "DIF (madde × grup)"}.get)
        if st.button("Grup testini çalıştır", key="rpm_group", disabled=cov == "—"):
            try: saved["group"] = {kind: RP.group_test(res, df, person, cov, kind)}
            except MFRMError as e: st.error(str(e))
        for kk, tabs in (saved.get("group") or {}).items():
            for name, t in tabs.items():
                st.markdown(f"**{kk.upper()} — {name}**"); st.dataframe(t, hide_index=True)
    with st.expander("7. Grafikler ve raporlar", expanded=False):
        if st.button("Grafik ve raporları hazırla (CSV/XLSX/DOCX/ZIP)", key="rpm_exports"):
            extra = {"Model seçimi": saved["selection"]} if saved.get("selection") is not None else {}
            for kk, tabs in (saved.get("group") or {}).items():
                extra.update({f"{kk.upper()} {n}"[:31]: t for n, t in tabs.items() if isinstance(t, pd.DataFrame)})
            with st.spinner("Hazırlanıyor..."):
                saved["files"] = RP.exports(res, config, show_ids, extra)
        files = saved.get("files") or {}
        for name in [n for n in files if n.endswith(".png")]:
            st.image(files[name], caption=name)
        cs = st.columns(3)
        for i, (name, b) in enumerate((n, b) for n, b in files.items() if not n.endswith(".png") and not n.endswith(".csv")):
            cs[i % 3].download_button(name, b, file_name=name, key=f"rpm_dl_{name}", use_container_width=True)
