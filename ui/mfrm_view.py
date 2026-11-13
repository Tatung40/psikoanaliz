"""MTK → Çok Yüzeyli Rasch (MFRM): Turkish page. Engine: R TAM (no substitute estimator)."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import core.mfrm as M

EX = Path(__file__).resolve().parents[1] / "examples/mfrm"
FIX = json.loads((EX / "fixtures.json").read_text())["fixtures"]
EXAMPLES = {"balanced_3facet": "Dengeli: 300 birey × 4 ölçüt × 3 puanlayıcı (hoşgörülü, ortalama, katı)",
            "unbalanced_linked": "Dengesiz bağlantılı: her bireyi 5 puanlayıcıdan 2'si puanlıyor",
            "interaction_bias": "Etkileşim: P2 puanlayıcısı K3 ölçütünde katı",
            "sparse_category": "Seyrek üst kategori (uyarı örneği)", "disconnected": "Bağlantısız tasarım (hata örneği)",
            "invalid_duplicate_missing": "Tekrarlanan satır ve eksik puan (hata örneği)"}
EQUATION = "log[P(X = k) / P(X = k − 1)] = θₙ − δᵢ − λⱼ − τₖ,  θ ~ N(μ, σ²)"
ENGINE_HELP = ("Bu analiz yalnız doğrulanmış R TAM 4.3-25 ile yapılır; R/TAM yokken Python ile yerine kestirim üretilmez. "
               "Streamlit Community Cloud'da TAM için desteklenen bir sistem paketi yoktur (Debian'da r-cran-tam bulunmaz; "
               "packages.txt yalnız apt paketleri kurar). Çalıştırmak için depodaki Dockerfile ile oluşturulan kapsayıcıyı "
               "veya R ve TAM 4.3-25 kurulu bir ortamı kullanın: docs/MFRM_PRODUCTION_DEPLOYMENT.md.")


def _csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def _plots(result):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = {}; P = result["params"]; el = result["elements"]; persons = result["persons"]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.hist(persons["WLE"] if "WLE" in persons else persons["EAP"], bins=25, orientation="horizontal", color="#9db6cc")
    xmax = ax.get_xlim()[1]
    for k, (f, colr) in enumerate((("item", "#2b5f8a"), ("rater", "#b5452f"), ("facet1", "#4c8c4a"))):
        sub = el[el.facet == f]
        for r in sub.itertuples():
            ax.annotate(r.Düzey, (xmax * (1.15 + .35 * k), r._4), color=colr, fontsize=8, annotation_clip=False)
    for t in P["tau"]:
        ax.axhline(t, color="#777", lw=.6, ls=":")
    ax.set_ylabel("Logit"); ax.set_xlabel("Birey sayısı"); ax.set_title("Wright haritası (bireyler | maddeler | puanlayıcılar)")
    fig.subplots_adjust(right=.62); out["wright_haritasi.png"] = _png(fig, plt)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150); z = np.linspace(-5, 5, 201)
    lp = M._log_probs(z, np.array([0.]), P["tau"])[0]
    for k in range(lp.shape[1]):
        ax.plot(z, np.exp(lp[:, k]), label=f"Kategori {k}")
    ax.set_xlabel("θ − δ − λ (logit)"); ax.set_ylabel("Olasılık"); ax.legend(fontsize=8); ax.set_title("Kategori olasılık eğrileri")
    out["kategori_egrileri.png"] = _png(fig, plt)
    if M.shown("facet_ci"):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150); y = np.arange(len(el))
        ax.errorbar(el["Ölçü (logit)"], y, xerr=1.96 * el["SE"], fmt="o", color="#2b5f8a", capsize=3)
        ax.set_yticks(y, [f"{a}: {b}" for a, b in zip(el.Yüzey, el.Düzey)], fontsize=8); ax.axvline(0, color="#999", lw=.6)
        ax.set_xlabel("Ölçü (logit) ± 1.96 SE (Wald)"); ax.set_title("Yüzey ölçüleri ve %95 aralık")
        out["yuzey_olculeri.png"] = _png(fig, plt)
    return out


def _png(fig, plt):
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig); return buf.getvalue()


def _docx(result, view, plots):
    from docx import Document
    doc = Document(); doc.add_heading("Çok Yüzeyli Rasch Ölçme (MFRM) Raporu", 0)
    doc.add_heading("Model ve kestirim", 1)
    doc.add_paragraph(f"Derecelendirme ölçeği çok yüzeyli Rasch modeli: {EQUATION}. Kestirim marjinal en çok olabilirlik "
                      "(R TAM::tam.mml.mfr, EM, 61 düğüm, Ramsay hızlandırması). Tanımlama: madde, puanlayıcı ve eşik parametreleri "
                      "sıfıra toplanır; gizil ortalama ve varyans kestirilir. Birey ölçüleri WLE ve EAP olarak verilir.")
    doc.add_paragraph("Bu çıktı FACETS yazılımının JMLE çıktısıyla eşdeğer değildir: tahmin edici (MML), birey ölçüsü ve uyum "
                      "istatistiklerinin tanımı (EAP tabanlı artık) farklıdır. Yalnız tanımı doğrulanmış çıktılar raporlanmıştır.")
    r = result["report"]
    doc.add_paragraph(f"Veri: {r['n_rows']} puan, {r['n_persons']} birey, düzey sayıları {r['levels']}; eksik puan "
                      f"{r['missing_scores_excluded']} satır analiz dışı; kategoriler {r['categories']} (sıklık {r['category_counts']}).")
    for name, t in view["tables"].items():
        doc.add_heading(name, 1)
        t = t.head(60) if name == "Bireyler" else t
        tab = doc.add_table(rows=1, cols=len(t.columns)); tab.style = "Table Grid"
        for i, c in enumerate(t.columns): tab.rows[0].cells[i].text = str(c)
        for row in t.itertuples(index=False):
            cells = tab.add_row().cells
            for i, v in enumerate(row): cells[i].text = f"{v:.4f}" if isinstance(v, float) else str(v)
    doc.add_heading("Yorum", 1)
    el = result["elements"]; rr = el[el.facet == "rater"].sort_values("Ölçü (logit)")
    doc.add_paragraph(f"En hoşgörülü puanlayıcı {rr.Düzey.iloc[0]} ({rr['Ölçü (logit)'].iloc[0]:.2f} logit), en katı puanlayıcı "
                      f"{rr.Düzey.iloc[-1]} ({rr['Ölçü (logit)'].iloc[-1]:.2f} logit). Katılık farkları ortak logit ölçeğinde "
                      "yorumlanır; uyum değerlerinin kabul aralığı evrensel değildir ve çalışmanın amacına göre değerlendirilmelidir.")
    doc.add_heading("Uyarılar ve sınırlılıklar", 1)
    for n in view["notes"]:
        doc.add_paragraph(n, style="List Bullet")
    doc.add_paragraph("Kanıt durumu yalnız depodaki deterministik fikstürler için geçerlidir (validation/mfrm/evidence.json).")
    for name, b in plots.items():
        doc.add_picture(io.BytesIO(b))
    doc.add_paragraph("Sürümler: " + "; ".join(result["versions"]))
    buf = io.BytesIO(); doc.save(buf); return buf.getvalue()


def _exports(result, view, config):
    plots = _plots(result); files = {}
    for name, t in view["tables"].items():
        files[f"{name}.csv"] = _csv(t)
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf) as w:
        for name, t in view["tables"].items():
            t.to_excel(w, sheet_name=name[:31], index=False)
    files["mfrm_sonuclari.xlsx"] = xbuf.getvalue(); files.update(plots)
    files["mfrm_raporu.docx"] = _docx(result, view, plots)
    files["analiz_yapilandirmasi.json"] = json.dumps(dict(config, versions=result["versions"], identity=result["identity"]),
                                                     ensure_ascii=False, indent=1, default=str).encode()
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, b in files.items(): zf.writestr(name, b)
    files["mfrm.zip"] = z.getvalue()
    return files


def render():
    st.header("Çok Yüzeyli Rasch Ölçme (MFRM)")
    st.caption(f"Model: {EQUATION}. Motor: R TAM {M.TAM_VERSION} (MML). FACETS ile tam eşdeğerlik iddia edilmez.")
    with st.expander("1. Veri yapısı ve örnek veriler", expanded="mfrm_source" not in st.session_state):
        st.markdown("Veri **uzun biçimde** olmalıdır: her satır tek bir puanlama olayıdır (bir birey, bir madde/ölçüt, bir "
                    "puanlayıcı ve puan). Zorunlu sütunlar: birey, madde/ölçüt, puanlayıcı, puan. Puanlar 0'dan başlayan sıralı tam "
                    "sayılardır. Aynı birey–madde–puanlayıcı birleşimi bir kez bulunmalıdır; puanlar ortalanmaz.")
        st.dataframe(pd.read_csv(EX / "balanced_3facet.csv").head(6), hide_index=True)
        ex = st.selectbox("Örnek veri", list(EXAMPLES), format_func=EXAMPLES.get, key="mfrm_example")
        c1, c2 = st.columns(2)
        if c1.button("Örnek veriyi yükle", key="mfrm_load", use_container_width=True):
            st.session_state["mfrm_source"] = dict(kind="example", id=ex, sha=FIX[ex]["sha256"]); st.session_state.pop("mfrm_result", None)
        c2.download_button("Örnek veriyi indir (CSV)", (EX / f"{ex}.csv").read_bytes(), file_name=f"{ex}.csv", key="mfrm_dl_ex",
                           use_container_width=True)
        c2.download_button("Tüm örnekler (XLSX)", (EX / "mfrm_examples.xlsx").read_bytes(), file_name="mfrm_examples.xlsx",
                           key="mfrm_dl_xlsx", use_container_width=True)
    up = st.file_uploader("Uzun biçimli CSV/XLSX yükleyin", type=["csv", "xlsx"], key="mfrm_upload")
    if up is not None:
        raw = up.getvalue(); sha = hashlib.sha256(raw).hexdigest()
        df = pd.read_excel(io.BytesIO(raw)) if up.name.lower().endswith("xlsx") else pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
        if st.session_state.get("mfrm_source", {}).get("sha") != sha:
            st.session_state["mfrm_source"] = dict(kind="upload", name=up.name, sha=sha); st.session_state.pop("mfrm_result", None)
    elif st.session_state.get("mfrm_source", {}).get("kind") == "example":
        df = pd.read_csv(EX / f"{st.session_state['mfrm_source']['id']}.csv")
    else:
        st.info("Bir dosya yükleyin veya örnek veri seçin."); return
    with st.expander("2. Yüzey rolleri ve model", expanded=True):
        cols = list(df.columns); pick = lambda label, guess, key: st.selectbox(label, cols, index=cols.index(guess) if guess in cols else 0, key=key)
        person = pick("Birey", "person_id", "mfrm_person"); item = pick("Madde / ölçüt", "item_id", "mfrm_item")
        rater = pick("Puanlayıcı", "rater_id", "mfrm_rater"); score = pick("Puan", "score", "mfrm_score")
        interaction = st.checkbox("Madde × puanlayıcı etkileşimini (yanlılık) sına", key="mfrm_interaction")
        show_ids = st.checkbox("Gerçek birey kimliklerini göster ve dışa aktar (varsayılan: takma ad)", key="mfrm_ids")
        st.caption("Ek yüzeyler (görev, oturum) bu sürümde desteklenmiyor: dört yüzeyli doğrulama TAM ile sayısal olarak başarısız oldu.")
    config = dict(person=person, item=item, rater=rater, score=score, interaction=interaction, show_ids=show_ids,
                  source=st.session_state.get("mfrm_source"))
    fp = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    data_error = None
    with st.expander("3. Veri denetimi", expanded=True):
        try:
            _, report = M.prepare(df, person, item, rater, score)
            st.dataframe(pd.DataFrame([{k: str(v) for k, v in report.items()}]).T.rename(columns={0: "Değer"}))
            if report["sparse_categories"]:
                st.warning(f"Seyrek kategori(ler): {report['sparse_categories']} (< 10 gözlem).")
        except M.MFRMError as e:
            data_error = str(e); st.error(data_error)
    ok, why = M.engine_available()
    if not ok:
        st.error("MFRM motoru kullanılamıyor: " + why)
        st.info(ENGINE_HELP)
    # Opt-in installation trial: only attempt after Rscript is present; never run on page load.
    if not ok and M.shutil.which("Rscript"):
        with st.expander("Ücretsiz Streamlit Cloud: R/TAM kurulum denemesi", expanded=False):
            st.warning("TAM kaynak koddan derlenecektir. İşlem uzun sürebilir veya ücretsiz sunucunun kaynak sınırına takılabilir. Kurulum yalnız bu sunucu örneği için geçerlidir.")
            if st.button("TAM 4.3-25 kurulumunu dene", key="mfrm_install_tam"):
                with st.spinner("R/TAM kuruluyor; bu işlem birkaç dakika sürebilir..."):
                    try:
                        import subprocess
                        import fcntl
                        import os
                        from pathlib import Path
                        # One installer per server instance, including concurrent Streamlit sessions.
                        lock_path = Path("/tmp/psikoanaliz_mfrm_install.lock")
                        with lock_path.open("a+") as lock_file:
                            try:
                                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                st.warning("R/TAM kurulumu başka bir oturumda devam ediyor. Birkaç dakika sonra sayfayı yenileyin.")
                                st.stop()
                            # R's 00LOCK-* directories can survive interrupted installs. Clean only
                            # known package locks while holding the installer mutex; never remove
                            # the entire user library or an unrelated package installation.
                            command = ('lib <- path.expand(Sys.getenv("R_LIBS_USER")); '
                                       'dir.create(lib, recursive=TRUE, showWarnings=FALSE); '
                                       '.libPaths(c(lib, .libPaths())); '
                                       'for (pkg in c("CDM","TAM")) { '
                                       'lock <- file.path(lib, paste0("00LOCK-",pkg)); '
                                       'if (dir.exists(lock)) unlink(lock, recursive=TRUE) }; '
                                       'options(repos=c(CRAN="https://cloud.r-project.org")); '
                                       'if (!requireNamespace("remotes",quietly=TRUE)) install.packages("remotes", lib=lib); '
                                       'remotes::install_version("TAM",version="4.3-25",upgrade="never",dependencies=NA,lib=lib); '
                                       'stopifnot(gsub("-",".",as.character(packageVersion("TAM",lib.loc=lib)))=="4.3.25")')
                            proc = subprocess.run(["Rscript", "-e", command], capture_output=True, text=True, timeout=900)
                            if proc.returncode:
                                st.error("TAM kurulamadı. Kurulum günlüğünün son bölümü:")
                                st.code((proc.stderr + "\\n" + proc.stdout)[-3500:])
                            else:
                                st.success("TAM kuruldu; motor yeniden kontrol ediliyor.")
                                st.rerun()
                    except subprocess.TimeoutExpired:
                        st.error("Kurulum 15 dakikalık sınırı aştı; ücretsiz ortam uygun olmayabilir.")
                    except OSError as exc:
                        st.error(f"Kurulum başlatılamadı: {exc}")
    blocked = data_error or (None if ok else why)
    # The run button is always rendered; it is disabled (never hidden) while data or engine checks fail.
    if st.button("MFRM kestirimini çalıştır", key="mfrm_run", type="primary", disabled=bool(blocked),
                 help=blocked or "R TAM ile marjinal ML kestirimi") and not blocked:
        try:
            res = M.analyze(df, person, item, rater, score, interaction=interaction, show_ids=show_ids)
            st.session_state["mfrm_result"] = {"fp": fp, "result": res}
        except M.MFRMError as e:
            st.session_state["mfrm_result"] = {"fp": fp, "error": str(e)}
    saved = st.session_state.get("mfrm_result")
    if saved and saved["fp"] != fp:
        st.session_state.pop("mfrm_result"); saved = None
    if not saved:
        return
    if "error" in saved:
        st.error(saved["error"]); return
    res = saved["result"]; view = M.present(res)
    for n in view["notes"]:
        st.warning(n)
    with st.expander("4. Sonuçlar", expanded=True):
        for name, t in view["tables"].items():
            st.markdown(f"**{name}**"); st.dataframe(t, hide_index=True, use_container_width=True)
    files = _exports(res, view, config)
    with st.expander("5. Grafikler", expanded=False):
        for name in [n for n in files if n.endswith(".png")]:
            st.image(files[name], caption=name)
    with st.expander("6. Raporlar ve dışa aktarma", expanded=False):
        cs = st.columns(3)
        for i, (name, b) in enumerate(files.items()):
            cs[i % 3].download_button(name, b, file_name=name, key=f"mfrm_dl_{name}", use_container_width=True)
