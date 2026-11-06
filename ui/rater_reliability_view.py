"""Puanlayıcı Güvenirliği ve Uyum — Turkish Streamlit page (30 preregistered methods, examples, data center)."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import core.rater_reliability as R

EX = Path(__file__).resolve().parents[1] / "examples/rater_reliability"
REG = json.loads((EX / "registry.json").read_text(encoding="utf-8"))
EXAMPLES = {e["id"]: e for e in REG["examples"]}
LAYOUTS = {"wide": "Geniş (satır = birim, sütun = puanlayıcı)", "long": "Uzun (birim, puanlayıcı, puan)",
           "counts": "Kategori-sıklık (satır = birim, sütun = kategori)"}
PANELS = [("kategorik", "5. Kategorik Uyum"), ("sirali_surekli", "6. Sıralı ve Sürekli Puanlar"),
          ("sistematik", "7. Sistematik Puanlayıcı Farklılıkları")]


def _csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def _read(upload):
    raw = upload.getvalue()
    return (pd.read_excel(io.BytesIO(raw)) if upload.name.lower().endswith((".xlsx", ".xls")) else
            pd.read_csv(io.BytesIO(raw), sep=None, engine="python")), hashlib.sha256(raw).hexdigest()


def _guide():
    st.markdown(
        "**Varsayılan biçim: GENİŞ.** Her **satır bir puanlanan birimdir** (öğrenci, kompozisyon, sınav yanıtı, hasta, "
        "görüşme, gözlem…). **Her sütun bir puanlayıcıdır.** İlk sütun birimin benzersiz kimliğidir; diğer sütunlar o "
        "puanlayıcının verdiği puanlardır. **Satırlar puanlayıcı değildir**; puanlanan birim ile puanı veren kişiyi "
        "karıştırmayın.")
    st.dataframe(pd.DataFrame({"Birim_ID": ["B001", "B002", "B003", "B004"], "Puanlayici_1": [1, 2, 3, 1],
                               "Puanlayici_2": [1, 2, 2, 1], "Puanlayici_3": [2, 2, 3, 1]}), hide_index=True)
    st.markdown("**Uzun biçim:** her satır tek bir birim–puanlayıcı gözlemidir. Aynı birim–puanlayıcı çifti iki kez "
                "yazılırsa analiz durur; puanlar **asla sessizce ortalanmaz**.")
    st.dataframe(pd.DataFrame({"Birim_ID": ["B001"] * 3 + ["B002"] * 3, "Puanlayici_ID": ["R1", "R2", "R3"] * 2,
                               "Puan": [1, 1, 2, 2, 2, 2]}), hide_index=True)
    st.markdown("**Kategori-sıklık biçimi (Fleiss κ, Gwet AC1, Brennan–Prediger, yüzde uyum):** her satır bir birim, her "
                "sütun bir kategori; hücre o kategoriyi seçen puanlayıcı sayısıdır ve satır toplamları eşit olmalıdır.")
    st.dataframe(pd.read_csv(EX / "fleiss_kappa_counts.csv").head(4), hide_index=True)
    st.markdown("**Hatalı veri örneği** (öğretim amaçlı): tekrarlanan kimlik, boş hücre ve metin olarak kodlanmış sayı.")
    bad = pd.read_csv(EX / "invalid_teaching.csv", dtype=object)
    st.dataframe(bad, hide_index=True)
    st.caption("Sorunlar: " + "; ".join(REG["teaching"]["invalid"]["problems"]) + ". Platform bunları düzeltmez, raporlar.")
    rows = [{"Yöntem": f"{m['name']} ({m['abbr']})", "Ölçme düzeyi": m["level"], "Puanlayıcı": m["raters"],
             "Veri düzeni": m["layout"], "Eksik veri": m["missing"]} for m in R.METHODS.values()]
    st.markdown("**Yöntemlere göre veri gereksinimleri**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("ICC tam çapraz, eksiksiz bir matris; ikili testler tam iki eşleşmiş sütun; Bland–Altman eşleşmiş nicel ölçüm; "
               "ağırlıklı yöntemler sıralı (sayısal) kategoriler gerektirir. Krippendorff alfa eksik puana izin verir.")
    c1, c2, c3 = st.columns(3)
    ex = c1.selectbox("Örnek", list(EXAMPLES), format_func=lambda k: f"{R.METHODS[EXAMPLES[k]['method']]['name']} — {EXAMPLES[k]['file']}",
                      key="rr_example_choice")
    if c1.button("Örnek Veriyi Yükle", key="rr_load_example", use_container_width=True):
        e = EXAMPLES[ex]; st.session_state["rr_source"] = dict(kind="example", id=ex, sha=e["sha256"])
        st.session_state["rr_cfg"] = dict(e["config"]); st.session_state["rr_methods"] = [e["method"]]
        st.session_state.pop("rr_results", None)
    e = EXAMPLES[ex]
    c2.download_button("Örnek Veriyi İndir (CSV)", (EX / e["file"]).read_bytes(), file_name=e["file"], mime="text/csv",
                       key="rr_dl_example", use_container_width=True)
    c2.download_button("Tüm örnekler (XLSX)", (EX / REG["workbook"]).read_bytes(), file_name=REG["workbook"],
                       key="rr_dl_workbook", use_container_width=True)
    tpl = c3.selectbox("Şablon", ["wide_template", "long_template", "counts_template"],
                       format_func=lambda k: {"wide_template": "Geniş", "long_template": "Uzun", "counts_template": "Kategori-sıklık"}[k],
                       key="rr_template_choice")
    c3.download_button("Veri Şablonunu İndir", (EX / f"{tpl}.csv").read_bytes(), file_name=f"{tpl}.csv", mime="text/csv",
                       key="rr_dl_template", use_container_width=True)
    st.caption(f"Seçili örnek: {e['scenario']} Sütunlar: " + "; ".join(f"**{c}** — {d}" for c, d in e["columns"].items()))


def _load_frame():
    src = st.session_state.get("rr_source")
    up = st.file_uploader("CSV veya XLSX yükleyin", type=["csv", "xlsx"], key="rr_upload")
    if up is not None:
        df, sha = _read(up)
        if not src or src.get("sha") != sha:
            st.session_state["rr_source"] = src = dict(kind="upload", name=up.name, sha=sha)
            st.session_state.pop("rr_cfg", None); st.session_state.pop("rr_results", None)
        return df
    platform = st.session_state.get("raw_df")
    if (not src or src["kind"] == "platform") and isinstance(platform, pd.DataFrame):
        if src or st.button("Platformda yüklü veriyi kullan", key="rr_use_platform"):
            sha = hashlib.sha256(pd.util.hash_pandas_object(platform, index=True).values.tobytes()).hexdigest()
            if not src or src.get("sha") != sha:
                st.session_state["rr_source"] = src = dict(kind="platform", sha=sha)
                idc = st.session_state.get("id_col") or platform.columns[0]
                raters = [c for c in (st.session_state.get("item_cols") or platform.columns[1:]) if c != idc]
                st.session_state["rr_cfg"] = {"layout": "wide", "id_col": idc, "rater_cols": list(raters)}
                st.session_state.pop("rr_results", None)
            st.info("Platformda yüklü veri kullanılıyor (geniş biçim varsayıldı; sütun eşlemesini aşağıdan denetleyin).")
            return platform
    if src and src["kind"] == "example":
        st.info(f"Örnek veri kullanılıyor: {EXAMPLES[src['id']]['file']}")
        return pd.read_csv(EX / EXAMPLES[src["id"]]["file"])
    return None


def _configure(df):
    cfg = st.session_state.setdefault("rr_cfg", {"layout": "wide", "id_col": df.columns[0], "rater_cols": list(df.columns[1:])})
    layouts = list(LAYOUTS)
    layout = st.radio("Veri biçimi", layouts, index=layouts.index(cfg.get("layout", "wide")), format_func=LAYOUTS.get,
                      horizontal=True, key="rr_layout")
    cols = list(df.columns)
    if layout == "long":
        a, b, c = st.columns(3)
        unit = a.selectbox("Birim sütunu", cols, index=cols.index(cfg.get("unit_col", cols[0])) if cfg.get("unit_col") in cols else 0, key="rr_unit")
        rater = b.selectbox("Puanlayıcı sütunu", cols, index=min(1, len(cols) - 1), key="rr_rater")
        score = c.selectbox("Puan sütunu", cols, index=min(2, len(cols) - 1), key="rr_score")
        new = dict(layout="long", unit_col=unit, rater_col=rater, score_col=score)
    else:
        idc = st.selectbox("Birim kimliği sütunu", cols, index=cols.index(cfg.get("id_col", cols[0])) if cfg.get("id_col") in cols else 0, key="rr_id")
        rest = [c for c in cols if c != idc]
        key = "category_cols" if layout == "counts" else "rater_cols"
        chosen = st.multiselect("Kategori sütunları" if layout == "counts" else "Puanlayıcı sütunları", rest,
                                default=[c for c in cfg.get(key, rest) if c in rest] or rest, key=f"rr_{key}")
        new = {"layout": layout, "id_col": idc, key: chosen}
    if new != cfg:
        st.session_state["rr_cfg"] = new; st.session_state.pop("rr_results", None)
    return new


def _data(df, cfg):
    if cfg["layout"] == "long":
        return R.from_long(df, cfg["unit_col"], cfg["rater_col"], cfg["score_col"])
    if cfg["layout"] == "counts":
        return R.from_counts(df, cfg["id_col"], cfg["category_cols"])
    return R.from_wide(df, cfg["id_col"], cfg["rater_cols"])


def _result_row(out):
    m = R.METHODS[out["method"]]; s = out["shown"]
    f = lambda v: "—" if v is None or not np.isfinite(v) else f"{v:.4f}"
    main = s.get("estimate", s.get("chi2", s.get("bias")))
    ci = s.get("ci95") or s.get("ci95_bias")
    stat = (f"χ²({s['df']}) = {s['chi2']:.4f}" if "chi2" in s and "df" in s else
            f"F({s['df1']:.0f}, {s['df2']:.0f}) = {s['F']:.4f}" if "F" in s else f"z₀ = {s['z0']:.4f}" if "z0" in s else "—")
    return {"Yöntem": m["name"], "Kısaltma": m["abbr"], "Değer": f(main), "SE": f(s.get("se")),
            "%95 GA": f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "—", "Test": stat, "p": f(s.get("p")),
            "Birim": out["n_units"], "Puanlayıcı": out["n_raters"], "Kanıt durumu": out["status"]}


def _results_table(results):
    return pd.DataFrame([_result_row(o) for o in results.values()])


def _ba_png(out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r = out["shown"]; fig, ax = plt.subplots(figsize=(6, 4), dpi=150)       # validated values only
    ax.scatter(r["mean"], r["difference"], s=14, color="#2b5f8a")
    for y, ls in ((r["bias"], "-"), (r["loa"][0], "--"), (r["loa"][1], "--")):
        ax.axhline(y, color="#b5452f" if ls == "--" else "#333", ls=ls, lw=1)
    ax.set_xlabel("İki ölçümün ortalaması"); ax.set_ylabel("Fark (1. − 2.)"); ax.set_title("Bland–Altman grafiği")
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig); return buf.getvalue()


def _docx(results, data, cfg):
    from docx import Document
    doc = Document(); doc.add_heading("Puanlayıcı Güvenirliği ve Uyum Raporu", 0)
    doc.add_heading("Amaç ve veri yapısı", 1)
    doc.add_paragraph("Bu rapor, aynı birimleri puanlayan puanlayıcılar arasındaki uyumu, güvenirliği, ilişkiyi ve sistematik "
                      "farklılıkları ayrı yöntemlerle özetler. Uyum (aynı puanı verme), güvenirlik (birimleri ayırt etme), ilişki "
                      "(birlikte değişme) ve sistematik yanlılık farklı kavramlardır.")
    for k, v in R.summary(data).items():
        doc.add_paragraph(f"{k}: {v}", style="List Bullet")
    doc.add_paragraph(f"Veri biçimi: {LAYOUTS[cfg['layout']]}. Eksik veri politikası: veri değiştirilmez; yalnız Krippendorff alfa eksik puanla çalışır.")
    doc.add_heading("Bulgular", 1)
    t = _results_table(results); table = doc.add_table(rows=1, cols=len(t.columns)); table.style = "Table Grid"
    for i, c in enumerate(t.columns): table.rows[0].cells[i].text = str(c)
    for row in t.itertuples(index=False):
        cells = table.add_row().cells
        for i, v in enumerate(row): cells[i].text = str(v)
    for out in results.values():
        m = R.METHODS[out["method"]]; doc.add_heading(f"{m['name']} ({m['abbr']})", 2)
        doc.add_paragraph(f"Ölçme düzeyi: {m['level']}; puanlayıcı: {m['raters']}; tahminci: {m['formula']}; motor: {m['engine']}.")
        doc.add_paragraph(out["interpretation"])
        doc.add_paragraph(f"Bilimsel kanıt durumu: {out['status']} (bağımsız referans: {m['reference']}).")
        if out["method"] == "bland_altman" and {"bias", "loa"} <= set(out["shown"]):
            doc.add_picture(io.BytesIO(_ba_png(out)))
    doc.add_heading("Sınırlılıklar", 1)
    doc.add_paragraph("Kanıt durumu yalnız depodaki deterministik örnek veriler için geçerlidir; genel sertifika değildir. SE/GA/p "
                      "yalnız bağımsız olarak doğrulandığı yöntemlerde verilir. Yorumda evrensel eşik kullanılmamıştır.")
    doc.add_paragraph("Sürümler: " + ", ".join(f"{k} {v}" for k, v in R.versions().items()))
    buf = io.BytesIO(); doc.save(buf); return buf.getvalue()


def _exports(results, data, cfg):
    t = _results_table(results)
    details = pd.DataFrame([{"method": k, **{a: json.dumps(b) if isinstance(b, (list, dict)) else b
                                            for a, b in o["shown"].items() if a not in ("mean", "difference")}}
                            for k, o in results.items()])
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf) as w:
        t.to_excel(w, sheet_name="Bulgular", index=False); details.to_excel(w, sheet_name="Ayrinti", index=False)
        pd.DataFrame([R.summary(data)]).to_excel(w, sheet_name="Veri_Yapisi", index=False)
    config = {"config": cfg, "methods": list(results), "source": st.session_state.get("rr_source"), "versions": R.versions(),
              "evidence": "validation/rater_reliability/evidence.json"}
    files = {"bulgular.csv": _csv(t), "ayrinti.csv": _csv(details), "bulgular.xlsx": xbuf.getvalue(),
             "rapor.docx": _docx(results, data, cfg), "analiz_yapilandirmasi.json": json.dumps(config, ensure_ascii=False, indent=1).encode()}
    if "bland_altman" in results and {"bias", "loa"} <= set(results["bland_altman"]["shown"]):
        files["bland_altman.png"] = _ba_png(results["bland_altman"])
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, b in files.items(): z.writestr(name, b)
    files["puanlayici_guvenirligi.zip"] = zbuf.getvalue()
    return files


def render():
    st.header("Puanlayıcı Güvenirliği ve Uyum")
    st.caption("30 yöntem; Python motorları, bağımsız R referanslarıyla örnek verilerde doğrulandı. Hiçbir yöntem CERTIFIED değildir.")
    with st.expander("1. Veri Setimi Nasıl Düzenlemeliyim?", expanded=st.session_state.get("rr_source") is None):
        _guide()
    with st.expander("2. Veri Yükle / Örnek Veri Seç", expanded=True):
        df = _load_frame()
        if df is None:
            st.info("Bir dosya yükleyin ya da 1. bölümden örnek veri seçin."); return
        cfg = _configure(df)
    try:
        data = _data(df, cfg)
    except R.RaterDataError as e:
        st.error(str(e)); return
    with st.expander("3. Veri Yapısı ve Puanlayıcılar", expanded=True):
        st.dataframe(pd.DataFrame([R.summary(data)]).T.rename(columns={0: "Değer"}).astype(str))
        for sev, msg in data.issues:
            (st.error if sev == "error" else st.warning)(msg)
        if not data.issues:
            st.success("Kimlik, kodlama ve eksik veri denetimleri sorun göstermedi.")
        if data.layout == "wide":
            st.download_button("Uzun biçime çevrilmiş veriyi indir", _csv(R.to_long(data)), file_name="uzun_bicim.csv", key="rr_dl_long")
            if not data.has_missing:
                c = R.counts_from_wide(data).reset_index(names="Birim_ID")
                st.download_button("Kategori-sıklık tablosunu indir", _csv(c), file_name="kategori_siklik.csv", key="rr_dl_counts")
    status = {m: R.applicable(m, data) for m in R.METHODS}
    with st.expander("4. Uygun Analizleri Öner", expanded=False):
        st.dataframe(pd.DataFrame([{"Yöntem": f"{R.METHODS[m]['name']} ({R.METHODS[m]['abbr']})", "Uygun": "Evet" if ok else "Hayır",
                                    "Gerekçe": why or R.METHODS[m]["level"], "Kanıt durumu": R.status(m)}
                                   for m, (ok, why) in status.items()]), hide_index=True, use_container_width=True)
    chosen = []
    for key, title in PANELS:
        groups = {"kategorik": {"kategorik"}, "sirali_surekli": {"sirali", "surekli"}, "sistematik": {"sistematik"}}[key]
        ids = [m for m, meta in R.METHODS.items() if meta["group"] in groups]
        with st.expander(title, expanded=any(m in st.session_state.get("rr_methods", []) for m in ids)):
            ok_ids = [m for m in ids if status[m][0]]
            sel = st.multiselect("Yöntemler", ok_ids, default=[m for m in st.session_state.get("rr_methods", []) if m in ok_ids],
                                 format_func=lambda m: f"{R.METHODS[m]['name']} ({R.METHODS[m]['abbr']})", key=f"rr_sel_{key}")
            chosen += sel
            for m in ids:
                if not status[m][0]:
                    st.caption(f"⛔ {R.METHODS[m]['name']}: {status[m][1]}")
    fp = hashlib.sha256(json.dumps([st.session_state.get("rr_source"), cfg, sorted(chosen)], default=str).encode()).hexdigest()
    if st.button("Seçili analizleri çalıştır", key="rr_run", type="primary", disabled=not chosen):
        st.session_state["rr_results"] = {"fp": fp, "out": {m: R.run(m, data) for m in chosen}}
    saved = st.session_state.get("rr_results")
    if saved and saved["fp"] != fp:
        st.session_state.pop("rr_results"); saved = None             # stale result after data/selection change
    with st.expander("8. Bulgular ve Akademik Yorum", expanded=bool(saved)):
        if not saved:
            st.info("Analiz seçip çalıştırın.")
        else:
            st.dataframe(_results_table(saved["out"]), hide_index=True, use_container_width=True)
            for m, out in saved["out"].items():
                meta = R.METHODS[m]
                st.markdown(f"**{meta['name']} ({meta['abbr']})** — {out['interpretation']}")
                st.caption(f"Amaç/model: {meta['formula']}. Düzey: {meta['level']}; puanlayıcı: {meta['raters']}; düzen: {meta['layout']}; "
                           f"eksik veri: {meta['missing']}. Motor: {meta['engine']}. Kanıt: {out['status']}.")
                if m == "bland_altman" and {"bias", "loa"} <= set(out["shown"]):
                    st.image(_ba_png(out), caption="Bland–Altman grafiği")
    with st.expander("9. Raporlar ve Dışa Aktarma", expanded=False):
        if saved:
            files = _exports(saved["out"], data, cfg)
            cols = st.columns(3)
            for i, (name, b) in enumerate(files.items()):
                cols[i % 3].download_button(name, b, file_name=name, key=f"rr_dl_{name}", use_container_width=True)
        else:
            st.info("Dışa aktarma için önce analiz çalıştırın.")
