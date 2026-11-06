"""Deterministic synthetic examples for the 30 rater-reliability methods (no personal data).
Run: PYTHONPATH=. python examples/rater_reliability/build.py   (rewrites CSV/XLSX and registry.json, keeping `expected`).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 20261106


def _rng(k):
    return np.random.default_rng(SEED + k)


def _ids(n):
    return [f"B{i + 1:03d}" for i in range(n)]


def _disagree(rng, truth, cats, p, shift=None):
    out = truth.copy()
    for i in range(len(out)):
        if rng.uniform() < p:
            if shift is not None:
                out[i] = min(max(out[i] + shift(rng), cats[0]), cats[-1])
            else:
                out[i] = rng.choice([c for c in cats if c != out[i]])
    return out


def datasets():
    d = {}
    r = _rng(1); cats = np.array(["Olumlu", "Nötr", "Olumsuz"], object)
    t = r.choice(cats, 40, p=[.6, .25, .15]); d["cohen_kappa_nominal"] = pd.DataFrame(
        {"Birim_ID": _ids(40), "Puanlayici_1": t, "Puanlayici_2": _disagree(r, t, list(cats), .25)})
    r = _rng(2); t = (r.uniform(size=50) < .85).astype(int); b = t.copy()
    b[(t == 0) & (r.uniform(size=50) < .6)] = 1; b[(t == 1) & (r.uniform(size=50) < .04)] = 0
    d["binary_two_raters"] = pd.DataFrame({"Birim_ID": _ids(50), "Puanlayici_1": t, "Puanlayici_2": b})
    r = _rng(3); t = r.choice([1, 2, 3, 4, 5], 40, p=[.1, .2, .35, .25, .1])
    d["ordinal_two_raters"] = pd.DataFrame({"Birim_ID": _ids(40), "Puanlayici_1": t,
        "Puanlayici_2": _disagree(r, t, [1, 2, 3, 4, 5], .55, shift=lambda g: int(g.choice([-1, 1, 1])))})
    r = _rng(4); t = r.choice([1, 2, 3], 30, p=[.5, .3, .2])
    d["nominal_multirater"] = pd.DataFrame({"Birim_ID": _ids(30), **{f"Puanlayici_{j + 1}": _disagree(r, t, [1, 2, 3], p)
                                            for j, p in enumerate([.1, .2, .3, .15])}})
    r = _rng(5); rows = []
    for i in range(25):
        probs = np.eye(4)[r.integers(4)] * .7 + .075
        rows.append(np.bincount(r.choice(4, 6, p=probs / probs.sum()), minlength=4))
    d["fleiss_kappa_counts"] = pd.DataFrame(rows, columns=["Kategori_A", "Kategori_B", "Kategori_C", "Kategori_D"]).assign(
        Birim_ID=_ids(25))[["Birim_ID", "Kategori_A", "Kategori_B", "Kategori_C", "Kategori_D"]]
    r = _rng(6); t = r.choice([1, 2, 3, 4, 5], 30, p=[.1, .2, .3, .25, .15])
    d["ordinal_multirater"] = pd.DataFrame({"Birim_ID": _ids(30), **{f"Puanlayici_{j + 1}": _disagree(
        r, t, [1, 2, 3, 4, 5], p, shift=lambda g: int(g.choice([-1, 1]))) for j, p in enumerate([.2, .3, .4])}})
    r = _rng(7); t = r.choice([1, 2, 3, 4, 5], 25)
    k = pd.DataFrame({f"Puanlayici_{j + 1}": _disagree(r, t, [1, 2, 3, 4, 5], .3, shift=lambda g: int(g.choice([-1, 1])))
                      for j in range(4)}).astype(float)
    mask = r.uniform(size=k.shape) < .15; mask[:, 0] &= False          # rater 1 complete; others with gaps
    d["krippendorff_alpha_missing"] = pd.concat([pd.DataFrame({"Birim_ID": _ids(25)}), k.mask(mask)], axis=1)
    r = _rng(8); true = r.normal(50, 10, 20); bias = [0, 2.5, -1.5, 4]
    d["icc_two_way"] = pd.DataFrame({"Birim_ID": _ids(20), **{f"Puanlayici_{j + 1}": np.round(true + bias[j] + r.normal(0, 4, 20), 1)
                                     for j in range(4)}})
    r = _rng(9); base = np.arange(1, 11, dtype=float); cols = {}
    for j in range(5):
        s = base + r.normal(0, 1.5, 10); s = np.round(s * 2) / 2          # half-point scores create ties
        cols[f"Puanlayici_{j + 1}"] = s
    d["kendall_w_rankings"] = pd.DataFrame({"Kompozisyon_ID": [f"K{i + 1:02d}" for i in range(10)], **cols})
    r = _rng(10); x = r.normal(120, 15, 30)
    d["bland_altman_paired"] = pd.DataFrame({"Birim_ID": _ids(30), "Yontem_A": np.round(x + r.normal(0, 4, 30), 1),
                                             "Yontem_B": np.round(x + 1.5 + .02 * (x - 120) + r.normal(0, 4, 30), 1)})
    lf = d["nominal_multirater"].melt(id_vars="Birim_ID", var_name="Puanlayici_ID", value_name="Puan")
    d["long_format_example"] = lf.sort_values(["Birim_ID", "Puanlayici_ID"]).reset_index(drop=True)
    bad = d["nominal_multirater"].head(8).copy().astype(object)
    bad.loc[3, "Birim_ID"] = bad.loc[2, "Birim_ID"]; bad.loc[5, "Puanlayici_2"] = np.nan; bad.loc[6, "Puanlayici_3"] = "2"
    d["invalid_teaching"] = bad
    d["wide_template"] = pd.DataFrame({"Birim_ID": ["B001", "B002"], "Puanlayici_1": [None, None], "Puanlayici_2": [None, None]})
    d["long_template"] = pd.DataFrame({"Birim_ID": ["B001", "B001"], "Puanlayici_ID": ["R1", "R2"], "Puan": [None, None]})
    d["counts_template"] = pd.DataFrame({"Birim_ID": ["B001", "B002"], "Kategori_1": [None, None], "Kategori_2": [None, None]})
    return d


COLS = {
    "Birim_ID": "Puanlanan birimin benzersiz kimliği (öğrenci, kompozisyon, hasta vb.); puanlayıcı DEĞİLDİR.",
    "Kompozisyon_ID": "Sıralanan nesnenin (kompozisyon) kimliği.",
    "Puanlayici_ID": "Uzun biçimde puanı veren puanlayıcının kimliği.", "Puan": "Uzun biçimde tek birim–puanlayıcı puanı.",
    "Yontem_A": "Birinci yöntem/puanlayıcının nicel ölçümü.", "Yontem_B": "İkinci yöntem/puanlayıcının aynı birim için nicel ölçümü.",
}
DESCRIBE = lambda c: COLS.get(c, "Bu puanlayıcının birime verdiği puan." if c.startswith("Puanlayici_") else
                              "Bu kategoriyi seçen puanlayıcı sayısı." if c.startswith("Kategori_") else "")

# method -> (dataset, layout, extra configuration, Turkish scenario)
PLAN = {
    "cohen_kappa": ("cohen_kappa_nominal", "wide", {}, "İki puanlayıcı 40 öğrenci yanıtını üç nominal kategoriye ayırıyor; prevalans dengesiz."),
    "cohen_kappa_linear": ("ordinal_two_raters", "wide", {}, "İki puanlayıcı 1–5 sıralı rubrik; komşu kategori farkları kısmen tolere edilir."),
    "cohen_kappa_quadratic": ("ordinal_two_raters", "wide", {}, "Aynı sıralı veri; büyük farklar karesel olarak cezalandırılır."),
    "fleiss_kappa": ("fleiss_kappa_counts", "counts", {}, "25 birimi altışar puanlayıcı dört kategoriye ayırıyor (sıklık tablosu)."),
    "conger_kappa": ("nominal_multirater", "wide", {}, "Dört sabit puanlayıcı 30 birimi üç kategoriye ayırıyor."),
    "light_kappa": ("nominal_multirater", "wide", {}, "Dört puanlayıcının tüm çiftlerindeki Cohen κ ortalaması."),
    "scott_pi": ("cohen_kappa_nominal", "wide", {}, "İki puanlayıcı, ortak marjinal varsayımıyla."),
    "gwet_ac1": ("nominal_multirater", "wide", {}, "Dört puanlayıcı; prevalansa daha az duyarlı katsayı."),
    "gwet_ac2": ("ordinal_multirater", "wide", {}, "Üç puanlayıcı 1–5 sıralı rubrik; karesel ağırlık."),
    "brennan_prediger": ("nominal_multirater", "wide", {}, "Dört puanlayıcı; şans uyumu kategori sayısından (1/q)."),
    "kripp_nominal": ("krippendorff_alpha_missing", "wide", {}, "Dört puanlayıcı, eksik puanlar var; kategoriler nominal kabul edilir."),
    "kripp_ordinal": ("krippendorff_alpha_missing", "wide", {}, "Aynı eksik veri; puanlar sıralı."),
    "kripp_interval": ("krippendorff_alpha_missing", "wide", {}, "Aynı eksik veri; eşit aralıklı puan."),
    "kripp_ratio": ("krippendorff_alpha_missing", "wide", {}, "Aynı eksik veri; oran düzeyi (pozitif puanlar)."),
    "percent_agreement": ("nominal_multirater", "wide", {}, "Dört puanlayıcı; çift bazlı tam uyum ve tümünün aynı olduğu birim oranı."),
    "pabak": ("binary_two_raters", "wide", {}, "İki puanlayıcı ikili karar; yüksek prevalans (≈%85 '1')."),
    "bangdiwala_b": ("cohen_kappa_nominal", "wide", {}, "İki puanlayıcı; uyum tablosunun köşegen alanı."),
    "icc_1_1": ("icc_two_way", "wide", {}, "20 birim dört puanlayıcı; puanlayıcılar birimler arasında farklı kabul edilir."),
    "icc_1_k": ("icc_two_way", "wide", {}, "Aynı veri; dört puanlayıcı ortalamasının güvenirliği."),
    "icc_2_1": ("icc_two_way", "wide", {}, "Aynı veri; puanlayıcılar rastgele örneklem, mutlak uyum."),
    "icc_2_k": ("icc_two_way", "wide", {}, "Aynı veri; rastgele puanlayıcılar, ortalama puan."),
    "icc_3_1": ("icc_two_way", "wide", {}, "Aynı veri; yalnız bu puanlayıcılar (sabit), tutarlılık."),
    "icc_3_k": ("icc_two_way", "wide", {}, "Aynı veri; sabit puanlayıcılar, ortalama puan."),
    "kendall_w": ("kendall_w_rankings", "wide", {}, "Beş puanlayıcı on kompozisyonu puanlıyor; puanlar sıraya çevrilir (bağlar var)."),
    "spearman": ("bland_altman_paired", "wide", {}, "İki yöntemin ölçümleri arasındaki sıra ilişkisi."),
    "pearson": ("bland_altman_paired", "wide", {}, "İki yöntemin ölçümleri arasındaki doğrusal ilişki."),
    "mcnemar": ("binary_two_raters", "wide", {}, "İkinci puanlayıcının '1' kararını daha sık vermesi (sistematik fark)."),
    "stuart_maxwell": ("ordinal_two_raters", "wide", {}, "İki puanlayıcının kategori marjinallerinin homojenliği."),
    "bhapkar": ("ordinal_two_raters", "wide", {}, "Aynı tablo; Bhapkar marjinal homojenlik testi."),
    "bland_altman": ("bland_altman_paired", "wide", {}, "Yöntem B ortalama +1.5 birim yüksek ölçüyor."),
}


def build():
    from core.rater_reliability import METHODS
    data = datasets(); files = {}
    for name, frame in data.items():
        path = HERE / f"{name}.csv"; frame.to_csv(path, index=False)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pd.ExcelWriter(HERE / "rater_reliability_examples.xlsx") as w:
        pd.DataFrame([{"Sayfa": n, "Dosya": f"{n}.csv", "Satır": len(f), "Açıklama": ", ".join(f"{c}: {DESCRIBE(c)}" for c in f.columns)}
                      for n, f in data.items()]).to_excel(w, sheet_name="Aciklama", index=False)
        for name, frame in data.items():
            frame.to_excel(w, sheet_name=name[:31], index=False)
    old = {}
    if (HERE / "registry.json").exists():
        old = {e["id"]: e for e in json.loads((HERE / "registry.json").read_text(encoding="utf-8"))["examples"]}
    examples = []
    for method, (ds, layout, extra, scenario) in PLAN.items():
        m = METHODS[method]; frame = data[ds]; idc = frame.columns[0]
        cfg = {"layout": layout, "id_col": idc}
        cfg.update({"category_cols": list(frame.columns[1:])} if layout == "counts" else {"rater_cols": list(frame.columns[1:])})
        ex_id = m["example"]
        examples.append({"id": ex_id, "method": method, "file": f"{ds}.csv", "sheet": ds[:31], "sha256": files[ds],
                         "config": cfg | extra, "scenario": scenario, "level": m["level"], "raters": m["raters"],
                         "layout": m["layout"], "missing": m["missing"],
                         "columns": {c: DESCRIBE(c) for c in frame.columns},
                         "expected": old.get(ex_id, {}).get("expected", {})})
    teaching = {"invalid": {"file": "invalid_teaching.csv", "sha256": files["invalid_teaching"],
                            "problems": ["B003 kimliği iki kez", "Puanlayici_2 boş hücre", "Puanlayici_3 '2' metin olarak kodlanmış"]},
                "long": {"file": "long_format_example.csv", "sha256": files["long_format_example"],
                         "config": {"layout": "long", "unit_col": "Birim_ID", "rater_col": "Puanlayici_ID", "score_col": "Puan"}},
                "templates": {k: f"{k}.csv" for k in ("wide_template", "long_template", "counts_template")}}
    out = {"seed": SEED, "workbook": "rater_reliability_examples.xlsx", "examples": examples, "teaching": teaching}
    (HERE / "registry.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    r = build(); print(len(r["examples"]), "examples;", len({e["file"] for e in r["examples"]}), "datasets")
