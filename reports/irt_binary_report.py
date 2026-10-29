"""Binary IRT (Rasch, 1PL–4PL) figures and report bundle from Python mirt results only.

Restores the scientific output types of the removed B13 report on the library route: ICC, IIC, TIF,
SEM, theta distribution, person–item (Wright) map, parameter distribution, asymptotes, 300 dpi PNGs,
item-parameter SE/Wald CI and an Excel/ZIP bundle. Every number comes from the fitted Python mirt
report (curves, information, parameters) and the person EAP table; nothing is re-estimated.
"""
import io
import zipfile

import numpy as np
import pandas as pd

KINDS = ("ICC", "IIC", "TIF", "SEM", "Theta dağılımı", "Wright haritası", "Parametre dağılımı", "Asimptotlar")
FREE = {"Rasch": "b", "1PL": "b", "2PL": "ab", "3PL": "abc", "4PL": "abcd"}


def kinds(model):
    return KINDS if model in ("3PL", "4PL") else KINDS[:-1]


def _theta(persons):
    column = next(c for c in persons.columns if c in ("Yetenek (θ)", "θ"))
    theta = np.asarray(persons[column], float)
    return theta[np.isfinite(theta)]


def png_bytes(report, persons, kind, selected=None):
    """Deterministic 300 dpi PNG of one chart; raises for a chart the model does not define."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    model = report.model
    if kind not in kinds(model):
        raise ValueError(f"{kind} bu model için tanımlı değil.")
    items = report.parameters
    names = items["Madde"].astype(str).tolist()
    info = report.information
    fig = Figure(figsize=(9, 5.5), layout="constrained")
    ax = fig.add_subplot(111)
    if kind in ("ICC", "IIC"):
        chosen = list(selected or names[:8])
        if set(chosen) - set(names):
            raise ValueError("Bilinmeyen madde.")
        for name in chosen:
            if kind == "ICC":
                curve = report.curves[(report.curves.Madde == name) & (report.curves.Kategori == "1")]
                ax.plot(curve["Theta"], curve["Olasılık"], label=name)
            else:
                ax.plot(info["θ"], info[f"I_{name}"], label=name)
        ax.legend(fontsize=7, ncol=2)
        ax.set_xlabel("Theta")
        ax.set_ylabel("P(X=1 | θ)" if kind == "ICC" else "Madde bilgisi")
    elif kind == "TIF":
        ax.plot(info["θ"], info["Test bilgisi"]); ax.set_xlabel("Theta"); ax.set_ylabel("Test bilgisi")
    elif kind == "SEM":
        ax.plot(info["θ"], info["SEM(θ)"]); ax.set_xlabel("Theta"); ax.set_ylabel("SEM(θ) = 1/√I(θ)")
    elif kind == "Theta dağılımı":
        ax.hist(_theta(persons), bins=30); ax.set_xlabel("θ (EAP)"); ax.set_ylabel("Kişi sayısı")
    elif kind == "Wright haritası":
        ax.hist(_theta(persons), bins=30, orientation="horizontal", color="#4C78A8")
        ax.set_xlabel("Kişi sayısı"); ax.set_ylabel("θ (EAP) ve madde b")
        for i, (name, b) in enumerate(zip(names, items["b"])):
            ax.annotate(name, xy=(0, float(b)), xytext=(-8 - (i % 3) * 10, 0),
                        textcoords="offset points", fontsize=6, ha="right")
            ax.axhline(float(b), color="#E45756", lw=.4, alpha=.6)
    elif kind == "Parametre dağılımı":
        free = FREE[model]
        ax.boxplot([np.asarray(items[p], float) for p in free])
        ax.set_xticks(range(1, len(free) + 1), list(free))
    else:
        ax.bar(names, items["c"], label="c (alt asimptot)")
        if model == "4PL":
            ax.scatter(names, items["d"], color="#E45756", label="d (üst asimptot)")
        ax.set_ylim(0, 1); ax.legend(); ax.tick_params(axis="x", rotation=90)
    ax.set_title(f"{model} — {kind} (Python mirt)")
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=300)
    return stream.getvalue()


def method_text(report, inference_reason):
    meta = report.metadata
    return ("PsikoAnaliz — İkili MTK rapor paketi\n"
            f"Model: {report.model}\nMotor: Python mirt 1.2.0 (tek kestirim altyapısı)\n"
            f"N: {meta.get('N')} | Madde: {meta.get('Madde')} | Yakınsama: {meta.get('Yakınsama')}\n"
            "Ölçek: θ ~ N(0,1); P = c + (d − c)·logistic(a(θ − b))\n"
            "Kişi puanları: EAP ve posterior SS; SEM(θ) = 1/√I_test(θ).\n"
            "Parametre SH: kütüphane Oakes gözlenen bilgisi; sınırdaki parametrelerde Wald GA yok.\n"
            + (f"SH durumu: {inference_reason}\n" if inference_reason else "")
            + "Wright haritası (2PL–4PL): b her modelde aynı θ ölçeğindedir; yalnız 1PL/2PL'de P = .5 noktasıdır.\n")


def bundle(report, persons, inference, inference_reason, itemfit=None):
    """(xlsx bytes, zip bytes) with every table, the method text and all applicable 300 dpi PNGs."""
    from reports.irt_b14_report import safe_frame
    tables = {"Model_Ozeti": pd.DataFrame([report.metadata]), "Madde_Parametreleri": report.parameters,
              "Kisi_Yetenekleri": persons, "Bilgi_SEM": report.information}
    if inference is not None:
        tables["Parametre_SH_GA"] = inference
    if itemfit is not None and not itemfit.empty:
        tables["Madde_Uyumu"] = itemfit
    xlsx = io.BytesIO()
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for name, frame in tables.items():
            safe_frame(frame).to_excel(writer, sheet_name=name[:31], index=False)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{report.model}_rapor.xlsx", xlsx.getvalue())
        z.writestr(f"{report.model}_yontem.txt", method_text(report, inference_reason))
        for name, frame in tables.items():
            z.writestr(f"tablolar/{name}.csv", safe_frame(frame).to_csv(index=False))
        for kind in kinds(report.model):
            z.writestr(f"grafikler/{kind}.png", png_bytes(report, persons, kind))
    return xlsx.getvalue(), archive.getvalue()
