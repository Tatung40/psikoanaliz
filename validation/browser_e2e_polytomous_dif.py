"""Real-browser (Playwright + installed Chromium) end-to-end checks for polytomous IRT, RSM and DIF.

Each flow starts a fresh Streamlit server, uploads an independently simulated CSV through the real
file input, selects the model in the UI, runs the analysis, checks that charts render, downloads the
CSV/XLSX through the browser, reopens the files and compares them with values computed directly in
this process from the same data (deterministic estimators). Synthetic data only; no respondent IDs.
Run: PYTHONPATH=. python validation/browser_e2e_polytomous_dif.py --output DIR
"""
import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


class Server:
    def __init__(self, log):
        self.port = free_port(); self.log = log
    def __enter__(self):
        self.proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", f"--server.port={self.port}",
                                      "--server.headless=true", "--server.fileWatcherType=none",
                                      "--browser.gatherUsageStats=false"], cwd=ROOT, stdout=self.log, stderr=self.log)
        end = time.monotonic() + 60
        while True:
            try:
                urlopen(f"http://127.0.0.1:{self.port}/_stcore/health", timeout=1).read(); return self
            except OSError:
                if time.monotonic() > end:
                    raise TimeoutError("Streamlit did not start")
                time.sleep(.3)
    def __exit__(self, *a):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def upload(page, port, csv):
    page.goto(f"http://127.0.0.1:{port}")
    page.get_by_role("button", name="📁 Veri Yükleme").click()
    page.locator("input[type=file]").first.set_input_files(str(csv))
    page.get_by_text("başarıyla yüklendi").wait_for(timeout=60_000)


def nav(page, label):
    page.get_by_role("button", name=label).first.click()
    page.wait_for_timeout(800)


def wait_idle(page, timeout=180_000):
    page.wait_for_function("() => !document.querySelector('[data-testid=stStatusWidget]')", timeout=timeout)


def download(page, label, outdir):
    wait_idle(page)
    with page.expect_download(timeout=60_000) as d:
        page.get_by_role("button", name=label).first.click()
    target = Path(outdir) / d.value.suggested_filename
    target.parent.mkdir(parents=True, exist_ok=True)
    d.value.save_as(target)
    return target


def ordinal_data(seed=1, n=500, k=8, C=4):
    from validation.polytomous_mc_study import probs
    from scipy.special import expit
    rng = np.random.default_rng(seed); th = rng.standard_normal(n)
    a = np.linspace(.8, 1.8, k); b = np.linspace(-1, 1, k)[:, None] + np.linspace(-1, 1, C - 1)[None, :]
    cum = np.concatenate([np.ones((n, k, 1)), expit(a[None, :, None] * (th[:, None, None] - b[None])), np.zeros((n, k, 1))], 2)
    P = cum[..., :-1] - cum[..., 1:]
    x = (rng.random((n, k))[..., None] > np.cumsum(P, 2)).sum(2)
    return pd.DataFrame(x, columns=[f"I{j + 1}" for j in range(k)])


def flow_polytomous(browser, outdir, log, models=("GRM", "PCM", "GPCM", "NRM")):
    import mirt
    data = ordinal_data(); csv = outdir / "poly_ordinal.csv"; data.to_csv(csv, index=False)
    nominal = data.replace({0: "A", 1: "B", 2: "C", 3: "D"}); ncsv = outdir / "poly_nominal.csv"
    nominal.to_csv(ncsv, index=False)
    records = []
    for model, src, frame in (("GRM", csv, data), ("PCM", csv, data), ("GPCM", csv, data), ("NRM", ncsv, nominal)):
        if model not in models:
            continue
        rec = {"flow": "B14 polytomous", "model": model, "status": "FAILED"}
        try:
            with Server(log) as srv:
                page = browser.new_page(accept_downloads=True)
                upload(page, srv.port, src)
                page.get_by_text("Python mirt Kütüphanesi — MTK").first.click()  # collapsed sidebar expander
                nav(page, "🧬 B14 Python mirt Çok Kategorili Rapor")
                page.get_by_text(model, exact=True).first.click(); wait_idle(page)
                if model == "NRM":
                    box = page.get_by_label("Nominal seçenekler (virgülle)")
                    box.fill("A,B,C,D"); box.press("Enter"); wait_idle(page)
                page.get_by_role("button", name="Python mirt ile B14 pilot analizi").click()
                page.get_by_role("button", name="Madde parametrelerini CSV indir").wait_for(timeout=180_000)
                wait_idle(page)
                charts = page.locator("[data-testid=stVegaLiteChart], [data-testid=stArrowVegaLiteChart], .vega-embed").count()
                path = download(page, "Madde parametrelerini CSV indir", outdir / model)
                got = pd.read_csv(path)
                from core.irt_python_mirt_adapter import fit_python_mirt  # the application's engine path
                ref_fit = fit_python_mirt(np.asarray(frame.replace({"A": 0, "B": 1, "C": 2, "D": 3}), int), model=model,
                                      n_categories=4).fit
                ref = ref_fit.coef_with_se().reset_index()
                num = [c for c in ref.columns if pd.api.types.is_numeric_dtype(ref[c])]
                diff = float(np.nanmax(np.abs(got[num].to_numpy(float) - ref[num].to_numpy(float))))
                ppath = download(page, "Kişi yeteneklerini CSV indir", outdir / model)
                xpath = download(page, "B14 tüm tablolar Excel", outdir / model)
                coded = np.asarray(frame.replace({"A": 0, "B": 1, "C": 2, "D": 3}), int)
                theta_ref = np.asarray(mirt.fscores(ref_fit, coded, method="EAP").theta, float).reshape(-1)
                dtheta = float(np.abs(pd.read_csv(ppath)["θ"].to_numpy() - theta_ref).max())
                sheets = pd.read_excel(xpath, sheet_name=None)
                plotly = page.locator("[data-testid=stPlotlyChart]").count()
                rec.update(charts=int(charts), plotly_charts=int(plotly), download=path.name, rows=len(got),
                           max_abs_diff_vs_direct_fit=diff, max_abs_diff_theta=dtheta, xlsx_sheets=list(sheets),
                           status="PASS" if diff < 1e-6 and dtheta < 1e-6 and len(got) == 8 and charts > 0
                           and plotly > 0 and {"Kişi Yetenekleri", "Bilgi Fonksiyonları"} <= set(sheets) else "FAIL")
                page.screenshot(path=str(outdir / f"{model}.png"), full_page=True); page.close()
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        records.append(rec)
    return records


def flow_rsm(browser, outdir, log):
    from core import irt_rsm as R
    x = R.simulate(np.linspace(-1.2, 1.2, 8), np.array([-1., 0., 1.]), 600, 1.1, seed=5)
    data = pd.DataFrame(x, columns=[f"I{j + 1}" for j in range(8)]); csv = outdir / "rsm.csv"; data.to_csv(csv, index=False)
    rec = {"flow": "RSM", "model": "RSM", "status": "FAILED"}
    try:
        with Server(log) as srv:
            page = browser.new_page(accept_downloads=True)
            upload(page, srv.port, csv)
            nav(page, "📏 RSM — Derecelendirme Ölçeği Modeli")
            page.get_by_role("button", name="▶️ RSM analizini başlat").click(); wait_idle(page)
            page.get_by_role("tab", name="📈 Grafikler").click()
            page.locator("[data-testid=stPlotlyChart]").first.wait_for(timeout=60_000)
            charts = page.locator("[data-testid=stPlotlyChart]").count()
            page.get_by_role("tab", name="📦 İndirme").click()
            xlsx = download(page, "XLSX — tüm RSM tabloları", outdir / "rsm")
            pcsv = download(page, "CSV — madde konumları", outdir / "rsm")
            sheets = pd.read_excel(xlsx, sheet_name=None)
            direct = R.fit(data)
            d_x = float(np.abs(sheets["Madde konumları"]["Konum (δ)"].to_numpy() - direct["delta"]).max())
            d_c = float(np.abs(pd.read_csv(pcsv)["Konum (δ)"].to_numpy() - direct["delta"]).max())
            d_t = float(np.abs(sheets["Ortak adımlar"]["Ortak adım (τ)"].to_numpy() - direct["tau"]).max())
            rec.update(charts=int(charts), sheets=list(sheets), max_abs_diff_delta_xlsx=d_x, max_abs_diff_delta_csv=d_c,
                       max_abs_diff_tau=d_t,
                       status="PASS" if max(d_x, d_c, d_t) < 1e-8 and charts >= 4 and "Kişiler" in sheets else "FAIL")
            page.screenshot(path=str(outdir / "RSM.png"), full_page=True); page.close()
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return [rec]


def flow_dif(browser, outdir, log):
    from core.dif_classical import dif_screen
    from scipy.special import expit
    rng = np.random.default_rng(9); n = 800
    g = rng.integers(0, 2, n); th = rng.standard_normal(n)
    b = np.tile(np.linspace(-1, 1, 10), (n, 1)); b[g == 1, 2] += .8
    x = (rng.random((n, 10)) < expit(th[:, None] - b)).astype(int)
    data = pd.DataFrame(x, columns=[f"M{j + 1}" for j in range(10)]); data.insert(0, "grup", np.where(g, "B", "A"))
    csv = outdir / "dif_binary.csv"; data.to_csv(csv, index=False)
    out = []
    for method, label in (("MH", "Mantel–Haenszel / Mantel"), ("LR", "Lojistik regresyon")):
        rec = {"flow": "Observed-score DIF", "model": method, "status": "FAILED"}
        try:
            with Server(log) as srv:
                page = browser.new_page(accept_downloads=True)
                upload(page, srv.port, csv)
                nav(page, "⚖️ Gözlenen Puan DIF (MH / Lojistik)")
                page.get_by_text(label, exact=True).click(); wait_idle(page)
                page.get_by_role("button", name="▶️ DIF analizini çalıştır").click(); wait_idle(page)
                page.locator("[data-testid=stPlotlyChart]").first.wait_for(timeout=60_000)
                path = download(page, "DIF tablosu — CSV", outdir / f"dif_{method}")
                xpath = download(page, "DIF tablosu — XLSX", outdir / f"dif_{method}")
                got = pd.read_csv(path); gx = pd.read_excel(xpath, sheet_name="DIF")
                ref, _ = dif_screen(data.iloc[:, 1:], data["grup"], "A", "B", method=method)
                diff = float(np.nanmax(np.abs(got["p"].to_numpy() - ref["p"].to_numpy())))
                diffx = float(np.nanmax(np.abs(gx["p"].to_numpy() - ref["p"].to_numpy())))
                flagged = got.loc[got["DIF (düzeltilmiş p < alpha)"], "Madde"].tolist()
                rec.update(max_abs_diff_p_csv=diff, max_abs_diff_p_xlsx=diffx, flagged=flagged,
                           status="PASS" if max(diff, diffx) < 1e-10 and "M3" in flagged else "FAIL")
                page.screenshot(path=str(outdir / f"DIF_{method}.png"), full_page=True); page.close()
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        out.append(rec)
    return out


def run(output):
    from playwright.sync_api import sync_playwright
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    records = []
    with open(output / "runtime.log", "w") as log, sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        try:
            for flow in (flow_rsm, flow_dif, flow_polytomous):
                records += flow(browser, output, log)
        finally:
            browser.close()
    report = {"browser": "Playwright Chromium (headless, real Streamlit server)",
              "status": "PASS" if records and all(r["status"] == "PASS" for r in records) else "FAIL",
              "runs": records}
    (output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(ap.parse_args().output), ensure_ascii=False, indent=1))
