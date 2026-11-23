"""Real-browser round trip: analysis (Python mirt 2PL) -> missing-data advisor -> Kayıp Veri İşlemleri -> apply ->
return -> run -> Excel export carries missing-data provenance.
python validation/browser_e2e_missing_roundtrip.py --output /tmp/missing-roundtrip   -> result.json
"""
import argparse, io, json, sys, traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from validation.browser_e2e_polytomous_dif import Server, download, nav, upload, wait_idle


def data():
    r = np.random.default_rng(20261123); th = r.normal(size=400)
    x = pd.DataFrame((r.random((400, 8)) < 1 / (1 + np.exp(-(th[:, None] - np.linspace(-1, 1, 8))))).astype(float),
                     columns=[f"V{j + 1}" for j in range(8)])
    x.loc[[2, 5], "V4"] = np.nan; return x


def run(out):
    from playwright.sync_api import sync_playwright
    out.mkdir(parents=True, exist_ok=True); csv = out / "missing_v4.csv"; data().to_csv(csv, index=False)
    res = {"status": "FAIL", "steps": []}; step = lambda s: res["steps"].append(s)
    with open(out / "server.log", "w") as log, Server(log) as srv, sync_playwright() as p:
        b = p.chromium.launch(args=["--no-sandbox"]); page = b.new_page(accept_downloads=True)
        try:
            upload(page, srv.port, csv); step("uploaded")
            nav(page, "📐 2PL Modeli"); wait_idle(page)
            page.get_by_text("Kayıp veri danışmanı").first.wait_for(timeout=60_000); step("advice shown on 2PL")
            page.get_by_role("button", name="Kayıp Veri İşlemleri'ne git").first.click()
            page.get_by_text("analizinden yönlendirildiniz").wait_for(timeout=60_000); wait_idle(page); step("missing page reached")
            page.get_by_role("button", name="Seçili yöntemi uygula").click(); wait_idle(page)
            page.get_by_text("İşlem tamamlandı").wait_for(timeout=60_000); step("listwise applied")
            page.get_by_role("button", name="Sonucu analizde kullan ve analiz ekranına dön").click()
            page.get_by_text("Kayıp veri işlemi uygulandı").wait_for(timeout=60_000)
            page.get_by_text("Python mirt Kütüphanesi — 2PL").wait_for(timeout=60_000); wait_idle(page); step("returned to 2PL with banner")
            page.get_by_role("button", name="Python mirt ile hesapla").click(); wait_idle(page)
            page.get_by_text("Kestirim yakınsadı").wait_for(timeout=180_000); step("2PL ran")
            xlsx = download(page, "Birey yetenekleri — Excel", out); sheets = pd.read_excel(xlsx, sheet_name=None)
            persons = [v for k, v in sheets.items() if k != "Kayıp veri işlemi"][0]
            assert len(persons) == 398, len(persons)                                  # analysed N after listwise
            prov = sheets.get("Kayıp veri işlemi")
            assert prov is not None and "Tam kayıt" in " ".join(map(str, prov.columns)), list(sheets)
            step("Excel export carries provenance"); res["status"] = "PASS"
        except Exception:
            res["error"] = traceback.format_exc()[-1500:]; page.screenshot(path=str(out / "failure.png"), full_page=True)
        finally:
            b.close()
    (out / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(json.dumps(res, ensure_ascii=False)[:600])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--output", default="/tmp/missing-roundtrip")
    sys.exit(0 if run(Path(ap.parse_args().output))["status"] == "PASS" else 1)
