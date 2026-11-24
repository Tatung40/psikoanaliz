"""Real-browser, mobile viewport (390x844): one "Python mirt ile hesapla" click on 2PL yields results, report bundle,
academic Word and Python code; routine notices sit in one collapsed technical section; no horizontal overflow.
python validation/browser_e2e_irt_single_click_mobile.py --output /tmp/irt-single-click-mobile   -> result.json
"""
import argparse, json, sys, traceback, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from validation.browser_e2e_polytomous_dif import Server, download, nav, upload, wait_idle

TITLE = "Teknik ve Bilimsel Doğrulama Ayrıntıları"


def data():
    r = np.random.default_rng(20261124); th = r.normal(size=400)
    return pd.DataFrame((r.random((400, 8)) < 1 / (1 + np.exp(-(th[:, None] - np.linspace(-1, 1, 8))))).astype(int),
                        columns=[f"V{j + 1}" for j in range(8)])


def run(out):
    from playwright.sync_api import expect, sync_playwright
    out.mkdir(parents=True, exist_ok=True); csv = out / "binary_2pl.csv"; data().to_csv(csv, index=False)
    res = {"status": "FAIL", "steps": [], "viewport": [390, 844]}; step = lambda s: res["steps"].append(s)
    with open(out / "server.log", "w") as log, Server(log) as srv, sync_playwright() as p:
        b = p.chromium.launch(args=["--no-sandbox"])
        page = b.new_page(accept_downloads=True, viewport=dict(width=390, height=844), is_mobile=True, has_touch=True)
        errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            upload(page, srv.port, csv); step("uploaded")
            nav(page, "📐 2PL Modeli"); wait_idle(page)
            if page.locator("[data-testid=stSidebar][aria-expanded=true], [data-testid=stSidebar][data-collapsed=false]").count():
                page.locator("[data-testid=stSidebarCollapseButton] button, [data-testid=stSidebarHeader] button").first.click()
                wait_idle(page); step("sidebar closed as on a phone")
            page.get_by_role("button", name="Python mirt ile hesapla").click(); wait_idle(page)   # the only click
            page.get_by_text("Kestirim yakınsadı").wait_for(timeout=180_000); wait_idle(page); step("one click")
            for name in ("Excel — tüm tablolar", "ZIP — rapor paketi", "2PL Akademik Rapor",
                         "Birey yetenekleri — Excel"):
                expect(page.get_by_role("button", name=name, exact=True)).to_have_count(1)
            step("bundle, Word and exports present without a second click")
            with zipfile.ZipFile(download(page, "ZIP — rapor paketi", out)) as archive:
                assert any(n.endswith(".png") for n in archive.namelist()), archive.namelist()
            section = page.locator("[data-testid=stExpander]").filter(has_text=TITLE)
            expect(section).to_have_count(1)
            assert section.locator("details").get_attribute("open") is None, "technical section must start collapsed"
            body = page.evaluate("""() => { const m = document.querySelector('[data-testid=stMain]').cloneNode(true);
                m.querySelectorAll('[data-testid=stCode], [data-testid=stJson]').forEach(e => e.remove()); return m.innerText; }""")
            hits = [w for w in ("UNSUPPORTED", "PythonMirtResult", "eski Word ve Python kodu") if w in body]
            res["visible_hits"] = [body[max(0, body.find(w) - 200): body.find(w) + 120] for w in hits]
            assert "UNSUPPORTED" not in body and "PythonMirtResult" not in body and "eski Word ve Python kodu" not in body
            step("routine notices collapsed; no internal class names or stale-code notice")
            page.get_by_role("tab", name="</> Python Kodları").click(); wait_idle(page)
            code = download(page, ".py indir", out).read_text()
            assert "fit_python_mirt" in code and "library_fit_diagnostics" in code
            step("python code downloadable")
            overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            assert overflow <= 1, f"horizontal overflow {overflow}px"
            assert not errors, errors
            page.screenshot(path=str(out / "mobile_single_click.png"), full_page=True); res["status"] = "PASS"
        except Exception:
            res["error"] = traceback.format_exc()[-1500:]; page.screenshot(path=str(out / "failure.png"), full_page=True)
        finally:
            b.close()
    (out / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False)); print(json.dumps(res, ensure_ascii=False)[:800])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--output", default="/tmp/irt-single-click-mobile")
    sys.exit(0 if run(Path(ap.parse_args().output))["status"] == "PASS" else 1)
