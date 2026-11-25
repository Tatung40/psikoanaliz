"""Production-like startup smoke test: `streamlit run app.py` in a fresh environment, then a real headless browser
must see the home page render. Prints startup timings, server RSS and the resolved package versions.
python validation/startup_smoke.py [--port 8501] [--budget 120] [--out startup_smoke.json]
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKGS = ("streamlit", "starlette", "uvicorn", "websockets", "pandas", "numpy", "pyarrow", "protobuf", "raschpy", "mirt")


def rss_mb(pid):
    total = 0
    for p in [pid] + [int(c) for c in subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()]:
        try: total += int(next(l for l in open(f"/proc/{p}/status") if l.startswith("VmRSS")).split()[1])
        except (OSError, StopIteration): pass
    return total // 1024


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", default="8599"); ap.add_argument("--budget", type=float, default=120)
    ap.add_argument("--out", default="startup_smoke.json"); a = ap.parse_args()
    out = {"python": sys.version.split()[0], "versions": {}}
    for p in PKGS:
        try: out["versions"][p] = metadata.version(p)
        except metadata.PackageNotFoundError: out["versions"][p] = None
    log = open(Path(a.out).with_suffix(".log"), "w"); t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", a.port, "--server.headless", "true"],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while True:
            if proc.poll() is not None: raise SystemExit(f"server exited with {proc.returncode}")
            if time.time() - t0 > a.budget: raise SystemExit("health endpoint not ready within budget")
            try: urllib.request.urlopen(f"http://127.0.0.1:{a.port}/_stcore/health", timeout=2); break
            except OSError: time.sleep(0.25)
        out["health_s"] = round(time.time() - t0, 2)
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch(); page = b.new_page(); errors = []
            page.on("pageerror", lambda e: errors.append(str(e)[:200]))
            t1 = time.time(); page.goto(f"http://127.0.0.1:{a.port}/", timeout=a.budget * 1000)
            page.wait_for_selector("text=PsikoAnaliz", timeout=a.budget * 1000); out["first_content_s"] = round(time.time() - t1, 2)
            page.wait_for_selector("[data-testid=stSidebar] button", timeout=a.budget * 1000)
            page.wait_for_function("!document.querySelector('[data-testid=stStatusWidget]')", timeout=a.budget * 1000)
            out["script_done_s"] = round(time.time() - t1, 2)
            out["exceptions_on_page"] = page.locator("[data-testid=stException]").count(); out["page_errors"] = errors
            b.close()
        out["server_rss_mb"] = rss_mb(proc.pid)
        out["ok"] = out["exceptions_on_page"] == 0 and out["script_done_s"] <= a.budget and out["server_rss_mb"] < 1024
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        Path(a.out).write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
    if not out.get("ok"): raise SystemExit("startup smoke FAILED")


if __name__ == "__main__":
    main()
