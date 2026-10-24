"""Real Chromium/CDP upload-fit-download smoke; NOT model certification.

Uses installed Chromium + websockets, no Playwright/browser download. Artifacts
contain synthetic data only. Run with the project Python and --output DIR.
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import socket
from urllib.request import urlopen
import numpy as np
import pandas as pd
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(output):
    from validation.known_truth_irt_recovery import simulate
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    x, _ = simulate("2PL", n_persons=400, n_items=8, seed=20261024)
    source = output / "synthetic_binary.csv"
    pd.DataFrame(x, columns=[f"M{i+1}" for i in range(8)]).to_csv(source, index=False)
    processes = []
    # Never attach to a different validation's browser or Streamlit session.
    for port in (8523, 9223):
        with socket.socket() as probe: probe.bind(("127.0.0.1", port))
    with tempfile.TemporaryDirectory(prefix="irt-browser-") as profile, open(output / "runtime.log", "w") as log:
        try:
            processes.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8523", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"], cwd=ROOT, stdout=log, stderr=log))
            processes.append(subprocess.Popen(["chromium", "--headless", "--no-sandbox", "--disable-dev-shm-usage", "--remote-debugging-port=9223", "--remote-allow-origins=*", f"--user-data-dir={profile}", "about:blank"], stdout=log, stderr=log))
            deadline = time.monotonic() + 45
            while True:
                try:
                    urlopen("http://127.0.0.1:8523/_stcore/health", timeout=1).read()
                    tabs = json.load(urlopen("http://127.0.0.1:9223/json", timeout=1)); break
                except OSError:
                    if time.monotonic() > deadline: raise TimeoutError("Browser/app startup failed")
                    time.sleep(.2)
            with connect(next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page"), max_size=None) as ws:
                serial = 0
                def call(method, params=None):
                    nonlocal serial
                    serial += 1; ident = serial
                    ws.send(json.dumps(dict(id=ident, method=method, params=params or {})))
                    while True:
                        result = json.loads(ws.recv(timeout=60))
                        if result.get("id") == ident:
                            if "error" in result: raise RuntimeError(result["error"])
                            return result.get("result", {})
                def evaluate(expression):
                    result = call("Runtime.evaluate", dict(expression=expression, returnByValue=True))
                    if "exceptionDetails" in result: raise RuntimeError(result["exceptionDetails"])
                    return result.get("result", {}).get("value")
                def wait(expression, seconds=60):
                    end = time.monotonic() + seconds
                    while not evaluate(expression):
                        if time.monotonic() > end:
                            (output / "failure_dom.txt").write_text(evaluate("document.body.innerText"), encoding="utf-8")
                            raise TimeoutError(expression)
                        time.sleep(.2)
                def click(text):
                    expression = "(()=>{const e=[...document.querySelectorAll('button,a')].find(e=>e.textContent.includes("+json.dumps(text)+"));if(!e)return false;e.click();return true})()"
                    wait(expression)
                call("Browser.setDownloadBehavior", dict(behavior="allow", downloadPath=str(output)))
                call("Page.navigate", dict(url="http://127.0.0.1:8523"))
                wait("document.body.innerText.includes('Veri Yükleme')")
                click("Veri Yükleme")
                wait("!!document.querySelector('input[type=file]')")
                doc = call("DOM.getDocument")
                node = call("DOM.querySelector", dict(nodeId=doc["root"]["nodeId"], selector="input[type=file]"))
                call("DOM.setFileInputFiles", dict(nodeId=node["nodeId"], files=[str(source)]))
                wait("document.body.innerText.includes('400 satır')")
                records = []
                for model in ("1PL", "2PL", "3PL", "4PL"):
                    click(f"{model} Modeli")
                    wait("document.body.innerText.includes("+json.dumps("Etkin model: " + model)+")")
                    time.sleep(.5)  # Let the model-specific widget tree finish replacing.
                    click("Python mirt ile analiz")
                    wait("document.body.innerText.includes("+json.dumps("Model: " + model)+")", 120)
                    wait("document.querySelectorAll('[data-testid=stPlotlyChart]').length>0")
                    before = set(output.glob("*"))
                    click("Madde parametreleri — CSV")
                    click("Madde parametreleri — Excel")
                    end = time.monotonic() + 30
                    while True:
                        new = [p for p in set(output.glob("*"))-before if p.suffix in (".csv", ".xlsx")]
                        if len(new) == 2: break
                        if time.monotonic() > end: raise TimeoutError("Download did not finish")
                        time.sleep(.2)
                    for path in new:
                        frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
                        assert len(frame) == 8 and {"a", "b"} <= set(frame)
                        assert np.isfinite(frame[["a", "b"]].to_numpy()).all()
                    records.append(dict(model=model, uploaded_rows=400, items=8, plots=True, downloads=[p.name for p in new]))
                png = call("Page.captureScreenshot", dict(format="png"))["data"]
                (output / "browser.png").write_bytes(base64.b64decode(png))
                report = dict(status="PASS", browser="Real Chromium CDP", scope="CSV upload, 1PL/2PL/3PL/4PL real library fits, plots, actual CSV/XLSX downloads and contents; not polytomous or DIF browser coverage or recovery certification", runs=records)
                (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                return report
        finally:
            for process in reversed(processes):
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), ensure_ascii=False))
