"""Free-hosting R/TAM trial: run the exact install command behind the page's 'TAM 4.3-25 kurulumunu dene' button as the
current (non-root) user, then check the engine. Writes a JSON report. No estimate is produced here."""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install_command():
    s = (ROOT / "ui/mfrm_view.py").read_text(encoding="utf-8")
    i = s.index("command = ("); j = s.index(")\n", s.index("stopifnot", i))
    return eval(s[i + len("command = "):j + 1])          # the literal string used by the button (no other source)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "trial_report.json"); t0 = time.monotonic()
    proc = subprocess.run(["Rscript", "-e", install_command()], capture_output=True, text=True, timeout=3600)
    sys.path.insert(0, str(ROOT)); import core.mfrm as M
    ok, why = M.engine_available()
    rep = {"install_returncode": proc.returncode, "install_seconds": round(time.monotonic() - t0, 1),
           "engine_available": ok, "engine_message": why, "log_tail": (proc.stderr or proc.stdout)[-4000:]}
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False)); print(json.dumps({k: v for k, v in rep.items() if k != "log_tail"}))
    sys.exit(0 if ok else 1)
