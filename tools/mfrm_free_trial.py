"""Free-hosting R/TAM trial: call the same core.mfrm.install_engine() used by the page's install button, as the current
(non-root) user, then report. No estimate is produced here."""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import core.mfrm as M

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "trial_report.json"); t0 = time.monotonic()
    state, log = M.install_engine(timeout=3000)
    ok, why = M.engine_available()
    rep = {"install_state": state, "install_seconds": round(time.monotonic() - t0, 1), "engine_available": ok,
           "engine_message": why, "log_tail": log[-4000:]}
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False)); print(json.dumps({k: v for k, v in rep.items() if k != "log_tail"}))
    sys.exit(0 if ok else 1)
