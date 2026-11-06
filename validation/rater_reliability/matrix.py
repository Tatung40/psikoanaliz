"""Method x output certification matrix from evidence.json (no recomputation). Writes CERTIFICATION_MATRIX.md."""
import json
from pathlib import Path

import core.rater_reliability as R

HERE = Path(__file__).resolve().parent


def matrix():
    ev = json.loads((HERE / "evidence.json").read_text(encoding="utf-8"))["status"]; rows = []
    for m, meta in R.METHODS.items():
        s = ev.get(m, {}); ok, bad = set(s.get("validated_outputs", [])), set(s.get("failed_outputs", []))
        for o in meta["outputs"]:
            st = "NUMERICALLY VALIDATED" if o in ok else "WITHHELD (referans uyuşmazlığı)" if o in bad else "NO EVIDENCE (gizli)"
            rows.append(dict(method=m, output=o, status=st, references="; ".join(s.get("references", []))))
    return rows


def summary(rows):
    by = {}
    for r in rows: by.setdefault(r["method"], []).append(r["status"])
    full = sum(all(x == "NUMERICALLY VALIDATED" for x in v) for v in by.values())
    return dict(methods=len(by), certified=0, partial=len(by) - full, fully_validated_outputs=full, not_certified=len(by))


if __name__ == "__main__":
    rows = matrix(); s = summary(rows)
    lines = ["# Puanlayıcı güvenirliği — yöntem × çıktı matrisi", "",
             "`evidence.json` kaydından üretildi (`matrix.py`), yeniden hesaplama yapılmadı. Doğrulama yalnız örnek veriler için geçerli;",
             f"**CERTIFIED: {s['certified']}**. Tüm çıktıları doğrulanmış yöntem: {s['fully_validated_outputs']}/{s['methods']}. Kısmi: {s['partial']}.",
             f"NOT CERTIFIED (genel kapsam): {s['not_certified']}. Kanıtsız ya da uyuşmayan çıktı ekranda ve dışa aktarımlarda gösterilmez.", "",
             "| Yöntem | Çıktı | Durum | Referanslar |", "|---|---|---|---|"]
    lines += [f"| {R.METHODS[r['method']]['name']} | {r['output']} | {r['status']} | {r['references']} |" for r in rows]
    (HERE / "CERTIFICATION_MATRIX.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); print(s)
    print([ (r['method'], r['output'], r['status']) for r in rows if r['status'] != 'NUMERICALLY VALIDATED'])
