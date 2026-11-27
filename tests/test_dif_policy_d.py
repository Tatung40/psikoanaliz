"""Policy D safeguards: shared notice, screening label in exports, SIBTEST limitation, no unsupported cutoff."""
import json
from pathlib import Path
from core.dif_classical import OBSERVED_SCORE_NOTICE, SCREENING_STATUS
from core.dif_center import LIMITATIONS


def test_notice_content():
    for s in ("impact", "ayırt edici", "saflaştırma", "çıkarımsal geçerlik", "TARAMA"):
        assert s in OBSERVED_SCORE_NOTICE
    assert "TARAMA" in SCREENING_STATUS and "sertifikalı değil" in SCREENING_STATUS
    assert "gerçek-puan regresyon" in LIMITATIONS and "bağımsız algoritma değildir" in LIMITATIONS
    assert "0.30" not in OBSERVED_SCORE_NOTICE and ".30" not in OBSERVED_SCORE_NOTICE


def test_ui_exports_carry_label_and_notice():
    src = Path("ui/dif_classical_view.py").read_text(encoding="utf-8")
    assert '"Bilimsel durum (p)": SCREENING_STATUS' in src and '("Bilimsel uyarı", OBSERVED_SCORE_NOTICE)' in src
    assert src.index('"Bilimsel durum (p)"') < src.index('to_excel(w, sheet_name="DIF"')


def test_evidence_counts_are_23_plus_3_empty():
    rep = json.loads(Path("validation/evidence/dif_difr/COMPARISON_v2.json").read_text())
    empty = [r for r in rep["rows"] if r["fixture"] == "dif2pl_none" and r["statistic"].startswith("sib_")]
    assert rep["agree_all"] and len(rep["rows"]) == 26 and len(empty) == 3 and len(rep["rows"]) - len(empty) == 23
