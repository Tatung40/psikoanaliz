"""Policy D notices (texts) and evidence counts; protected DIF sources must stay byte-identical."""
import json
from pathlib import Path
from core.dif_policy import OBSERVED_SCORE_NOTICE, SCREENING_STATUS, SIBTEST_NOTE


def test_notice_content():
    for s in ("impact", "ayırt edici", "saflaştırma", "çıkarımsal geçerlik", "TARAMA"):
        assert s in OBSERVED_SCORE_NOTICE
    assert "sertifikalı değil" in SCREENING_STATUS and "bağımsız algoritma değildir" in SIBTEST_NOTE
    assert ".30" not in OBSERVED_SCORE_NOTICE


def test_evidence_counts_are_23_plus_3_empty():
    rep = json.loads(Path("validation/evidence/dif_difr/COMPARISON_v2.json").read_text())
    empty = [r for r in rep["rows"] if r["fixture"] == "dif2pl_none" and r["statistic"].startswith("sib_")]
    assert rep["agree_all"] and len(rep["rows"]) - len(empty) == 23 and len(empty) == 3
