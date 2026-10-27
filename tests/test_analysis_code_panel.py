"""Code panel must preserve settings without embedding respondent data."""
import ast
from ui.analysis_code_panel import python_mirt_script


def test_mirt_script_compiles_and_preserves_model():
    settings = dict(model="2PL", items=["M01", "M02"], person="(satır indeksi)",
                    dims=2, efa=False, correlated="full", pattern=[[1, 0], [0, 1]],
                    labels=None, c=0.2, u=1.0, n_quadpts=9, max_iter=1000,
                    tol=1e-7)
    script = python_mirt_script(settings)
    ast.parse(script)
    assert "from core.irt_multidim import fit, report" in script
    assert "veri.csv" in script
    assert "M01" in script
    assert "person_id_123" not in script
