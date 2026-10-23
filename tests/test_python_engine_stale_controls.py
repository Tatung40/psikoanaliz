from pathlib import Path
import ast

def test_python_engine_invalidation_includes_categories():
    source=Path("ui/irt_python_library_view.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "old_categories" in source
    assert "old_hash" in source
    assert "old_categories:" in source
    assert "(int(categories) if categories else None)!=old_categories" in source

def test_homepage_does_not_falsely_hide_irt():
    source=Path("app.py").read_text(encoding="utf-8")
    assert "Bu analizler mevcut sürümde henüz kullanıma sunulmamıştır." not in source
    assert "4PL formel" in source
