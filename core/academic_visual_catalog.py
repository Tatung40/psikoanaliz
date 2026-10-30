"""Deterministic table and figure manifest for four academic report pilots.

This module describes existing numeric tables and chart inputs; it does not render
images or generate APA interpretations. Missing inputs never become fake figures.
"""
from core.academic_result_schema import validate_result


PLOT_SPECS = {
    "item_calibration": (
        ("category_curves", "Kategori yanıt olasılıkları", "Theta", "Olasılık"),
        ("information", "Madde ve test bilgi fonksiyonları", "θ", "Bilgi"),
    ),
    "DIF": (
        ("Curves", "Gruplara göre kategori olasılıkları", "Theta", "Probability"),
        ("Difference", "Gruplar arası olasılık farkları", "Theta", "Focal_minus_reference"),
        ("TIF", "Gruplara göre test bilgisi", "Theta", "Information"),
        ("DTF", "Diferansiyel test fonksiyonu", "Theta", "Focal_minus_reference"),
    ),
    "G_and_D_study": (
        ("variance", "Varyans bileşenleri", "component", "Variance"),
        ("d", "D çalışması katsayıları", "measurement_units_per_person", "rho^2"),
    ),
}


def build_visual_catalog(result):
    """Return an ordered, APA-ready manifest without modifying any source values."""
    source = validate_result(result)
    kind = source.get("metadata", {}).get("analysis_type")
    tables = source["tables"]
    catalog = {"tables": [], "figures": [], "warnings": []}
    for number, (key, rows) in enumerate(tables.items(), 1):
        if not isinstance(rows, list):
            raise ValueError(f"Table {key} is not a row collection")
        catalog["tables"].append({
            "id": key, "number": number, "title": key.replace("_", " "),
            "source_table": key, "row_count": len(rows),
            "columns": list(rows[0]) if rows else [],
            "apa_label": f"Tablo {number}",
        })
    for key, title, x, y in PLOT_SPECS.get(kind, ()):
        rows = tables.get(key)
        if not rows:
            catalog["warnings"].append(f"{key}: grafik kaynağı yok; görsel üretilmedi.")
            continue
        columns = set().union(*(row.keys() for row in rows))
        if x not in columns or y not in columns:
            catalog["warnings"].append(f"{key}: {x}/{y} eksen verileri eksik; görsel üretilmedi.")
            continue
        number = len(catalog["figures"]) + 1
        catalog["figures"].append({
            "id": f"figure_{key}", "number": number, "apa_label": f"Şekil {number}",
            "title": title, "source_table": key, "x": x, "y": y,
            "rendered_file": None, "render_status": "NOT_RENDERED",
            "note": "Grafik, özgün analiz değerlerinden üretilmelidir.",
        })
    return catalog
