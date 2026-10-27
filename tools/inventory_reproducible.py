"""Static, deterministic screen/API inventory: no model imports or fits."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inventory():
    records = []
    for path in [ROOT / 'app.py', *sorted((ROOT / 'ui').glob('*.py')), *sorted((ROOT / 'pages').glob('*.py'))]:
        if path.name == 'reproducible_view.py':
            continue
        tree = ast.parse(path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (node.module == 'core' or node.module.startswith('core.')) and 'reproducible' not in node.module:
                imports.extend(node.module + '.' + alias.name for alias in node.names)
        records.append({'screen': str(path.relative_to(ROOT)), 'core_imports': sorted(set(imports)),
                        'python': 'Runtime capture; GENERATED / UNVERIFIED only for faithfully serializable actual calls; otherwise UNSUPPORTED'})
    return records


if __name__ == '__main__':
    target = ROOT / 'validation' / 'REPRODUCIBLE_ANALYSIS_INVENTORY.json'
    target.write_text(json.dumps(inventory(), ensure_ascii=False, indent=2) + '\n')
    print(f'{len(inventory())} screen modules inventoried; names are not proof of API support.')
