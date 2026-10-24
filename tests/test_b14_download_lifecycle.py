"""Guard the CSV→CSV→XLSX sequence against download-triggered reruns."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from validation import browser_e2e_polytomous_dif as e2e


@pytest.mark.parametrize('key', ['b14_csv', 'b14_persons_csv', 'b14_xlsx'])
def test_b14_browser_exports_do_not_rerun(key):
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'ui/irt_b14_view.py').read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'download_button'
             and any(k.arg == 'key' and isinstance(k.value, ast.Constant) and k.value.value == key
                     for k in node.keywords)]
    assert len(calls) == 1
    assert any(k.arg == 'on_click' and isinstance(k.value, ast.Constant) and k.value.value == 'ignore'
               for k in calls[0].keywords)


def test_download_waits_for_idle_and_propagates_cancellation(tmp_path):
    events = []
    class Download:
        suggested_filename = 'report.xlsx'
        def save_as(self, target):
            assert target.parent.is_dir()
            events.append('save')
            raise RuntimeError('canceled')
    class Pending:
        value = Download()
        def __enter__(self):
            events.append('expect'); return self
        def __exit__(self, *args):
            pass
    page = SimpleNamespace(
        wait_for_function=lambda *a, **k: events.append('idle'),
        expect_download=lambda **k: Pending(),
        get_by_role=lambda *a, **k: SimpleNamespace(first=SimpleNamespace(click=lambda: events.append('click'))),
    )
    with pytest.raises(RuntimeError, match='canceled'):
        e2e.download(page, 'XLSX', tmp_path / 'nested')
    assert events == ['idle', 'expect', 'click', 'save']
