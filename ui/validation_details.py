"""One collapsed "Teknik ve Bilimsel Doğrulama Ayrıntıları" section for routine IRT notices.

Routine st.info/st.success notices, and captions that expose internal identifiers (module paths, snake_case engine
names, result-class or serialization diagnostics), are moved into a single collapsed expander created at the top of
the page. st.warning/st.error are never moved, and any info/success that signals a risk, failure, change or required
action stays visible. Per-session via a ContextVar (Streamlit runs each script in its own thread); no global state.
"""
import re
import sys
from contextvars import ContextVar
from functools import wraps

import streamlit as st

TITLE = "Teknik ve Bilimsel Doğrulama Ayrıntıları"
PAGES = ("MTK", "Boyutluluk ve Faktör Analizi", "Yerel Bağımsızlık ve Madde Uyumu", "Çok Boyutlu MTK (MIRT)")
RISK = re.compile(r"(?i)(kestirim yakınsadı|uyarı|risk|kayıp|yakınsamad|yakınsamıyor|geçersiz|değişti|yeniden|hata|başarısız|eksik|sınır|redded|kapatıldı|"
                  r"uydurul|yapılamadı|üretilemedi|hesaplanamadı|desteklenm|unsupported|dikkat|önemli)")
INTERNAL = re.compile(r"(\b(?:core|ui|reports)\.\w+|\b[A-Z]\w*(?:Result|Error|Exception)\b|\bpython_mirt\b|"
                      r"serializ|no faithful public|@attr|Traceback|\bspec_model\b)")
_BOX = sys.__dict__.setdefault("_validation_details_box", ContextVar("validation_details_box", default=None))


def routine(kind, body):
    text = str(body)
    if kind == "caption":
        return bool(INTERNAL.search(text))
    return not RISK.search(text)


def _install():
    for name in ("info", "success", "caption"):
        original = getattr(st, name)
        if getattr(original, "__validation_details__", False):
            continue

        def wrap(fn, kind):
            @wraps(fn)
            def call(body, *args, **kwargs):
                state = _BOX.get()
                if state is None or not routine(kind, body):
                    return fn(body, *args, **kwargs)
                if state["box"] is None:
                    state["box"] = state["holder"].expander(TITLE, expanded=False)
                return getattr(state["box"], kind)(body, *args, **kwargs)
            call.__validation_details__ = True
            return call
        setattr(st, name, wrap(original, name))


def begin(page):
    """Start routing for IRT pages; the expander is created lazily inside a top-of-page holder."""
    _install()
    _BOX.set({"holder": st.container(), "box": None} if str(page).startswith(PAGES) else None)


def end():
    _BOX.set(None)
