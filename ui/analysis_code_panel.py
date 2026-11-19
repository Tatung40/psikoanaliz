"""Reproducible Python script generator for the existing multidimensional route.

The R tab is intentionally gated: a comparable R model has not been
independently verified for every selectable structure and estimator.
"""
import json


def python_mirt_script(settings):
    """Generate a project-local, executable rerun; never embed respondent data."""
    allowed = {"c", "u", "items", "person", "model", "dims", "efa",
               "correlated", "pattern", "labels", "n_quadpts", "max_iter", "tol"}
    config = {key: value for key, value in settings.items() if key in allowed}
    # JSON literals are parsed by Python, so booleans/null are safe.
    encoded = json.dumps(config, ensure_ascii=False, allow_nan=False)
    return '''# PsikoAnaliz — çok boyutlu MTK yeniden üretim betiği
# Bu kod uygulamada çalıştırılan Python motoruna aynı argümanları iletir.
# Proje deposu ve requirements.txt bağımlılıkları gereklidir.
# Veriyi yerel "veri.csv" dosyasına koyun; kişisel veriler koda eklenmez.
import json
import pandas as pd
from core.irt_multidim import fit, report
from core import irt_multidim_factorized as factorized
from core import irt_multidim_rsm_stack as rsm
from core import irt_multidim_general_report as general

cfg = json.loads(''' + repr(encoded) + ''')
data = pd.read_csv("veri.csv")
items = cfg["items"]
person = cfg["person"]
ids = data.index.tolist() if person == "(satır indeksi)" else data[person].tolist()
x = data[items]
kwargs = dict(correlated=cfg["correlated"], exploratory=cfg["efa"],
              n_quadpts=cfg["n_quadpts"], max_iter=cfg["max_iter"], tol=cfg["tol"])
model = cfg["model"]
pattern = cfg["pattern"]
if model == "Bifactor":
    # UI, özgül grupları boyut dizisi olarak iletir.
    pass
if model in factorized.MODELS:
    result = factorized.report(factorized.fit(x, model, pattern,
        c=cfg["c"], u=cfg["u"], **kwargs), ids)
elif model in rsm.MODELS:
    result = rsm.report(rsm.fit(x, model, pattern, **kwargs), ids)
elif model in general.MODELS:
    result = general.fit_report(x, model, pattern, c=cfg["c"], u=cfg["u"],
        person_ids=ids, **kwargs)
else:
    result = report(fit(x, model, pattern, labels=cfg["labels"], **kwargs), ids)
print(result["metadata"])
for name, table in result["tables"].items():
    print(name)
    print(table)
'''


def render_mirt_code_panel(settings):
    import streamlit as st
    script = python_mirt_script(settings)
    with st.expander("Analiz Kodları ve Tekrarlanabilirlik", expanded=False):
        py, r = st.tabs(["Python — gerçek motor", "R — eşdeğer kod"])
        with py:
            st.caption("Bu betik aynı PsikoAnaliz Python fonksiyonlarını çağırır. "
                       "Oturumdaki analiz yeniden çalıştırılmaz; yerel veri.csv gerekir.")
            st.code(script, language="python")
            st.download_button("Python kodunu indir (.py)", script,
                               "mirt_tekrarla.py", mime="text/x-python",
                               key="md_repro_python", on_click="ignore")
        with r:
            st.info("Bu modelin seçili faktör yapısı, kısıtları ve kestirim "
                    "ayarları için bağımsız doğrulanmış, çalıştırılabilir R "
                    "yeniden üretim betiği henüz yok. Yanıltıcı R kodu üretilmez.")
