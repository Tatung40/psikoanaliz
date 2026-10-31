"""Shared Python syntax panel: no fit, code execution, network call or installs."""
import streamlit as st
from core.reproducible import bundle, python_code


def render(recorder):
    with st.tabs(['</> Python Kodları'])[0]:
        if recorder is None or not recorder.calls:
            st.info('Bu sayfada kaydedilmiş bir hesaplama çağrısı bulunamadı. Analizi bu oturumda yeniden çalıştırın; kod üretilmeden boş bir alan gösterilmez.')
            return
        manifest = recorder.manifest()
        python = python_code(manifest)
        if not python:
            st.warning('Bu analiz için doğrulanabilir Python yeniden üretim kodu oluşturulamadı. Eksik veya desteklenmeyen adımlar:')
            for reason in manifest.get('unsupported', []):
                st.caption(str(reason))
            return
        st.caption('Gerçek hesaplama girdileri kullanılır. Kod üretimi bağımsız bilimsel doğrulama değildir. '
                   'Ön işleme uygulanmış veri tekrar kodlanmaz; seed yalnız açıkça verilmişse taşınır. '
                   'Kod bloğunun kopyala düğmesini kullanabilirsiniz. İndirmeler verinizi içerir.')
        st.caption('Bağımsız çalıştırma: ZIP paketini açın; requirements-repro.txt kurulumundan sonra analysis.py çalıştırın. Python yalnız aynı mevcut motoru tekrar çağırır.')
        st.code(python, language='python')
        st.download_button('.py indir', python, 'analysis.py', 'text/x-python', key='repro_py', on_click='ignore')
        with st.expander('Paketler / sürümler / veri ve analiz ayarları'):
            st.json(manifest)
        st.download_button('Veri + kod + mevcut motor — ZIP indir', bundle(manifest, python),
                           'psikoanaliz-analysis.zip', 'application/zip', key='repro_zip', on_click='ignore')
