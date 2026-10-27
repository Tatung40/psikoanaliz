"""Shared syntax panel: no fit, code execution, network call or automatic installs."""
import streamlit as st
from core.reproducible import bundle, python_code
from core.reproducible_r import r_code


def render(recorder):
    with st.tabs(['</> Python / R Kodları'])[0]:
        if recorder is None or not recorder.calls:
            st.info('UNSUPPORTED — Bu çalıştırmada taşınabilir bir hesaplama çağrısı kaydedilmedi. Analizi çalıştırın; saklanmış sonuçlar doğrulanmış kod sayılmaz.')
            return
        manifest = recorder.manifest()
        python = python_code(manifest)
        r, reason = r_code(manifest) if python else (None, 'Python input mapping is unsupported')
        st.caption('Gerçek hesaplama girdileri kullanılır. Kod üretimi bağımsız bilimsel doğrulama değildir. '
                   'Ön işleme uygulanmış veri tekrar kodlanmaz; seed yalnız açıkça verilmişse taşınır. '
                   'Kod bloğunun kopyala düğmesini kullanabilirsiniz. İndirmeler verinizi içerir.')
        py_tab, r_tab = st.tabs(['Python', 'R'])
        with py_tab:
            st.write('GENERATED / UNVERIFIED' if python else 'UNSUPPORTED')
            if python:
                st.caption('Bağımsız çalıştırma: ZIP paketini açın; requirements-repro.txt kurulumundan sonra analysis.py çalıştırın. Python yalnız aynı mevcut motoru tekrar çağırır.')
                st.code(python, language='python')
                st.download_button('.py indir', python, 'analysis.py', 'text/x-python', key='repro_py', on_click='ignore')
            else:
                st.warning('\n'.join(manifest['unsupported']))
        with r_tab:
            st.write('GENERATED / UNVERIFIED' if r else 'UNSUPPORTED')
            if r:
                st.code(r, language='r')
                st.download_button('.R indir', r, 'analysis.R', 'text/plain', key='repro_r', on_click='ignore')
            else:
                st.info(reason)
        with st.expander('Paketler / sürümler / veri ve analiz ayarları'):
            st.json(manifest)
        if python:
            st.download_button('Veri + kod + mevcut motor — ZIP indir', bundle(manifest, python, r),
                               'psikoanaliz-analysis.zip', 'application/zip', key='repro_zip', on_click='ignore')
