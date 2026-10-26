"""Public Python mirt EM controls shared by its two existing UI entry points."""
import streamlit as st


def solver_controls(prefix):
    with st.expander('Python mirt — sayısal çözücü',expanded=False):
        max_iter=st.number_input('Maksimum EM iterasyonu',min_value=1,value=500,step=1,key=prefix+'_max_iter')
        tol=st.number_input('Mutlak EM amaç fonksiyonu toleransı',min_value=0.,value=1e-4,format='%.8g',key=prefix+'_tol')
        n_quadpts=st.number_input('Gauss–Hermite düğüm sayısı',min_value=1,value=21,step=1,key=prefix+'_n_quadpts')
        st.caption('ML: marjinal logLik; MAP: logposterior değişimi. Gerçek mirt.fit_mirt ayarları; değişiklikten sonra yeniden kestirim gerekir. Düğüm sayısı doğruluğu etkiler; küçük grid önerilmez.')
    return dict(max_iter=int(max_iter),tol=float(tol),n_quadpts=int(n_quadpts))
