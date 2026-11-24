"""Public Python mirt EM controls shared by its two existing UI entry points."""
import streamlit as st


def solver_controls(prefix,advanced_key=None,n_quadpts=21,tol=1e-4,max_iter=500):
    # 4PL screens pass tol=1e-10, max_iter=50000: the setting validated by certification amendment D4.
    # Binary screens pass 61 (R mirt default): 21 nodes bias large-N/k estimates (binary certification amendment A2).
    default_quadpts=n_quadpts
    with st.expander('Python mirt — sayısal çözücü',expanded=False):
        if advanced_key is not None:
            enabled=st.checkbox('Gelişmiş sayısal ayarları etkinleştir',value=False,key=advanced_key)
            if not enabled:
                st.caption(f'Standart mod: max_iter={max_iter}, tol={tol:g}, n_quadpts={default_quadpts}.')
                return dict(max_iter=max_iter,tol=tol,n_quadpts=default_quadpts)
        max_iter=st.number_input('Maksimum EM iterasyonu',min_value=1,value=max_iter,step=1,key=prefix+'_max_iter')
        tol=st.number_input('Mutlak EM amaç fonksiyonu toleransı',min_value=0.,value=tol,format='%.8g',key=prefix+'_tol')
        n_quadpts=st.number_input('Gauss–Hermite düğüm sayısı',min_value=1,value=default_quadpts,step=1,key=prefix+'_n_quadpts')
        st.caption('ML: marjinal logLik; MAP: logposterior değişimi. Gerçek mirt.fit_mirt ayarları; değişiklikten sonra yeniden kestirim gerekir. Düğüm sayısı doğruluğu etkiler; küçük grid önerilmez.')
    return dict(max_iter=int(max_iter),tol=float(tol),n_quadpts=int(n_quadpts))


D4_4PL = dict(tol=1e-10, max_iter=50000)


def effective_solver(model, estimation, options):
    """4PL: the D4-validated setting applies to MAP (the certified estimation); ML 4PL (outside the certified scope) keeps
    standard precision unless the user changed the values."""
    if model == "4PL" and estimation == "ML" and options.get("tol") == D4_4PL["tol"] and options.get("max_iter") == D4_4PL["max_iter"]:
        return dict(options, tol=1e-4, max_iter=500)
    return options
