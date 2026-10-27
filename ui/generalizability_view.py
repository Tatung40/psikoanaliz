"""Turkish G/D study screen; content-addressed results and safe exports."""
import io
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from core import generalizability as gt
from core.python_mirt_dif import fingerprint
from core.reproducible import capture_view
from ui.dimensionality_localfit_view import _safe, _xlsx, _zip, _png_lines

PAGE = 'Genellenebilirlik Kuramı (G ve D Çalışmaları)'


def exports(result, target, tables=None):
    tables = gt.report_tables(result, target) if tables is None else tables
    report = (f"{PAGE}\nDurum: {result['status']}\n" + result['method'] +
              '\n\nBirey katsayıları:\n' + result['g'].to_string() +
              '\nHata varyansları:\n' + result['errors'].to_string() +
              f'\nHedef: {target:.3f}; D senaryoları:\n' + tables['d'].to_string(index=False))
    files = {f'{k}.csv': _safe(v).to_csv(index=True).encode('utf-8-sig') for k, v in tables.items()}
    files['G_D.xlsx'] = _xlsx(tables)
    files['RAPOR.txt'] = report.encode('utf-8')
    files['settings.json'] = json.dumps(result['settings'], ensure_ascii=False, indent=2).encode()
    files['engine.log'] = result['log'].encode()
    files['variance.png'] = _png_lines(list(range(len(result['variance']))),
        {'Ham varyans': result['variance'].Variance.to_numpy(),
         'Katsayılarda kullanılan': result['variance'].coefficient_variance.to_numpy()},
        'Varyans bileşenleri (CSV sırası)', 'Bileşen', 'Varyans')
    files['D.png'] = _png_lines(list(range(1, len(result['d']) + 1)),
        {'Eρ²': result['d']['rho^2'], 'Φ': result['d']['phi^2']},
        'D çalışması', 'Senaryo', 'Katsayı')
    return tables, files, report


def _levels(text):
    values = [int(s.strip()) for s in text.split(',')]
    if not values or len(values) != len(set(values)) or any(n < 2 or n > 100 for n in values):
        raise ValueError('D düzeyleri: virgülle ayrılmış, benzersiz 2–100 tamsayılar.')
    return values


@capture_view
def render(data, item_cols=()):
    st.header(PAGE)
    st.caption('GeneralizIT 0.1.2 · ANOVA / Henderson I · Rastgele yüzeyler · CERTIFIED değildir')
    tabs = st.tabs(['Veri ve Desen', 'G Çalışması', 'Varyans Bileşenleri',
                    'D Çalışması', 'Grafikler', 'Akademik Rapor'])
    with tabs[0]:
        st.info('CSV/XLSX için ortak Veri Yükleme ekranını kullanın. Eksik/dengesiz, sabit yüzeyli '
                've (p:r) × i desenleri doğrulanmadığından kapalıdır; görev gibi yüzeyler i/r rollerine atanabilir.')
        form = st.radio('Veri biçimi', ['Uzun', 'Geniş (birey × madde)'], key='gt_format', horizontal=True)
        source = data
        if form.startswith('Geniş'):
            person = st.selectbox('Birey kimliği', list(data.columns), key='gt_wide_person')
            items = st.multiselect('Madde sütunları', [c for c in data.columns if c != person],
                default=[c for c in item_cols if c in data.columns and c != person], key='gt_wide_items')
            try:
                source = gt.wide_to_long(data, person, items)
            except (ValueError, KeyError) as exc:
                st.session_state.pop('gt_result', None)
                st.error(str(exc))
                return
            design, columns, score = 'p x i', {'p': 'p', 'i': 'i'}, 'score'
        else:
            design = st.selectbox('Desen', list(gt.DESIGNS), key='gt_design')
            columns = {}
            for f in gt.DESIGNS[design]:
                options = list(source.columns)
                default = options.index(f) if f in options else min(len(columns), len(options) - 1)
                columns[f] = st.selectbox({'p': 'Birey', 'i': 'Madde / görev', 'r': 'Puanlayıcı / yüzey'}[f],
                    options, index=default, key='gt_col_' + f)
            options = list(source.columns)
            score = st.selectbox('Puan', options,
                index=options.index('score') if 'score' in options else len(options) - 1, key='gt_score')
        st.checkbox('Sabit yüzeyler (NOT VALIDATED)', value=False, disabled=True, key='gt_fixed')
        try:
            prepared, counts = gt.prepare(source, design, columns, score)
        except (ValueError, KeyError, TypeError) as exc:
            st.session_state.pop('gt_result', None)
            st.error(str(exc))
            return
        st.dataframe(pd.DataFrame({'Yüzey': list(counts), 'Düzey (yuvalı ise ebeveyn başına)': list(counts.values())}),
                     hide_index=True)
        st.caption(f'{len(prepared)} gözlem; eksik/tekrarlı hücre yok; negatif bileşen politikası: '
                   'ham değeri koru, yalnız katsayı hesabında kütüphane sıfırlaması.')
        scenarios = {'p': [counts['p']]}
        try:
            for f in counts:
                if f != 'p':
                    # Separate keys prevent carrying a crossed total into a nested per-parent count.
                    text = st.text_input(f'D çalışması {f} düzeyleri (virgülle)', str(counts[f]),
                                         key=f'gt_d_{design}_{f}')
                    scenarios[f] = _levels(text)
        except ValueError as exc:
            st.session_state.pop('gt_result', None)
            st.error(str(exc))
            return
        target = float(st.number_input('Hedef güvenirlik (Eρ² ve Φ)', 0.01, 0.99, 0.80, 0.01, key='gt_target'))
        key = fingerprint(data, (form, design, columns, score, scenarios, target, '0.1.2',
                                 (person, tuple(items)) if form.startswith('Geniş') else ()))
        if st.session_state.get('gt_key') != key:
            st.session_state.pop('gt_result', None)
            st.session_state['gt_key'] = key
        if st.button('G ve D çalışmalarını çalıştır', type='primary', key='gt_run'):
            st.session_state.pop('gt_result', None)
            try:
                with st.spinner('G/D çalışmaları hesaplanıyor…'):
                    st.session_state['gt_result'] = gt.analyze(source, design, columns, score, scenarios)
            except (ValueError, TypeError, ArithmeticError) as exc:
                st.error(f'Analiz engellendi: {exc}')
    result = st.session_state.get('gt_result')
    if result is None:
        return
    tables = gt.report_tables(result, target)
    with tabs[1]:
        st.subheader('G çalışması — birey evreni')
        st.metric('Eρ² (göreli)', f"{result['g'].loc['p', 'rho^2']:.4f}")
        st.metric('Φ (mutlak)', f"{result['g'].loc['p', 'phi^2']:.4f}")
        st.dataframe(result['anova'])
        st.caption('SS/df/MS: dengeli ortogonal ANOVA; en yüksek etkileşim artıkla karışıktır. '
                   'Rastgele-etki F/p ve doğrulanmamış güven aralıkları verilmez.')
        with st.expander('GeneralizIT özgün Henderson tablosu (T ≠ SS)'):
            st.dataframe(result['henderson'])
        st.dataframe(result['errors'])
    with tabs[2]:
        st.dataframe(result['variance'])
        if (result['variance'].Variance < 0).any():
            st.warning('Negatif bileşenler var. Ham değerler tabloda korunur; katsayılarda sıfıra kırpılır. '
                       'Bu sınır durumu yorumu ve D kestirimlerini sınırlar.')
    with tabs[3]:
        d = tables['d']
        st.dataframe(d, hide_index=True)
        candidates = d[d.target_met].sort_values('measurement_units_per_person')
        if not candidates.empty:
            st.success('Girilen senaryolar içinde hedefe en az ölçme birimiyle ulaşan: ' +
                       ', '.join(f'{f}={int(candidates.iloc[0][f])}' for f in counts if f != 'p'))
        else:
            st.info('Girilen senaryolarda her iki hedefi karşılayan tasarım yok.')
        st.caption('En az birim yalnız girilen adaylar arasındadır; evrensel optimizasyon veya maliyet eşitliği değildir.')
    with tabs[4]:
        st.plotly_chart(px.bar(result['variance'].reset_index(names='source'), x='source', y='Variance',
                               title='Ham varyans bileşenleri'), use_container_width=True)
        st.plotly_chart(px.line(d.reset_index(names='scenario'), x='scenario', y=['rho^2', 'phi^2'],
                                markers=True, title='D çalışması katsayıları'), use_container_width=True)
        other = [f for f in counts if f != 'p']
        if len(other) == 2 and all(d[f].nunique() > 1 for f in other):
            st.plotly_chart(px.density_heatmap(d, x=other[0], y=other[1], z='phi^2', histfunc='avg',
                                              title='D çalışması Φ'), use_container_width=True)
    with tabs[5]:
        _, files, report = exports(result, target, tables)
        st.text(report)
        st.download_button('XLSX indir', files['G_D.xlsx'], 'G_D.xlsx', key='gt_xlsx', on_click='ignore')
        st.download_button('ZIP (CSV/XLSX/PNG/rapor)', _zip(files), 'G_D.zip', key='gt_zip', on_click='ignore')
        st.download_button('Varyans CSV indir', files['variance.csv'], 'variance.csv', key='gt_csv', on_click='ignore')
        st.download_button('Grafik PNG indir', files['variance.png'], 'variance.png', key='gt_png', on_click='ignore')
        st.caption('Çalıştırılan veri ve ayarları taşıyan .py/ZIP: ortak </> Python Kodları sekmesinde.')
