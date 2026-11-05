# Universal academic Word and faithful Python exports

Scope: **38 real analysis routes**, including the standalone missing-data page;
6 auxiliary routes have explicit exclusions below. The existing module inventory
also lists helpers and is not an analysis-screen count.

Word is available only after a current, recorded calculation with reportable outputs.
No fit is run to make a report. A failed/nonconverged fit or an opaque result without
its actual producer cannot create a valid academic download. No claim of scientific
certification or VERIFIED code execution is made by the UI.

| Analysis route | Word producer | Python status for a valid recorded calculation |
|---|---|---|
| 3B. Betimsel İstatistikler | Shared academic_docx; DESCRIPTIVE | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| 4. Test İstatistikleri | Shared academic_docx; CTT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| 5. Madde Analizi | Shared academic_docx; CTT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| 6. Çeldirici Analizi | Shared academic_docx; CTT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| 7. Güvenirlik & Ölçme Hatası | Shared academic_docx; CTT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| 8. Grup Bazlı Analizler | Shared academic_docx; CTT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| 9. Bilimsel Grafikler | Shared academic_docx; DESCRIPTIVE | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| KTK: Çift Motor Rapor | Existing CTT Word | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Rasch | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: 1PL | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: 2PL | Existing academic 2PL Word | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: 3PL | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: 4PL | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Karşılaştırma | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Varsayım Denetimi | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Çok Modelli DIF Taraması | Shared academic_docx; DIF | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Çok Kategorili B14 | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: RSM | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Python mirt DIF | Existing academic DIF Word | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: DIF Analiz Merkezi | Shared academic_docx; DIF | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: GRM Formel DIF | Shared academic_docx; DIF | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Çok Kategorili Formel DIF | Shared academic_docx; DIF | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Gözlenen Puan DIF | Shared academic_docx; DIF | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: 3PL/4PL Bootstrap DIF | Shared academic_docx; DIF | UNSUPPORTED: raw bootstrap replay would bypass UI inference gates |
| Boyutluluk ve Faktör Analizi | Shared academic_docx; FACTOR | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| Yerel Bağımsızlık ve Madde Uyumu | Shared academic_docx; FACTOR | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| Genellenebilirlik Kuramı (G ve D Çalışmaları) | Shared academic_docx; GT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| Çok Boyutlu MTK (MIRT) | Shared academic_docx; MIRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| Eksik Veri ve Atama | Shared academic_docx; DATA | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Python mirt Rasch | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Python mirt 1PL | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Python mirt 2PL | Existing academic 2PL Word | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Python mirt 3PL | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Python mirt 4PL | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Python mirt GRM | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Python mirt PCM | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Python mirt GPCM | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |
| MTK: Python mirt NRM | Shared academic_docx; IRT | GENERATED / UNVERIFIED; exact public-API inputs and producer references |

## Auxiliary scope

- **1. Ana Sayfa**: Tanıtım; hesaplanan analiz sonucu yok.
- **2. Veri Yükleme**: Yardımcı dosya yükleme/eşleme ekranı; bağımsız bilimsel analiz değildir.
- **3. Puanlama & Veri Hazırlama**: Yardımcı puanlama/ön işleme; işlem kodu korunur, bağımsız akademik sonuç raporu değildir.
- **10. Raporlama**: Mevcut raporları birleştirir; yeni bağımsız analiz üretmez.
- **11. Formüller ve Yöntemler**: Başvuru/yöntem metni; veri üzerinde analiz üretmez.
- **12. Kullanım Kılavuzu**: Kullanım yardımı; veri üzerinde analiz üretmez.

The aggregate report retains its original content/API and now uses the same final formatting standard (reports.word_export). Existing .py/ZIP controls
remain where their actual computations can be recorded. Exclusions do not remove
existing application functionality.

## Root cause and reproducibility

A real 2PL AppTest reproduced `PythonMirtResult/FitResult: no faithful public
serialization` when navigating away and returning to the fitted page. `begin`
received the latest other-page history; the recorder then excluded its own page
from `others`, losing the real producer. The router now supplies its **own latest
recording** first. No opaque-object serializer, pickle, guessed output or fake
producer was added. Old sessions without that producer require a real recalculation.

Cold lazy imports (including dimensionality and G theory) are instrumented after
import. Only core computation aliases and an explicit B14 computation allowlist
inside reports are captured; export functions are not indiscriminately wrapped.
Multistart records the existing public selection call and its actual selected result,
not an invented single-fit recipe. Pure public ui.charts computations and their
exact source are included in the portable ZIP. Lord replay also carries its original
reference-gate JSON and cohort fixture, with SHA-256 integrity checks.

The reporting facade resides in UI, so the core recorder cannot mistake its own
instrumentation for an analysis. Existing core recorder/schema/adapters/catalog and
all production estimation, covariance, inference and reference-evidence files remain
byte-identical. The additive screen-domain adapter delegates common invariants to
the existing validator and preserves the actual domain in documents.

## Validity and report content

Session-local observation captures the tables, metrics, notices and figures actually
shown, respecting hidden dataframe columns. It never recalculates a psychometric
statistic. Reports contain the actual API input sample basis, model, scalar settings,
observed findings/tables, source warnings, versions and source identity. Input N is
explicitly distinguished from any later method-specific exclusions. Large tables
show their first 100 rows with an explicit omission note and retain the original
screen/analysis exports. New shared tables have APA-style horizontal rules, font
and margins; existing specialized exporters keep their established format.

Full data-content hashes (including unchanged-shape edits), selected inputs and
analysis widgets, including editable model matrices, protect report/code freshness.
Run/download buttons do not invalidate a valid result. Cached postprocessing is not
a new calibration. A first comparison visit accepts imported fits only when their
ML input matrices exactly match the current data. Deferred existing Word buttons
are restored only after these checks. Opaque/stale recipes never become GENERATED.

Matplotlib figures are embedded directly. Supported Plotly numerical traces and
surfaces are rendered from their existing points; unsupported trace types (e.g.
interactive histogram/box rendering) produce an explicit omission warning rather
than a fabricated or partial image. No plotting dependency is downloaded.

All source scientific restrictions remain: 3PL/4PL inference, nominal categories,
DIF-FWER NOT CERTIFIED, free-SD 1PL Lord and closed fixed-SD inference, GH101/GH161,
and bootstrap replay UNSUPPORTED. Code generation is not independent R validation.
No R calibration, Monte Carlo study or scientific archive is changed by this PR.

## Executable checks

- `tests/test_universal_academic_word.py`: actual computations across all 38 routes,
  Word downloads, python-docx opening, actual numeric cells, script compilation,
  unchanged reruns, same-shape data and item-setting invalidation; bootstrap stays
  UNSUPPORTED. Uses existing teaching/B10 fixtures, not new recovery simulations.
- `tests/test_universal_academic_capture.py`: real 2PL navigation and pre-deploy
  opaque state, actual eight-start 3PL selection, standalone chart ZIP replay with
  exact original observations outside the checkout.
- `tests/test_academic_screen_results.py`: actual NA suppression, domain invariants,
  APA formatting, numerical surface rendering and exact route inventory.
- Existing portable descriptives/IRT replay, specialized 2PL Word, stale-state,
  archived-source identity and scientific gate tests remain in regression/CI.

Targeted results and the single final regression are recorded in the PR description.
The coverage table describes supported runtime paths, not scientific certification
of every model/design or execution of every possible option.


## Shared reference-2PL Word standard (v2)

All 38 analysis routes above and the aggregate reporting route use the same
`reports.academic_docx.standardize_document` finalizer. Five preparation/help routes
remain auxiliary. No new estimator or separate model-specific DOCX engine is added.
Existing content producers keep their actual tables, figures, notes and APIs.

The reference contract is versioned in `WORD_REFERENCE_CONTRACT.json` (original
file SHA-256, layout and headings). The original DOCX contains 3 tables/17 figures;
its data are not committed or used as fabricated calibration evidence.

Hierarchy: Methods/settings → observed findings → model fit (only supplied fit
metrics) → tables → figures → limitations → references (if supplied) → reproducibility.
The 2PL ICC and IIF/TIF sections retain every plot and become figure subsections;
limitations and reproducibility finish the document. Inapplicable sections are
omitted with an explicit reason. No missing fit statistic, p/CI or reference result
is created. The 2PL reference's Letter size, margins, heading sizes and blue palette
are shared; body Calibri 11pt, captions 9pt. APA tables use horizontal top/header/
bottom rules, repeating headers, sequential cached Word SEQ fields, and fit figures
to the 6-inch text area. Numeric values are not inferred from nominal labels.

| Existing content producer | Entry points / scope | Shared formatting |
|---|---|---|
| reports.academic_docx | 34 generic analysis routes: KTK, IRT, MIRT/bifactor, DIF, G theory, factors, missingness | v2; actual displayed outputs |
| reports.irt_2pl_academic_docx | Both 2PL routes | v2; grouped item ICC + IIF/TIF preserved |
| reports.ctt_guided_docx | Dual CTT button and its release ZIP | v2; native/reference checks retained |
| reports.dif_academic_docx | Python mirt DIF button | v2; original four-panel figure retained |
| reports.word_export | Aggregate KTK reporting route | v2; existing API and source content retained |
| reports.grm_dif_report | GRM DIF DOCX inside ZIP | v2; source cautions, tables, PNG unchanged |
| reports.polytomous_dif_report | PCM/GPCM/RSM/NRM DIF DOCX inside ZIP | v2; nominal restrictions retained |

The same data/settings/producer fingerprints still gate downloads. Renderer-version
keys additionally invalidate old 2PL DOCX/generic Word and prepared GRM/poly ZIP
bytes after a format upgrade, without recalibrating models. CSV/XLSX/PNG remain
unchanged. The existing 38-route real-result tests now assert the common hierarchy,
caption sequences, page dimensions, image width, table rules, Turkish text and
reproducibility. `tests/test_word_standard.py` covers mixed figure sources and legacy
normalization; existing 2PL, DIF, CTT and bundle safeguards remain active.

Reference execution: all 3 table cell grids and all 17 image SHA-256 values were preserved; normalization was checked for idempotence. Native Word/LibreOffice pagination was not rendered in this environment; OOXML dimensions, table widths, caption order and image placement bounds are tested.


## Compact result panels and contextual downloads

All output tables, Plotly/Matplotlib/native charts and JSON diagnostics use the
existing screen observer and `ui.result_panels`. New result expanders start closed;
existing result expanders are reused, never nested. Controls/metrics stay in their
original layout. Warnings/errors and downloads inside existing expanders are
routed to visible page-level anchors. Actual data, report content, widget keys,
producer recording and input/cache fingerprints are unchanged. Open/close is a
frontend action and does not trigger calibration.

Only DOCX button labels and download filenames change. CSV/XLSX/PNG/ZIP payloads,
labels and internal filenames remain. Dynamic models are read from the active
route's actual selection after existing result-validity gates; naming never grants
scientific certification.

| Analysis | Academic DOCX label | Filename | Result panels | Evidence |
|---|---|---|---|---|
| 3B. Betimsel İstatistikler | Betimsel İstatistikler Akademik Rapor | betimsel_istatistikler_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| 4. Test İstatistikleri | Test İstatistikleri Akademik Rapor | test_istatistikleri_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| 5. Madde Analizi | Madde Analizi Akademik Rapor | madde_analizi_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| 6. Çeldirici Analizi | Çeldirici Analizi Akademik Rapor | celdirici_analizi_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| 7. Güvenirlik & Ölçme Hatası | Güvenirlik & Ölçme Hatası Akademik Rapor | guvenirlik_olcme_hatasi_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| 8. Grup Bazlı Analizler | Grup Bazlı Analizler Akademik Rapor | grup_bazli_analizler_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| 9. Bilimsel Grafikler | Bilimsel Grafikler Akademik Rapor | bilimsel_grafikler_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| KTK: Çift Motor Rapor | KTK Çift Motor Akademik Rapor | ktk_cift_motor_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Rasch | Rasch Akademik Rapor | rasch_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: 1PL | 1PL Akademik Rapor | 1pl_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: 2PL | 2PL Akademik Rapor | 2pl_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: 3PL | 3PL Akademik Rapor | 3pl_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: 4PL | 4PL Akademik Rapor | 4pl_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Karşılaştırma | MTK Model Karşılaştırması Akademik Rapor | mtk_model_karsilastirmasi_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Varsayım Denetimi | Varsayım Denetimi Akademik Rapor | varsayim_denetimi_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Çok Modelli DIF Taraması | Çok Modelli DIF Taraması Akademik Rapor | cok_modelli_dif_taramasi_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Çok Kategorili B14 | {b14_model} Akademik Rapor | {model}_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: RSM | RSM Akademik Rapor | rsm_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Python mirt DIF | {pmd_model} DIF Akademik Rapor | {model}_dif_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: DIF Analiz Merkezi | DIF Analiz Merkezi Akademik Rapor | dif_analiz_merkezi_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: GRM Formel DIF | GRM DIF Akademik Rapor | grm_dif_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Çok Kategorili Formel DIF | {poly_model} DIF Akademik Rapor | {model}_dif_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Gözlenen Puan DIF | Gözlenen Puan DIF Akademik Rapor | gozlenen_puan_dif_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: 3PL/4PL Bootstrap DIF | {b34_model} Bootstrap DIF Akademik Rapor | {model}_bootstrap_dif_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| Boyutluluk ve Faktör Analizi | Boyutluluk ve Faktör Analizi Akademik Rapor | boyutluluk_ve_faktor_analizi_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| Yerel Bağımsızlık ve Madde Uyumu | Yerel Bağımsızlık ve Madde Uyumu Akademik Rapor | yerel_bagimsizlik_ve_madde_uyumu_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| Genellenebilirlik Kuramı (G ve D Çalışmaları) | Genellenebilirlik Kuramı Akademik Rapor | genellenebilirlik_kurami_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| Çok Boyutlu MTK (MIRT) | {md_model} MIRT Akademik Rapor | {model}_mirt_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| Eksik Veri ve Atama | Kayıp Veri ve Atama Akademik Rapor | kayip_veri_ve_atama_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Python mirt Rasch | Rasch Akademik Rapor | rasch_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Python mirt 1PL | 1PL Akademik Rapor | 1pl_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Python mirt 2PL | 2PL Akademik Rapor | 2pl_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Python mirt 3PL | 3PL Akademik Rapor | 3pl_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Python mirt 4PL | 4PL Akademik Rapor | 4pl_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Python mirt GRM | GRM Akademik Rapor | grm_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Python mirt PCM | PCM Akademik Rapor | pcm_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Python mirt GPCM | GPCM Akademik Rapor | gpcm_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| MTK: Python mirt NRM | NRM Akademik Rapor | nrm_akademik_rapor.docx | Closed initially; independent; existing panels reused | Real-result route AppTest + panel/DOCX checks |
| 10. Raporlama | KTK Toplu Rapor Akademik Rapor | ktk_toplu_rapor_akademik_rapor.docx | Shared presentation; existing aggregate content | Shared adapter/name tests |

`tests/test_result_panels.py` checks every route/model naming rule, column-method
placement, one expander ancestor, initially closed state, visible controls/critical
warnings/downloads, unchanged tables/figures and DOCX contents. Existing real-result
tests now assert the same panel contract for all 38 analysis routes.
The existing browser E2E flows additionally open/close a result panel, verify another
panel stays closed, check the DOCX button remains accessible, then open panels for
the original numerical/export checks. No model/test/assertion or numeric tolerance
was removed, and no extra calibration is added. Actual frontend toggle evidence is
produced by CI (local Playwright/browser is not installed).
