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

The aggregate report retains its original Word exporter. Existing .py/ZIP controls
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
