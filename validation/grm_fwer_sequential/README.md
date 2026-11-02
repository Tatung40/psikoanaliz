# Fixed-look GRM-LR FWER study

Protocol9d01c0f was committed before new outcomes. Only the same two latent-null
conditions are studied: N500/500, J10/C4, fixed P01-P08 anchors, P09/P10 family,
impact-.5/0. GH101 cold fits and independent GH161 gates are unchanged. Three
unique omnibus models replace seven component-family fits solely because no
threshold/slope component testing was requested. No old ordinal LR is rerun.
No historical GRM fits are pooled: source/environment equivalence failed.

Execute opt-in on the registered Linux/Python environment with single BLAS
threads: `python -m validation.grm_fwer_sequential.runner`. There is no automatic
retry, seed substitution or overwrite. Checkpoint/progress records preserve all
attempts. Only fixed500/1000/1500 looks can stop for statistical classification;
resource/integrity stops remain INCONCLUSIVE if no registered look is reached.

Six two-sided CP intervals use tail alpha.05/12, giving joint coverage>=95%
over both conditions and all looks without independence between conditions.
SCOPED CONTROL means the conservative upper bound<=.075, NOT nominal .05
control. SCOPED INFLATION means the missing-outcome-robust lower bound>.05.
Both can mathematically coexist. No near-nominal, superiority or general FWER
certificate is issued. Invalid fits are conservative operational events; they
cannot create evidence of inferential Type-I inflation. Valid-only estimates
are conditional descriptions, not a missingness correction.

Canonical compact records preserve every seed, data/RNG hash, attempted fit,
parameter vector, full gradient/derivative vectors, information eigenvalues and
conditions, LR/df/p, corrections, decisions, failures, time and memory record.
Only deterministically reproducible dense-information matrices are omitted as
redundant caches; each retains its shape and float64 SHA256. The original full
execution archive is preserved unchanged in the execution workspace, with its
digest in projection.json. It is not misrepresented as committed. All omitted
matrices can be rebuilt from saved estimates/data/source, without optimization.
The first30 pairs of original NPZ response matrices are bundled, not regenerated
for archiving; all later matrices replay deterministically and must match the
original archived ordinal-response and actual random-draw hashes.

After completion: `python -m validation.grm_fwer_sequential.archive`, then
`python -m validation.grm_fwer_sequential.audit`. The audit independently
recomputes every Holm/BH decision and binomial CP endpoint, checks every saved
response hash against frozen original records, verifies paired actual draw
hashes and source/old-archive identity; it never calls a fit or regenerates
responses by default. Full data replay is separately opt-in.
Routine CI checks canonical evidence and small fixtures only. Production
algorithms, inference gates and scientific-status labels are not promoted by
this study. Lord/Wald/SIBTEST/Raju/selected-anchor FWER remain outside scope.

The logical compact gzip is stored as two binary parts to respect the delivery
API limit. records.compact.parts.json records individual and reconstructed
SHA256 identities. record_stream transparently reassembles exact bytes and
fails on corruption. No scientific values are serialized again or changed.
The archive command needs the original workspace cache; a published checkout
can audit the committed parts directly, without fitting or regenerating data.
