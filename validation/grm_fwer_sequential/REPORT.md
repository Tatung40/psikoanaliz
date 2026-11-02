# GRM-LR sequential FWER — user-stopped evidence

**INCONCLUSIVE / NOT CERTIFIED.** Explicit user stop; no new fits or resumed1000/1500 stages.

Completed852 seed pairs,1704 recorded analyses, all1704 numerically valid;5112 recorded unique omnibus fits. Total wall4535.65s. Original protocol and acceptance bounds are unchanged. Only registered look500 was reached;1000/1500 were NOT EXECUTED.

| Registered condition | n | Holm events | FWER | MCSE | Simultaneous CP interval | Decision |
|---|---:|---:|---:|---:|---|---|
| impact−.5 | 500 | 29 | 0.0580 | 0.010453 | [0.033928, 0.091237] | INCONCLUSIVE |
| impact0 | 500 | 25 | 0.0500 | 0.009747 | [0.027860, 0.081499] | INCONCLUSIVE |

These intervals retain the original .05/12 tail allocation over both conditions
and all three planned looks. Neither upper<=.075 nor lower>.05 was established
at look500. No nominal .05 control, near-nominal or general certification claim.

## Preserved incomplete prefix

- impact−.5: 56/852 Holm events (0.065728); BH event rate0.068075. Descriptive only.
- impact0: 54/852 Holm events (0.063380); BH event rate0.065728. Descriptive only.

The unplanned852-pair prefix and its producer fixed-n intervals are preserved
as descriptive records, not a registered sequential look or certificate.
A853rd pair (seed2026110952, both impacts) was dispatched when the user stop
interrupted Future.result. Executor shutdown drained workers, but outputs
were uncollected: UNKNOWN, not converged/failed results invented by the auditor.
user_stop.json and summary.json explicitly retain both unknown analyses and
worst-case empirical bounds (one possible event per condition). No retry or
replacement. The original producer-summary stop label was an exception-path
default; its snapshot is retained and the final summary records the user stop.

## Integrity and resources

Independent raw audit: INDEPENDENT RAW AUDIT PASS; Holm/BH, paired MCSE, CP endpoint equations, LR/df, stored gradients/information gates and source/old-archive identities verified. No fits or response regeneration during final audit.

Actual first30 matrix pairs are bundled losslessly. All primary raw results
remain in records.compact.jsonl.gz; only reproducible dense-matrix caches are
projected, with shapes/eigenvalues/conditions/full matrix hashes retained.
The unchanged full execution archive remains in the workspace; its SHA256
is in projection.json. No previous PR evidence is edited or pooled.

Observed producer parent lifetime high-water mark781.16MiB; worker114.31MiB.
Two workers, one BLAS thread each; recorded total wall4535.65s, below18000s.
CPU time overlapped the authorized regression run; timing is environment-specific.

## Scope

Only N500/500,J10/C4, latent-null GRM, fixed P01-P08 anchors, P09/P10 omnibus
df4 family, Holm alpha.05, impacts−.5/0. GH101/GH161 gates unchanged.
BH and paired differences are secondary descriptions, not method-superiority
or broad FDR claims. Lord/Wald/SIBTEST/Raju/selected-anchor FWER remain
unvalidated. No production GRM inference gates or scientific labels promoted.

The logical compact gzip is stored as two binary parts to respect the delivery
API limit. records.compact.parts.json records individual and reconstructed
SHA256 identities. record_stream transparently reassembles exact bytes and
fails on corruption. No scientific values are serialized again or changed.
The archive command needs the original workspace cache; a published checkout
can audit the committed parts directly, without fitting or regenerating data.
