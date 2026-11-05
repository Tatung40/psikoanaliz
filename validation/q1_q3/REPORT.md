# Q1–Q3 certification batch

Base main `b72d6d4`: merged PR141 confirmed. **Zero new certified cells. Zero
Monte Carlo studies. Q4–Q10 implementation deferred.** No production source changed.

## Q1 — actual independent R evidence archived

Four JSON archives preserve original numerical bytes with SHA256 and lossless
compression (including fixed input CSVs), reference estimates/curves and versions,
settings, existing tolerances, convergence, numerical differences and run provenance:

| File under `validation/evidence/` | Actual R source | Scope/outcome |
|---|---|---|
| `r_polytomous.json` | run40957313137, artifact12933083519 | GRM/PCM/GPCM/NRM; four numerical agreement fixtures |
| `r_rsm.json` | run41226545056, artifact13051448638 | Three RSM cases; native likelihoods, not common-grid likelihoods |
| `r_34.json` | same run; also run40151828898/artifact12582966948 | 3PL interior fixture numerical agreement; preserve 4PL NOT VERIFIED and matched-R FAIL/nonconvergence |
| `r_gt.json` | run42013874602, artifact13407453866, main b72d6d4 | Five existing balanced random-facet ANOVA/G/D reference cases |

No R fit was repeated. Where original comparison reports omitted Python coefficients,
exactly eight necessary deterministic Python replays used the original stored data
and existing comparator APIs/settings: four polytomous, three RSM, one3PL. These
replays are explicitly distinguished from original runs. Old reports remain intact.
RSM step estimates absent from the old R result are derived explicitly from its
actual adjacent-category log-odds at theta0, not invented raw coefficient records.

`reference_sources.json` verifies all six historical R/comparison scripts against
current source byte-for-byte. `manifest.json` binds files, production sources and
unchanged p4/p8 evidence. Original negative results remain inspectable.

All existing comparison checks pass for the executed successful cases. This does
not satisfy every preaudit certificate requirement: native quadratures differ,
some original gradient/boundary diagnostics were not recorded, and4PL failed.
Hence Q1's scientific comparison status is PARTIAL, archive integrity is PASS,
and no new general model/SE/coverage certificate is awarded.

## Q2 — exact Git chronology

| Family | Criteria commit | Results commit | Result |
|---|---|---|---|
| p4 | `81b3cfef2024b7106b68ba3da440c236583be6ec` | `7951225dfb0787587ddb7d9c5bcd497e0b429b9d` | VERIFIED ancestry, unchanged thresholds |
| p8 | `b77b6a19ae6453c59ad33331695166f81ed3fc8e` | `467ca897c63c8d22360f86bf722e167337689ed5` | VERIFIED ancestry, unchanged thresholds |

Raw original criteria sources, exact parent chains, timestamps and evidence digests
are in `preregistration.json`. p4 requires retained PR62 ancestry: main squash
`db16a5a` co-commits source/results; main-only history cannot establish the order.
p4 original committer timestamps coincide; the directed Git graph establishes commit
order, not actual computation wall time. Reproduce with `git merge-base --is-ancestor`
for each listed pair after fetching retained commits.

p4's NRM Oakes/adapter repair followed the observed default-SE failure: **post-hoc
method repair**, with unchanged preregistered thresholds, not a new blind validation.
Its source applies coverage/curve criteria only at N2000, not N500. Pooled interval
CP uncertainty is not replicate-cluster coverage uncertainty. The November preaudit
criteria must not be retrospectively described as the October p4 preregistration.
p8 remains NOT MET; verified ordering does not cure N500 recovery failure.

## Q3 — deterministic diagnosis

See `../FOUR_PL_ML_DECISION.md`: ten fixed profile values; five polished stationary
GH61 solutions, none stationary at GH101. Strong fixture-specific practical weak
identification plus quadrature sensitivity; no structural/global/permanent4PL ban.
Three technical protocols were committed before their respective executions. Both
failed scalar-stopping attempts remain archived. No unsupported SE/CI were opened.

## Reproduction without downloads or Monte Carlo

```sh
python -m validation.q1_q3.archive --restore-to /tmp/q123-restored
python -m pytest -q tests/test_certification_q123.py
```

The first command only restores verified original bytes. Optional **explicit**
`--artifacts /tmp/q123-restored` replays the eight existing Python comparisons;
it never calls R or generates Monte Carlo data. Profile commands/protocol commits
are recorded in the three protocol JSONs; use the restored complete R parameter
vector at `2/refit/4PL/R_96wsqzr8/parameters.csv` with `profile.py`/`polish.py`.
Do not overwrite the committed originals when exploring new settings.

Targeted tests verify archive bytes, actual R mapping, unchanged criteria/parent
chains, failed fits retained and fixed-coefficient stationarity without refits.
The existing CI full-regression opt-in is requested once. To honor ZERO MONTE CARLO,
the explicit PR marker selects archived-evidence mode only after a fail-closed
changed-path whitelist. Original quick-study commands remain unchanged for all
other changes; evidence and every browser check remain required. No test or
scientific threshold is removed. CI success is not scientific certification.
