# 4PL P3: scoped covariance-standardized item-fit inference

## Decision
One new 4PL **Q** method is SCOPED CERTIFIED for the registered reference design.
It is **not ordinary S-X²** and uses no unvalidated chi-square p-value.
All14 P1, six P2 and the prior3PL P3 groups remain unchanged (21→22 relevant groups).
4PL c/d score confidence sets remain; withdrawn c/d Wald inference stays NOT CERTIFIED.

| Independent evidence | Result |
|---|---:|
| Null / power datasets |6000 /600 per predeclared alternative |
| Native convergence / refusals |7800/7800;0 |
| Pooled Type-I / dataset-cluster MCSE |.03381944 /.000696831 |
| Worst one-sided simultaneous itemwise CP upper |.04452608 ≤.07 |
| Nonmonotone power / adjusted CP lower |117/600=.195 /.15769145 |
| Local-dependence power / adjusted CP lower |366/600=.610 /.56085567 |
| Two-stage shape power / adjusted CP interval |20/600=.033333 [.01825950,.05524138]; FAIL |
| Native S-X² identity, all datasets |7.11e-13 ≤1e-8 |
| Post-fit reference kernel median/P95 |.16394 /.25739 seconds |

The null size tolerance remains upper≤.07, pooled95% lower≥.03, MCSE≤.004 and refusal≤.01.
This method is conservative in this design; no exact nominal5% assertion is made.
Size CP alpha=.05/24 covers the24 old/new model-items. Power CP alpha=.025/3
covers the three prospective detection claims. These are separate families;
their individual confidence levels are not a joint95% guarantee for every reported quantity.
A successful detection claim requires lower>.10. The failed third class is never dropped,
strengthened, replaced or certified. No method superiority or FWER/FDR claim follows.

## Protocol, data and independence
Protocol e7f8b3b13501723cd638a07d8055d2cf0167b874 preceded every new dataset; selection e4448a2991b89aa2fd9146fd601ce176b1e874a3
preceded all actual confirmation fits. A rest-score dtype precheck failed before confirmation;
dc4fa11ce917839cbc3d2727c92ea4cb08895045 corrected the exact integer index only. No fits occurred in that failed precheck.
Canonical actual calibration run: https://github.com/Tatung40/psikoanaliz/actions/runs/45833050428 .
Actual PR checkout: 50127d6ce26352d92e445eea6747c9bc02e27aa2; all7800 records independently verified.
N3000/J12, reference truth, latent N(0,1), mirt1.2.0 cold native MAP, GH61, max500, tol1e-4, native default priors.
Native item-fit uses its actual GH41 conditional probabilities; their discrepancy is retained in residual means.
Null seed [202611264201,4,r,0], r0..5999.
Power seeds [202611264202,4,r,k], k1=nonmonotone,k2=dependence,k3=two-stage,r0..599.
Analysis seeds [202611264203,4,r,k] are independent of response generation.
Development used separate roots264101/264102/264103; no3PL campaign or old failed experiment was repeated.

All alternatives were fixed before pilot: item1 quadratic adjusted logit gamma=.25;
conditional dependence lambda1.0 on item2's centered response in the adjusted component logit;
or .6/.4 logistic-stage mixture at locations−1/+1 with slopes2/3. Exact equations are in dgp.py.
The underlying component-logit coefficient is not misreported as an observed Bernoulli odds ratio.
Nonmonotone baseline IRF RMS departure is .13463; two-stage RMS .14893, but nuisance refitting
absorbs most of the latter. Effects were not chosen against confirmation power.

## Why the old power alternative was futile
The historical generator independently redraws theta for item1.
m=E[.1+.8 logistic(1.6(theta+1.25))]=.7422646031.
The feasible 4PL null a1=5,b1=−6,c1=.1,d1=m gives, by common-uniform coupling,
TV(Q^N,P^N)≤N*.8 E exp(−5theta−30)=3000*.8 exp(−17.5).
Any test controlling .05 **at this feasible null** has power≤**.05006026398**.
Independent adaptive integration gives per-person disagreement1.3332883e−8,
below the analytic2.0087993e−8. This is not a claim of uniform size from a single reference cell.
Old data, failed bootstrap/sandwich/analytic experiments and their limits are unchanged.

## Distinguishability without pretending a local minimum is global
All admissible positive-slope, locally independent4PL laws imply monotone
P(X1=1|rest score=t). Successive rest-score likelihood ratios are
e_(t+1)(odds(theta))/e_t(odds(theta)); coordinate derivatives are nonnegative by
Newton's elementary-symmetric inequalities. Thus rest-score likelihoods are TP2,
theta|score is MLR ordered, and the increasing target IRF has increasing conditional expectation.
Data processing plus weighted Bernoulli-KL isotonic projection supplies a **global**
KL lower bound for any admissible null, independently of optimizer convergence.

For the nonmonotone alternative the lower bound is .0005528187783 per person.
Adaptive quadrature independently agrees with GH121/GH241 (normal tail beyond±12 negligible).
A feasible full48-parameter null gives upper .0006688875417; hence closest-null KL lies
approximately in [.00055282,.00066889], N times [1.65846,2.00666].
Its local optimizer reached700 iterations; no global optimum or completed optimization is claimed.
Feasible-null GH121/GH241 probability L1 sensitivity is1.41e−7.

For dependence, feasible-null KL upper .00737009 (N×=22.11); the monotone relaxation
lower bound is0 and is uninformative here. We do not invent a positive global lower bound.
For the two-stage shape, feasible-null upper .0000333094 (N×=.09993);
the very low measured power is consistent with nuisance absorption, not structural nonidentifiability.
Population fits are diagnostics only and never enter confirmation decisions.

## Method and production
The existing frozen joint score/residual Gaussian law is reused.
The full physical native box and joint nuisance dependence enter the batched active-set MAP perturbation;
the prior drift/full prior curvature remain. No estimator, bound, stopping rule or point estimate changes.
Residual means (including prior/quadrature effects) are retained.
Training draws estimate covariance and center, and the separately named Q whitens the residual.
4095 independent reference draws yield p=(1+exceedances)/4096.
KKT, covariance rank and deterministic null-space discrepancies fail closed.
Independent Mahalanobis solves and constrained scipy optimizations verify the algebra.
The ML asymptotics motivating Han et al.'s standardized S-X² (doi:10.1177/01466216221108077)
are **not** asserted to transfer to MAP; independent finite-sample confirmation supplies the scoped evidence.

Only native MAP/N3000/J12/default-cold/GH61/max500/tol1e-4 solutions can show new Q/p/decision columns.
The existing4PL UI D4 defaults max50000/tol1e-10 remain unchanged and OUTSIDE this Q certificate;
the explicitly selected standard-native solver profile is required. No solver/point certificate is silently generalized.
Ordinary S-X² remains visible descriptively; its old p-values remain NA.
No generic user item design/latent distribution is certified.
The same separate labels, seed, scope and refusals flow to UI/CSV/XLSX/ZIP/DOCX and portable Python.
Actual response values, fitted coefficients, method, source/proof fingerprint and persistent independent seed
bind caches, including each actual analysis seed in report bundles. Proof withdrawal invalidates all downloads without recalibration.
Post-fit public API timing is additionally measured by targeted regression; kernel latency above is not mislabeled as whole-page latency.

## Permanent evidence and verification
Independent arithmetic: actual successful run https://github.com/Tatung40/psikoanaliz/actions/runs/45835088987 .
Full original archive SHA256 c93d1f7494b3313766b21d81bf672a3f8eff9751950d8ecdafdefb79c99e8115;
complete native coefficients/pattern counts/logs stay in canonical artifacts.
The selected-method ledger keeps every dataset, p numerator (exact /4096), Q, both S-X² values,
convergence, seeds, hashes, failures, KKT and timing without numerical rounding.
Committed ledger SHA256 1462180d605ac9a9129bbc6e81a54c4bbbe79562b67bef8914fd3353354e6a3d; Git blob 2e40fa8fdc4647655aceb99b6977ed7a92063290.
No historical archive was altered; large-log transport compaction performed no validation/refitting.

Run: python -m validation.p3_4pl_closure.frozen .
Final archival CI checks committed ledgers and source hashes only; it never recalibrates missing caches.
Preserved source/certificate blobs are enumerated in PRESERVED.json.
No full local suite or unrelated model/R/Monte-Carlo campaign was run.
