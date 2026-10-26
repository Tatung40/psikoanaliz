# Saved 4PL observed-information audit — no refits

Actual derivative calculations executed in Python against the **saved R solution**
from run 40151828898 / commit 498be38 / artifact 12582966948. N=3000, J=12,
unchanged 61-GH respondent marginal logLik. Input/code hashes, software versions,
all stencils and results are in report.json. Raw matrices/eigenvectors are in
information_matrices.npz (`allow_pickle=False` suffices). Prior evidence untouched.

Mean0/variance1 are fixed and excluded: 48 physical item coordinates (a,b,c,u),
with u called d only in the existing machine schema. This is the **observed**
negative Hessian: analytic marginal scores, including posterior changes, are
numerically differentiated; it is not expected/complete-data information.

Two second-order derivative calculations used h=cuberoot(machine epsilon)
×max(1,|parameter|) and h/2. Central stencils apply in the interior; second-order
feasible one-sided stencils apply at exact bounds. No bound or likelihood floor
was changed. The existing stable log-probability implementation was reused.

There are 11 exact active bounds: four c=0, six u=1, M9 a=5. Their physical
scores all point out of the feasible box, consistent with active-bound signs.
These coordinates are held fixed for the **37-coordinate active-face** result.
The full 48-coordinate one-sided Hessian is also provided. Its arbitrary
normal-direction eigenvectors are not feasible two-sided perturbations.

| Quantity | Full 48 coordinates | Active face, 37 coordinates |
|---|---:|---:|
| Smallest eigenvalues | .0185760, .0220115, .0752291 | .0378910, .1069814, .2881509, .4202179 |
| Spectral condition number | 879872 | 430814 |
| Two-step spectral difference | 1.0363e-5 | 1.0356e-5 |
| Antisymmetric component norm | 1.2146e-6 | 5.0895e-7 |

Both matrices are positive definite at both step sizes. Active-face relative
step difference is 6.34e-10; its smallest eigenvalue changes by 5.01e-10.
The smallest eigenvalue exceeds the combined observed step/asymmetry scale
1.0865e-5. This is a matrix-perturbation comparison, **not an invented statistical
acceptance threshold or a rigorous two-step error bound**. No material derivative
instability is detected at these steps; that is not a convergence certificate.

The weakest physical mode has eigenvalue .0378910 and **99.14% squared loading
on M12**: principally M12 a=-.93925, b=.30563, u=.12558 (sign arbitrary).
The fourth mode, eigenvalue .420218, has **85.73% loading on M7**, principally
a=-.89150, b=.21517, u=.12365. All modes are saved; leading loadings for the
first four are reported. These contributions depend on physical coordinate
units. Diagonal information scaling reduces the condition number to **2349.5**,
but still shows small eigenvalues (.001373, .001709): units alone do not explain
the near-collinearity. Scaled and physical vectors must not be conflated.

Maximum raw physical score is 4.8053 at an active upper bound; that blocked
outward score is not itself a KKT violation. Maximum feasible box-tangent score
is .00169485; diagonally scaled interior score is .00030591. These are reported
without a new stationarity cutoff. The interior solution is not certified KKT.

Only because the active-face Hessian is positive definite and its positivity
exceeds observed step perturbations was a local quadratic prediction computed:
coarse **3.508795012e-6**, fine **3.508795059e-6** logLik improvement. The predicted
physical displacement stays feasible, holds active coordinates fixed, and includes
M12 da=.0127753, db=-.00411148, du=-.00168955. It was **not applied or refitted**;
no nonlinear likelihood evaluation of this candidate was made. This is a local
Taylor estimate, not a demonstrated remaining maximum or global bound.

Scientific interpretation: stable but ill-conditioned local curvature supports
**weak local identification/coupled slow EM**, especially M12; no exact local
rank deficiency or major finite-difference instability is demonstrated. This is
not a proof of global identification, optimizer uniqueness, or valid SE/CI at
boundaries. Runtime evidence still gives **NOT CONVERGED**: 29/50 changes fail
1e-8, maximum change 2.164495e-6, final logLik increment about 2.3e-9. A tiny
objective increment is compatible with parameter drift along weak directions.

Two targeted tests verify cross-curvature at physical boundaries and block
quadratic eligibility for indefinite/step-unstable matrices. Together with the
two existing score checks: **4 passed**. No model fitting or Monte Carlo occurred.

Reproduction (do not run again merely to inspect):
`OPENBLAS_NUM_THREADS=1 python -m validation.r_mirt_phase1.information_audit --source EXTRACTED/refit --output NEW_DIR`
