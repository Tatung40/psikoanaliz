# Independent R mirt — rating scale model

Compares the local RSM estimator (`core.irt_rsm`, Rasch identification: slopes 1, latent mean 0, σ free) with
R mirt 1.48 `itemtype = "rsm"` under the same identification, on RSM example datasets irt_027–029. Both engines
define θ on the same scale, so R's own `probtrace`, σ, the maximized log-likelihood and the parameter count
are compared without any parameter mapping. Design and criteria: docstring of `compare.py` (fixed before any
R run). R runs only in `.github/workflows/r-mirt-34.yml`.
