"""Optional native R mirt 1.48 backend for whitelisted advanced MIRT specifications.

* Optional: Python workflows never require R. When R mirt 1.48 is missing, the backend refuses;
  it never falls back to another engine and never installs packages.
* Only fixed, validated specifications run (``SPECS``). User-supplied R code is never executed:
  requests are validated and serialised as JSON/CSV for the fixed script ``mirt_r_backend.R``.
* Subprocess: argument vector (no shell), own session/process group, bounded timeout, whole group
  killed on timeout, private temporary directory removed afterwards.
* Outputs are EXPERIMENTAL / NOT CERTIFIED. Item SE, confidence intervals and p-values are not produced.
  Agreement between this adapter and direct R mirt calls is adapter fidelity, not independent
  validation; independence comes only from comparisons against a different engine (e.g. SciPy).
"""
from __future__ import annotations

import functools, json, os, shutil, signal, subprocess, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

SCRIPT = Path(__file__).with_name("mirt_r_backend.R")
MIRT_VERSION = "1.48"
STATUS = "EXPERIMENTAL / NOT CERTIFIED"
SPECS = ("M3PL_fixed", "M4PL_fixed", "bfactor")
BFACTOR_TYPES = ("2PL", "graded", "gpcm")
SUPPRESSED = "Item SE, confidence intervals and inferential p-values are not computed (not validated)."


class RBackendUnavailable(RuntimeError):
    """R or the pinned R mirt version is unavailable; no other engine is substituted."""


@functools.lru_cache(maxsize=1)
def engine_available() -> tuple[bool, str]:
    r = shutil.which("Rscript")
    if not r:
        return False, "Rscript not found; the native R mirt backend is unavailable (no engine substitution)."
    try:
        p = subprocess.run([r, "--vanilla", "-e", 'cat(as.character(packageVersion("mirt")))'],
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False, "R mirt version check timed out."
    version = p.stdout.strip()
    if p.returncode or not version:
        return False, "R mirt is not installed; the native R backend is unavailable (no runtime installation)."
    if version != MIRT_VERSION:
        return False, f"R mirt {version} found; the backend requires exactly {MIRT_VERSION}."
    return True, f"R mirt {version}"


def _binary(x: pd.DataFrame) -> np.ndarray:
    v = x.to_numpy()
    if not np.isin(v, [0, 1]).all() or any(np.unique(v[:, j]).size != 2 for j in range(v.shape[1])):
        raise ValueError("Complete 0/1 data with both categories per item required")
    return v.astype(int)


def validate(data, spec: str, *, pattern=None, groups=None, itemtype=None, c=None, u=None,
             n_quadpts=15, max_iter=3000, tol=1e-8) -> dict:
    """Return a validated, JSON-serialisable request; raise ValueError on anything outside the whitelist."""
    if spec not in SPECS:
        raise ValueError(f"Unsupported backend specification: {spec!r}")
    x = pd.DataFrame(data)
    if len(x) < 30 or x.isna().any().any() or not x.columns.is_unique:
        raise ValueError("At least 30 complete rows and unique item names required")
    if not (isinstance(max_iter, (int, np.integer)) and 1 <= max_iter <= 5000):
        raise ValueError("max_iter must be an integer in [1, 5000]")
    if not (np.isfinite(tol) and 1e-12 <= tol <= 1e-3):
        raise ValueError("tol must be in [1e-12, 1e-3]")
    J = x.shape[1]
    req = dict(spec=spec, max_iter=int(max_iter), tol=float(tol), n_quadpts=int(n_quadpts), items=[str(c_) for c_ in x.columns])
    if spec == "bfactor":
        if itemtype not in BFACTOR_TYPES:
            raise ValueError(f"bfactor itemtype must be one of {BFACTOR_TYPES}")
        g = np.asarray(groups)
        if g.shape != (J,) or not np.issubdtype(g.dtype, np.integer):
            raise ValueError("One integer specific-factor code per item required")
        uniq = np.unique(g)
        if len(uniq) < 2 or any(np.count_nonzero(g == k) < 3 for k in uniq):
            raise ValueError("At least two specific factors with >=3 items each required")
        if not 5 <= n_quadpts <= 41:
            raise ValueError("n_quadpts must be in [5, 41]")
        if itemtype == "2PL":
            v = _binary(x)
        else:
            v = x.to_numpy()
            if not np.issubdtype(v.dtype, np.integer) or v.min() < 0:
                raise ValueError("Polytomous bifactor requires non-negative integer categories")
            if any(set(np.unique(v[:, j])) != set(range(v[:, j].max() + 1)) or v[:, j].max() < 2 for j in range(J)):
                raise ValueError("Each item must observe all categories 0..K (K>=2)")
        req.update(itemtype=itemtype, groups=[int(k) + 1 for k in np.searchsorted(uniq, g)])
        return dict(request=req, data=pd.DataFrame(v, columns=req["items"]), q=1 + len(uniq))
    v = _binary(x)
    P = np.asarray(pattern)
    if P.ndim != 2 or P.shape[0] != J or not np.isin(P, [0, 1]).all() or P.shape[1] not in (2, 3):
        raise ValueError("0/1 loading pattern for 2 or 3 correlated dimensions required")
    if (P.sum(1) == 0).any() or any(((P[:, d] == 1) & (P.sum(1) == 1)).sum() < 3 for d in range(P.shape[1])):
        raise ValueError("Every item must load; each dimension needs >=3 pure anchor items")
    if not 5 <= n_quadpts or n_quadpts ** P.shape[1] > 4000 or len(v) * J * n_quadpts ** P.shape[1] > 20_000_000:
        raise ValueError("Gauss-Hermite budget exceeded")
    cvec = np.broadcast_to(np.asarray(.0 if c is None else c, float), (J,)).copy()
    uvec = np.broadcast_to(np.asarray(1. if (u is None or spec == "M3PL_fixed") else u, float), (J,)).copy()
    if spec == "M3PL_fixed" and u is not None and not np.allclose(u, 1):
        raise ValueError("M3PL has upper asymptote 1")
    if not (np.all((0 <= cvec) & (cvec <= .5)) and np.all((.5 <= uvec) & (uvec <= 1)) and np.all(cvec < uvec)):
        raise ValueError("Fixed asymptotes must satisfy 0<=c<=.5, .5<=u<=1, c<u")
    req.update(pattern=P.astype(int).tolist(), c=cvec.tolist(), u=uvec.tolist())
    return dict(request=req, data=pd.DataFrame(v, columns=req["items"]), q=P.shape[1])


def _run(script_args, root: Path, timeout: float):
    proc = subprocess.Popen(script_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True, cwd=root)
    try:
        log, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise TimeoutError(f"R mirt backend exceeded {timeout:.0f} s and was terminated")
    if proc.returncode:
        raise RuntimeError(f"R mirt backend failed (exit {proc.returncode}): " + (log or "")[-2000:])
    return log


def fit(data, spec: str, *, trace=None, timeout: float = 900, **kwargs) -> dict:
    """Fit one whitelisted specification with native R mirt; return structured results."""
    ok, why = engine_available()
    if not ok:
        raise RBackendUnavailable(why)
    v = validate(data, spec, **kwargs)
    req, x, q = v["request"], v["data"], v["q"]
    theta = np.vstack([np.zeros(q), np.eye(q), -np.eye(q)]) if trace is None else np.asarray(trace, float)
    if theta.ndim != 2 or theta.shape[1] != q or not np.isfinite(theta).all():
        raise ValueError("trace must be a finite (n, q) array")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mirt_r_") as tmp:
        root = Path(tmp)
        x.to_csv(root / "data.csv", index=False)
        pd.DataFrame(theta, columns=[f"F{d+1}" for d in range(q)]).to_csv(root / "trace.csv", index=False, float_format="%.17g")
        grid = None
        if spec != "bfactor":
            from mirt.estimation.quadrature import GaussHermiteQuadrature
            grid = GaussHermiteQuadrature(req["n_quadpts"], q)
            g = pd.DataFrame(grid.nodes, columns=[f"F{d+1}" for d in range(q)]); g["weight"] = grid.weights
            g.to_csv(root / "integration.csv", index=False, float_format="%.17g")
        (root / "request.json").write_text(json.dumps(req))
        _run([shutil.which("Rscript"), "--vanilla", str(SCRIPT), str(root)], root, timeout)
        out = json.loads((root / "result.json").read_text())
    out["wall_seconds"] = time.perf_counter() - started
    out["status"] = STATUS
    out["inference"] = SUPPRESSED
    out["validation_scope"] = "Adapter fidelity vs direct R mirt is not independent validation"
    out["values"] = pd.DataFrame(out["values"])
    out["trace"] = theta
    if grid is not None:
        out.update(_scores(x.to_numpy(), np.asarray(out["integration_probabilities"])[..., 1].T, grid, np.asarray(out["covariance"])))
    return out


def _scores(v, p, grid, cov):
    """Common-grid marginal logLik, EAP and posterior SD from R-native item probabilities (binary items)."""
    inv = np.linalg.inv(cov); _, ld = np.linalg.slogdet(cov); nodes = grid.nodes
    logw = np.log(grid.weights) - ld / 2 - np.einsum("kr,rs,ks->k", nodes, inv - np.eye(len(cov)), nodes) / 2
    logw -= logsumexp(logw)
    p = np.clip(p, 1e-300, 1 - 1e-16)
    ell = v @ np.log(p).T + (1 - v) @ np.log1p(-p).T + logw
    norm = logsumexp(ell, axis=1); post = np.exp(ell - norm[:, None])
    mean = post @ nodes; var = post @ nodes ** 2 - mean ** 2
    return dict(common_grid_logLik=float(norm.sum()), EAP=mean, posterior_SD=np.sqrt(np.maximum(var, 0)),
                posterior_note="Posterior SD is person-score uncertainty conditional on item estimates, not item-parameter SE")
