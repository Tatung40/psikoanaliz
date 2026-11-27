"""One prospectively registered cold NRM pair; no retries or advanced fits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import mirt
from mirt.results.fit_result import FitResult
from validation.mirt_stage1 import compare as H

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path("validation/mirt_core_production/PROTOCOL.md")
PROTOCOL_COMMIT = "af024e73e4be99d0db1a2a0df400039f98ab6095"
CRITERIA_SHA256 = "5b666c899ee9cc53e3683b99fe6d192816fdcc0731ca0a32b61607f612910d44"
DATA_SHA256 = "e43e687f10bf1231ba1c386a4d68bba4fc5dd22f81d31f92690e5e550e742454"
TIMEOUT = 300


def _write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_protocol():
    current = (ROOT / PROTOCOL).read_bytes()
    committed = subprocess.check_output(
        ["git", "show", PROTOCOL_COMMIT + ":" + PROTOCOL.as_posix()], cwd=ROOT)
    if current != committed or b"Prospective optional C1: one cold-start NRM" not in committed:
        raise ValueError("Prospective C1 protocol differs from its committed version")
    if hashlib.sha256((H.ROOT / "criteria.json").read_bytes()).hexdigest() != CRITERIA_SHA256:
        raise ValueError("Original preregistered numerical criteria changed")
    return dict(commit=PROTOCOL_COMMIT, path=PROTOCOL.as_posix(),
                sha256=hashlib.sha256(committed).hexdigest())


def _bounded(worker, out, label, extra=()):
    """Bound a worker and all its R descendants; preserve both log streams."""
    stdout_path = out / (label + "_worker_stdout.log")
    stderr_path = out / (label + "_worker_stderr.log")
    code = ("import sys; from validation.mirt_core_production.nrm_cold import " +
            worker + "; " + worker + "(*sys.argv[1:])")
    command = [sys.executable, "-c", code, str(out.resolve()), *extra]
    started = time.monotonic()
    timed_out = False
    code = None
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                    start_new_session=True)
            try:
                code = proc.wait(timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = proc.wait()
        if timed_out:
            raise TimeoutError(label + " exceeded preregistered 300 seconds")
        if code:
            raise RuntimeError(label + " exited " + str(code) + "; original stderr retained")
        return time.monotonic() - started
    finally:
        _write(out / (label + "_runtime.json"), dict(
            elapsed_seconds=time.monotonic() - started, limit_seconds=TIMEOUT,
            timed_out=timed_out, returncode=code))


def _r_worker(out, mode):
    # Keep the original native R reference and r_run implementation byte-identical.
    # The outer process group limit bounds its original 1800s child timeout to300s.
    H.r_run(Path(out), mode)



def _python_admissibility(outcome):
    from mirt.estimation.base import _parameter_bounds
    model = outcome["fit"].model
    pattern = np.asarray(outcome["pattern"], bool)
    params = model.parameters
    slopes = np.asarray(params["slopes"])
    intercepts = np.asarray(params["intercepts"])
    fixed = (np.array_equal(slopes[:, 0, :], np.zeros_like(slopes[:, 0, :])) and
             np.array_equal(intercepts[:, 0], np.zeros_like(intercepts[:, 0])) and
             np.all(slopes[~np.broadcast_to(pattern[:, None, :], slopes.shape)] == 0))
    at_bound = []
    inside = True
    bounds = {}
    for name, value in params.items():
        lower, upper = _parameter_bounds(model, name)
        bounds[name] = [lower, upper]
        free = np.asarray(value)[model.free_parameter_masks[name]]
        inside = inside and bool(np.isfinite(free).all() and np.all(free >= lower) and np.all(free <= upper))
        for coordinate in np.argwhere(model.free_parameter_masks[name]):
            index = tuple(coordinate)
            v = float(value[index])
            if min(v - lower, upper - v) <= np.sqrt(np.finfo(float).eps) * max(1, abs(v)):
                at_bound.append(dict(parameter=name, coordinate=list(map(int, coordinate)), value=v))
    covariance = np.asarray(outcome["covariance"])
    latent = bool(np.linalg.eigvalsh(covariance).min() > 0 and
                  np.allclose(np.diag(covariance), 1, atol=1e-12, rtol=0))
    return dict(valid=bool(fixed and inside and latent and not at_bound),
                reference_and_structural_zeros=bool(fixed), inside_native_boxes=bool(inside),
                identified_latent_metric=latent, at_bound=at_bound, actual_native_bounds=bounds)


def _r_admissibility(directory):
    raw = pd.read_csv(directory / "raw_parameters.csv")
    free = raw.loc[raw["est"].astype(str).str.lower() == "true"]
    at_bound = []
    inside = True
    for _, row in free.iterrows():
        value, lower, upper = (float(row[name]) for name in ("value", "lbound", "ubound"))
        inside = inside and bool(np.isfinite(value) and lower <= value <= upper)
        if min(value - lower, upper - value) <= np.sqrt(np.finfo(float).eps) * max(1, abs(value)):
            at_bound.append(dict(item=str(row["item"]), parameter=str(row["name"]),
                                 parnum=int(row["parnum"]), value=value))
    return dict(valid=bool(inside and not at_bound), free_parameters=len(free),
                inside_native_boxes=bool(inside), at_bound=at_bound)


def _python_worker(out):
    out = Path(out)
    cfg = json.loads((out / "settings.json").read_text())
    data = pd.read_csv(out / "data.csv")
    # Public production estimator, unchanged cold starts; no saved item solution.
    outcome = H.fit(data, "NRM", cfg["pattern"], correlated="full",
                    labels=["0", "1", "2", "3"], n_quadpts=15,
                    max_iter=3000, tol=1e-9)
    fit = outcome["fit"]
    np.savez(out / "python_parameters.npz", **fit.model.parameters,
             covariance=outcome["covariance"])
    raw = outcome["raw_fit"].model.parameters
    np.savez(out / "python_native_raw_parameters.npz", **raw)
    _write(out / "python_fit.json", fit.to_dict())
    _write(out / "python_outcome.json", dict(
        covariance=np.asarray(outcome["covariance"]).tolist(),
        orientation=outcome["orientation"]))
    _write(out / "python_metadata.json", dict(
        engine="Python mirt " + mirt.__version__, python=platform.python_version(),
        converged=bool(fit.converged), logLik=fit.log_likelihood,
        AIC=fit.aic, BIC=fit.bic, n_parameters=fit.n_parameters,
        iterations=fit.n_iterations, admissibility=_python_admissibility(outcome)))


def run(out):
    protocol = verify_protocol()  # before fixture generation or estimation
    if out.exists():
        raise ValueError("Output already exists; a second attempt/overwrite is prohibited")
    out.mkdir(parents=True)
    result = dict(profile="NRM_2D_COLD_C1", status="BLOCKED",
                  R_execution="NOT EXECUTED", Python_execution="NOT EXECUTED",
                  protocol=protocol, timing_seconds={}, calibration_attempts=dict(Python=0, R=0),
                  provenance="Exactly one prospective cold pair; no warm estimates, retries or continuations",
                  limitations="Finite registered fixture only; no population inference, global optimum or quadrature-accuracy claim.")
    try:
        if mirt.__version__ != "1.2.0":
            raise ValueError("Unregistered Python mirt version")
        cfg, data, engine, grid = H.inputs("NRM_2D", out)
        if cfg["data_sha256"] != DATA_SHA256 or cfg["criteria_sha256"] != CRITERIA_SHA256:
            raise ValueError("Original dataset/criteria hashes changed")
        cfg.update(max_iter=3000, tol=1e-9,
                   initialization="Cold native nominal_start; zero category intercepts; identity latent covariance in BOTH engines",
                   protocol_commit=PROTOCOL_COMMIT)
        _write(out / "settings.json", cfg)
        _write(out / "PROTOCOL_SNAPSHOT.json", protocol)
        (out / "PROTOCOL.md").write_bytes((ROOT / PROTOCOL).read_bytes())
        _write(out / "sources.json", {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (H.ROOT / "compare.py", H.ROOT / "reference.R", H.ROOT / "criteria.json",
                         ROOT / "core/irt_multidim.py", Path(__file__).resolve())})
        starts = dict(slopes=H.nominal_start(cfg["pattern"], 4).tolist(),
                      intercepts=np.zeros((cfg["k"], 4)).tolist(),
                      covariance=np.eye(cfg["q"]).tolist())
        _write(out / "cold_starts.json", starts)
        result["settings"] = cfg
        result["R_execution"] = "PREFLIGHT ATTEMPTED"
        _write(out / "report.json", result)
        result["timing_seconds"]["R_preflight"] = _bounded("_r_worker", out, "R_preflight", ("preflight",))
        result["R_execution"] = "FIXED PARAMETER PREFLIGHT EXECUTED"
        engine.set_parameters(slopes=np.asarray(starts["slopes"]),
                              intercepts=np.asarray(starts["intercepts"]))
        trace = pd.read_csv(out / "trace.csv").to_numpy()
        p, info = H.native_outputs(engine, trace)
        rp, ri = H.arrays(out / "preflight", cfg)
        covariance = np.full((cfg["q"], cfg["q"]), .25)
        np.fill_diagonal(covariance, 1)
        criteria = json.loads((H.ROOT / "criteria.json").read_text())
        checks = dict(
            probabilities=H.discrepancies(p, rp, criteria["mapping_probability"]),
            information=H.discrepancies(info, ri, criteria["mapping_information"]),
            prior=H.discrepancies(H.masses(grid.nodes, grid.weights, covariance),
                                 pd.read_csv(out / "preflight/prior.csv").weight.to_numpy(),
                                 criteria["mapping_prior_mass"]))
        result["preflight"] = checks
        if any(c["status"] != "PASS" for c in checks.values()):
            raise ValueError("NOT COMPARABLE: original fixed mapping gates failed; zero calibrations")
        result["Python_execution"] = "CALIBRATION ATTEMPTED"
        result["calibration_attempts"]["Python"] = 1
        _write(out / "report.json", result)
        result["timing_seconds"]["Python"] = _bounded("_python_worker", out, "Python")
        result["Python_execution"] = "EXECUTED"
        result["Python"] = json.loads((out / "python_metadata.json").read_text())
        if not result["Python"]["converged"]:
            result["status"] = "NOT CONVERGED"
            raise ValueError("Python did not converge; independent R fit cannot repair production convergence")
        if not result["Python"]["admissibility"]["valid"]:
            result["status"] = "BOUNDARY OR INADMISSIBLE SOLUTION / NOT VALIDATED"
            raise ValueError("Python admissibility/boundary gate failed; no R production promotion attempted")
        result["R_execution"] = "CALIBRATION ATTEMPTED"
        result["calibration_attempts"]["R"] = 1
        _write(out / "report.json", result)
        result["timing_seconds"]["R"] = _bounded("_r_worker", out, "R", ("fit",))
        result["R_execution"] = "CALIBRATION EXECUTED"
        fit = FitResult.from_dict(json.loads((out / "python_fit.json").read_text()))
        native = json.loads((out / "python_outcome.json").read_text())
        outcome = dict(fit=fit, covariance=np.asarray(native["covariance"]))
        result = H.compare_pair(outcome, out / "fit", cfg, data, grid, result, criteria)
        result["R_admissibility"] = _r_admissibility(out / "fit")
        if not result["R_admissibility"]["valid"]:
            result["status"] = "BOUNDARY OR INADMISSIBLE SOLUTION / NOT VALIDATED"
        # Existing comparator requires native/common LL identity <=1e-6.
        # Any failure is retained, never repaired by a second fit or hidden rounding.
    except Exception as exc:
        result["blocker"] = type(exc).__name__ + ": " + str(exc)
        if isinstance(exc, TimeoutError):
            result["status"] = "TIMEOUT / NOT VALIDATED"
    finally:
        result["worker_runtime"] = {path.stem: json.loads(path.read_text()) for path in sorted(out.glob("*_runtime.json"))}
        _write(out / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output)
    print("NRM_COLD_REPORT=" + json.dumps(report, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if report["status"] == "NUMERICALLY VALIDATED / NOT CERTIFIED" else 1)
