"""Permanent offline audit of the committed actual cold-NRM archive.

No model fits, subprocesses, downloads, extraction dependency or regenerated
reference outputs. Missing evidence fails closed. Archived binary-file hashes
whose original bytes were not embedded are retained, explicitly not re-created.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RAW = HERE / "NRM_COLD_RAW.json"
REFERENCE = HERE / "NRM_COLD_REFERENCE.json"
RAW_GIT_BLOB = "f0c0e70ad4cbd1b5fc5ef159ed3132ee4c6f497f"
PROTOCOL_COMMIT = "af024e73e4be99d0db1a2a0df400039f98ab6095"
CRITERIA_SHA256 = "5b666c899ee9cc53e3683b99fe6d192816fdcc0731ca0a32b61607f612910d44"
DATA_SHA256 = "e43e687f10bf1231ba1c386a4d68bba4fc5dd22f81d31f92690e5e550e742454"
# Existing mathematical-identity tolerance, distinct from unchanged scientific gates.
IDENTITY_ATOL = 1e-10
TEXT_FILES = (
    "data.csv", "cold_starts.json", "sources.json", "trace.csv", "python_fit.json",
    "python_metadata.json", "fit/parameters.csv", "fit/raw_parameters.csv",
    "fit/metadata.json", "fit/eap.csv", "fit/probabilities.csv", "fit/information.csv")
SOURCES = frozenset((
    "validation/mirt_stage1/compare.py", "validation/mirt_stage1/reference.R",
    "validation/mirt_stage1/criteria.json", "core/irt_multidim.py",
    "validation/mirt_core_production/nrm_cold.py"))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def _matrix(text):
    rows = list(csv.reader(io.StringIO(text)))
    return rows[0], np.asarray(rows[1:], float)


def _close(actual, expected, name, tolerance=IDENTITY_ATOL):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    _require(actual.shape == expected.shape, name + ": shape mismatch")
    _require(np.isfinite(actual).all() and np.isfinite(expected).all(),
             name + ": nonfinite number")
    _require(np.allclose(actual, expected, atol=tolerance, rtol=0),
             name + ": numerical parity failed")


def _max_error(actual, expected, report, name):
    value = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    _close([value], [report["comparison"][name]["max_absolute"]], name + " recorded discrepancy")
    return value


def verify(payload_path=RAW, reference_path=REFERENCE, project=PROJECT):
    payload_path, reference_path, project = Path(payload_path), Path(reference_path), Path(project)
    raw_bytes = payload_path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw_bytes)).encode() + b"\0" + raw_bytes).hexdigest()
    _require(blob == RAW_GIT_BLOB, "Canonical raw archive blob changed")
    payload = json.loads(raw_bytes)
    reference = json.loads(reference_path.read_text())
    provenance = reference["provenance"]
    _require(provenance["run_id"] == 45960533214 and provenance["job_id"] == 140377033939,
             "Canonical execution provenance changed")
    _require(provenance["protocol_commit"] == PROTOCOL_COMMIT, "Protocol provenance changed")
    _require(payload["new_fits"] == 0, "Archival operation cannot claim new fits")
    report = payload["report"]
    settings = payload["settings"]
    _require(report == reference["report"], "Actual raw report differs from pinned reference summary")
    _require(settings == report["settings"], "Raw settings differ from actual report")
    _require(report["status"] == "NUMERICALLY VALIDATED / NOT CERTIFIED",
             "Numerical evidence must not become an inferential certificate")
    _require(report["protocol"]["commit"] == PROTOCOL_COMMIT, "Unregistered protocol")
    _require(report["calibration_attempts"] == {"Python": 1, "R": 1}, "Unexpected fit/retry count")
    _require(report["Python_execution"] == "EXECUTED" and report["R_execution"] == "CALIBRATION EXECUTED",
             "Actual independent execution missing")
    _require(report["Python"]["engine"] == "Python mirt 1.2.0" and report["R"]["version"] == "1.48",
             "Canonical engine versions changed")
    _require(report["Python"]["converged"] is True and report["R"]["converged"] is True,
             "Canonical fit did not converge")
    _require(report["Python"]["admissibility"]["valid"] is True and
             report["R_admissibility"]["valid"] is True and report["R"]["constraints_valid"] is True,
             "Canonical admissibility failed")
    _require((settings["model"], settings["N"], settings["k"], settings["q"],
              settings["seed"], settings["n_quadpts"], settings["max_iter"], settings["tol"]) ==
             ("NRM", 400, 10, 2, 20261132, 15, 3000, 1e-9), "Cold design changed")
    _require(settings["criteria_sha256"] == CRITERIA_SHA256 and
             settings["data_sha256"] == DATA_SHA256, "Original data/criteria hash changed")
    _require("Cold native nominal_start" in settings["initialization"], "Warm starts substituted")
    for worker in report["worker_runtime"].values():
        _require(worker["limit_seconds"] == 300 and worker["returncode"] == 0 and
                 worker["timed_out"] is False and 0 <= worker["elapsed_seconds"] <= 300,
                 "Canonical bounded worker did not complete")
    files = payload["file_sha256"]
    captured = {}
    for filename in TEXT_FILES:
        _require(isinstance(payload.get(filename), str), "Missing actual captured file: " + filename)
        digest = hashlib.sha256(payload[filename].encode("utf-8")).hexdigest()
        _require(digest == files[filename], "Captured-file hash mismatch: " + filename)
        captured[filename] = digest
    _require(captured["data.csv"] == DATA_SHA256, "Canonical response matrix changed")
    sources = json.loads(payload["sources.json"])
    _require(set(sources) == SOURCES, "Canonical source set changed")
    for relative, expected in sources.items():
        path = Path(relative)
        _require(not path.is_absolute() and ".." not in path.parts, "Unsafe source path")
        _require(hashlib.sha256((project / path).read_bytes()).hexdigest() == expected,
                 "Frozen scientific source hash changed: " + relative)
    criteria = json.loads((project / "validation/mirt_stage1/criteria.json").read_text())
    threshold_keys = dict(slopes="calibration_parameters", intercepts="calibration_parameters",
        covariance="calibration_covariance", probabilities="calibration_probability",
        information="calibration_information", EAP="calibration_EAP",
        posterior_SD="calibration_posterior_SD", common_logLik="common_logLik",
        Python_native_LL="native_common_logLik", R_native_LL="native_common_logLik")
    _require(set(report["comparison"]) == set(threshold_keys), "Incomplete original comparison gates")
    for name, key in threshold_keys.items():
        gate = report["comparison"][name]
        _require(gate["status"] == "PASS" and gate["tolerance"] == criteria[key] and
                 np.isfinite(gate["max_absolute"]) and 0 <= gate["max_absolute"] <= criteria[key],
                 "Original numerical gate failed/changed: " + name)
    for name, key in dict(probabilities="mapping_probability", information="mapping_information",
                          prior="mapping_prior_mass").items():
        gate = report["preflight"][name]
        _require(gate["status"] == "PASS" and gate["tolerance"] == criteria[key] and
                 0 <= gate["max_absolute"] <= criteria[key], "Original preflight gate changed: " + name)
    names, responses = _matrix(payload["data.csv"])
    pattern = np.asarray(settings["pattern"], bool)
    _require(responses.shape == (400, 10) and pattern.shape == (10, 2), "Invalid data/model shape")
    _require(np.isin(responses, range(4)).all(), "Invalid nominal response codes")
    starts = json.loads(payload["cold_starts.json"])
    expected_start = np.zeros((10, 4, 2))
    expected_start[:, 1, :] = pattern
    for category in (2, 3):
        expected_start[:, category, :] = pattern * (.8 * np.sin(category * np.arange(1, 3)))
    _close(starts["slopes"], expected_start, "cold nominal_start")
    _close(starts["intercepts"], np.zeros((10, 4)), "cold intercepts")
    _close(starts["covariance"], np.eye(2), "cold latent covariance")
    native = json.loads(payload["python_fit.json"])
    python_metadata = json.loads(payload["python_metadata.json"])
    r_metadata = json.loads(payload["fit/metadata.json"])
    _require(python_metadata == report["Python"] and r_metadata == report["R"],
             "Raw engine metadata differs from actual report")
    _require(native["model"]["name"] == "NRM" and native["model"]["n_factors"] == 2 and
             native["model"]["item_names"] == names, "Native Python model/category alignment changed")
    _require(native["n_parameters"] == 67 and native["n_observations"] == 400 and native["converged"] is True,
             "Native Python fit metadata changed")
    arrays = payload["python_parameters"]
    _require(set(arrays) == {"slopes", "intercepts", "covariance"}, "Missing actual NPZ coefficient arrays")
    py_a, py_d, py_c = (np.asarray(arrays[k], float) for k in ("slopes", "intercepts", "covariance"))
    _require(py_a.shape == (10, 4, 2) and py_d.shape == (10, 4) and py_c.shape == (2, 2),
             "Invalid coefficient shapes")
    for key in ("slopes", "intercepts"):
        _require(np.array_equal(np.asarray(arrays[key]), np.asarray(native["parameters"][key])),
                 "Python JSON/NPZ coefficient identity failed: " + key)
    _require(np.array_equal(py_c, np.asarray(native["latent_covariance"])), "Python covariance JSON/NPZ differs")
    _close(py_a[:, 0], np.zeros((10, 2)), "Python reference-category slopes", 0)
    _close(py_d[:, 0], np.zeros(10), "Python reference-category intercepts", 0)
    _close(py_a[~np.broadcast_to(pattern[:, None, :], py_a.shape)],
           np.zeros(np.count_nonzero(~np.broadcast_to(pattern[:, None, :], py_a.shape))),
           "Python structural zeros", 0)
    r_rows = _rows(payload["fit/parameters.csv"])
    _require(len(r_rows) == 40, "Incomplete R coefficients")
    r_a, r_d = np.full((10, 4, 2), np.nan), np.full((10, 4), np.nan)
    seen = set()
    for row in r_rows:
        item, category = int(row["Item"]), int(row["Category"])
        _require((item, category) not in seen and 0 <= item < 10 and 0 <= category < 4,
                 "Invalid/duplicate R item-category coordinate")
        seen.add((item, category))
        r_a[item, category] = [float(row["a1"]), float(row["a2"])]
        r_d[item, category] = float(row["d"])
    _require(np.isfinite(r_a).all() and np.isfinite(r_d).all(), "Missing R category coefficient")
    raw_r = _rows(payload["fit/raw_parameters.csv"])
    orientation = np.asarray(r_metadata["orientation"])
    for item, label in enumerate(names):
        rows = [row for row in raw_r if row["item"] == label]
        item_slopes = np.array([float(next(row["value"] for row in rows if row["name"] == f"a{j+1}"))
                                for j in range(2)])
        scores = [float(row["value"]) for row in rows if row["name"].startswith("ak")]
        intercepts = [float(row["value"]) for row in rows if row["name"].startswith("d")]
        _require(len(scores) == 8 and len(intercepts) == 4, "Incomplete native gpcm_mats coefficients")
        transformed = np.asarray(scores).reshape((4, 2), order="F") * item_slopes * orientation
        _close(transformed, r_a[item], "R native gpcm_mats/CSV slope transformation")
        _close(intercepts, r_d[item], "R native/CSV intercept identity")
    r_c = np.asarray(r_metadata["covariance"], float)
    differences = dict(slopes=_max_error(py_a, r_a, report, "slopes"),
                       intercepts=_max_error(py_d, r_d, report, "intercepts"),
                       covariance=_max_error(py_c, r_c, report, "covariance"))
    score_columns, scores = _matrix(payload["fit/eap.csv"])
    _require(score_columns == ["F1", "F2", "SD1", "SD2"] and scores.shape == (16, 4) and
             np.isfinite(scores).all() and (scores[:, 2:] >= 0).all(), "Invalid captured R EAP/SD table")
    for engine, metadata in (("Python", report["Python"]), ("R", report["R"])):
        ll, aic, bic = (float(metadata[k]) for k in ("logLik", "AIC", "BIC"))
        _close([aic], [-2 * ll + 2 * 67], engine + " AIC parameter count", criteria["native_common_logLik"])
        _close([bic], [-2 * ll + np.log(400) * 67], engine + " BIC parameter count",
               criteria["native_common_logLik"])
    return dict(archive_integrity="PASS", scientific_status=report["status"], new_fits=0,
        canonical_run=provenance["run_id"], captured_text_files_verified=len(captured),
        frozen_sources_verified=len(sources), coefficient_parity=differences,
        nonembedded_original_file_digests=sorted(set(files) - set(captured)),
        limitations=("Only embedded original text bytes are hash-verified here. Original NPZ arrays are checked "
                     "against native JSON coefficients, not reconstructed ZIP bytes. Integration-grid/raw-log "
                     "digests remain recorded where original bytes were not embedded; no missing evidence is "
                     "fabricated, downloaded, fitted or silently regenerated. Recorded EAP/LL gates retain their "
                     "actual independent execution provenance; no new inferential certificate is created."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=RAW)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    print("NRM_OFFLINE_ARCHIVE=" + json.dumps(verify(args.payload, args.reference),
                                              ensure_ascii=False, allow_nan=False))
