"""Policy D runtime view of the existing, immutable MIRT numerical reports.

This is not a new certification registry or estimator. A label identifies the
tested finite-grid implementation profile, never certifies the user's DGP.
"""
import copy
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PINNED = {
    "validation/mirt_stage1/criteria.json": "dc13c876887e56ca932890bb08118c0d058bb30e",
    "validation/mirt_stage1/expansion_original_evidence.json": "493ad69808d88846e83740ad41d58c8d03899f00",
    "validation/mirt_stage1/phase1_evidence.json": "a2d4d4e4c89f07f7418b05a6d56813a5b5bc472d",
    "validation/mirt_core_production/NRM_COLD_REFERENCE.json": "82642212d8260a875053a948cde5d9e441305c16"
}
OWNED = ("2PL_2D", "2PL_3D", "GRM_2D", "GRM_3D", "GPCM_3D", "Rasch_2D", "PCM_2D", "NRM_2D_COLD_C1")
NUMERICAL = "NUMERICALLY CERTIFIED / SCOPED"
EXPERIMENTAL = "EXPERIMENTAL / NOT CERTIFIED"
WITHHELD = "WITHHELD"
COLD_PATH = "validation/mirt_core_production/NRM_COLD_REFERENCE.json"
COLD_PROTOCOL = "af024e73e4be99d0db1a2a0df400039f98ab6095"
COLD_INITIALIZATION = "NATIVE COLD / nominal_start + zero intercepts + identity covariance"
CHECKS = {
    "slopes": "calibration_parameters", "intercepts": "calibration_parameters",
    "covariance": "calibration_covariance", "probabilities": "calibration_probability",
    "information": "calibration_information", "EAP": "calibration_EAP",
    "posterior_SD": "calibration_posterior_SD", "common_logLik": "common_logLik",
    "Python_native_LL": "native_common_logLik", "R_native_LL": "native_common_logLik",
}
PREFLIGHT = {"probabilities": "mapping_probability", "information": "mapping_information",
             "prior": "mapping_prior_mass"}
LIMITATIONS = (
    "Sayısal sertifika yalnız kayıtlı sonlu GH profilinin uygulamasına aittir; kullanıcı verisinin "
    "üreteci veya popülasyon çıkarımı sertifikalanmaz. Yeni N/J/kategori/yapı/çözücü kapsamına genellenmez. "
    "Parametre SH/GA/p, global veya madde uyumu çıkarımı ve biçimsel karşılaştırma kapalıdır. "
    "EAP ve posterior SD karşılaştırması ilk 16 kişiyle sınırlıdır; posterior kovaryans için bağımsız "
    "karşılaştırma veya sıkçı kapsama kanıtı yoktur. GRM/GPCM parametre karşılaştırması "
    "d=-sum(a)*b/adım dönüşümünü kullanır. Yakınsama durağanlık/global optimum veya quadrature doğruluğu değildir."
)


def _package_version():
    try:
        return version("mirt")
    except PackageNotFoundError:
        return "NOT INSTALLED"


def evidence_fingerprint():
    """Changes to evidence, policy source or native package invalidate derived caches."""
    h = hashlib.sha256()
    for path in [Path(__file__), *(ROOT / p for p in PINNED)]:
        h.update(str(path.name).encode())
        try:
            raw = path.read_bytes()
        except OSError:
            raw = b"MISSING"
        h.update(hashlib.sha256(raw).digest())
    h.update(_package_version().encode())
    return h.hexdigest()


def _passes(block, expected, criteria):
    if set(block) != set(expected):
        return False
    for name, key in expected.items():
        check = block[name]
        tol = criteria[key]
        value = float(check["max_absolute"])
        if (check["status"] != "PASS" or check["tolerance"] != tol or
                not np.isfinite(value) or not 0 <= value <= tol):
            return False
    return True


def _records():
    """Read the original reports; changed/withdrawn/missing evidence fails closed."""
    try:
        documents = {}
        for path, expected in PINNED.items():
            try:
                raw = (ROOT / path).read_bytes()
            except OSError:
                if path == COLD_PATH:
                    continue
                return {}
            actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            if actual != expected:
                if path == COLD_PATH:
                    continue
                return {}
            documents[path] = json.loads(raw)
        criteria = documents["validation/mirt_stage1/criteria.json"]
        if criteria["status"] != "PREREGISTERED_NUMERICAL_ONLY":
            return {}
        digest = hashlib.sha256((ROOT / "validation/mirt_stage1/criteria.json").read_bytes()).hexdigest()
        records = {}
        for path in ("validation/mirt_stage1/phase1_evidence.json",
                     "validation/mirt_stage1/expansion_original_evidence.json"):
            document = documents[path]
            for report in document["reports"]:
                name = report["profile"]
                if name not in OWNED:
                    continue
                cfg, py, ref = report["settings"], report["Python"], report["R"]
                if (cfg["name"] != name or cfg["criteria_sha256"] != digest or
                        report["status"] != "NUMERICALLY VALIDATED / NOT CERTIFIED" or
                        report["R_execution"] != "CALIBRATION EXECUTED" or
                        report["Python_execution"] != "EXECUTED" or
                        py["engine"] != "Python mirt 1.2.0" or ref["version"] != "1.48" or
                        py["converged"] is not True or ref["converged"] is not True or
                        ref["constraints_valid"] is not True or ref["logPrior"] != 0 or
                        not _passes(report["comparison"], CHECKS, criteria) or
                        not _passes(report["preflight"], PREFLIGHT, criteria)):
                    continue
                records[name] = dict(settings=cfg, run=document["run"])
        cold = _cold_record(documents.get(COLD_PATH), criteria, digest)
        if cold is not None:
            records["NRM_2D_COLD_C1"] = cold
        return records
    except (OSError, ValueError, KeyError, TypeError):
        return {}



def _cold_record(document, criteria, digest):
    """Only the actual prospective cold pair can support the cold NRM profile."""
    try:
        proof, report = document["provenance"], document["report"]
        cfg, py, ref = report["settings"], report["Python"], report["R"]
        pa, ra = py["admissibility"], report["R_admissibility"]
        if (proof["run_id"] != 45960533214 or proof["job_id"] != 140377033939 or
                proof["source_head"] != "1ae9d8321f8a392bbcf49e3cee25bb7dda1aee55" or
                proof["actual_checkout"] != "8f1848afc92febd5f2140bae743dcaf081f33877" or
                proof["protocol_commit"] != COLD_PROTOCOL or
                report["protocol"]["commit"] != COLD_PROTOCOL or
                report["protocol"]["sha256"] != "016abb32bada448a70cde66ed2397e550db9c5d33307cdc4b7936036e639e336" or
                report["calibration_attempts"] != {"Python": 1, "R": 1} or
                report["profile"] != "NRM_2D_COLD_C1" or cfg["name"] != "NRM_2D" or
                cfg["model"] != "NRM" or cfg["q"] != 2 or cfg["k"] != 10 or cfg["N"] != 400 or
                cfg["correlated"] != "full" or cfg["n_quadpts"] != 15 or
                cfg["max_iter"] != 3000 or cfg["tol"] != 1e-9 or
                cfg["protocol_commit"] != COLD_PROTOCOL or cfg["criteria_sha256"] != digest or
                cfg["data_sha256"] != "e43e687f10bf1231ba1c386a4d68bba4fc5dd22f81d31f92690e5e550e742454" or
                report["status"] != "NUMERICALLY VALIDATED / NOT CERTIFIED" or
                report["R_execution"] != "CALIBRATION EXECUTED" or report["Python_execution"] != "EXECUTED" or
                py["engine"] != "Python mirt 1.2.0" or ref["version"] != "1.48" or
                py["converged"] is not True or ref["converged"] is not True or
                ref["constraints_valid"] is not True or ref["logPrior"] != 0 or
                pa["valid"] is not True or pa["reference_and_structural_zeros"] is not True or
                pa["inside_native_boxes"] is not True or pa["identified_latent_metric"] is not True or
                pa["at_bound"] or pa["actual_native_bounds"] != {"slopes": [-5, 5], "intercepts": [-6, 6]} or
                ra["valid"] is not True or ra["inside_native_boxes"] is not True or ra["at_bound"] or
                ra["free_parameters"] != 67 or py["n_parameters"] != 67 or
                not _passes(report["comparison"], CHECKS, criteria) or
                not _passes(report["preflight"], PREFLIGHT, criteria)):
            return None
        for name in ("Python_runtime", "R_runtime", "R_preflight_runtime"):
            worker = report["worker_runtime"][name]
            if worker["timed_out"] is not False or worker["returncode"] != 0 or worker["elapsed_seconds"] > 300:
                return None
        settings = copy.deepcopy(cfg)
        settings["name"] = "NRM_2D_COLD_C1"  # alias only; frozen report remains byte-identical
        return dict(settings=settings, run=proof["run_url"])
    except (KeyError, TypeError, ValueError):
        return None


def profiles(model=None):
    """Independent accepted cold-CFA settings; callers receive independent copies."""
    return {name: copy.deepcopy(record["settings"]) for name, record in _records().items()
            if model is None or record["settings"]["model"] == model}


def _admissible(outcome):
    """Use actual fitted native parameters/masks, not merely the reported model name."""
    from mirt.estimation.base import _parameter_bounds

    fit, pattern = outcome["fit"], np.asarray(outcome["pattern"], bool)
    if not bool(fit.converged) or not np.isfinite(fit.log_likelihood):
        return False
    parameters, masks = fit.model.parameters, fit.model.free_parameter_masks
    if outcome["model"] == "NRM":
        a = np.asarray(parameters["slopes"], float)
        d = np.asarray(parameters["intercepts"], float)
        categories = len(outcome["labels"])
        expected = np.broadcast_to(pattern[:, None, :], (len(pattern), categories, pattern.shape[1])).copy()
        expected[:, 0, :] = False
        intercept_mask = np.ones((len(pattern), categories), bool)
        intercept_mask[:, 0] = False
        if (a.shape != expected.shape or d.shape != intercept_mask.shape or
                np.any(a[~expected] != 0) or np.any(d[:, 0] != 0) or
                not np.array_equal(masks["slopes"], expected) or
                not np.array_equal(masks["intercepts"], intercept_mask)):
            return False
        for j in range(pattern.shape[1]):
            anchors = np.flatnonzero(pattern[:, j] & (pattern.sum(1) == 1))
            if not len(anchors) or a[anchors[0], 1, j] <= np.sqrt(np.finfo(float).eps):
                return False
        if outcome.get("initialization") != COLD_INITIALIZATION:
            return False  # upstream stamps this only when its actual call has no supplied starts
    else:
        key = "slopes" if outcome["model"] in ("2PL", "Rasch") else "discrimination"
        a = np.asarray(parameters[key], float)
        if a.shape != pattern.shape or np.any(a[~pattern] != 0) or np.any(a[pattern] <= 0):
            return False
        if outcome["model"] in ("Rasch", "PCM") and not np.array_equal(a, pattern.astype(float)):
            return False
    covariance = np.asarray(outcome["covariance"], float)
    q = pattern.shape[1]
    if (covariance.shape != (q, q) or not np.isfinite(covariance).all() or
            not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0) or
            not np.allclose(np.diag(covariance), 1, atol=1e-12, rtol=0) or
            np.linalg.eigvalsh(covariance).min() <= 0):
        return False
    allowed = np.zeros((q, q), bool)
    if outcome["correlated"] == "full":
        allowed[:] = True
    elif outcome["correlated"]:
        allowed[0, 1] = allowed[1, 0] = True
    np.fill_diagonal(allowed, True)
    if np.any(covariance[~allowed] != 0):
        return False
    mean = getattr(fit, "latent_mean", None)
    if mean is not None and not np.allclose(mean, 0, atol=1e-12, rtol=0):
        return False
    for name, raw in parameters.items():
        values = np.asarray(raw, float)
        mask = np.asarray(masks[name], bool)
        if mask.shape != values.shape or not np.isfinite(values).all():
            return False
        lo, hi = _parameter_bounds(fit.model, name)
        free = values[mask]
        gap = np.minimum(free - lo, hi - free)
        if np.any(gap <= np.sqrt(np.finfo(float).eps) * np.maximum(1, np.abs(free))):
            return False
    if outcome["model"] == "GRM" and np.any(np.diff(parameters["thresholds"], axis=1) <= 1e-6):
        return False
    return True


def numerical_scope(outcome):
    """Metadata only; never fits, scores, estimates uncertainty or changes a result."""
    metadata = dict(
        Status=EXPERIMENTAL, Numerical_status=EXPERIMENTAL, Inferential_status=WITHHELD,
        Certification_layer="NUMERICAL IMPLEMENTATION PROFILE, not independent certification of the current data/calibration",
        Current_data_independent_R_fit="NOT EXECUTED BY APPLICATION; Evidence_run identifies the historical reference fixture",
        Reference_DGP="Archived fixed generator: unit latent variances; zero means; true free factor correlations .25, others0. Native item/category truth is fixed by the unchanged original generator, not estimated/verified from user data.",
        Item_parameter_SE=WITHHELD, Item_parameter_CI=WITHHELD, Inferential_p=WITHHELD,
        Global_fit_inference=WITHHELD, Item_fit_inference=WITHHELD, Formal_model_comparison=WITHHELD,
        Posterior_uncertainty="CONDITIONAL POSTERIOR ONLY; not item SE or frequentist CI",
        Evidence_profile="NONE", Evidence_run="NONE",
        Applicable_conditions="No exact observable accepted cold-CFA profile matched.",
        Unverifiable_assumptions="Latent Gaussian distribution, item response family and local independence "
                                "cannot be established directly from user data.",
        Limitations=LIMITATIONS,
    )
    try:
        if _package_version() != "1.2.0" or outcome.get("exploratory", False):
            return metadata
        data = outcome["coded"]
        labels = outcome["labels"]
        if data.isna().any().any() or not data.columns.is_unique:
            return metadata
        categories = len(labels)
        if any(set(data[c]) != set(range(categories)) for c in data):
            return metadata
        for name, record in _records().items():
            cfg = record["settings"]
            expected_categories = 2 if cfg["model"] in ("2PL", "Rasch") else 4
            settings = {k: cfg[k] for k in ("max_iter", "tol", "n_quadpts")}
            if (outcome["model"] != cfg["model"] or len(data) != cfg["N"] or
                    data.shape[1] != cfg["k"] or categories != expected_categories or
                    outcome["settings"] != settings or
                    type(outcome["correlated"]) is not type(cfg["correlated"]) or
                    outcome["correlated"] != cfg["correlated"] or
                    not np.array_equal(outcome["pattern"], cfg["pattern"]) or
                    not _admissible(outcome)):
                continue
            metadata.update(Status=NUMERICAL, Numerical_status=NUMERICAL,
                            Evidence_profile=name, Evidence_run=record["run"],
                            Applicable_conditions=json.dumps({
                                "N": cfg["N"], "J": cfg["k"], "categories": expected_categories,
                                "pattern": cfg["pattern"], "correlated": cfg["correlated"],
                                "settings": settings, "estimation": "ML, no priors",
                                "identification": ("zero means; unit variances; category0 reference; canonical pure-anchor signs"
                                                   if cfg["model"] == "NRM" else "zero means; unit variances; positive CFA structural masks"),
                                "initialization": cfg.get("initialization", "registered cold CFA starts"),
                                "reference_seed": cfg["seed"], "reference_data_sha256": cfg["data_sha256"],
                                "scope": "finite-grid implementation profile; not certification of the user's DGP"
                            }, ensure_ascii=False, sort_keys=True))
            return metadata
    except (AttributeError, KeyError, ValueError, TypeError, ImportError, np.linalg.LinAlgError):
        pass
    return metadata
