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
    "validation/mirt_stage1/phase1_evidence.json": "a2d4d4e4c89f07f7418b05a6d56813a5b5bc472d"
}
OWNED = ("2PL_2D", "2PL_3D", "GRM_2D", "GRM_3D", "GPCM_3D", "Rasch_2D", "PCM_2D")
NUMERICAL = "NUMERICALLY CERTIFIED / SCOPED"
EXPERIMENTAL = "EXPERIMENTAL / NOT CERTIFIED"
WITHHELD = "WITHHELD"
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
            raw = (ROOT / path).read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            if actual != expected:
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
        return records
    except (OSError, ValueError, KeyError, TypeError):
        return {}


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
                                "identification": "zero means; unit variances; positive CFA structural masks",
                                "scope": "finite-grid implementation profile; not certification of the user's DGP"
                            }, ensure_ascii=False, sort_keys=True))
            return metadata
    except (AttributeError, KeyError, ValueError, TypeError, ImportError, np.linalg.LinAlgError):
        pass
    return metadata
