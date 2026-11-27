"""No-fit policy contracts on real native model objects, not calibration evidence."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core import irt_mirt_policy as P


def _outcome(name):
    """Exercise scope classification; this fixture does not estimate parameters."""
    cfg = P.profiles()[name]
    pattern = np.asarray(cfg["pattern"], bool)
    q, k, model = cfg["q"], cfg["k"], cfg["model"]
    categories = 2 if model in ("2PL", "Rasch") else 4
    unit = model in ("Rasch", "PCM")
    a = pattern.astype(float) * (1 if unit else 1.2)
    if categories == 2:
        from mirt.models.multidimensional import MultidimensionalModel
        engine = MultidimensionalModel(k, q, model_type="confirmatory", loading_pattern=pattern)
        engine.set_parameters(slopes=a, intercepts=np.zeros(k))
        loading_key = "slopes"
    elif model == "NRM":
        from mirt.models.polytomous import NominalResponseModel
        from core.irt_multidim import nominal_start
        engine = NominalResponseModel(k, categories, n_factors=q)
        engine.set_parameters(slopes=nominal_start(pattern, categories), intercepts=np.zeros((k, categories)))
        loading_key = "slopes"
    else:
        from mirt.models.polytomous import GradedResponseModel, GeneralizedPartialCredit
        constructor = GradedResponseModel if model == "GRM" else GeneralizedPartialCredit
        engine = constructor(k, n_factors=q, n_categories=4)
        engine.set_parameters(**{"discrimination": a,
                                "thresholds" if model == "GRM" else "steps":
                                    np.tile([-1., 0., 1.], (k, 1))})
        loading_key = "discrimination"
    masks = {key: np.ones_like(value, dtype=bool) for key, value in engine.parameters.items()}
    if model == "NRM":
        masks["slopes"] = np.broadcast_to(pattern[:, None, :], (k, categories, q)).copy()
        masks["slopes"][:, 0, :] = False
        masks["intercepts"][:, 0] = False
    else:
        masks[loading_key] = np.zeros_like(pattern) if unit else pattern.copy()
    engine.set_free_parameter_masks(masks)
    data = pd.DataFrame((np.arange(cfg["N"])[:, None] + np.arange(k)) % categories,
                        columns=[f"M{j+1}" for j in range(k)])
    return dict(model=model, fit=SimpleNamespace(model=engine, converged=True, log_likelihood=-100.,
                                                 latent_mean=np.zeros(q)),
                coded=data, labels=tuple(map(str, range(categories))), pattern=pattern,
                covariance=np.eye(q), correlated=cfg["correlated"], exploratory=False,
                initialization=P.COLD_INITIALIZATION if model == "NRM" else None,
                settings={key: cfg[key] for key in ("max_iter", "tol", "n_quadpts")})


def test_exact_original_profiles_and_criteria_are_reused():
    profiles = P.profiles()
    assert set(profiles) == set(P.OWNED)
    assert set(P.profiles("GPCM")) == {"GPCM_3D"}
    assert set(P.profiles("NRM")) == {"NRM_2D_COLD_C1"}
    assert P.profiles("Bifactor") == {} and P.profiles("3PL") == {}
    assert profiles["GPCM_3D"]["correlated"] == "full"
    profiles["GPCM_3D"]["N"] = 999
    assert P.profiles()["GPCM_3D"]["N"] == 400


@pytest.mark.parametrize("name", P.OWNED)
def test_exact_scoped_numerical_status_never_promotes_inference(name):
    out = _outcome(name)
    meta = P.numerical_scope(out)
    assert meta["Status"] == P.NUMERICAL
    assert meta["Evidence_profile"] == name
    assert meta["Evidence_run"].startswith("https://github.com/Tatung40/psikoanaliz/actions/runs/")
    for key in ("Inferential_status", "Item_parameter_SE", "Item_parameter_CI", "Inferential_p",
                "Global_fit_inference", "Item_fit_inference", "Formal_model_comparison"):
        assert meta[key] == P.WITHHELD
    assert "CONDITIONAL POSTERIOR ONLY" in meta["Posterior_uncertainty"]
    assert "cannot be established directly from user data" in meta["Unverifiable_assumptions"]
    assert "not certification of the user's DGP" in meta["Applicable_conditions"]
    assert "ilk 16" in meta["Limitations"]


@pytest.mark.parametrize("change", ["N", "J", "categories", "pattern", "correlated", "settings",
                                    "exploratory", "not_converged", "nonfinite_LL", "zero_mean",
                                    "covariance", "missing", "boundary", "structural_zero"])
def test_observable_mismatches_invalid_results_and_bounds_fail_closed(change):
    out = _outcome("2PL_2D")
    if change == "N": out["coded"] = out["coded"].iloc[:-1]
    elif change == "J": out["coded"] = out["coded"].iloc[:, :-1]
    elif change == "categories": out["labels"] = ("0", "1", "2")
    elif change == "pattern":
        out["pattern"] = out["pattern"].copy(); out["pattern"][-1] = [False, True]
    elif change == "correlated": out["correlated"] = False
    elif change == "settings": out["settings"] = dict(out["settings"], tol=1e-9)
    elif change == "exploratory": out["exploratory"] = True
    elif change == "not_converged": out["fit"].converged = False
    elif change == "nonfinite_LL": out["fit"].log_likelihood = np.nan
    elif change == "zero_mean": out["fit"].latent_mean = np.ones(2)
    elif change == "covariance": out["covariance"][0, 0] = 2
    elif change == "missing": out["coded"].iloc[0, 0] = np.nan
    else:
        from mirt.estimation.base import _parameter_bounds
        a = out["fit"].model.parameters["slopes"].copy()
        if change == "boundary":
            a[0, 0] = _parameter_bounds(out["fit"].model, "slopes")[1]
            out["fit"].model.set_parameters(slopes=a)
        else:
            # The real setter canonicalizes structural zeros. Simulate a corrupted
            # returned result explicitly, so this test actually exercises refusal.
            native = out["fit"].model
            a[0, 1] = .1
            out["fit"].model = SimpleNamespace(parameters=dict(native.parameters, slopes=a),
                free_parameter_masks=native.free_parameter_masks, model_name=native.model_name)
            assert out["fit"].model.parameters["slopes"][0, 1] == .1
    assert P.numerical_scope(out)["Status"] == P.EXPERIMENTAL


def test_no_silent_rasch_unit_loading_relaxation():
    out = _outcome("Rasch_2D")
    a = out["fit"].model.parameters["slopes"].copy()
    a[0, 0] = 1.01
    out["fit"].model.set_parameters(slopes=a)
    assert P.numerical_scope(out)["Status"] == P.EXPERIMENTAL


def test_native_version_is_read_and_unknown_versions_refuse(monkeypatch):
    out = _outcome("GRM_2D")
    assert P._package_version() == "1.2.0"
    before = P.evidence_fingerprint()
    monkeypatch.setattr(P, "version", lambda package: "UNKNOWN")
    assert P.evidence_fingerprint() != before
    assert P.numerical_scope(out)["Status"] == P.EXPERIMENTAL


def test_withdrawn_or_missing_evidence_invalidates_cached_scope(tmp_path, monkeypatch):
    out = _outcome("2PL_2D")
    original = P.ROOT
    for path in P.PINNED:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / path).read_bytes())
    monkeypatch.setattr(P, "ROOT", tmp_path)
    assert P.numerical_scope(out)["Status"] == P.NUMERICAL
    before = P.evidence_fingerprint()
    target = tmp_path / "validation/mirt_stage1/phase1_evidence.json"
    doc = json.loads(target.read_text())
    doc["reports"][0]["comparison"]["probabilities"]["status"] = "FAIL"
    target.write_text(json.dumps(doc))
    assert P.evidence_fingerprint() != before
    assert P.profiles() == {}
    assert P.numerical_scope(out)["Status"] == P.EXPERIMENTAL
    target.unlink()
    assert P.profiles() == {} and P.numerical_scope(out)["Status"] == P.EXPERIMENTAL


def test_even_pass_label_cannot_override_changed_frozen_archive(tmp_path, monkeypatch):
    original = P.ROOT
    for path in P.PINNED:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / path).read_bytes())
    target = tmp_path / "validation/mirt_stage1/phase1_evidence.json"
    doc = json.loads(target.read_text())
    doc["reports"][0]["comparison"]["probabilities"]["max_absolute"] /= 2
    target.write_text(json.dumps(doc))
    monkeypatch.setattr(P, "ROOT", tmp_path)
    assert P.profiles() == {}


@pytest.mark.parametrize("bound", [.1, 5.])
def test_actual_native_binary_slope_bounds_refuse_inference_profile(bound):
    from mirt.estimation.base import _parameter_bounds
    out = _outcome("2PL_2D")
    assert _parameter_bounds(out["fit"].model, "slopes") == (.1, 5.)
    a = out["fit"].model.parameters["slopes"].copy()
    a[0, 0] = bound
    out["fit"].model.set_parameters(slopes=a)
    assert P.numerical_scope(out)["Status"] == P.EXPERIMENTAL


@pytest.mark.parametrize("name", ["Rasch_2D", "PCM_2D"])
def test_fixed_unit_loadings_and_structural_zeros_are_not_active_bounds(name):
    out = _outcome(name)
    key = "slopes" if name.startswith("Rasch") else "discrimination"
    assert not np.asarray(out["fit"].model.free_parameter_masks[key]).any()
    assert P.numerical_scope(out)["Status"] == P.NUMERICAL


@pytest.mark.parametrize("initialization", [None, "warm", "saved item solution"])
def test_nrm_cold_claim_requires_explicit_actual_initialization_marker(initialization):
    out = _outcome("NRM_2D_COLD_C1")
    out["initialization"] = initialization
    assert P.numerical_scope(out)["Status"] == P.EXPERIMENTAL


@pytest.mark.parametrize("bound", [-5., 5.])
def test_nrm_signed_category_slopes_and_true_native_bounds(bound):
    from mirt.estimation.base import _parameter_bounds
    out = _outcome("NRM_2D_COLD_C1")
    assert _parameter_bounds(out["fit"].model, "slopes") == (-5., 5.)
    assert (out["fit"].model.parameters["slopes"] < 0).any()
    assert P.numerical_scope(out)["Status"] == P.NUMERICAL
    a = out["fit"].model.parameters["slopes"].copy()
    a[0, 2, 0] = bound
    out["fit"].model.set_parameters(slopes=a)
    assert P.numerical_scope(out)["Status"] == P.EXPERIMENTAL


@pytest.mark.parametrize("bad", ["reference_slope", "reference_intercept", "structural_zero", "anchor_sign"])
def test_nrm_reference_structure_and_canonical_orientation_are_required(bad):
    out = _outcome("NRM_2D_COLD_C1")
    native = out["fit"].model
    a, d = native.parameters["slopes"].copy(), native.parameters["intercepts"].copy()
    if bad == "reference_slope": a[0, 0, 0] = .1
    elif bad == "reference_intercept": d[0, 0] = .1
    elif bad == "structural_zero": a[0, 1, 1] = .1
    else: a[0, 1, 0] = -.1
    # Native setters may normalize reference/structural zeros; simulate a damaged
    # returned snapshot explicitly, without pretending this was a calibrated fit.
    out["fit"].model = SimpleNamespace(parameters=dict(native.parameters, slopes=a, intercepts=d),
        free_parameter_masks=native.free_parameter_masks, model_name=native.model_name)
    assert P.numerical_scope(out)["Status"] == P.EXPERIMENTAL


def test_warm_archive_cannot_replace_missing_or_withdrawn_cold_proof(tmp_path, monkeypatch):
    out = _outcome("NRM_2D_COLD_C1")
    original = P.ROOT
    for path in P.PINNED:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / path).read_bytes())
    monkeypatch.setattr(P, "ROOT", tmp_path)
    assert len(P.profiles()) == 8
    target = tmp_path / P.COLD_PATH
    before = P.evidence_fingerprint()
    target.unlink()
    assert P.evidence_fingerprint() != before
    assert set(P.profiles()) == set(P.OWNED) - {"NRM_2D_COLD_C1"}
    assert P.profiles("NRM") == {} and P.numerical_scope(out)["Status"] == P.EXPERIMENTAL
    doc = json.loads((original / P.COLD_PATH).read_text())
    doc["report"]["calibration_attempts"]["R"] = 2
    target.write_text(json.dumps(doc))
    assert P.profiles("NRM") == {}  # even a PASS label cannot replace frozen actual one-shot evidence
    assert len(P.profiles()) == 7
