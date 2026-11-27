"""Native CFA derived-output identities; no calibration or item inference.

Uses only public mirt scoring/model APIs. A derived identity is not an
independent R certificate or a frequentist coverage claim.
"""
import importlib
import numpy as np
import pandas as pd

OWNED = frozenset(("2PL", "GRM", "GPCM", "PCM", "Rasch", "1PL (a=1)", "NRM"))
# Same numerical identity tolerance used by existing multidimensional IIF/TIF checks.
IDENTITY_ATOL = 1e-10
MAX_POSTERIOR_CELLS = 2_000_000  # technical memory bound, not a scientific criterion
_KEYS = ("LoadingMatrix", "Correlation", "PosteriorCovariance",
         "Directional_IIF", "Directional_TIF", "Directional_Curves", "DerivedDiagnostics")
NOTE = ("Türetilmiş çıktılar matematiksel/kütüphane özdeşlikleridir; bağımsız R "
        "sertifikası veya çıkarımsal kapsama kanıtı değildir. Posterior kovaryans "
        "sabit madde parametrelerine koşulludur; madde parametresi SH/GA değildir. "
        "Yükler lojistik IRT katsayılarıdır, standartlaştırılmış AFA yükleri değildir. "
        "Yön bilgisi e' I e'dir; 1/sqrt(e' I e), sqrt(e' I^-1 e) ile aynı değildir.")


def _owned(outcome):
    return (outcome.get("model") in OWNED and
            not outcome.get("exploratory", False))



def native_boundaries(fit):
    """Authoritative installed native optimizer boxes, free coordinates only.

    Corrects the legacy report's dictionary fallback for slopes:
    multidimensional 2PL uses [.1,5], while NRM slopes use [-5,5].
    No estimation, bounds, constraints or scientific thresholds are changed.
    """
    from mirt.estimation.base import _parameter_bounds
    rows = []
    model = fit.model
    for name, values in model.parameters.items():
        value = np.asarray(values, float)
        mask = np.asarray(model.free_parameter_masks[name], bool)
        if mask.shape != value.shape:
            raise ValueError("Parametre/serbestlik maskesi boyutları uyuşmuyor")
        lower, upper = _parameter_bounds(model, name)
        for index in np.ndindex(value.shape):
            if not mask[index]:
                continue
            v = float(value[index])
            gap = min(v - lower, upper - v)
            rows.append(dict(Parameter=name, Coordinate=str(index), Value=v,
                Lower=lower, Upper=upper, Distance_to_bound=gap,
                At_bound=bool(not np.isfinite(v) or
                    gap <= np.sqrt(np.finfo(float).eps) * max(1, abs(v)))))
    return pd.DataFrame(rows, columns=["Parameter", "Coordinate", "Value", "Lower",
                                      "Upper", "Distance_to_bound", "At_bound"])


def _loadings(outcome):
    params = outcome["fit"].model.parameters
    q = outcome["pattern"].shape[1]
    items = list(outcome["coded"].columns)
    if outcome["model"] == "NRM":
        slopes = np.asarray(params["slopes"], float)
        if slopes.shape != (len(items), len(outcome["labels"]), q):
            raise ValueError("Nominal yük boyutları uyuşmuyor")
        rows = [dict(Item=item, Category=str(outcome["labels"][c]), Reference=c == 0,
                     **{f"a{j+1}": slopes[i, c, j] for j in range(q)})
                for i, item in enumerate(items) for c in range(slopes.shape[1])]
    else:
        name = "slopes" if outcome["model"] in ("2PL", "Rasch", "1PL (a=1)") else "discrimination"
        slopes = np.asarray(params[name], float)
        if slopes.shape != (len(items), q):
            raise ValueError("Yük boyutları uyuşmuyor")
        rows = [dict(Item=item, **{f"a{j+1}": slopes[i, j] for j in range(q)})
                for i, item in enumerate(items)]
    if not np.isfinite(slopes).all():
        raise ValueError("Sonlu olmayan yük")
    return pd.DataFrame(rows)


def _correlation(outcome):
    q = outcome["pattern"].shape[1]
    covariance = np.asarray(outcome["covariance"], float)
    if (covariance.shape != (q, q) or not np.isfinite(covariance).all() or
            np.any(np.diag(covariance) <= 0)):
        raise ValueError("Geçersiz örtük kovaryans")
    np.testing.assert_allclose(covariance, covariance.T, atol=IDENTITY_ATOL, rtol=0)
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError("Örtük kovaryans pozitif tanımlı değil")
    sd = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sd, sd)
    return pd.DataFrame(correlation, columns=[f"F{j+1}" for j in range(q)]).assign(
        Factor=[f"F{j+1}" for j in range(q)])[["Factor"] + [f"F{j+1}" for j in range(q)]]


def _directions(frame, q, itemwise):
    keys = ["Node", "Item"] if itemwise else ["Node"]
    if frame.empty:
        raise ValueError("Bilgi matrisi tablosu boş")
    wide = frame.pivot(index=keys, columns=["Row", "Column"], values="Information")
    coordinates = pd.MultiIndex.from_product([range(1, q + 1), range(1, q + 1)])
    if set(wide.columns) != set(coordinates):
        raise ValueError("Bilgi matrisi koordinatları eksik")
    matrices = wide.reindex(columns=coordinates).to_numpy(float).reshape(-1, q, q)
    if not np.isfinite(matrices).all():
        raise ValueError("Sonlu olmayan bilgi matrisi")
    np.testing.assert_allclose(matrices, matrices.transpose(0, 2, 1),
                               atol=IDENTITY_ATOL, rtol=0)
    eigenvalues = np.linalg.eigvalsh(matrices)
    if (eigenvalues < -IDENTITY_ATOL).any():
        raise ValueError("Bilgi matrisi pozitif yarı tanımlı değil")
    directions = np.vstack([np.eye(q), np.ones((1, q)) / np.sqrt(q)])
    names = [f"F{j+1}" for j in range(q)] + ["Birim köşegen"]
    information = np.einsum("dr,nrs,ds->nd", directions, matrices, directions)
    # Joint contrast variance requires the full matrix to be stably invertible.
    # Singular single-item matrices are expected and are never pseudoinverted.
    stable = eigenvalues[:, 0] > np.sqrt(np.finfo(float).eps) * np.maximum(eigenvalues[:, -1], 1)
    joint = np.full_like(information, np.nan)
    if stable.any():
        inverse = np.linalg.solve(matrices[stable], np.broadcast_to(np.eye(q), matrices[stable].shape))
        variance = np.einsum("dr,nrs,ds->nd", directions, inverse, directions)
        if (variance < 0).any():
            raise ValueError("Geçersiz ortak Fisher varyansı")
        joint[stable] = np.sqrt(variance)
    keys_frame = wide.index.to_frame(index=False)
    rows = []
    for n in range(len(wide)):
        identity = keys_frame.iloc[n].to_dict()
        for d, name in enumerate(names):
            value = float(information[n, d])
            # Never silently clip a negative or zero directional-information result.
            positive = value > 0
            rows.append(dict(**identity, Direction=name,
                **{f"Direction_F{j+1}": directions[d, j] for j in range(q)},
                Information=value if value >= 0 else np.nan,
                Direction_fixed_orthogonal_Fisher_SD=1 / np.sqrt(value) if positive else np.nan,
                Joint_contrast_Fisher_SD=joint[n, d], Joint_matrix_stable=bool(stable[n]),
                Note="" if positive else "Sıfır/sayısal olarak geçersiz yön bilgisi; yön SH gösterilmedi."))
    return pd.DataFrame(rows)



def _directional_curves(outcome):
    """Native response probabilities along explicitly named unit-vector paths."""
    q = outcome["pattern"].shape[1]
    items = list(outcome["coded"].columns)
    labels = outcome["labels"]
    directions = np.vstack([np.eye(q), np.ones((1, q)) / np.sqrt(q)])
    names = [f"F{j+1}" for j in range(q)] + ["Birim köşegen"]
    values = np.linspace(-3, 3, 25)
    rows = []
    for vector, name in zip(directions, names):
        theta = values[:, None] * vector[None, :]
        probability = np.asarray(outcome["fit"].model.probability(theta), float)
        if probability.ndim == 2:
            probability = np.stack([1 - probability, probability], axis=-1)
        if (probability.shape != (len(values), len(items), len(labels)) or
                not np.isfinite(probability).all() or (probability < 0).any() or
                (probability > 1).any()):
            raise ValueError("Geçersiz yönsel kategori olasılıkları")
        np.testing.assert_allclose(probability.sum(-1), 1, atol=IDENTITY_ATOL, rtol=0)
        for n, value in enumerate(values):
            for i, item in enumerate(items):
                for c, label in enumerate(labels):
                    rows.append(dict(Direction=name, Theta=value, Item=item, Category=str(label),
                        Probability=probability[n, i, c],
                        **{f"Direction_F{j+1}": vector[j] for j in range(q)}))
    return pd.DataFrame(rows)


def _posterior_covariance(outcome, tables, person_ids):
    scoring = importlib.import_module("mirt.scoring")
    posterior_api = getattr(scoring, "ability_posterior", None)
    if not callable(posterior_api):
        raise ImportError("Kurulu mirt public ability_posterior API'si yok")
    x = outcome["coded"]
    q = outcome["pattern"].shape[1]
    n_quadpts = outcome["settings"]["n_quadpts"]
    if len(x) * n_quadpts ** q > MAX_POSTERIOR_CELLS:
        raise ValueError("Tam posterior için teknik bellek sınırı aşıldı; model kestirimi korunur")
    eap = tables.get("EAP")
    if eap is None or len(eap) != len(x):
        raise ValueError("Aynı kestirime ait EAP tablosu gerekli")
    ids = eap["ID"].tolist() if person_ids is None else list(person_ids)
    if len(ids) != len(x) or ids != eap["ID"].tolist():
        raise ValueError("Kişi kimlikleri/sırası EAP ile uyuşmuyor")
    posterior = posterior_api(outcome["fit"], x.to_numpy(), n_quadpts=n_quadpts,
                              person_ids=ids, batch_size=128)
    points = np.asarray(posterior.points, float)
    weights = np.asarray(posterior.weights, float)
    if (points.ndim != 2 or points.shape[1] != q or
            weights.shape != (len(x), len(points)) or
            not np.isfinite(points).all() or not np.isfinite(weights).all() or
            (weights < 0).any() or list(posterior.person_ids) != ids):
        raise ValueError("Geçersiz public posterior çıktısı")
    np.testing.assert_allclose(weights.sum(1), 1, atol=IDENTITY_ATOL, rtol=0)
    mean = np.asarray(posterior.mean).reshape(len(x), q)
    sd = np.asarray(posterior.standard_error).reshape(len(x), q)
    np.testing.assert_allclose(mean, eap[[f"F{j+1}" for j in range(q)]],
                               atol=IDENTITY_ATOL, rtol=0)
    np.testing.assert_allclose(sd, eap[[f"Posterior_SD_F{j+1}" for j in range(q)]],
                               atol=IDENTITY_ATOL, rtol=0)
    batch = max(1, min(128, 250_000 // max(1, len(points) * q)))
    rows = []
    for start in range(0, len(x), batch):
        stop = min(start + batch, len(x))
        centered = points[None, :, :] - mean[start:stop, None, :]
        covariance = np.einsum("np,npr,nps->nrs", weights[start:stop], centered, centered,
                               optimize=True)
        if not np.isfinite(covariance).all() or (np.linalg.eigvalsh(covariance) < -IDENTITY_ATOL).any():
            raise ValueError("Geçersiz posterior kovaryans")
        np.testing.assert_allclose(np.diagonal(covariance, axis1=1, axis2=2), sd[start:stop] ** 2,
                                   atol=IDENTITY_ATOL, rtol=0)
        for n in range(start, stop):
            for r in range(q):
                for c in range(q):
                    rows.append(dict(ID=ids[n], Row=r+1, Column=c+1,
                                     Covariance=covariance[n-start, r, c]))
    return pd.DataFrame(rows)


def enrich(outcome, result, person_ids=None):
    """Add derived native-CFA tables to an existing report; never fit a model.

    Caller owns content-bound, session-isolated caching. Advanced/EFA/bifactor
    results are deliberately left completely unchanged.
    """
    if not _owned(outcome):
        return result
    tables = result["tables"]
    metadata = result["metadata"]
    for name in _KEYS:
        tables.pop(name, None)
    entries = []
    def record(name, status, note):
        entries.append(dict(Output=name, Status=status, Scope=note))
    try:
        tables["Boundaries"] = native_boundaries(outcome["fit"])
        metadata["At_box_bound"] = bool(tables["Boundaries"]["At_bound"].any())
        metadata["Boundary_status"] = "DESCRIPTIVE IDENTITY"
        record("Boundaries", "DESCRIPTIVE IDENTITY",
               "Kurulu native optimizasyon kutuları; yalnız serbest koordinatlar. Parametre/fit değişmedi.")
    except (ImportError, AttributeError, KeyError, TypeError, ValueError) as exc:
        tables.pop("Boundaries", None)
        metadata["At_box_bound"] = None
        metadata["Boundary_status"] = "WITHHELD"
        record("Boundaries", "WITHHELD", str(exc))
    if not bool(getattr(outcome["fit"], "converged", False)):
        for name in _KEYS[:-1]:
            record(name, "WITHHELD", "Yakınsamama: yalnız geçici parametreler ve tanılar korunur.")
    else:
        operations = [
            ("LoadingMatrix", lambda: _loadings(outcome), "Ham lojistik IRT yükleri; standartlaştırılmış AFA yükleri değil."),
            ("Correlation", lambda: _correlation(outcome), "Korelasyon = kovaryans / marjinal SD çarpımı."),
            ("PosteriorCovariance", lambda: _posterior_covariance(outcome, tables, person_ids),
             "Public mirt posterior moment özdeşliği; sabit maddelere koşullu. Bağımsız R/çıkarımsal sertifika yok."),
            ("Directional_IIF", lambda: _directions(tables["IIF"], outcome["pattern"].shape[1], True),
             "e' I_item e; diğer yönler sabit. Tek madde genellikle tekil; ortak SH bastırılır."),
            ("Directional_TIF", lambda: _directions(tables["TIF"], outcome["pattern"].shape[1], False),
             "e' I_test e ile sqrt(e' I_test^-1 e) ayrıdır; ters yalnız kararlı pozitif tanımlı matriste."),
            ("Directional_Curves", lambda: _directional_curves(outcome),
             "Public model olasılıkları: theta=t e, ||e||=1, t=[-3,3]; diğer yönler sıfır. Çıkarım yok.")
        ]
        for name, operation, scope in operations:
            try:
                tables[name] = operation()
                record(name, "DESCRIPTIVE IDENTITY", scope)
            except (ImportError, AttributeError, KeyError, TypeError, ValueError,
                    AssertionError, np.linalg.LinAlgError) as exc:
                record(name, "WITHHELD", str(exc))
    tables["DerivedDiagnostics"] = pd.DataFrame(entries)
    metadata["Derived_diagnostics_note"] = NOTE
    posterior_status = next(row for row in entries if row["Output"] == "PosteriorCovariance")
    metadata["Posterior_covariance_status"] = posterior_status["Status"]
    metadata["Posterior_covariance_reason"] = posterior_status["Scope"]
    metadata["Posterior_covariance_independent_R_certification"] = "NOT CERTIFIED"
    return result
