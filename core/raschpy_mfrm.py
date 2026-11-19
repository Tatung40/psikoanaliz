"""Independent RaschPy 1.4.0 MFRM service (experimental). The R/TAM pipeline in core/mfrm.py is not used for estimation;
only its data validation (prepare) is reused. RaschPy estimates by regularised pairwise composite likelihood (RCL),
which is not equivalent to TAM's marginal ML: values are never presented as interchangeable."""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from core.mfrm import MFRMError, prepare

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "validation/raschpy_mfrm/evidence.json"
SIMULATION = ROOT / "validation/raschpy_mfrm/simulation.json"
SE_BAND = (0.80, 1.25)                     # preregistered V10 band for SE / empirical SD (PROTOCOL amendment)
BOOT_MODELS = ("bivector", "centrality", "bistretch")
RASCHPY_VERSION = "1.4.0"
MODELS = {  # RaschPy MFRM._MODELS, in its order; label, rater term in the log-numerator of category k
    "global": ("Genel (global) katılık", "k·λ_r"),
    "items": ("Madde bazında katılık (puanlayıcı × madde)", "k·λ_{r,i}"),
    "thresholds": ("Eşik bazında katılık (puanlayıcı × eşik)", "Σ_{j≤k} λ_{r,j}"),
    "bivector": ("Çift vektör (madde + eşik)", "k·λ_{r,i} + Σ_{j≤k} λ_{r,j}"),
    "matrix": ("Tam matris (puanlayıcı × madde × eşik)", "Σ_{j≤k} λ_{r,i,j}"),
    "centrality": ("Merkezilik / eşik germe (Jin & Wang)", "Σ_{j≤k} [λ_r + (ω_r − 1)·τ_j]"),
    "pseudo_halo": ("Sözde hale / madde germe", "k·[λ_r + (ω_r − 1)·δ_i]"),
    "bistretch": ("Çift germe (madde + eşik)", "Σ_{j≤k} λ_{r,i,j}; λ, ω_madde, ω_eşik"),
}
STRETCH = {"centrality": ("lambda", "omega"), "pseudo_halo": ("lambda", "omega"),
           "bistretch": ("lambda", "omega_items", "omega_thresholds")}
EQUATION = "log P(X=k) ∝ k·(θ_n − δ_i) − Σ_{j≤k} τ_j − [puanlayıcı terimi]"
OUTPUT_LABELS = {
    "parameters": "Madde, eşik ve puanlayıcı parametreleri", "se": "Puanlayıcı sandviç SE (yalnız benzetimde güvenilir modellerde)",
    "cell_bootstrap_se": "Puanlayıcı hücreleri bootstrap SE / %95 GA",
    "persons": "Birey ölçüleri (Warm WLE) ve CSEM", "fit": "Infit/Outfit MnSq", "probabilities": "Kategori olasılıkları ve beklenen puan",
    "category_counts": "Kategori sıklıkları (betimsel)", "category_stats": "Kategori genişliği (bootstrap)",
    "test_stats": "Test düzeyi özetler (RaschPy)", "model_selection": "Model seçimi (AIC/BIC, RaschPy)",
    "dif": "DIF (RaschPy dif_test)", "drf": "DRF — farklı puanlayıcı işlevi (RaschPy drf_test)",
}
UNSUPPORTED = {
    "andersen_lr_test": "RaschPy 1.4.0 bu testi kendisi devre dışı bırakmıştır (NotImplementedError): RCL/PAIR bir olabilirlik "
                        "kestiricisi olmadığından LR istatistiği geçersizdir.",
    "extra_facets": "RaschPy MFRM yalnız bir puanlayıcı yüzeyi destekler (veri: (Puanlayıcı, Birey) × madde); görev/oturum gibi "
                    "ek yüzeyler kestirilemez.",
    "interaction_lr_tam": "TAM etkileşim LR testi bu motorda yoktur; puanlayıcı × madde etkileri 'items'/'matrix' modelleriyle "
                          "doğrudan kestirilir.",
}


def evidence():
    try:
        return json.loads(EVIDENCE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def simulation():
    try:
        return json.loads(SIMULATION.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def rater_se_reliable(model):
    """Rater-side sandwich SE are shown only if, in V9, SE / empirical SD stayed inside SE_BAND for every reported cell in
    both designs. Item/threshold/category SE have no coverage evidence and are never shown."""
    v = [simulation().get("V9", {}).get(f"{model}:{d}", {}).get("sandwich_se_over_sd_range") for d in ("dense", "linked")]
    return all(r is not None and SE_BAND[0] <= r[0] and r[1] <= SE_BAND[1] for r in v)


def cell_bootstrap_validated(model):
    return model in BOOT_MODELS and bool(simulation().get("V10", {}).get(model, {}).get("gate"))


def status(model, output):
    return evidence().get("models", {}).get(model, {}).get(output, {}).get("status", "NOT CERTIFIED")


# ---------------------------------------------------------------- data
def to_wide(df, person, item, rater, score):
    """Validate with core.mfrm.prepare (duplicates, missing, categories, connectivity) and pivot to RaschPy layout."""
    d, report = prepare(df, person, item, rater, score)
    w = d.pivot(index=["rater", "person"], columns="item", values="score").astype(float)
    w.index.names = ["Rater", "Person"]; w.columns.name = None
    report["max_score"] = int(max(report["categories"]))
    return d, w, report


def _model(w, K):
    import raschpy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return raschpy.MFRM(w, max_score=K)


def fit(df, person, item, rater, score, model="global", se_method="sandwich", n_boot=200, seed=20261118):
    if model not in MODELS:
        raise MFRMError(f"Desteklenmeyen RaschPy MFRM modeli: {model}")
    d, w, report = to_wide(df, person, item, rater, score)
    K = report["max_score"]; m = _model(w, K); t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m.calibrate(model=model)
            if se_method == "bootstrap":
                np.random.seed(seed); m.std_errors(model=model, se_method="bootstrap", no_of_samples=int(n_boot))
            else:
                m.std_errors(model=model, se_method="sandwich")
            m.fit_statistics(model=model)
        except Exception as e:                                  # RaschPy numerical failure: report, never substitute
            raise MFRMError(f"RaschPy {model} kestirimi başarısız: {type(e).__name__}: {e}") from e
    secs = time.perf_counter() - t0
    g = lambda a: getattr(m, f"{a}_{model}", None)
    res = dict(model=model, se_method=se_method, n_boot=n_boot if se_method == "bootstrap" else None, seed=seed,
               report=report, seconds=secs, K=K, estimation=g("estimation"), convergence=g("convergence"),
               items=m.item_stats_df(model=model), thresholds=m.threshold_stats_df(model=model),
               raters=m.rater_stats_df(model=model), persons=m.person_stats_df(model=model),
               test=m.test_stats_df(model=model), category_counts=m.category_counts_df(),
               rater_params=_rater_params(m, model), versions=_versions())
    res["category_stats"] = m.category_stats_df(model=model) if se_method == "bootstrap" else None
    res["cell_bootstrap"] = None
    res["identity"] = identity_checks(m, model, d)
    res["_m"] = m; res["_d"] = d
    return res


def _versions():
    import raschpy, scipy, sklearn
    return [f"raschpy {getattr(raschpy, '__version__', RASCHPY_VERSION)}", f"numpy {np.__version__}", f"pandas {pd.__version__}",
            f"scipy {scipy.__version__}", f"scikit-learn {sklearn.__version__}"]


def _rater_params(m, model):
    """Long table of the rater-side parameters RaschPy estimates for `model`, with their SE when RaschPy reports them."""
    if model in STRETCH:
        rows = []
        for p in STRETCH[model]:
            est = getattr(m, f"{p}_{model}"); se = getattr(m, f"{p}_se_{model}", None)
            for r, v in est.items():
                rows.append(dict(Puanlayıcı=r, Parametre=p, Kestirim=float(v),
                                 SE=float(se[r]) if se is not None and r in se else np.nan))
        return pd.DataFrame(rows)
    est = getattr(m, f"raters_{model}"); se = getattr(m, f"rater_se_{model}", None)
    if isinstance(est, pd.Series):
        return pd.DataFrame(dict(Puanlayıcı=est.index, Parametre="λ", Kestirim=est.to_numpy(),
                                 SE=se.reindex(est.index).to_numpy() if isinstance(se, pd.Series) else np.nan))
    long = est.stack().rename("Kestirim").reset_index()
    if isinstance(se, pd.DataFrame) and se.shape == est.shape:
        long["SE"] = se.stack().to_numpy()
    else:
        long["SE"] = np.nan
    long.columns = ["Puanlayıcı", *[f"Düzey{i}" for i in range(long.shape[1] - 3)], "Kestirim", "SE"][:long.shape[1]]
    return long


# ---------------------------------------------------------------- independent identity computations (no RaschPy code)
def offsets(m, model, d):
    """Per-observation cumulative offsets c_k (k = 0..K) with log num_k = k·θ − c_k, built from RaschPy's fitted parameters
    with the documented log-numerators (cat_prob docstring; Jin & Wang stretch substitution)."""
    delta = getattr(m, f"items_{model}"); tau = np.asarray(getattr(m, f"thresholds_{model}"), float)
    tau = tau[1:] if len(tau) == d.score.max() + 1 or len(tau) == int(m.max_score) + 1 else tau
    K = len(tau); k = np.arange(K + 1); cum = lambda v: np.concatenate([[0.], np.cumsum(v)])
    base = np.outer(d.item.map(delta).to_numpy(float), k) + cum(tau)[None, :]
    R = getattr(m, f"raters_{model}"); out = np.empty_like(base)
    for n, (i, r) in enumerate(zip(d.item, d.rater)):
        if model == "global":
            term = k * R[r]
        elif model in ("items", "pseudo_halo"):
            term = k * R.loc[r, i]
        elif model in ("thresholds", "centrality"):
            term = cum(R.loc[r].to_numpy(float))
        else:
            term = cum(R.loc[(r, i)].to_numpy(float))
        out[n] = base[n] + term
    return out


def stretch_reconstruction(m, model):
    """max |RaschPy full-vector rater parameters − λ + (ω − 1)·reference| for the stretch models."""
    if model not in STRETCH:
        return None
    R = getattr(m, f"raters_{model}"); lam = getattr(m, f"lambda_{model}")
    tau = np.asarray(getattr(m, f"thresholds_{model}"), float); tau = tau[-int(m.max_score):]
    delta = getattr(m, f"items_{model}")
    if model == "centrality":
        om = getattr(m, f"omega_{model}"); want = pd.DataFrame({r: lam[r] + (om[r] - 1) * tau for r in lam.index}).T
        return float(np.abs(R.to_numpy(float) - want.to_numpy()).max())
    if model == "pseudo_halo":
        om = getattr(m, f"omega_{model}"); want = pd.DataFrame({r: lam[r] + (om[r] - 1) * delta for r in lam.index}).T
        return float(np.abs(R.to_numpy(float) - want[R.columns].to_numpy()).max())
    oi, ot = getattr(m, f"omega_items_{model}"), getattr(m, f"omega_thresholds_{model}")
    diff = 0.
    for (r, i), row in R.iterrows():
        want = lam[r] + (oi[r] - 1) * delta[i] + (ot[r] - 1) * tau
        diff = max(diff, float(np.abs(row.to_numpy(float) - want).max()))
    return diff


def _moments(theta, c):
    k = np.arange(c.shape[1]); ln = k[None, :] * np.asarray(theta)[:, None] - c
    p = np.exp(ln - ln.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    m1 = p @ k; m2 = p @ k ** 2
    return p, m1, m2 - m1 ** 2, p @ k ** 3, m2


def wle_generic(c, y, groups, tol=1e-10, maxit=200):
    """Warm WLE per person for arbitrary category offsets: solve Σ(x − E) + J/(2I) = 0 by bisection-safeguarded Newton."""
    k = np.arange(c.shape[1]); out = {}
    for pid, ix in pd.Series(range(len(y))).groupby(groups, sort=False).indices.items():
        cc, yy = c[ix], y[ix]
        def g(t):
            p, m1, v, m3, m2 = _moments(np.full(len(ix), t), cc)
            return (yy - m1).sum() + (m3 - 3 * m1 * m2 + 2 * m1 ** 3).sum() / (2 * v.sum())
        lo, hi = -15., 15.
        if g(lo) <= 0 or g(hi) >= 0:
            out[pid] = np.nan; continue
        for _ in range(maxit):                                   # g is decreasing in θ
            mid = (lo + hi) / 2
            if g(mid) > 0: lo = mid
            else: hi = mid
            if hi - lo < tol: break
        out[pid] = (lo + hi) / 2
    return pd.Series(out)


def residual_fit_generic(c, y, theta, keys):
    _, E, W, _, _ = _moments(theta, c); res = y - E
    f = pd.DataFrame({"res2": res ** 2, "W": W, "z2": res ** 2 / W, "key": keys})
    g = f.groupby("key"); return pd.DataFrame({"infit": g.res2.sum() / g.W.sum(), "outfit": g.z2.mean()})


def identity_checks(m, model, d):
    """V1 (probabilities/expected scores), V1b (stretch substitution), V2 (WLE), V3 (fit) — independent code paths."""
    out = {}
    try:
        c = offsets(m, model, d); persons = getattr(m, f"persons_{model}")
        th = d.person.map(persons).to_numpy(float); ok = np.isfinite(th)
        _, E, _, _, _ = _moments(th[ok], c[ok])
        exp = getattr(m, f"exp_score_df_{model}").stack()
        theirs = np.array([exp.get((r, p, i), np.nan) for r, p, i in zip(d.rater[ok], d.person[ok], d.item[ok])], float)
        fin = np.isfinite(theirs)
        out["V1_expected_score_max_diff"] = float(np.abs(E[fin] - theirs[fin]).max()) if fin.any() else None
        out["V1b_stretch_reconstruction_max_diff"] = stretch_reconstruction(m, model)
        y = d.score.to_numpy(float); maxs = d.groupby("person").score.transform("count") * c.shape[1] - d.groupby("person").score.transform("count")
        tot = d.groupby("person").score.transform("sum"); nonext = ((tot > 0) & (tot < maxs)).to_numpy()
        w = wle_generic(c[nonext], y[nonext], d.person.to_numpy()[nonext])
        both = w.dropna().index.intersection(persons.index)
        out["V2_wle_max_diff"] = float(np.abs(w[both] - persons[both]).max()) if len(both) else None
        out["V2_n_persons"] = int(len(both))
        th_w = d.person.map(w).to_numpy(float); okc = np.isfinite(th_w) & nonext       # CSEM = 1/sqrt(test information)
        _, _, var, _, _ = _moments(th_w[okc], c[okc])
        mine = 1 / np.sqrt(pd.Series(var).groupby(d.person.to_numpy()[okc]).sum())
        csem = getattr(m, f"csem_vector_{model}").reindex(mine.index)
        out["V2b_csem_max_diff"] = float(np.abs(mine - csem).max())
        okf = np.isfinite(th)                                      # RaschPy fit includes extreme persons (adjusted WLE)
        diffs = []
        for key, facet in (("item", "item"), ("rater", "rater")):
            mine = residual_fit_generic(c[okf], y[okf], th[okf], d[facet].to_numpy()[okf])
            for kind in ("infit", "outfit"):                      # unrounded RaschPy attributes (tables round to 3 dp)
                theirs = getattr(m, f"{key}_{kind}_ms_{model}")
                diffs.append(float(np.abs(mine[kind] - theirs.reindex(mine.index)).max()))
        out["V3_fit_max_diff"] = max(diffs)
    except Exception as e:                                         # an identity that cannot be computed is reported, not passed
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def cell_bootstrap(res, B=50, seed=None):
    """V10-validated person-cluster bootstrap for the rater-side cells of bivector/centrality/bistretch (RaschPy reports
    no per-cell SE): resample persons with replacement, refit, SE = SD of the B refits, normal 95% CI."""
    model = res["model"]
    if not cell_bootstrap_validated(model):
        raise MFRMError("Hücre bootstrap SE'si bu model için doğrulanmadı; raporlanmaz.")
    d = res["_d"]; rng = np.random.default_rng(res["seed"] if seed is None else seed); ix = d.groupby("person").indices
    persons = list(ix); est = _cells(res["_m"], model); reps = []
    for _ in range(int(B)):
        pick = rng.choice(len(persons), len(persons), replace=True)
        rows = np.concatenate([ix[persons[j]] for j in pick])
        lab = np.repeat(np.arange(len(pick)), [len(ix[persons[j]]) for j in pick])
        db = d.iloc[rows].assign(person=[f"B{j:05d}" for j in lab])
        w = db.pivot(index=["rater", "person"], columns="item", values="score").astype(float)
        w.index.names = ["Rater", "Person"]; w.columns.name = None; mb = _model(w, res["K"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try: mb.calibrate(model=model); reps.append(_cells(mb, model))
            except Exception: pass
    bt = pd.DataFrame(reps); se = bt.std(ddof=1)
    out = pd.DataFrame({"Kestirim": pd.Series(est), "Bootstrap SE": se})
    out["%95 GA alt"] = out.Kestirim - 1.959964 * out["Bootstrap SE"]; out["%95 GA üst"] = out.Kestirim + 1.959964 * out["Bootstrap SE"]
    out["B (başarılı)"] = len(bt)
    return out.rename_axis("Parametre").reset_index()


def _cells(m, model):
    Rv = getattr(m, f"raters_{model}"); out = {}
    for idx, row in Rv.iterrows():
        for c, v in row.items(): out[f"{idx}:{c}"] = float(v)
    for p in STRETCH.get(model, ())[1:]:
        out.update({f"{p}:{r}": float(v) for r, v in getattr(m, f"{p}_{model}").items()})
    return out


# ---------------------------------------------------------------- optional RaschPy analyses
def model_selection(res, test="AIC"):
    m = res["_m"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); m.model_selection(test=test, seed=res["seed"])
    return getattr(m, f"model_comparison_mfrm_{test.lower()}_summary").reset_index()


def group_test(res, df, person, covariate, kind="drf", n_boot=200):
    """DIF (person groups × items) or DRF (person groups × raters) with a person-level covariate column."""
    cov = df[[person, covariate]].dropna().astype({person: str}).drop_duplicates()
    if cov[person].duplicated().any():
        raise MFRMError("Grup değişkeni her birey için tek bir değer almalıdır.")
    if cov[covariate].nunique() < 2:
        raise MFRMError("Grup değişkeni en az iki grup içermelidir.")
    ex = cov.set_index(person)[[covariate]]; ex.columns = ["group"]
    import raschpy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = raschpy.MFRM(_rewide(res), max_score=res["K"], exogenous=ex)
        try:
            if kind == "dif":
                m.dif_test("group", model=res["model"])
            else:
                np.random.seed(res["seed"]); m.drf_test("group", model=res["model"], no_of_samples=int(n_boot))
        except Exception as e:
            raise MFRMError(f"RaschPy {kind}_test başarısız: {type(e).__name__}: {e}") from e
    tabs = {"tablo": getattr(m, f"{kind}_table", None), "omnibus": getattr(m, f"{kind}_omnibus_table", None)}
    if kind == "drf":
        tabs["eşik"] = getattr(m, "drf_threshold_table", None)
    return {k: (v.reset_index() if isinstance(v, pd.DataFrame) else v) for k, v in tabs.items() if v is not None}


def _rewide(res):
    d = res["_d"]; w = d.pivot(index=["rater", "person"], columns="item", values="score").astype(float)
    w.index.names = ["Rater", "Person"]; w.columns.name = None; return w


# ---------------------------------------------------------------- plots (RaschPy figures + own Wright map of RaschPy estimates)
def plots(res, max_items=6, fast=False):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    m, model = res["_m"], res["model"]; out = {}
    kw = dict(font="DejaVu Sans")
    def png(fig):
        import io
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig); return buf.getvalue()
    out["wright_haritasi.png"] = png(wright_map(res))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, call in [("kategori_egrileri_{i}.png", lambda i: m.crcs(model=model, item=i, **kw)),
                           ("madde_karakteristik_{i}.png", lambda i: m.icc(i, model=model, **kw))]:
            for i in list(res["items"].index)[:max_items]:
                try: out[name.format(i=i)] = png(call(i))
                except Exception: pass
        if fast:
            out = {k: v for k, v in out.items() if not k.startswith("madde_")}; return out
        for name, call in [("test_karakteristik.png", lambda: m.tcc(model=model, **kw)),
                           ("test_bilgi.png", lambda: m.test_info(model=model, **kw)),
                           ("standart_artiklar.png", lambda: m.std_residuals_plot(model=model, **kw))]:
            try: out[name] = png(call())
            except Exception: pass
    return out


def rater_location(res):
    """Rater position for the Wright map: λ_r (global), λ_r (stretch models) or the mean of the rater's λ cells (labelled)."""
    m, model = res["_m"], res["model"]
    if model == "global": return m.raters_global
    if model in STRETCH: return getattr(m, f"lambda_{model}")
    R = getattr(m, f"raters_{model}")
    return R.groupby(level=0).mean().mean(axis=1) if isinstance(R.index, pd.MultiIndex) else R.mean(axis=1)


def wright_map(res):
    import matplotlib.pyplot as plt
    p = res["persons"]["Estimate"].dropna(); it = res["items"]["Estimate"]; rt = rater_location(res)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 5), sharey=True, gridspec_kw=dict(width_ratios=[2, 2]))
    a.hist(p, bins=25, orientation="horizontal", color="#9db6cc"); a.invert_xaxis(); a.set_xlabel("Birey"); a.set_ylabel("Logit")
    for x, (s, col, lab) in enumerate(((it, "#2b5f8a", "Madde"), (rt, "#b5452f", "Puanlayıcı"))):
        b.scatter([x] * len(s), s.to_numpy(float), color=col, s=14)
        for n, v in s.items(): b.annotate(str(n), (x + .06, float(v)), fontsize=7, color=col)
    for t in np.asarray(res["thresholds"]["Estimate"], float): b.axhline(t, color="#888", lw=.5, ls=":")
    b.set_xticks([0, 1], ["Madde", "Puanlayıcı"]); b.set_xlim(-.4, 1.8)
    note = "" if res["model"] == "global" or res["model"] in STRETCH else " (puanlayıcı: λ hücre ortalaması)"
    fig.suptitle(f"Wright haritası — RaschPy {res['model']} (RCL){note}"); return fig


# ---------------------------------------------------------------- tables and exports
def tables(res, show_ids=False):
    persons = res["persons"].copy()
    if not show_ids:
        persons.index = [f"Kişi_{i + 1:04d}" for i in range(len(persons))]
    nose = lambda df: df.drop(columns=[c for c in df.columns if c == "SE" or (isinstance(c, tuple) and "SE" in c)])
    rater_ok = rater_se_reliable(res["model"]); rp = res["rater_params"].set_index("Puanlayıcı")
    t = {"Maddeler": nose(res["items"]), "Eşikler": nose(res["thresholds"]),
         "Puanlayıcılar": res["raters"] if rater_ok else nose(res["raters"]),
         "Puanlayıcı parametreleri": rp if rater_ok else rp.drop(columns="SE"), "Bireyler": persons,
         "Test özetleri": res["test"], "Kategori sıklıkları": res["category_counts"]}
    if res["category_stats"] is not None:
        t["Kategori genişliği (bootstrap)"] = nose(res["category_stats"])
    if res.get("cell_bootstrap") is not None:
        t["Puanlayıcı hücreleri (bootstrap SE/GA)"] = res["cell_bootstrap"]
    flat = lambda v: v.set_axis([" | ".join(map(str, c)) if isinstance(c, tuple) else c for c in v.columns], axis=1)
    return {k: flat(v.reset_index()).rename(columns={"index": "Öğe"}) for k, v in t.items()}


def status_table(model):
    rows = [dict(Çıktı=lab, Anahtar=k, Durum=status(model, k)) for k, lab in OUTPUT_LABELS.items()]
    rows += [dict(Çıktı=k, Anahtar=k, Durum="UNSUPPORTED", Not=v) for k, v in UNSUPPORTED.items()]
    return pd.DataFrame(rows)


def exports(res, config, show_ids=False, extra_tables=None, figs=None):
    import io, zipfile
    T = tables(res, show_ids); T.update(extra_tables or {}); T["Kanıt durumu"] = status_table(res["model"])
    figs = figs if figs is not None else plots(res); files = {f"{k}.csv": v.to_csv(index=False).encode("utf-8-sig") for k, v in T.items()}
    xb = io.BytesIO()
    with pd.ExcelWriter(xb, engine="openpyxl") as w:
        for k, v in T.items(): v.to_excel(w, sheet_name=k[:31], index=False)
    files["raschpy_mfrm.xlsx"] = xb.getvalue(); files.update(figs)
    files["raschpy_mfrm_raporu.docx"] = docx_report(res, figs, config, show_ids, extra_tables)
    files["analiz_yapilandirmasi.json"] = json.dumps(dict(config, model=res["model"], se_method=res["se_method"],
        n_boot=res["n_boot"], seed=res["seed"], versions=res["versions"], identity=res["identity"],
        convergence=res["convergence"]), ensure_ascii=False, indent=1, default=str).encode()
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in files.items(): zf.writestr(k, v)
    files["raschpy_mfrm.zip"] = z.getvalue()
    return files


STATUS_TR = {"CERTIFIED": "CERTIFIED (kanıt kapsamında doğrulandı)", "PARTIAL": "PARTIAL (kısmi kanıt)",
             "NOT CERTIFIED": "NOT CERTIFIED (doğrulanmadı)", "UNSUPPORTED": "UNSUPPORTED (desteklenmiyor)"}
EVIDENCE_SCOPE = ("Kanıt kapsamı: depodaki örnek veri setleri (dengeli ve bağlantılı tasarımlar) ile benzetim koşulları "
                  "(4 madde, 4 kategori, en çok 5 puanlayıcı). Durumlar başka tasarımlara kendiliğinden genellenmez.")
GENERATED_NOTE = ("Raporun ve kodun üretilmiş olması (GENERATED) bilimsel doğrulama değildir; bilimsel durum yalnız "
                  "CERTIFIED / PARTIAL / NOT CERTIFIED etiketleriyle ve belirtilen kanıt kapsamıyla gösterilir.")
FIT_BAND = (0.5, 1.5)


def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def summary(res):
    r = res["report"]; p = res["persons"]
    extreme = int(((p["Score"] == 0) | (p["Score"] == p["Max score"])).sum())
    return dict(n_persons=int(r["n_persons"]), n_items=int(res["_d"].item.nunique()), n_raters=int(res["_d"].rater.nunique()),
                n_obs=int(r["n_rows"]), missing=int(r["missing_scores_excluded"]), extreme=extreme,
                categories=list(map(int, r["categories"])), model=res["model"], model_label=MODELS[res["model"]][0],
                status=status(res["model"], "parameters"))


def rater_fit(res):
    m, model = res["_m"], res["model"]
    return pd.DataFrame({"Infit MnSq": getattr(m, f"rater_infit_ms_{model}"), "Outfit MnSq": getattr(m, f"rater_outfit_ms_{model}")})


def academic_tables(res, show_ids=False, max_persons=None):
    """Ordered (key, title, table, explanation) for the screen and the Word report; text uses computed values only."""
    model = res["model"]; out = []; it = res["items"]; th = res["thresholds"]; S = summary(res)
    items = pd.DataFrame({"Madde": it.index, "Güçlük (logit)": it["Estimate"].to_numpy(float), "Gözlem": it["Count"].to_numpy(int),
                          "Ortalama puan / en yüksek puan": it["Facility"].to_numpy(float),
                          "Infit MnSq": it["Infit MS"].to_numpy(float), "Outfit MnSq": it["Outfit MS"].to_numpy(float)})
    hard, easy = it["Estimate"].idxmax(), it["Estimate"].idxmin()
    misfit = int(((items[["Infit MnSq", "Outfit MnSq"]] < FIT_BAND[0]) | (items[["Infit MnSq", "Outfit MnSq"]] > FIT_BAND[1])).any(axis=1).sum())
    out.append(("items", "Madde Parametreleri", items,
                f"Güçlük, maddenin logit ölçeğindeki konumudur; yüksek değer daha zor maddeyi gösterir. En zor madde {hard} "
                f"({_f(it['Estimate'][hard])}), en kolay madde {easy} ({_f(it['Estimate'][easy])}). Infit veya Outfit değeri yaygın "
                f"kullanılan {FIT_BAND[0]}–{FIT_BAND[1]} aralığı dışında kalan madde sayısı: {misfit}."))
    tau = np.asarray(th["Estimate"], float)
    thr = pd.DataFrame({"Eşik": [f"{k - 1}→{k}" for k in range(1, len(tau) + 1)], "Konum (logit)": tau,
                        "Infit MnSq": th["Infit MS"].to_numpy(float), "Outfit MnSq": th["Outfit MS"].to_numpy(float)})
    ordered = bool(np.all(np.diff(tau) > 0))
    out.append(("thresholds", "Eşik Parametreleri", thr,
                "Eşik, iki komşu kategorinin eşit olasılıkla seçildiği konumdur (Andrich eşiği). Kestirilen eşikler "
                + ("artan sıradadır." if ordered else "artan sırada değildir (sırasız eşik); kategori işleyişi incelenmelidir.")))
    loc = rater_location(res); fit = rater_fit(res).reindex(loc.index)
    n_obs = res["_d"].groupby("rater").size().reindex(loc.index)
    lab = {"global": "Katılık λ (logit)"}.get(model, "Katılık λ (logit)" if model in STRETCH else "Ortalama katılık (λ hücre ort., logit)")
    raters = pd.DataFrame({"Puanlayıcı": loc.index, lab: loc.to_numpy(float), "Gözlem": n_obs.to_numpy(int),
                           "Infit MnSq": fit["Infit MnSq"].to_numpy(float), "Outfit MnSq": fit["Outfit MnSq"].to_numpy(float)})
    if rater_se_reliable(model) and (model == "global" or model in STRETCH):
        se = res["rater_params"]; se = se[se.Parametre.isin(["λ", "lambda"])].set_index("Puanlayıcı")["SE"]
        raters.insert(2, "SE", se.reindex(loc.index).to_numpy(float))
    sev, len_ = loc.idxmax(), loc.idxmin()
    out.append(("raters", "Puanlayıcı Katılıkları", raters,
                f"Yüksek katılık değeri, aynı bireye ve maddeye daha düşük puan verme eğilimini gösterir. En katı puanlayıcı {sev} "
                f"({_f(loc[sev])}), en cömert puanlayıcı {len_} ({_f(loc[len_])}); aralık {_f(loc[sev] - loc[len_])} logit."
                + ("" if model == "global" or model in STRETCH else " Bu modelde puanlayıcı başına tek katılık yoktur; sütun, "
                   "puanlayıcının madde/eşik hücrelerinin ortalamasıdır.")))
    if model != "global":
        rp = res["rater_params"].copy()
        names = {"items": ["Madde"], "thresholds": ["Eşik"], "bivector": ["Madde", "Eşik"], "matrix": ["Madde", "Eşik"],
                 "bistretch": ["Madde", "Eşik"]}.get(model, [])
        rp = rp.rename(columns={f"Düzey{i}": n for i, n in enumerate(names)})
        if "Parametre" in rp:
            rp["Parametre"] = rp["Parametre"].map({"lambda": "λ (konum)", "omega": "ω (germe)", "omega_items": "ω madde (germe)",
                                                   "omega_thresholds": "ω eşik (germe)"}).fillna(rp["Parametre"])
        if not rater_se_reliable(model): rp = rp.drop(columns="SE")
        out.append(("rater_params", "Modele Özgü Puanlayıcı Parametreleri", rp,
                    f"{MODELS[model][0]} modelinin puanlayıcı tarafı parametreleri (puanlayıcı terimi {MODELS[model][1]}). "
                    + ("ω = 1 germe olmadığını, ω < 1 sıkışmayı (merkeze yığılma), ω > 1 yayılmayı gösterir." if model in STRETCH else
                       "Değerler logit ölçeğindedir.")))
    if res.get("cell_bootstrap") is not None:
        out.append(("cell_bootstrap", "Puanlayıcı Hücreleri için Bootstrap Standart Hataları", res["cell_bootstrap"],
                    "Birey kümeli bootstrap (bireyler yerine koyarak yeniden örneklenir, model her örneklemde yeniden kestirilir); "
                    "%95 güven aralığı normal yaklaşımla hesaplanmıştır. Bu yöntem benzetimde bu model için kapsama kapısını geçmiştir."))
    p = res["persons"]; est = p["Estimate"]
    ps = pd.DataFrame({"İstatistik": ["Birey sayısı", "Ortalama (logit)", "Standart sapma", "En küçük", "En büyük",
                                      "Ortalama CSEM", "Uç puanlı birey (0 veya tam puan)"],
                       "Değer": [str(S["n_persons"]), *(_f(float(v)) for v in (est.mean(), est.std(), est.min(), est.max(),
                                                                                 p["CSEM"].mean())), str(S["extreme"])]})
    out.append(("persons_summary", "Birey Ölçüleri Özeti", ps,
                "Birey ölçüleri Warm ağırlıklı olabilirlik (WLE) kestirimleridir; CSEM koşullu ölçme standart hatasıdır. "
                f"Uç puanlı {S['extreme']} birey için RaschPy düzeltilmiş uç kestirim raporlar; bu değerler diğerlerinden daha belirsizdir."))
    pt = p.copy()
    pt.index = pt.index if show_ids else [f"Kişi_{i + 1:04d}" for i in range(len(pt))]
    pt = pd.DataFrame({"Birey": pt.index, "Ölçü (logit)": pt["Estimate"].to_numpy(float), "CSEM": pt["CSEM"].to_numpy(float),
                       "Ham puan": pt["Score"].to_numpy(), "En yüksek puan": pt["Max score"].to_numpy(),
                       "Infit MnSq": pt["Infit MS"].to_numpy(float), "Outfit MnSq": pt["Outfit MS"].to_numpy(float)})
    out.append(("persons", "Birey Ölçüleri", pt if max_persons is None else pt.head(max_persons),
                "Her birey için WLE ölçüsü, CSEM, ham puan ve uyum değerleri." +
                ("" if max_persons is None or len(pt) <= max_persons else
                 f" Raporda ilk {max_persons} birey gösterilmiştir; {len(pt)} bireyin tam tablosu Excel/CSV dosyasındadır.")))
    cc = res["category_counts"].copy(); cats = [c for c in cc.columns if c not in ("Total", "Missing")]
    cc = cc[cats].T; cc.index = [f"Kategori {c}" for c in cc.index]; cc["Toplam"] = cc.sum(axis=1)
    cc = cc.reset_index().rename(columns={"index": "Kategori"})
    out.append(("category_counts", "Kategori Sıklıkları", cc,
                "Her kategorinin madde bazında gözlenen sıklığıdır (betimsel). "
                + (f"10'dan az gözlemli kategori(ler): {res['report']['sparse_categories']}." if res["report"]["sparse_categories"]
                   else "Tüm kategorilerde en az 10 gözlem vardır.")))
    t = res["test"].T.reset_index().rename(columns={"index": "Yüzey"})
    t["Yüzey"] = t["Yüzey"].map({"Items": "Maddeler", "Persons": "Bireyler"}).fillna(t["Yüzey"])
    t = t.rename(columns={"Mean": "Ortalama", "SD": "SS", "Separation ratio": "Ayırma oranı", "Strata": "Tabaka",
                          "Reliability": "Güvenirlik"})
    out.append(("test", "Ayırma ve Güvenirlik Özetleri", t,
                "RaschPy tarafından raporlanan ayırma oranı, tabaka sayısı ve güvenirlik. Bu özetler bağımsız olarak yeniden "
                "hesaplanmamıştır; yorumda bu sınırlılık dikkate alınmalıdır."))
    return out


def evidence_rows(model):
    rows = [dict(Çıktı=lab, Durum=STATUS_TR.get(status(model, k), status(model, k))) for k, lab in OUTPUT_LABELS.items()]
    return pd.DataFrame(rows)


TABLE_STATUS = {"items": "parameters", "thresholds": "parameters", "raters": "parameters", "rater_params": "parameters",
                "cell_bootstrap": "cell_bootstrap_se", "persons_summary": "persons", "persons": "persons",
                "category_counts": "category_counts", "test": "test_stats"}


def docx_report(res, figs, config=None, show_ids=False, extra_tables=None):
    """Academic Word report built only from computed values (no HTML, API calls or debug output)."""
    import io
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt
    doc = Document(); model = res["model"]; S = summary(res); n_tab = [0]; n_fig = [0]
    st_ = doc.styles["Normal"]; st_.font.name = "Times New Roman"; st_.font.size = Pt(11)
    from docx.shared import RGBColor
    for h in ("Title", "Heading 1", "Heading 2"):
        doc.styles[h].font.name = "Times New Roman"; doc.styles[h].font.color.rgb = RGBColor(0, 0, 0)
    def para(text, italic=False, size=None, bold=False):
        p = doc.add_paragraph(); run = p.add_run(text); run.italic = italic; run.bold = bold
        if size: run.font.size = Pt(size)
        return p
    def table(title, df, note, key=None):
        n_tab[0] += 1
        para(f"Tablo {n_tab[0]}", bold=True).paragraph_format.space_after = Pt(0)
        para(title, italic=True)
        t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, c in enumerate(df.columns):
            cell = t.rows[0].cells[i]; cell.text = str(c); cell.paragraphs[0].runs[0].bold = True
        for row in df.itertuples(index=False):
            cells = t.add_row().cells
            for i, v in enumerate(row):
                num = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                cells[i].text = (_f(float(v)) if isinstance(v, (float, np.floating)) else str(v))
                if num: cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for row in t.rows:
            for c in row.cells:
                for r_ in c.paragraphs[0].runs: r_.font.size = Pt(9)
        lab = STATUS_TR.get(status(model, TABLE_STATUS[key]), "") if key in TABLE_STATUS else ""
        para("Not. " + note + (f" Bilimsel durum: {lab}." if lab else ""), size=9)
    doc.add_heading("Çok Yüzeyli Rasch Modeli (MFRM) Analiz Raporu", 0)
    para(f"RaschPy {RASCHPY_VERSION} ile bağımsız deneysel analiz — model: {S['model']} ({S['model_label']})", italic=True)
    doc.add_heading("1. Amaç ve Kapsam", 1)
    para("Bu rapor, puanlayıcı katılığını madde güçlüğü ve birey yeteneğiyle aynı logit ölçeğinde kestiren çok yüzeyli Rasch "
         "modelinin sonuçlarını sunar. Analiz deneysel RaschPy modülüyle yapılmıştır; platformun R/TAM tabanlı MFRM "
         "analizinden bağımsızdır.")
    doc.add_heading("2. Yöntem", 1)
    doc.add_heading("2.1 Veri ve Örneklem", 2)
    para(f"Analize {S['n_obs']} puan, {S['n_persons']} birey, {S['n_items']} madde ve {S['n_raters']} puanlayıcı girmiştir. "
         f"Puan kategorileri: {', '.join(map(str, S['categories']))}. Eksik puan nedeniyle dışlanan satır: {S['missing']}. "
         f"Uç puanlı (0 veya tam puan) birey: {S['extreme']}. Tasarım bağlantılılığı, tekrar eden puanlar ve kategori kullanımı "
         "analiz öncesinde denetlenmiştir.")
    doc.add_heading("2.2 Model", 2)
    para(f"{EQUATION}. Puanlayıcı terimi: {MODELS[model][1]} ({S['model_label']}).")
    doc.add_heading("2.3 Kestirim", 2)
    se_text = ("Puanlayıcı standart hataları sandviç yöntemiyle hesaplanmış ve benzetimde güvenilir kapsama gösterdiği için raporlanmıştır."
               if rater_se_reliable(model) else "Puanlayıcı standart hataları benzetimde güvenilir kapsama göstermediği için raporlanmamıştır.")
    para("Parametreler RaschPy MFRM.calibrate ile düzenlileştirilmiş ikili bileşik koşullu olabilirlik (RCL) yöntemiyle "
         "kestirilmiştir; bu yöntemde birey parametreleri koşullama ile elenir ve ridge cezası aşırı değerleri sıfıra doğru çeker. "
         "Birey ölçüleri Warm WLE ile kestirilmiştir. Bu kestirim TAM'ın marjinal en çok olabilirlik kestirimiyle eşdeğer değildir; "
         f"iki motorun değerleri doğrudan karşılaştırılamaz. {se_text} Madde, eşik ve kategori standart hataları için kapsama kanıtı "
         "bulunmadığından raporlanmamıştır.")
    doc.add_heading("3. Bulgular", 1)
    for key, title, df, note in academic_tables(res, show_ids, max_persons=15):
        table(title, df, note, key)
    for name, df in (extra_tables or {}).items():
        if isinstance(df, pd.DataFrame) and len(df):
            table(name, df.head(30), "RaschPy tarafından raporlanmıştır; bağımsız olarak doğrulanmamıştır (NOT CERTIFIED).")
    if figs:
        doc.add_heading("4. Şekiller", 1)
        for name, png in figs.items():
            n_fig[0] += 1
            doc.add_picture(io.BytesIO(png), width=Inches(5.8)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            para(f"Şekil {n_fig[0]}. {fig_title(name)}", italic=True, size=9)
    doc.add_heading(f"{5 if figs else 4}. Bilimsel Durum ve Sınırlılıklar", 1)
    table("Çıktıların Bilimsel Durumu", evidence_rows(model), EVIDENCE_SCOPE)
    limits = [GENERATED_NOTE,
              "NOT CERTIFIED olarak işaretlenen çıktılar doğrulanmış sonuç gibi yorumlanmamalıdır.",
              "Uç puanlı bireylerin ölçüleri düzeltilmiş kestirimlerdir ve daha yüksek belirsizlik taşır.",
              f"Eksik puanlar analizden dışlanmıştır ({S['missing']} satır); eksiklik mekanizması incelenmemiştir.",
              "RaschPy yalnız bir puanlayıcı yüzeyini destekler; görev/oturum gibi ek yüzeyler modellenmemiştir. Andersen LR testi "
              "RaschPy'de devre dışıdır."]
    if S["status"] == "NOT CERTIFIED":
        limits.insert(1, f"Bu modelin ({model}) parametreleri NOT CERTIFIED durumundadır: önceden kaydedilmiş geri kazanım "
                         "ölçütü örnek veri setlerinde karşılanmamıştır. Bulgular betimsel olarak değerlendirilmelidir.")
    for x in limits: doc.add_paragraph(x, style="List Bullet")
    doc.add_heading(f"{6 if figs else 5}. Tekrarlanabilirlik", 1)
    rep = pd.DataFrame({"Öğe": ["Yazılım", "Model", "Standart hata yöntemi", "Rastgele tohum", "Bağımsız özdeşlik denetimleri"],
                        "Değer": [", ".join(res["versions"]), model, res["se_method"] + (f" (B = {res['n_boot']})" if res["n_boot"] else ""),
                                  str(res["seed"]), "geçti" if identity_ok(res) else "başarısız veya hesaplanamadı"]})
    src = (config or {}).get("source")
    if isinstance(src, dict) and src.get("sha"):
        rep.loc[len(rep)] = ["Veri parmak izi (SHA-256)", config["source"]["sha"][:16] + "…"]
    table("Analiz Ayarları ve Yazılım", rep, "Aynı veri, model ve tohumla analiz aynı sonuçları üretir. Tam tablolar Excel/CSV dosyalarındadır.")
    buf = io.BytesIO(); doc.save(buf); return buf.getvalue()


FIGURE_TITLES = {"wright_haritasi": "Wright haritası: bireyler, maddeler ve puanlayıcılar aynı logit ölçeğinde",
                 "kategori_egrileri": "Kategori olasılık eğrileri", "madde_karakteristik": "Madde karakteristik eğrisi",
                 "test_karakteristik": "Test karakteristik eğrisi", "test_bilgi": "Test bilgi fonksiyonu",
                 "standart_artiklar": "Standartlaştırılmış artıklar"}


def fig_title(name):
    stem = name[:-4] if name.endswith(".png") else name
    for k, v in FIGURE_TITLES.items():
        if stem == k: return v
        if stem.startswith(k + "_"): return f"{v} — {stem[len(k) + 1:]}"
    return stem
IDENTITY_TOL = {"V1_expected_score_max_diff": 1e-12, "V1b_stretch_reconstruction_max_diff": 1e-12, "V2_wle_max_diff": 1e-4,
                "V2b_csem_max_diff": 1e-4, "V3_fit_max_diff": 1e-6}


def identity_ok(res):
    i = res["identity"]
    return "error" not in i and all(i.get(k) is None or i[k] <= t for k, t in IDENTITY_TOL.items())


def identity_table(res):
    lab = {"V1_expected_score_max_diff": "Beklenen puan (bağımsız formül)", "V1b_stretch_reconstruction_max_diff": "Germe parametre yapısı",
           "V2_wle_max_diff": "Warm WLE (bağımsız çözüm)", "V2b_csem_max_diff": "CSEM (bağımsız bilgi fonksiyonu)",
           "V3_fit_max_diff": "Infit/Outfit (bağımsız artıklar)"}
    i = res["identity"]
    rows = [dict(Denetim=v, **{"En büyük fark": i[k], "Tolerans": IDENTITY_TOL[k], "Sonuç": "geçti" if i[k] <= IDENTITY_TOL[k] else "başarısız"})
            for k, v in lab.items() if i.get(k) is not None]
    if "error" in i: rows.append(dict(Denetim="Hesaplama hatası", **{"En büyük fark": np.nan, "Tolerans": np.nan, "Sonuç": i["error"]}))
    return pd.DataFrame(rows)
