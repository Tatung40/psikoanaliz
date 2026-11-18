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
def plots(res, max_items=6):
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


def exports(res, config, show_ids=False, extra_tables=None):
    import io, zipfile
    T = tables(res, show_ids); T.update(extra_tables or {}); T["Kanıt durumu"] = status_table(res["model"])
    figs = plots(res); files = {f"{k}.csv": v.to_csv(index=False).encode("utf-8-sig") for k, v in T.items()}
    xb = io.BytesIO()
    with pd.ExcelWriter(xb, engine="openpyxl") as w:
        for k, v in T.items(): v.to_excel(w, sheet_name=k[:31], index=False)
    files["raschpy_mfrm.xlsx"] = xb.getvalue(); files.update(figs)
    files["raschpy_mfrm_raporu.docx"] = docx_report(res, T, figs)
    files["analiz_yapilandirmasi.json"] = json.dumps(dict(config, model=res["model"], se_method=res["se_method"],
        n_boot=res["n_boot"], seed=res["seed"], versions=res["versions"], identity=res["identity"],
        convergence=res["convergence"]), ensure_ascii=False, indent=1, default=str).encode()
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in files.items(): zf.writestr(k, v)
    files["raschpy_mfrm.zip"] = z.getvalue()
    return files


def docx_report(res, T, figs):
    import io
    from docx import Document
    from docx.shared import Inches
    doc = Document(); model = res["model"]; r = res["report"]
    doc.add_heading("RaschPy Çok Yüzeyli Rasch (MFRM) Raporu — Deneysel", 0)
    doc.add_paragraph(f"Model: {model} — {MODELS[model][0]}. {EQUATION}; puanlayıcı terimi {MODELS[model][1]}.")
    doc.add_paragraph(f"Kestirim: RaschPy {RASCHPY_VERSION} MFRM.calibrate (yöntem {res['estimation']}: düzenlileştirilmiş ikili "
                      f"bileşik olabilirlik). SE: {res['se_method']}" + (f" ({res['n_boot']} örneklem, tohum {res['seed']})" if res["n_boot"] else "")
                      + ". Birey ölçüleri Warm WLE.")
    doc.add_paragraph("Bu motor TAM'ın marjinal en çok olabilirlik kestirimiyle eşdeğer değildir; değerler doğrudan karşılaştırılamaz. "
                      "Ridge düzenlileştirmesi uç parametreleri sıfıra doğru çeker.")
    doc.add_paragraph(f"Veri: {r['n_rows']} puan, {r['n_persons']} birey, düzeyler {r['levels']}; eksik puan {r['missing_scores_excluded']}; "
                      f"kategoriler {r['categories']} (sıklık {r['category_counts']}).")
    for k, t in T.items():
        doc.add_heading(k, 1); t = t.head(40)
        tab = doc.add_table(rows=1, cols=len(t.columns)); tab.style = "Table Grid"
        for i, c in enumerate(t.columns): tab.rows[0].cells[i].text = str(c)
        for row in t.itertuples(index=False):
            cells = tab.add_row().cells
            for i, v in enumerate(row): cells[i].text = f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    doc.add_heading("Özdeşlik denetimleri", 1)
    for k, v in res["identity"].items(): doc.add_paragraph(f"{k}: {v}", style="List Bullet")
    for k, b in figs.items():
        doc.add_paragraph(k); doc.add_picture(io.BytesIO(b), width=Inches(5.5))
    doc.add_paragraph("Kanıt kapsamı: validation/raschpy_mfrm/evidence.json (yalnız depodaki fikstürler). Sürümler: " + "; ".join(res["versions"]))
    buf = io.BytesIO(); doc.save(buf); return buf.getvalue()
