"""Many-facet Rasch measurement (rating-scale MFRM) — R TAM::tam.mml.mfr engine with Python validation and diagnostics.

Model (marginal ML, theta ~ N(mu, sigma^2)):
    log P(X = k) / P(X = k - 1) = theta_n - delta_i - lambda_j [- gamma_t] - tau_k,   k = 1..K
Identification (TAM constraint = "items"): item, every facet and the thresholds sum to zero; mu and sigma^2 free.
R is required: without Rscript and TAM the analysis is refused (no other estimator is substituted).
"""
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/mfrm_tam.R"
TAM_VERSION = "4.3.25"                    # pinned engine version (packageVersion("TAM") 4.3-25)
NODES = 61


class MFRMError(ValueError):
    """Fail-closed input or estimation problem with a Turkish explanation."""


def prepare(df, person, item, rater, score, extra=()):
    """Validate long data and return (clean frame with canonical column names, report dict)."""
    cols = [person, item, rater, *extra, score]
    if len(set(cols)) != len(cols) or any(c not in df for c in cols):
        raise MFRMError("Birey, madde/ölçüt, puanlayıcı ve puan için veride bulunan farklı sütunlar seçilmelidir.")
    facets = ["rater", *[f"facet{k + 1}" for k in range(len(extra))]]
    d = df[cols].copy(); d.columns = ["person", "item", *facets, "score"]
    for c in d.columns[:-1]:
        if d[c].isna().any():
            raise MFRMError(f"'{cols[list(d.columns).index(c)]}' sütununda boş kimlik var.")
        d[c] = d[c].astype(str)
    report = {"n_rows_input": len(d)}
    s = pd.to_numeric(d["score"], errors="coerce")
    bad = d["score"].notna() & s.isna()
    if bad.any():
        raise MFRMError("Puan sütununda sayı olmayan değer var (ör. " + repr(d.loc[bad, "score"].iloc[0]) + ").")
    missing = int(s.isna().sum()); d = d.assign(score=s)[s.notna()]
    report["missing_scores_excluded"] = missing
    if not np.isfinite(d.score).all() or (d.score != np.round(d.score)).any() or (d.score < 0).any():
        raise MFRMError("Puanlar 0'dan başlayan sıralı tam sayı kategoriler olmalıdır.")
    d["score"] = d.score.astype(int)
    keys = ["person", "item", *facets]
    dup = d[d.duplicated(keys, keep=False)]
    if len(dup):
        raise MFRMError(f"Aynı birey–madde–puanlayıcı birleşimi için birden fazla puan var ({len(dup)} satır). "
                        "Puanlar ortalanmaz; tekrarları kaldırın.")
    cats = sorted(d.score.unique()); K = max(cats)
    if K < 1:
        raise MFRMError("En az iki puan kategorisi gerekir.")
    empty = [k for k in range(K + 1) if k not in cats]
    if empty:
        raise MFRMError(f"Ara kategori(ler) hiç kullanılmamış: {empty}. Eşik parametreleri tanımlanamaz; kategorileri "
                        "yeniden kodlamak araştırmacının kararıdır ve otomatik yapılmaz.")
    for f in ("item", *facets):
        if d[f].nunique() < 2:
            raise MFRMError(f"'{f}' yüzeyinde en az iki düzey gerekir.")
    comps = components(d, facets)
    if comps > 1:
        raise MFRMError(f"Tasarım bağlantılı değil ({comps} ayrı bileşen): yüzey parametreleri ortak bir ölçekte "
                        "tanımlanamaz. Bileşenleri ortak puanlayıcı veya madde ile bağlayın.")
    counts = d.score.value_counts().reindex(range(K + 1), fill_value=0)
    report.update(n_rows=len(d), n_persons=d.person.nunique(), categories=list(range(K + 1)),
                  category_counts=counts.astype(int).tolist(), facets=facets,
                  levels={f: int(d[f].nunique()) for f in ("item", *facets)},
                  sparse_categories=[int(k) for k, c in counts.items() if c < 10],
                  extreme_persons=_extremes(d, K))
    return d.reset_index(drop=True), report


def _extremes(d, K):
    g = d.groupby("person").score.agg(["sum", "count"])
    return int(((g["sum"] == 0) | (g["sum"] == K * g["count"])).sum())


def components(d, facets):
    """Connected components of the person–element graph (elements = items and facet levels)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for row in d[["person", "item", *facets]].itertuples(index=False):
        a = find(("p", row[0]))
        for k, v in enumerate(row[1:]):
            b = find((k, v))
            if a != b:
                parent[b] = a
    return len({find(x) for x in list(parent)})


def engine_available():
    r = shutil.which("Rscript")
    if not r:
        return False, "Rscript bulunamadı. MFRM kestirimi R TAM gerektirir; başka bir kestirici ikame edilmez."
    try:
        v = subprocess.run([r, "-e", 'cat(as.character(packageVersion("TAM")))'], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False, "R yanıt vermedi."
    if v.returncode != 0:
        return False, "R TAM paketi kurulu değil."
    if v.stdout.strip().replace("-", ".") != TAM_VERSION:
        return False, f"R TAM sürümü {v.stdout.strip()}; doğrulanan sürüm {TAM_VERSION}. Sonuçlar raporlanmadı."
    return True, ""


# ---------------------------------------------------------------- model identities (Python, independent of TAM code)
def _log_probs(theta, eta, tau):
    """log P(X = k | theta) for rows x nodes; eta = -(delta_i + lambda_j + ...); tau thresholds (K)."""
    K = len(tau); k = np.arange(K + 1); cum = np.concatenate([[0.], np.cumsum(tau)])
    num = k[None, None, :] * (np.asarray(theta)[None, :, None] + eta[:, None, None]) - cum[None, None, :]
    return num - np.logaddexp.reduce(num, axis=2, keepdims=True)


def _design(d, facets):
    return [d["item"].to_numpy()] + [d[f].to_numpy() for f in facets]


def _eta(d, params, facets):
    eta = -d["item"].map(params["item"]).to_numpy(float)
    for f in facets:
        eta = eta - d[f].map(params[f]).to_numpy(float)
    return eta


def nodes_weights(mean, variance, n=NODES):
    z = np.linspace(-6, 6, n); w = np.exp(-(z - mean) ** 2 / (2 * variance)); return z, w / w.sum()


def person_loglik(d, params, facets, z):
    lp = _log_probs(z, _eta(d, params, facets), params["tau"])
    rows = np.take_along_axis(lp, d.score.to_numpy()[:, None, None].repeat(len(z), 1), 2)[:, :, 0]
    return pd.DataFrame(rows).groupby(d.person.to_numpy(), sort=False).sum()


def marginal_loglik(d, params, facets, mean, variance):
    z, w = nodes_weights(mean, variance); per = person_loglik(d, params, facets, z).to_numpy()
    m = per.max(axis=1, keepdims=True)
    return float((np.log(np.exp(per - m) @ w) + m[:, 0]).sum())


def eap(d, params, facets, mean, variance):
    z, w = nodes_weights(mean, variance); per = person_loglik(d, params, facets, z)
    post = np.exp(per.to_numpy() - per.to_numpy().max(axis=1, keepdims=True)) * w; post /= post.sum(axis=1, keepdims=True)
    m = post @ z; sd = np.sqrt(post @ z ** 2 - m ** 2)
    return pd.DataFrame({"eap": m, "eap_sd": sd}, index=per.index)


def wle(d, params, facets, tol=1e-10, maxit=200):
    """Warm (1989) weighted likelihood estimate and SE = 1/sqrt(information), solved per person by Newton steps."""
    eta = _eta(d, params, facets); tau = np.asarray(params["tau"]); K = len(tau); k = np.arange(K + 1)
    out = {}
    for pid, ix in d.groupby("person", sort=False).indices.items():
        e, y = eta[ix], d.score.to_numpy()[ix]; th = 0.
        for _ in range(maxit):
            p = np.exp(_log_probs([th], e, tau)[:, 0, :]); m1 = p @ k; m2 = p @ k ** 2; m3 = p @ k ** 3
            info = (m2 - m1 ** 2).sum(); j = (m3 - 3 * m1 * m2 + 2 * m1 ** 3).sum()      # dI/dtheta
            g = (y - m1).sum() + j / (2 * info)
            dinfo = sum(((p * (k - m1[:, None]) ** 4).sum(1) - 3 * (m2 - m1 ** 2) ** 2))  # d j / dtheta
            h = -info + (dinfo * info - j ** 2) / (2 * info ** 2)
            step = g / h; th -= np.clip(step, -1, 1)
            if abs(step) < tol: break
        out[pid] = (th, 1 / np.sqrt(info))
    return pd.DataFrame(out, index=["wle", "wle_se"]).T


def unpack(xsi, facets):
    """TAM xsi.facets (parameter, facet, xsi) -> {'item': {...}, facet: {...}, 'tau': array}."""
    params = {"item": {}, **{f: {} for f in facets}, "tau": []}
    for r in xsi.itertuples():
        if r.facet == "step": params["tau"].append(r.xsi)
        elif r.facet == "item": params["item"][r.parameter] = r.xsi
        else: params[r.facet][r.parameter[len(r.facet):]] = r.xsi
    params["tau"] = np.array(params["tau"]); return params


def observed_information_se(d, params, facets, mean, variance, h=1e-4):
    """SE of every level under sum-to-zero constraints: numerical Hessian of the marginal logLik over free parameters
    (all levels but the last of each block, plus mean and log-variance), then delta method for the dependent level."""
    blocks = [("item", sorted(params["item"]))] + [(f, sorted(params[f])) for f in facets] + [("tau", list(range(len(params["tau"]))))]
    free = [(b, lv) for b, lvs in blocks for lv in lvs[:-1]]

    def build(x):
        p = {b: dict(params[b]) for b in ["item", *facets]}; p["tau"] = np.array(params["tau"], float); i = 0
        for b, lvs in blocks:
            vals = list(x[i:i + len(lvs) - 1]); i += len(lvs) - 1; vals.append(-sum(vals))
            if b == "tau": p["tau"] = np.array(vals)
            else: p[b] = dict(zip(lvs, vals))
        return p, x[i], math.exp(x[i + 1])

    x0 = np.array([params[b][lv] if b != "tau" else params["tau"][lv] for b, lv in free] + [mean, math.log(variance)])
    f = lambda x: marginal_loglik(d, *build(x)[:1], facets, *build(x)[1:])
    n = len(x0); H = np.zeros((n, n)); f0 = f(x0)
    for a in range(n):
        for b in range(a, n):
            ea, eb = np.eye(n)[a] * h, np.eye(n)[b] * h
            H[a, b] = H[b, a] = (f(x0 + ea + eb) - f(x0 + ea - eb) - f(x0 - ea + eb) + f(x0 - ea - eb)) / (4 * h * h)
    eig = np.linalg.eigvalsh(-H)
    if eig[0] <= 0:
        raise MFRMError("Gözlenen bilgi matrisi pozitif tanımlı değil; standart hatalar raporlanmadı.")
    V = np.linalg.inv(-H); se = {}; i = 0
    for b, lvs in blocks:
        m = len(lvs) - 1; C = np.vstack([np.eye(m), -np.ones(m)]); Vb = C @ V[i:i + m, i:i + m] @ C.T; i += m
        se[b] = dict(zip(lvs if b != "tau" else range(1, len(lvs) + 1), np.sqrt(np.diag(Vb))))
    return se, float(eig[-1] / eig[0])


def residual_fit(d, params, facets, theta):
    """Wright & Masters residual mean squares per element with person measures `theta` (Series by person):
    outfit = mean(z^2), infit = sum((x - E)^2) / sum(W). Definition depends on the person estimator supplied."""
    eta = _eta(d, params, facets); k = np.arange(len(params["tau"]) + 1)
    th = d.person.map(theta).to_numpy(float)
    lp = np.exp(np.stack([_log_probs([t], np.array([e]), params["tau"])[0, 0] for t, e in zip(th, eta)]))
    E = lp @ k; W = lp @ k ** 2 - E ** 2; res = d.score.to_numpy() - E
    frame = pd.DataFrame({"res2": res ** 2, "W": W, "z2": res ** 2 / W, "obs": d.score.to_numpy(), "exp": E})
    out = []
    for f in ("item", *facets):
        g = frame.groupby(d[f].to_numpy())
        out.append(pd.DataFrame({"facet": f, "level": list(g.groups), "infit": (g.res2.sum() / g.W.sum()).to_numpy(),
                                 "outfit": g.z2.mean().to_numpy(), "observed_average": g.obs.mean().to_numpy(),
                                 "expected_average": g.exp.mean().to_numpy(), "n": g.size().to_numpy()}))
    return pd.concat(out, ignore_index=True)


def separation(estimates, se):
    """Rasch separation statistics: observed variance (ddof = 1), error variance = mean SE^2."""
    est, se = np.asarray(estimates, float), np.asarray(se, float)
    obs = est.var(ddof=1); err = float(np.mean(se ** 2)); true = max(obs - err, 0.)
    G = math.sqrt(true / err) if err > 0 else float("nan")
    return {"SD": math.sqrt(obs), "RMSE": math.sqrt(err), "TrueSD": math.sqrt(true), "Separation": G,
            "Strata": (4 * G + 1) / 3, "Reliability": true / obs if obs > 0 else float("nan")}


# ---------------------------------------------------------------- engine run and assembled results
EVIDENCE = ROOT / "validation/mfrm/evidence.json"
OUTPUT_LABELS = {
    "facet_measures": "Madde, puanlayıcı ve ek yüzey ölçüleri (logit)", "facet_se": "Yüzey ölçülerinin standart hatası",
    "facet_ci": "Yüzey ölçülerinin %95 Wald aralığı", "thresholds": "Derecelendirme ölçeği eşikleri",
    "threshold_se": "Eşik standart hataları", "person_wle": "Birey WLE ölçüsü ve SE", "person_eap": "Birey EAP ve sonsal SS",
    "fit_mnsq": "Infit/Outfit MnSq (EAP tabanlı artık)", "fit_zstd": "Standartlaştırılmış uyum (ZSTD)",
    "observed_expected": "Gözlenen ve beklenen ortalama puan", "separation": "Ayırma, tabaka ve güvenirlik",
    "model_fit": "Log-olabilirlik, sapma, AIC, BIC", "latent": "Gizil dağılım ortalaması ve varyansı",
    "interaction_lr": "Madde × puanlayıcı etkileşimi LR testi", "interaction_estimates": "Etkileşim (yanlılık) kestirimleri",
    "interaction_se": "Etkileşim kestirimi SE", "fair_average": "Adil ortalama (diğer yüzeyler ortalamada)",
    "category_diagnostics": "Kategori sıklıkları ve ortalama ölçüler", "rater_ranking": "Puanlayıcı katılık sıralaması"}


def evidence():
    try:
        return json.loads(EVIDENCE.read_text(encoding="utf-8"))["outputs"]
    except (OSError, KeyError, ValueError):
        return {}


def status(output):
    return evidence().get(output, {}).get("status", "NOT CERTIFIED")


def shown(output):
    return status(output) in ("CERTIFIED", "PARTIAL")


def run_tam(d, facets, interaction=False, timeout=600):
    ok, why = engine_available()
    if not ok:
        raise MFRMError(why)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); d.to_csv(tmp / "long.csv", index=False)
        try:
            run = subprocess.run([shutil.which("Rscript"), str(SCRIPT), str(tmp / "long.csv"), str(tmp / "out"), ",".join(facets),
                                  str(NODES), "1" if interaction else "0"], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise MFRMError("R TAM kestirimi zaman sınırını aştı; sonuç raporlanmadı.")
        if run.returncode != 0:
            err = [l for l in run.stderr.splitlines() if l.startswith("Error")] or run.stderr.strip().splitlines()[-1:] or ["R hatası"]
            raise MFRMError("R TAM kestirimi başarısız oldu; sonuç raporlanmadı (" + err[0][:240] + ").")
        o = tmp / "out"
        res = {"xsi": pd.read_csv(o / "parameters.csv"), "persons": pd.read_csv(o / "persons.csv", dtype={"person": str}),
               "summary": pd.read_csv(o / "summary.csv").set_index("key").value.to_dict(),
               "versions": (o / "versions.txt").read_text().splitlines()}
        if (o / "interaction.csv").exists():
            res["interaction"] = pd.read_csv(o / "interaction.csv")
        return res


SUPPORTED_EXTRA_FACETS = 0   # four-facet TAM run failed under the preregistered controls (validation/mfrm/RESULT.md)


def analyze(df, person, item, rater, score, extra=(), interaction=False, show_ids=False, engine=None, allow_extra=False):
    if len(extra) > SUPPORTED_EXTRA_FACETS and not allow_extra:
        raise MFRMError("Dört ve daha fazla yüzeyli model bu sürümde desteklenmiyor: önceden kaydedilen TAM ayarlarıyla dört "
                        "yüzeyli doğrulama fikstürü sayısal olarak başarısız oldu. Yalnız Birey × Madde × Puanlayıcı modeli kullanılabilir.")
    d, report = prepare(df, person, item, rater, score, extra)
    facets = report["facets"]
    raw = (engine or run_tam)(d, facets, interaction)
    s = raw["summary"]
    if not s.get("converged"):
        raise MFRMError(f"TAM {int(s.get('iterations', 0))} iterasyonda yakınsamadı; hiçbir kestirim raporlanmadı.")
    P = unpack(raw["xsi"], facets); mean, var = float(s["mean"]), float(s["variance"])
    ll = marginal_loglik(d, P, facets, mean, var)
    if abs(ll - float(s["loglik"])) > 1e-5:
        raise MFRMError("Parametre eşlemesi TAM log-olabilirliğini yeniden üretmedi; sonuçlar raporlanmadı.")
    se, cond = observed_information_se(d, P, facets, mean, var)
    pers = raw["persons"].set_index("person"); e = eap(d, P, facets, mean, var)
    w = wle(d, P, facets)
    identity = {"loglik": abs(ll - float(s["loglik"])), "eap": float((e.eap - pers.eap.loc[e.index]).abs().max()),
                "wle": float((w.wle - pers.wle.loc[w.index]).abs().max()), "information_condition": cond}
    fit = residual_fit(d, P, facets, e.eap)
    rows = []
    for f in ("item", *facets):
        for lv, m in P[f].items():
            r = fit[(fit.facet == f) & (fit.level == lv)].iloc[0]
            rows.append({"Yüzey": "Madde" if f == "item" else ("Puanlayıcı" if f == "rater" else f), "facet": f, "Düzey": lv,
                         "Ölçü (logit)": m, "SE": se[f][lv], "%95 alt": m - 1.96 * se[f][lv], "%95 üst": m + 1.96 * se[f][lv],
                         "Infit MnSq": r.infit, "Outfit MnSq": r.outfit, "Gözlenen ort.": r.observed_average,
                         "Beklenen ort.": r.expected_average, "N": int(r.n),
                         "Adil ortalama": _fair(P, mean, own=(f, m))})
    elements = pd.DataFrame(rows)
    raters = elements.facet == "rater"
    elements.loc[raters, "Katılık sırası"] = elements.loc[raters, "Ölçü (logit)"].rank(ascending=False).astype(int)
    K = len(P["tau"]); counts = report["category_counts"]
    eta = _eta(d, P, facets); meas = d.person.map(e.eap).to_numpy() + eta
    avg = pd.Series(meas).groupby(d.score.to_numpy()).mean()
    thr = pd.DataFrame({"Kategori": range(K + 1), "Sıklık": counts, "%": np.array(counts) / sum(counts) * 100,
                        "Ortalama ölçü": [avg.get(k, np.nan) for k in range(K + 1)],
                        "Eşik (τ)": [np.nan] + list(P["tau"]), "Eşik SE": [np.nan] + [se["tau"][k] for k in range(1, K + 1)]})
    disordered = bool(np.any(np.diff(P["tau"]) <= 0))
    persons = pd.DataFrame({"WLE": w.wle, "WLE SE": w.wle_se, "EAP": e.eap, "Sonsal SS": e.eap_sd,
                            "Ham puan": pers.score.loc[w.index].to_numpy(), "Azami": pers["max"].loc[w.index].to_numpy()}, index=w.index)
    persons["Uç puan"] = (persons["Ham puan"] == 0) | (persons["Ham puan"] == persons["Azami"])
    persons["Adil ortalama"] = [_fair(P, t) for t in persons.EAP]
    persons.index = persons.index if show_ids else [f"Kişi_{i + 1:04d}" for i in range(len(persons))]
    persons.index.name = "Birey"
    sep = {f: separation(elements.loc[elements.facet == f, "Ölçü (logit)"], elements.loc[elements.facet == f, "SE"]) for f in ("item", *facets)}
    sep["person"] = separation(e.eap, e.eap_sd)
    out = {"report": report, "facets": facets, "elements": elements, "thresholds": thr, "disordered_thresholds": disordered,
           "persons": persons, "separation": pd.DataFrame(sep).T, "identity": identity,
           "model": {"logLik": float(s["loglik"]), "Sapma": float(s["deviance"]), "AIC": float(s["AIC"]), "BIC": float(s["BIC"]),
                     "Parametre": int(s["npars"]), "Birey (BIC n)": int(s["n_persons"]), "Gizil ortalama": mean, "Gizil varyans": var,
                     "İterasyon": int(s["iterations"])},
           "params": P, "versions": raw["versions"], "show_ids": show_ids}
    if interaction:
        if not s.get("interaction_converged"):
            out["interaction"] = {"reason": "Etkileşim modeli yakınsamadı; LR testi raporlanmadı."}
        else:
            from scipy.stats import chi2
            lr = float(s["deviance"] - s["interaction_deviance"]); df_ = int(s["interaction_npars"] - s["npars"])
            out["interaction"] = {"LR": lr, "df": df_, "p": float(chi2.sf(lr, df_)) if lr >= -1e-6 else float("nan"),
                                  "table": raw["interaction"][["parameter", "xsi"]].rename(columns={"parameter": "Madde:Puanlayıcı", "xsi": "Etkileşim (logit)"})}
    return out


def _fair(P, theta, own=None):
    """Expected rating with all facets at their (zero) mean except `own`; persons: theta with every facet at zero."""
    eta = -(own[1] if own else 0.)
    lp = _log_probs([theta if own is None else theta], np.array([eta]), P["tau"])[0, 0]
    return float(np.exp(lp) @ np.arange(len(P["tau"]) + 1))


WITHHELD = {"NOT CERTIFIED": "Bağımsız sayısal kanıt yok; çıktı raporlanmadı."}


def status_table():
    ev = evidence(); rows = []
    for key, label in OUTPUT_LABELS.items():
        e = ev.get(key, {}); st = e.get("status", "NOT CERTIFIED")
        rows.append({"Çıktı": label, "Durum": st, "Kanıt": e.get("basis", "kanıt kaydı yok"),
                     "En büyük fark": e.get("max_diff")})
    rows.append({"Çıktı": "4+ yüzeyli model", "Durum": "NOT CERTIFIED", "Kanıt": ev.get("four_facet_model", {}).get("basis", ""),
                 "En büyük fark": None})
    return pd.DataFrame(rows)


def present(result):
    """Tables exactly as displayed and exported: withheld outputs are removed and explained; never zero-filled."""
    el = result["elements"].copy(); notes = []
    gates = {"facet_se": ["SE"], "facet_ci": ["%95 alt", "%95 üst"], "fit_mnsq": ["Infit MnSq", "Outfit MnSq"],
             "observed_expected": ["Gözlenen ort.", "Beklenen ort."], "fair_average": ["Adil ortalama"],
             "rater_ranking": ["Katılık sırası"]}
    if not shown("facet_measures"):
        return {"notes": ["Yüzey ölçüleri doğrulanmadığı için hiçbir MFRM tablosu gösterilmiyor."], "tables": {}}
    for out, cols in gates.items():
        if not shown(out):
            el = el.drop(columns=cols); notes.append(f"{OUTPUT_LABELS[out]}: {WITHHELD['NOT CERTIFIED']}")
    thr = result["thresholds"].copy()
    if not shown("threshold_se"):
        thr = thr.drop(columns=["Eşik SE"])
    if result["disordered_thresholds"]:
        notes.append("Eşikler sıralı değil (τ artmıyor): kategori yapısı gözden geçirilmelidir.")
    if result["report"]["sparse_categories"]:
        notes.append(f"Seyrek kategori(ler) (< 10 gözlem): {result['report']['sparse_categories']}; eşikler kararsız olabilir.")
    if result["report"]["extreme_persons"]:
        notes.append(f"{result['report']['extreme_persons']} bireyin puanı uç (en düşük/en yüksek); WLE bu bireyler için sonlu olsa da yorum sınırlıdır.")
    persons = result["persons"].copy()
    keep = [c for c, o in (("WLE", "person_wle"), ("WLE SE", "person_wle"), ("EAP", "person_eap"), ("Sonsal SS", "person_eap"),
                           ("Adil ortalama", "fair_average")) if not shown(o)]
    persons = persons.drop(columns=keep)
    tables = {"Yüzey ölçüleri": el.drop(columns=["facet"]), "Eşikler ve kategoriler": thr, "Bireyler": persons.reset_index(),
              "Ayırma ve güvenirlik": result["separation"].reset_index(names="Yüzey") if shown("separation") else None,
              "Model uyumu": pd.DataFrame([result["model"]]) if shown("model_fit") else None,
              "Kanıt durumu": status_table()}
    inter = result.get("interaction")
    if inter is not None:
        if "LR" in inter and shown("interaction_lr"):
            tables["Etkileşim LR"] = pd.DataFrame([{k: inter[k] for k in ("LR", "df", "p")}])
        else:
            notes.append("Madde × puanlayıcı etkileşimi: " + inter.get("reason", WITHHELD["NOT CERTIFIED"]))
        if not shown("interaction_estimates"):
            notes.append("Etkileşim (yanlılık) kestirimleri: bağımsız kanıt yok; raporlanmadı.")
    if not result["show_ids"]:
        notes.append("Birey kimlikleri takma adla gösteriliyor (Kişi_0001 …); gerçek kimlikler yalnız açık onayla gösterilir.")
    return {"notes": notes, "tables": {k: v for k, v in tables.items() if v is not None}}
