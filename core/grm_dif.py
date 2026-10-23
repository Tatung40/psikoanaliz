"""Concurrent two-group GRM MML; isolated from completed binary DIF.

P(Y>=c|theta)=logistic(a*(theta-b_c)); positive slopes, ordered thresholds.
Reference theta~N(0,1); focal mean/SD free. Predeclared anchors share both
slopes and thresholds. Same ordered categories in every item/group; MAR
likelihood skips missing responses. No score matching or nominal recoding.

Scope remains experimental until independent calibration and simulation gates
are recorded. There is no automatic or data-selected anchor purification.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp
from scipy.stats import chi2, norm
from core.rasch import _grid
from core.rasch_dif import benjamini_hochberg
from core.irt_dif import _nested_lr

STATUS = "EXPERIMENTAL — GRM formel model; doğrulama kapsamı raporda belirtilmelidir"


def category_terms(theta, a, b):
    """Stable log category probabilities and exact physical-parameter scores.

    Axes: nodes x items x categories; db adds a final threshold axis.
    Middle probabilities use a log difference identity, avoiding cancellation.
    """
    theta, a, b = np.asarray(theta, float), np.asarray(a, float), np.asarray(b, float)
    if (theta.ndim != 1 or b.ndim != 2 or b.shape[1] < 1 or a.shape != (len(b),)
            or not all(np.isfinite(v).all() for v in (theta, a, b))
            or np.any(a <= 0) or np.any(np.diff(b, axis=1) <= 0)):
        raise ValueError("GRM pozitif eğim ve kesin sıralı eşikler gerektirir.")
    z = a[None, :, None] * (theta[:, None, None] - b[None, :, :])
    s = expit(z)
    q, k, t = z.shape
    lp = np.empty((q, k, t + 1)); da = np.zeros_like(lp); dt = np.zeros_like(lp)
    db = np.zeros((q, k, t + 1, t))
    lp[:, :, 0] = log_expit(-z[:, :, 0])
    lp[:, :, -1] = log_expit(z[:, :, -1])
    da[:, :, 0] = -s[:, :, 0] * (theta[:, None] - b[None, :, 0])
    da[:, :, -1] = (1 - s[:, :, -1]) * (theta[:, None] - b[None, :, -1])
    dt[:, :, 0] = -a * s[:, :, 0]
    dt[:, :, -1] = a * (1 - s[:, :, -1])
    db[:, :, 0, 0] = a * s[:, :, 0]
    db[:, :, -1, -1] = -a * (1 - s[:, :, -1])
    for c in range(1, t):
        gap = b[:, c] - b[:, c - 1]
        h = 1 / np.expm1(a * gap)
        lp[:, :, c] = log_expit(z[:, :, c - 1]) + log_expit(-z[:, :, c]) + np.log(-np.expm1(-a * gap))
        da[:, :, c] = ((1 - s[:, :, c - 1]) * (theta[:, None] - b[None, :, c - 1])
                      - s[:, :, c] * (theta[:, None] - b[None, :, c]) + gap * h)
        db[:, :, c, c - 1] = -a * (1 - s[:, :, c - 1] + h)
        db[:, :, c, c] = a * (s[:, :, c] + h)
        dt[:, :, c] = a * (1 - s[:, :, c - 1] - s[:, :, c])
    return lp, da, db, dt


class Design:
    """Threshold vectors use b0, log positive gaps: ordering is never sorted post-fit."""
    def __init__(self, roles, categories):
        self.roles, self.categories = tuple(roles), categories
        k, t = len(roles), categories - 1
        self.a = np.empty((2, k), int); self.b = np.empty((2, k, t), int)
        self.labels, self.bounds = [], []
        for i, role in enumerate(roles):
            if role not in ("anchor", "a", "b", "ab"):
                raise ValueError("Geçersiz GRM kısıtı.")
            for g in ((0, 1) if role in ("a", "ab") else (-1,)):
                index = len(self.labels)
                self.labels.append(("a", g, i, 0)); self.bounds.append((.1, 4.))
                if g == -1:
                    self.a[:, i] = index
                else:
                    self.a[g, i] = index
            for g in ((-1,) if role in ("anchor", "a") else (0, 1)):
                indices = []
                for c in range(t):
                    indices.append(len(self.labels)); self.labels.append(("b", g, i, c))
                    self.bounds.append((-6., 4.) if c == 0 else (np.log(.05), np.log(6.)))
                if g == -1:
                    self.b[:, i] = indices
                else:
                    self.b[g, i] = indices
        self.latent = len(self.labels)
        self.bounds += [(-3., 3.), (np.log(.3), np.log(3.))]
        self.size = self.latent + 2

    def unpack(self, p):
        a = p[self.a]
        u = p[self.b]
        b = np.concatenate([u[:, :, :1], u[:, :, :1] + np.cumsum(np.exp(u[:, :, 1:]), axis=2)], axis=2)
        return a, b, float(p[-2]), float(np.exp(p[-1]))

    def encode(self, a, b, mu=0., sd=1.):
        p = np.zeros(self.size)
        for j, (kind, g, i, c) in enumerate(self.labels):
            g = max(0, g)
            p[j] = a[g, i] if kind == "a" else b[g, i, 0] if c == 0 else np.log(b[g, i, c] - b[g, i, c - 1])
        p[-2:] = mu, np.log(sd)
        return p


def validate_inputs(data, group, reference, focal, anchors, categories=None, min_cell=5):
    data = pd.DataFrame(data)
    if data.shape[1] < 3 or not data.columns.is_unique:
        raise ValueError("En az üç benzersiz adlı madde gereklidir.")
    if reference == focal or len(group) != len(data):
        raise ValueError("İki farklı grup ve eşleşen satır sayısı gereklidir.")
    if isinstance(group, pd.Series) and group.name in data.columns:
        raise ValueError("Grup sütunu madde olarak kullanılamaz.")
    if isinstance(anchors, (str, bytes)) or anchors is None:
        raise ValueError("Önceden belirlenmiş sabit çapaları liste olarak belirtin.")
    anchors = list(anchors)
    if len(set(anchors)) != len(anchors) or any(a not in data for a in anchors):
        raise ValueError("Çapa adları benzersiz olmalı ve maddelerde bulunmalıdır.")
    if not 2 <= len(anchors) < data.shape[1]:
        raise ValueError("En az iki sabit çapa ve bir test maddesi gereklidir.")
    if isinstance(min_cell, bool) or not isinstance(min_cell, (int, np.integer)) or min_cell < 2:
        raise ValueError("Kategori hücre alt sınırı en az 2 olmalıdır.")
    groups = pd.Series(group).reset_index(drop=True)
    selected = groups.isin([reference, focal]).to_numpy()
    try:
        x = data.loc[selected].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("GRM yalnızca gerekçeli sıralı 0,1,... puanları kabul eder; nominal etiketler dönüştürülmez.") from exc
    if np.isinf(x).any():
        raise ValueError("Sonsuz yanıt geçersizdir; eksik yanıt için NaN kullanın.")
    observed = np.isfinite(x)
    values = x[observed]
    if not values.size or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Yanıtlar 0'dan başlayan sıralı tamsayılar olmalıdır.")
    if categories is None:
        categories = int(values.max()) + 1
    if isinstance(categories, bool) or not isinstance(categories, (int, np.integer)) or not 3 <= categories <= 10:
        raise ValueError("Bu GRM uygulaması ortak 3–10 kategori destekler.")
    if np.any(values >= categories):
        raise ValueError("Yanıt tanımlı kategori aralığının dışında.")
    keep = observed.any(axis=1)
    retained = np.flatnonzero(selected)[keep]
    x, observed = x[keep], observed[keep]
    g = (groups.iloc[retained] == focal).to_numpy(int)
    sizes = [int((g == i).sum()) for i in (0, 1)]
    if min(sizes) < 100:
        raise ValueError("Bu deneysel GRM kalibrasyonu her grupta en az 100 kişi gerektirir.")
    counts = np.array([[(x[g == gg, j] == c).sum() for j in range(x.shape[1])
                        for c in range(categories)] for gg in (0, 1)]).reshape(2, x.shape[1], categories)
    if counts.min() < min_cell:
        raise ValueError("Her madde × grup × kategori hücresinde yeterli yanıt yok; eşik çıkarımı yapılmadı.")
    x = np.where(observed, x, -1).astype(int)
    return dict(x=x, groups=g, names=list(data.columns), anchors=[data.columns.get_loc(a) for a in anchors],
                categories=categories, sizes=sizes, retained_rows=retained, excluded_rows=len(data) - len(x),
                missingness=np.array([(~observed[g == gg]).sum(axis=0) for gg in (0, 1)]), min_cell=min_cell)


def pattern_parts(x, groups, categories):
    result = []
    for g in (0, 1):
        patterns, counts = np.unique(x[groups == g], axis=0, return_counts=True)
        indicators = np.eye(categories)[np.maximum(patterns, 0)] * (patterns >= 0)[:, :, None]
        result.append((indicators.reshape(len(patterns), -1), counts))
    return result


def objective(p, design, parts, grid):
    a, b, mu, sd = design.unpack(p)
    z, lw = grid
    gradient = np.zeros(design.size); ll = 0.; n = sum(counts.sum() for _, counts in parts)
    for g, (indicators, counts) in enumerate(parts):
        theta = z if g == 0 else mu + sd * z
        lp, da, db, dt = category_terms(theta, a[g], b[g])
        joint = indicators @ lp.reshape(len(z), -1).T + lw
        marginal = logsumexp(joint, axis=1)
        ll += counts @ marginal
        posterior = np.exp(joint - marginal[:, None]) * counts[:, None]
        expected = (indicators.T @ posterior).T.reshape(lp.shape)
        gradient[design.a[g]] += np.einsum("qic,qic->i", expected, da)
        gb = np.einsum("qic,qict->it", expected, db)
        gu = np.flip(np.cumsum(np.flip(gb, axis=1), axis=1), axis=1)
        gu[:, 1:] *= np.exp(p[design.b[g, :, 1:]])
        gradient[design.b[g].ravel()] += gu.ravel()
        if g:
            gradient[-2] += np.sum(expected * dt)
            gradient[-1] += np.sum(expected * dt * (sd * z)[:, None, None])
    return -ll / n, -gradient / n


class Fitter:
    def __init__(self, inputs, points=41):
        self.inputs = inputs
        self.grid = _grid(points)
        self.audit_grid = _grid(max(points + 20, 81))
        self.parts = pattern_parts(inputs["x"], inputs["groups"], inputs["categories"])
        self.cache = {}
        self.optimizations = 0

    def fit(self, roles, start_fit=None, information=True, max_iter=2000):
        key = tuple(roles)
        cached = self.cache.get(key)
        if (cached is not None and cached["converged"] and (not information or cached["cov"] is not None)
                and (start_fit is None or cached["loglik"] >= start_fit["loglik"] - 1e-6)):
            return cached
        if cached is not None and (start_fit is None or cached["loglik"] > start_fit["loglik"]):
            start_fit = cached
        d = Design(roles, self.inputs["categories"])
        if start_fit is not None:
            start = d.encode(*start_fit["design"].unpack(start_fit["params"]))
        else:
            x = self.inputs["x"]
            tails = np.array([[(x[:, i] >= c).sum() / (x[:, i] >= 0).sum()
                               for c in range(1, d.categories)] for i in range(x.shape[1])])
            b = -np.log(np.clip(tails, .02, .98) / (1 - np.clip(tails, .02, .98)))
            start = d.encode(np.ones((2, len(roles))), np.stack([b, b]))
        lo, hi = np.array(d.bounds).T
        fn = lambda p: objective(p, d, self.parts, self.grid)
        self.optimizations += 1
        result = minimize(fn, np.clip(start, lo, hi), jac=True, method="L-BFGS-B", bounds=d.bounds,
                          options=dict(maxiter=max_iter, ftol=1e-14, gtol=1e-10))
        value, score = fn(result.x)
        at_bound = np.minimum(result.x - lo, hi - result.x).min() < 1e-4
        converged = bool(np.isfinite(value) and np.max(np.abs(score)) < 2e-6 and not at_bound)
        audit_value, audit_score = objective(result.x, d, self.parts, self.audit_grid)
        quadrature_error = abs(audit_value - value) * len(self.inputs["x"])
        quadrature_ok = bool(quadrature_error <= 1e-3 and np.max(np.abs(audit_score)) <= 1e-5)
        out = dict(design=d, params=result.x, loglik=-value * len(self.inputs["x"]),
                   converged=converged, at_bound=bool(at_bound), score=float(np.max(np.abs(score))),
                   iterations=result.nit, cov=None, reason="", information_min_eigenvalue=np.nan,
                   information_condition=np.nan)
        out.update(quadrature_error=quadrature_error, quadrature_ok=quadrature_ok,
                   audit_score=float(np.max(np.abs(audit_score))))
        if not converged:
            out["reason"] = "GRM yakınsamadı veya bir kestirim sınırda; çıkarım kullanılamıyor."
        elif information:
            hessian = np.empty((d.size, d.size))
            for j in range(d.size):
                h = 1e-5 * max(1., abs(result.x[j]))
                step = np.zeros(d.size); step[j] = h
                hessian[:, j] = (fn(result.x + step)[1] - fn(result.x - step)[1]) * len(self.inputs["x"]) / (2 * h)
            hessian = (hessian + hessian.T) / 2
            eigen = np.linalg.eigvalsh(hessian)
            out["information_min_eigenvalue"] = float(eigen[0])
            out["information_condition"] = float(eigen[-1] / eigen[0]) if eigen[0] > 0 else np.inf
            if np.isfinite(eigen).all() and eigen[0] > 1e-7 * max(eigen[-1], 1.):
                out["cov"] = np.linalg.inv(hessian)
            else:
                out["reason"] = "Gözlenen bilgi matrisi tekil/pozitif tanımlı değil; çıkarım kullanılamıyor."
        if not quadrature_ok:
            out["reason"] += " Daha yoğun quadrature ile olabilirlik/skor kararsız; çıkarım kullanılamıyor."
        if cached is not None:
            cached.update(out)
            return cached
        self.cache[key] = out
        return out


def _item_jacobian(fit, item, theta):
    """Jacobian of focal-minus-reference category probabilities at matched theta."""
    d, p = fit["design"], fit["params"]
    a, b, _, _ = d.unpack(p)
    jac = np.zeros((len(theta), d.categories, d.size))
    probabilities = []
    theta_derivative = np.zeros((len(theta), d.categories))
    for g, sign in ((0, -1), (1, 1)):
        lp, da, db, dt = category_terms(theta, a[g, item:item + 1], b[g, item:item + 1])
        prob = np.exp(lp[:, 0]); probabilities.append(prob)
        jac[:, :, d.a[g, item]] += sign * prob * da[:, 0]
        for c, index in enumerate(d.b[g, item]):
            multiplier = 1. if c == 0 else np.exp(p[index])
            jac[:, :, index] += sign * prob * db[:, 0, :, c:].sum(axis=-1) * multiplier
        theta_derivative += sign * prob * dt[:, 0]
    return probabilities[0], probabilities[1], jac, theta_derivative


def parameter_contrasts(fit, item, usable=True):
    d, p = fit["design"], fit["params"]
    a, b, _, _ = d.unpack(p)
    rows = []
    for component in range(d.categories):
        jac = np.zeros(d.size)
        for g, sign in ((0, -1), (1, 1)):
            if component == 0:
                jac[d.a[g, item]] += sign
            else:
                for c in range(component):
                    index = d.b[g, item, c]
                    jac[index] += sign * (1 if c == 0 else np.exp(p[index]))
        estimate = a[1, item] - a[0, item] if component == 0 else b[1, item, component - 1] - b[0, item, component - 1]
        se = np.sqrt(max(0., jac @ fit["cov"] @ jac)) if usable and fit["cov"] is not None else np.nan
        rows.append({"Parametre": "Eğim a" if component == 0 else f"Eşik b{component}",
                     "Odak − referans": estimate, "SH": se,
                     "%95 GA alt": estimate - norm.ppf(.975) * se, "%95 GA üst": estimate + norm.ppf(.975) * se})
    return pd.DataFrame(rows)


def conditional_curves(fit, item, theta=None, usable=True):
    theta = np.linspace(-4, 4, 161) if theta is None else np.asarray(theta, float)
    if theta.ndim != 1 or len(theta) < 2 or not np.isfinite(theta).all():
        raise ValueError("Sonlu, tek boyutlu theta ızgarası gereklidir.")
    pr, pf, jac, _ = _item_jacobian(fit, item, theta)
    difference = pf - pr
    cov = fit["cov"] if usable else None
    se = np.sqrt(np.maximum(np.einsum("qci,ij,qcj->qc", jac, cov, jac), 0)) if cov is not None else np.full_like(difference, np.nan)
    category = pd.DataFrame({"Theta": np.repeat(theta, pr.shape[1]), "Kategori": np.tile(np.arange(pr.shape[1]), len(theta)),
        "P (referans)": pr.ravel(), "P (odak)": pf.ravel(), "Olasılık farkı": difference.ravel(), "SH": se.ravel(),
        "%95 GA alt": np.maximum(-1, difference - norm.ppf(.975) * se).ravel(),
        "%95 GA üst": np.minimum(1, difference + norm.ppf(.975) * se).ravel()})
    scores = np.arange(pr.shape[1])
    delta = difference @ scores
    j_score = np.einsum("qci,c->qi", jac, scores)
    se_score = np.sqrt(np.maximum(np.einsum("qi,ij,qj->q", j_score, cov, j_score), 0)) if cov is not None else np.full(len(theta), np.nan)
    score = pd.DataFrame({"Theta": theta, "Beklenen puan farkı": delta, "SH": se_score,
        "%95 GA alt": delta - norm.ppf(.975) * se_score, "%95 GA üst": delta + norm.ppf(.975) * se_score})
    return category, score


def signed_effect(fit, item, points=61, usable=True):
    _, _, mu, sd = fit["design"].unpack(fit["params"])
    z, lw = _grid(points); w = np.exp(lw)
    pr, pf, jac, dtheta = _item_jacobian(fit, item, mu + sd * z)
    scores = np.arange(pr.shape[1])
    estimate = w @ ((pf - pr) @ scores)
    gradient = np.einsum("q,qci,c->i", w, jac, scores)
    gradient[-2] += w @ (dtheta @ scores)
    gradient[-1] += (w * sd * z) @ (dtheta @ scores)
    se = np.sqrt(max(0., gradient @ fit["cov"] @ gradient)) if usable and fit["cov"] is not None else np.nan
    return {"İşaretli etki (puan)": estimate, "SH (etki)": se,
            "Etki %95 GA alt": estimate - norm.ppf(.975) * se, "Etki %95 GA üst": estimate + norm.ppf(.975) * se}


def test_dif(data, group, reference, focal, anchors, *, categories=None, alpha=.05, points=41, min_cell=5):
    """Fixed-anchor global threshold and conditional slope LR hypotheses.

    H0: a,b equal; Hb: b free/a equal; Ha: a free/b equal; H2: a,b free.
    Global H2/H0: C df; thresholds H2/Ha: C-1 df; slope H2/Hb: 1 df.
    Each component hypothesis conditions on the other component being free.
    Threshold changes need not be equal across categories; they are not one b shift.
    """
    if not 0 < alpha < .5 or isinstance(points, bool) or not isinstance(points, int) or points < 21:
        raise ValueError("Geçerli alfa ve en az 21 quadrature noktası gereklidir.")
    inputs = validate_inputs(data, group, reference, focal, anchors, categories, min_cell)
    fitter = Fitter(inputs, points)
    roles = ["anchor" if i in inputs["anchors"] else "ab" for i in range(len(inputs["names"]))]
    fits = {}
    for i in range(len(roles)):
        if roles[i] == "anchor":
            continue
        null_roles = roles.copy(); null_roles[i] = "anchor"
        m0 = fitter.fit(null_roles)
        threshold_roles = roles.copy(); threshold_roles[i] = "b"
        mb = fitter.fit(threshold_roles, m0)
        slope_roles = roles.copy(); slope_roles[i] = "a"
        ma = fitter.fit(slope_roles, m0)
        m2 = fitter.fit(roles, max((mb, ma), key=lambda f: f["loglik"]))
        fits[i] = {"H0": m0, "Hb": mb, "Ha": ma, "H2": m2}
    table = []
    for i, name in enumerate(inputs["names"]):
        row = {"Madde": name, "Sabit çapa": i in inputs["anchors"], "p": np.nan, "q (BH)": np.nan,
               "p (eşikler)": np.nan, "p (eğim | eşikler)": np.nan, "DIF işareti": False}
        if i not in fits:
            row.update(Karar="Sabit çapa (test edilmedi)", Not="Çapa değişmezliği varsayılır; bu analizle kanıtlanmaz.")
        else:
            m0, mb, ma, m2 = (fits[i][h] for h in ("H0", "Hb", "Ha", "H2"))
            regular = all(f["converged"] and not f["at_bound"] and f["quadrature_ok"]
                          and f["cov"] is not None for f in (m0, mb, ma, m2))
            notes = [f["reason"] for f in (m0, mb, ma, m2) if f["reason"]]
            for label, alt, null, df in (("genel", m2, m0, inputs["categories"]),
                                         ("eşikler", m2, ma, inputs["categories"] - 1),
                                         ("eğim | eşikler", m2, mb, 1)):
                stat, note = _nested_lr(alt["loglik"], null["loglik"])
                row[f"LR ({label})"], row[f"sd ({label})"] = stat, df
                row["p" if label == "genel" else f"p ({label})"] = chi2.sf(stat, df) if regular and not note else np.nan
                if note:
                    notes.append(f"{label}: {note}")
            for label, f in zip(("H0", "Hb", "Ha", "H2"), (m0, mb, ma, m2)):
                row[f"logL ({label})"] = f["loglik"]
            row["Not"] = " ".join(dict.fromkeys(notes))
        table.append(row)
    table = pd.DataFrame(table)
    tested = ~table["Sabit çapa"]
    for pcol, qcol in (("p", "q (BH)"), ("p (eşikler)", "q (BH, eşikler)"),
                       ("p (eğim | eşikler)", "q (BH, eğim | eşikler)")):
        table.loc[tested, qcol] = benjamini_hochberg(table.loc[tested, pcol])
    table.loc[tested, "DIF işareti"] = table.loc[tested, "q (BH)"] < alpha
    table.loc[tested, "Karar"] = ["Çıkarım kullanılamıyor" if not np.isfinite(q) else
        "İçerik incelemesi gerekli" if q < alpha else "Gözlenen DIF yok" for q in table.loc[tested, "q (BH)"]]
    contrasts = []
    for i, fits_for_item in fits.items():
        usable = bool(np.isfinite(table.loc[i, "p"]))
        frame = parameter_contrasts(fits_for_item["H2"], i, usable)
        frame.insert(0, "Madde", inputs["names"][i]); contrasts.append(frame)
        for key, value in signed_effect(fits_for_item["H2"], i, usable=usable).items():
            table.loc[i, key] = value
    return dict(table=table, parameters=pd.concat(contrasts, ignore_index=True), fits=fits, inputs=inputs,
                anchors=[inputs["names"][i] for i in inputs["anchors"]], alpha=alpha, points=points,
                reference=reference, focal=focal, status=STATUS, n_fits=fitter.optimizations, n_models=len(fitter.cache),
                method="Eşzamanlı GRM MML; sabit çapalar; referans N(0,1), odak ortalama/SS serbest",
                warnings=["Sabit çapalar bağımsız kanıt gerektirir; seçim belirsizliği düzeltilmez.",
                          "Genel/eşik/eğim testlerinde ayrı BH aileleri; aileler arasında ortak FDR garantisi yok.",
                          "Tek boyutluluk, koşullu yerel bağımsızlık ve eksiklik için MAR varsayılır; bu DIF testi bunları doğrulamaz.",
                          "Olasılık/puan farkı ve parametre aralıkları noktasal normal yaklaşımıdır; eşzamanlı kapsam garantisi yok.",
                          "DIF madde yanlılığı kanıtı değildir; genel tip-I hata/güç garantisi verilmez."])
