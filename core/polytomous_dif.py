"""Experimental, fixed-anchor PCM/GPCM/RSM/NRM concurrent MML.

The four designs share only categorical likelihood machinery. Their constraints,
parameters and hypotheses differ. Completed binary and GRM modules are read-only.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import chi2, norm
from core.grm_dif import validate_inputs as _ordinal_inputs, pattern_parts
from core.rasch import _grid
from core.rasch_dif import benjamini_hochberg
from core.irt_dif import _nested_lr

MODELS = ("PCM", "GPCM", "RSM", "NRM")
STATUS = "EXPERIMENTAL — bağımsız doğrulama kapsamı ve sınırlılıklar raporda belirtilmelidir"


def validate_inputs(data, group, reference, focal, anchors, model="PCM", categories=3,
                    min_cell=5, category_labels=None):
    if model not in MODELS:
        raise ValueError("Model PCM/GPCM/RSM/NRM olmalıdır.")
    if isinstance(categories, bool) or not isinstance(categories, (int, np.integer)) or not 3 <= categories <= 10:
        raise ValueError("Ortak kategori sayısı 3–10 tamsayı olmalıdır.")
    frame = pd.DataFrame(data)
    labels = list(range(categories)) if category_labels is None else list(category_labels)
    if len(labels) != categories or len(set(labels)) != categories or any(pd.isna(v) for v in labels):
        raise ValueError("Kategori etiketleri eksiksiz ve benzersiz olmalıdır.")
    if model == "NRM":
        # Explicit label map, never category order, matching score or an answer key.
        selected = pd.Series(group).reset_index(drop=True).isin([reference, focal]).to_numpy()
        values = frame.loc[selected]
        if any(not values[c].dropna().isin(labels).all() for c in frame):
            raise ValueError("NRM yanıtı beyan edilen kategori etiketlerinde yok.")
        frame = frame.apply(lambda col: col.map(dict(zip(labels, range(categories)))))
    elif category_labels is not None:
        raise ValueError("Sıralı modeller gerekçeli 0,1,... puanları gerektirir; nominal etiket dönüştürülmez.")
    try:
        inputs = _ordinal_inputs(frame, group, reference, focal, anchors, categories, min_cell)
    except ValueError as exc:
        raise ValueError(str(exc).replace("GRM", model).replace("eşik çıkarımı", "parametre çıkarımı")) from exc
    inputs.update(model=model, category_labels=labels)
    return inputs


class Design:
    """Identified canonical logits eta_c=A_c*theta+D_c, eta_0=0.

    PCM/GPCM steps are unconstrained in order. RSM has common centered steps
    across EVERY item and group; only locations can exhibit item DIF.
    NRM uses a reference category and a predeclared anchor/category sign convention.
    """
    def __init__(self, model, roles, categories, orientation_item=0):
        if model not in MODELS or any(r not in ("anchor", "a", "b", "ab") for r in roles):
            raise ValueError("Geçersiz model veya kısıt.")
        if model in ("PCM", "RSM") and any(r == "a" for r in roles):
            raise ValueError("Sabit eğimli modelde eğim DIF testi yoktur.")
        self.model, self.roles, self.categories = model, tuple(roles), categories
        self.orientation_item = orientation_item
        self.labels, self.bounds = [], []
        t = categories - 1
        for i, role in enumerate(roles):
            if model in ("GPCM", "NRM"):
                for g in ((0, 1) if role in ("a", "ab") else (-1,)):
                    for c in range(1 if model == "GPCM" else t):
                        self.labels.append(("a", g, i, c))
                        self.bounds.append((.1, 4.) if model == "GPCM" else
                                           (.05, 4.) if i == orientation_item and c == 0 else (-4., 4.))
            for g in ((0, 1) if role in ("b", "ab") else (-1,)):
                for c in range(1 if model == "RSM" else t):
                    self.labels.append(("b", g, i, c)); self.bounds.append((-8., 8.))
        if model == "RSM":
            for c in range(t - 1):
                self.labels.append(("tau", -1, -1, c)); self.bounds.append((-4., 4.))
        self.latent = len(self.labels)
        self.bounds += [(-3., 3.), (np.log(.3), np.log(3.))]
        self.size = self.latent + 2

    def physical(self, p):
        k, t = len(self.roles), self.categories - 1
        a = np.ones((2, k, 1 if self.model != "NRM" else t))
        b = np.zeros((2, k, 1 if self.model == "RSM" else t))
        tau = np.zeros(t)
        for j, (kind, g, i, c) in enumerate(self.labels):
            if kind == "tau": tau[c] = p[j]
            else:
                target = a if kind == "a" else b
                if g == -1: target[:, i, c] = p[j]
                else: target[g, i, c] = p[j]
        if self.model == "RSM": tau[-1] = -tau[:-1].sum()
        return a, b, tau

    def coefficients(self, p, jacobian=False):
        a, b, tau = self.physical(p)
        k, c = len(self.roles), self.categories
        scores = np.arange(c)
        A, D = np.zeros((2, k, c)), np.zeros((2, k, c))
        if self.model == "NRM": A[:, :, 1:], D[:, :, 1:] = a, b
        else:
            A[:] = a * scores
            steps = b + tau if self.model == "RSM" else b
            D[:, :, 1:] = -a * np.cumsum(steps, axis=2)
        if not jacobian: return A, D
        JA, JD = np.zeros((*A.shape, self.size)), np.zeros((*A.shape, self.size))
        for j, (kind, g, i, h) in enumerate(self.labels):
            for gg in ((0, 1) if g == -1 else (g,)):
                if kind == "tau":
                    JD[gg, :, h + 1:, j] -= 1
                    JD[gg, :, -1, j] += 1
                elif self.model == "NRM":
                    (JA if kind == "a" else JD)[gg, i, h + 1, j] = 1
                elif kind == "a":
                    JA[gg, i, :, j] = scores
                    JD[gg, i, 1:, j] = -np.cumsum(b[gg, i])
                elif self.model == "RSM": JD[gg, i, :, j] = -scores
                else: JD[gg, i, h + 1:, j] = -a[gg, i, 0]
        return A, D, JA, JD

    def unpack(self, p):
        return (*self.coefficients(p), float(p[-2]), float(np.exp(p[-1])))

    def encode(self, A, D, mu=0., sd=1.):
        p = np.zeros(self.size)
        a = A[:, :, -1:] / (self.categories - 1)
        steps = -np.diff(D, axis=2) / a if self.model != "NRM" else None
        for j, (kind, g, i, c) in enumerate(self.labels):
            gg = max(0, g)
            if kind == "tau":
                p[j] = steps[0, 0, c] - steps[0, 0].mean()
            elif self.model == "NRM": p[j] = (A if kind == "a" else D)[gg, i, c + 1]
            elif kind == "a": p[j] = a[gg, i, 0]
            elif self.model == "RSM": p[j] = steps[gg, i].mean()
            else: p[j] = steps[gg, i, c]
        p[-2:] = mu, np.log(sd)
        return p


def log_probabilities(theta, A, D):
    eta = np.asarray(theta)[:, None, None] * A[None] + D[None]
    return eta - logsumexp(eta, axis=2, keepdims=True)


def objective(p, design, parts, grid):
    A, D, JA, JD = design.coefficients(p, True)
    mu, sd = p[-2], np.exp(p[-1]); z, lw = grid
    n = sum(counts.sum() for _, counts in parts)
    ll, gradient = 0., np.zeros(design.size)
    for g, (indicators, counts) in enumerate(parts):
        theta = z if g == 0 else mu + sd * z
        lp = log_probabilities(theta, A[g], D[g])
        joint = indicators @ lp.reshape(len(z), -1).T + lw
        marginal = logsumexp(joint, axis=1); ll += counts @ marginal
        posterior = np.exp(joint - marginal[:, None]) * counts[:, None]
        expected = (indicators.T @ posterior).T.reshape(lp.shape)
        residual = expected - expected.sum(axis=2, keepdims=True) * np.exp(lp)
        gradient += np.einsum("qic,icj->j", residual * theta[:, None, None], JA[g])
        gradient += np.einsum("qic,icj->j", residual, JD[g])
        if g:
            gradient[-2] += np.sum(residual * A[g])
            gradient[-1] += np.sum(residual * A[g] * (sd * z)[:, None, None])
    return -ll / n, -gradient / n


class Fitter:
    def __init__(self, inputs, points=61):
        self.inputs, self.grid, self.audit_grid = inputs, _grid(points), _grid(max(points + 20, 81))
        self.parts = pattern_parts(inputs["x"], inputs["groups"], inputs["categories"])
        self.cache, self.optimizations = {}, 0

    def fit(self, roles, start_fit=None, max_iter=2000):
        key = tuple(roles); old = self.cache.get(key)
        if old is not None and old["converged"] and (start_fit is None or old["loglik"] >= start_fit["loglik"] - 1e-6):
            return old
        d = Design(self.inputs["model"], roles, self.inputs["categories"], self.inputs["anchors"][0])
        if old is not None and (start_fit is None or old["loglik"] > start_fit["loglik"]): start_fit = old
        if start_fit is None:
            k, c = len(roles), d.categories
            A = np.broadcast_to(np.arange(c, dtype=float), (2, k, c)).copy()
            D = np.broadcast_to(-np.r_[0., np.cumsum(np.linspace(-.7, .7, c - 1))], (2, k, c)).copy()
            start = d.encode(A, D)
        else: start = d.encode(*start_fit["design"].unpack(start_fit["params"]))
        lo, hi = np.array(d.bounds).T
        fn = lambda p: objective(p, d, self.parts, self.grid)
        self.optimizations += 1
        result = minimize(fn, np.clip(start, lo, hi), jac=True, method="L-BFGS-B", bounds=d.bounds,
                          options=dict(maxiter=max_iter, ftol=1e-14, gtol=1e-10))
        if d.model == "NRM" and start_fit is None and (np.minimum(result.x - lo, hi - result.x).min() < 1e-4
                or np.max(np.abs(fn(result.x)[1])) >= 2e-6):
            # A nominal one-hot SVD supplies a second numerical start only.
            # It is invariant to label-column permutations; no scored matching
            # variable enters the marginal likelihood or the DIF hypotheses.
            x = self.inputs["x"]; c = d.categories
            indicators = np.eye(c)[np.maximum(x, 0)] * (x >= 0)[:, :, None]
            matrix = indicators.reshape(len(x), -1)
            matrix -= matrix.mean(axis=0)
            u, s, _ = np.linalg.svd(matrix, full_matrices=False)
            proxy = u[:, 0] * np.sqrt(len(x))
            A = np.zeros((2, len(roles), c)); D = A.copy()
            for i in range(len(roles)):
                means = np.array([proxy[x[:, i] == h].mean() for h in range(c)])
                probabilities = np.array([(x[:, i] == h).sum() for h in range(c)], float)
                A[:, i] = means - means[0]
                D[:, i] = np.log(probabilities / probabilities[0])
            if A[0, d.orientation_item, 1] < 0: A = -A
            seed = np.clip(d.encode(A, D), lo + .001, hi - .001)
            self.optimizations += 1
            retry = minimize(fn, seed, jac=True, method="L-BFGS-B", bounds=d.bounds,
                options=dict(maxiter=max_iter, ftol=1e-14, gtol=1e-10))
            if retry.fun < result.fun: result = retry
        value, score = fn(result.x)
        at_bound = bool(np.minimum(result.x - lo, hi - result.x).min() < 1e-4)
        converged = bool(np.isfinite(value) and np.max(np.abs(score)) < 2e-6 and not at_bound)
        audit_value, audit_score = objective(result.x, d, self.parts, self.audit_grid)
        error = abs(audit_value - value) * len(self.inputs["x"])
        quadrature_ok = bool(error <= .001 and np.max(np.abs(audit_score)) <= 1e-5)
        out = dict(design=d, params=result.x, loglik=-value * len(self.inputs["x"]), converged=converged,
                   at_bound=at_bound, score=float(np.max(np.abs(score))), iterations=result.nit,
                   cov=None, reason="", quadrature_error=error, quadrature_ok=quadrature_ok,
                   information_min_eigenvalue=np.nan, information_condition=np.nan)
        if not converged: out["reason"] = "Yakınsama/sınır denetimi başarısız; çıkarım kullanılamıyor."
        else:
            H = np.empty((d.size, d.size))
            for j in range(d.size):
                h = 1e-5 * max(1., abs(result.x[j])); step = np.zeros(d.size); step[j] = h
                H[:, j] = (fn(result.x + step)[1] - fn(result.x - step)[1]) * len(self.inputs["x"]) / (2 * h)
            H = (H + H.T) / 2; eigen = np.linalg.eigvalsh(H)
            out["information_min_eigenvalue"] = float(eigen[0])
            out["information_condition"] = float(eigen[-1] / eigen[0]) if eigen[0] > 0 else np.inf
            if np.isfinite(eigen).all() and eigen[0] > 1e-7 * max(eigen[-1], 1.): out["cov"] = np.linalg.inv(H)
            else: out["reason"] = "Gözlenen bilgi matrisi tekil/pozitif tanımlı değil; çıkarım kullanılamıyor."
        if not quadrature_ok: out["reason"] += " Quadrature kararsız; çıkarım kullanılamıyor."
        out.update(points=len(self.grid[0]), audit_points=len(self.audit_grid[0]))
        if converged and not quadrature_ok and len(self.grid[0]) < 161:
            # Refine only failed integral audits. Successful fits are reused.
            refined = Fitter(self.inputs, min(len(self.grid[0]) + 20, 161))
            out = refined.fit(roles, out, max_iter)
            self.optimizations += refined.optimizations
        # Preserve shared full-fit references if a later nested start improves it.
        if old is not None: old.update(out); return old
        self.cache[key] = out
        return out


def item_jacobian(fit, item, theta):
    d, p = fit["design"], fit["params"]
    A, D, JA, JD = d.coefficients(p, True)
    probabilities, jac, dt = [], np.zeros((len(theta), d.categories, d.size)), np.zeros((len(theta), d.categories))
    for g, sign in ((0, -1), (1, 1)):
        pr = np.exp(log_probabilities(theta, A[g, item:item + 1], D[g, item:item + 1]))[:, 0]
        raw = np.asarray(theta)[:, None, None] * JA[g, item] + JD[g, item]
        centered = raw - np.einsum("qc,qcj->qj", pr, raw)[:, None]
        jac += sign * pr[:, :, None] * centered
        dt += sign * pr * (A[g, item] - pr @ A[g, item][:, None])
        probabilities.append(pr)
    return *probabilities, jac, dt


def conditional_curves(fit, item, theta=None, usable=True):
    theta = np.linspace(-4, 4, 161) if theta is None else np.asarray(theta, float)
    pr, pf, jac, _ = item_jacobian(fit, item, theta)
    cov = fit["cov"] if usable else None
    difference = pf - pr
    se = np.sqrt(np.maximum(np.einsum("qci,ij,qcj->qc", jac, cov, jac), 0)) if cov is not None else np.full_like(difference, np.nan)
    category = pd.DataFrame({"Theta": np.repeat(theta, pr.shape[1]), "Kategori": np.tile(np.arange(pr.shape[1]), len(theta)),
        "P (referans)": pr.ravel(), "P (odak)": pf.ravel(), "Olasılık farkı": difference.ravel(), "SH": se.ravel(),
        "%95 GA alt": np.maximum(-1, difference - norm.ppf(.975) * se).ravel(),
        "%95 GA üst": np.minimum(1, difference + norm.ppf(.975) * se).ravel()})
    if fit["design"].model == "NRM": return category, pd.DataFrame()
    scores = np.arange(pr.shape[1]); delta = difference @ scores
    j = np.einsum("qci,c->qi", jac, scores)
    s = np.sqrt(np.maximum(np.einsum("qi,ij,qj->q", j, cov, j), 0)) if cov is not None else np.full(len(theta), np.nan)
    return category, pd.DataFrame({"Theta": theta, "Beklenen puan farkı": delta, "SH": s,
        "%95 GA alt": delta - norm.ppf(.975) * s, "%95 GA üst": delta + norm.ppf(.975) * s})


def weighted_effects(fit, item, usable=True):
    d, p = fit["design"], fit["params"]
    _, _, mu, sd = d.unpack(p); z, lw = _grid(81); w = np.exp(lw)
    pr, pf, jac, dt = item_jacobian(fit, item, mu + sd * z)
    gradient = np.einsum("q,qci->ci", w, jac)
    gradient[:, -2] += w @ dt
    gradient[:, -1] += (w * sd * z) @ dt
    estimates = w @ (pf - pr)
    if d.model != "NRM":
        scores = np.arange(d.categories)
        estimates, gradient = np.array([estimates @ scores]), (scores @ gradient)[None]
    cov = fit["cov"] if usable else None
    se = np.sqrt(np.maximum(np.einsum("ci,ij,cj->c", gradient, cov, gradient), 0)) if cov is not None else np.full(len(estimates), np.nan)
    return pd.DataFrame({"Etki": [f"Kategori {c} olasılık farkı" for c in range(d.categories)] if d.model == "NRM" else ["Beklenen puan farkı"],
        "Odak − referans": estimates, "SH": se, "%95 GA alt": estimates - norm.ppf(.975) * se,
        "%95 GA üst": estimates + norm.ppf(.975) * se})


def parameter_contrasts(fit, item, usable=True):
    d, p = fit["design"], fit["params"]
    rows = []
    for kind in (("a", "b") if d.model in ("GPCM", "NRM") else ("b",)):
        count = d.categories - 1 if d.model == "NRM" or (kind == "b" and d.model != "RSM") else 1
        for c in range(count):
            j = np.zeros(d.size)
            for index, label in enumerate(d.labels):
                if label[0] == kind and label[2:] == (item, c) and label[1] >= 0: j[index] = -1 if label[1] == 0 else 1
            estimate = j @ p
            se = np.sqrt(max(0., j @ fit["cov"] @ j)) if usable and fit["cov"] is not None else np.nan
            label = (f"Kategori {c + 1} {'eğim' if kind == 'a' else 'kesişim'}" if d.model == "NRM" else
                     "Eğim a" if kind == "a" else "Konum b" if d.model == "RSM" else f"Adım b{c + 1}")
            rows.append({"Parametre": label, "Odak − referans": estimate, "SH": se,
                         "%95 GA alt": estimate - norm.ppf(.975) * se, "%95 GA üst": estimate + norm.ppf(.975) * se})
    return pd.DataFrame(rows)


def test_dif(data, group, reference, focal, anchors, *, model="PCM", categories=3, category_labels=None,
             alpha=.05, points=61, min_cell=5):
    if not 0 < alpha < .5 or isinstance(points, bool) or not isinstance(points, int) or points < 21:
        raise ValueError("Geçerli alfa ve en az 21 quadrature noktası gereklidir.")
    inputs = validate_inputs(data, group, reference, focal, anchors, model, categories, min_cell, category_labels)
    fitter = Fitter(inputs, points)
    roles = ["anchor" if i in inputs["anchors"] else "ab" for i in range(len(inputs["names"]))]
    fits = {}
    for i, role in enumerate(roles):
        if role == "anchor": continue
        null = roles.copy(); null[i] = "anchor"
        m0 = fitter.fit(null)
        models = {"H0": m0}
        if model in ("GPCM", "NRM"):
            br = roles.copy(); br[i] = "b"; ar = roles.copy(); ar[i] = "a"
            models.update(Hb=fitter.fit(br, m0), Ha=fitter.fit(ar, m0))
        models["H2"] = fitter.fit(roles, max(models.values(), key=lambda f: f["loglik"]))
        fits[i] = models
    table, contrasts, effects = [], [], []
    for i, name in enumerate(inputs["names"]):
        row = {"Madde": name, "Sabit çapa": i in inputs["anchors"], "p": np.nan, "q (BH)": np.nan, "DIF işareti": False}
        if i not in fits:
            row.update(Karar="Sabit çapa (test edilmedi)", Not="Çapa değişmezliği varsayılır; kanıtlanmaz.")
        else:
            models = fits[i]; m2, m0 = models["H2"], models["H0"]
            regular = all(f["converged"] and not f["at_bound"] and f["quadrature_ok"] and f["cov"] is not None for f in models.values())
            comparisons = [("genel", m2, m0)]
            if model in ("GPCM", "NRM"):
                comparisons += [("adım/kesişim | eğim", m2, models["Ha"]), ("eğim | adım/kesişim", m2, models["Hb"])]
            notes = [f["reason"] for f in models.values() if f["reason"]]
            for label, alt, null in comparisons:
                stat, note = _nested_lr(alt["loglik"], null["loglik"])
                df = alt["design"].size - null["design"].size
                row[f"LR ({label})"], row[f"sd ({label})"] = stat, df
                row["p" if label == "genel" else f"p ({label})"] = chi2.sf(stat, df) if regular and not note else np.nan
                if note: notes.append(f"{label}: {note}")
            for label, f in models.items(): row[f"logL ({label})"] = f["loglik"]
            row["Not"] = " ".join(dict.fromkeys(notes))
        table.append(row)
    table = pd.DataFrame(table); tested = ~table["Sabit çapa"]
    for col in ["p"] + [c for c in table if c.startswith("p (")]:
        qcol = "q (BH)" if col == "p" else "q (BH, " + col[3:]
        table.loc[tested, qcol] = benjamini_hochberg(table.loc[tested, col])
    table.loc[tested, "DIF işareti"] = table.loc[tested, "q (BH)"] < alpha
    table.loc[tested, "Karar"] = ["Çıkarım kullanılamıyor" if not np.isfinite(q) else "İçerik incelemesi gerekli" if q < alpha else "Gözlenen DIF yok" for q in table.loc[tested, "q (BH)"]]
    for i, models in fits.items():
        usable = np.isfinite(table.loc[i, "p"])
        for destination, frame in ((contrasts, parameter_contrasts(models["H2"], i, usable)),
                                   (effects, weighted_effects(models["H2"], i, usable))):
            frame.insert(0, "Madde", inputs["names"][i]); destination.append(frame)
    warnings = ["Önceden belirlenmiş çapalar bağımsız kanıt gerektirir; veriyle çapa seçimi/arındırma yapılmaz.",
        "Referans N(0,1); odak ortalama/SS serbest. Impact DIF değildir. Tek boyutluluk/yerel bağımsızlık/MAR varsayılır.",
        "Her hipotez türünün ayrı BH ailesi vardır; ortak veya genel FDR kontrolü kanıtlanmamıştır.",
        "Aralıklar noktasal Wald/delta yaklaşımıdır; eşdeğerlik veya madde yanlılığı kanıtı değildir; otomatik madde silinmez.",
        "Sınır, bilgi matrisi, yakınsama veya quadrature sorunu varsa p ve güven aralıkları verilmez."]
    if model == "RSM": warnings += ["Kategori adımları tüm maddelerde VE gruplarda ortaktır, toplamları sıfırdır. Test yalnızca madde konumu DIF'idir; ayrı kategori-adımı DIF'i sınanmaz."]
    if model == "NRM": warnings += ["Kategoriler sırasızdır; beyan edilen ilk etiket referans kategoridir. İlk çapanın ikinci/referans kategori eğim farkı pozitif olacak şekilde eksen yönü tanımlanır. Bu bir başarı sırası değildir; puan/avantaj yorumu yoktur."]
    return dict(model=model, status=STATUS, method=f"Eşzamanlı {model} MML; sabit çapalar", inputs=inputs,
        table=table, parameters=pd.concat(contrasts, ignore_index=True), effects=pd.concat(effects, ignore_index=True),
        fits=fits, anchors=list(anchors), reference=reference, focal=focal, alpha=alpha, points=points,
        n_fits=fitter.optimizations, n_models=len(fitter.cache), warnings=warnings)
