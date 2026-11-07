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
TAM_VERSION = "4.3.25"                     # pinned engine version (packageVersion("TAM") 4.3-25)
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
