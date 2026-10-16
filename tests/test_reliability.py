"""
PsikoAnaliz — Güvenirlik ve Ölçme Hatası Doğrulama Testleri (tests/test_reliability.py)
KR-20 ve Cronbach Alfa Sayısal Eşdeğerlik Kanıtı ve Güvenirlik Analizleri.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

import pytest
import numpy as np
import pandas as pd
from core.reliability import compute_reliability_indices


def test_kr20_and_cronbach_alpha_exact_equivalence():
    """
    KRİTİK BİLİMSEL VE MATEMATİKSEL DOĞRULAMA:
    Eksiksiz (kayıp verisiz) ikili (dichotomous 0/1) bir veri setinde,
    örneklem varyansı (ddof=1) standardı tutarlı kullanıldığında:
    Cronbach Alfa ile KR-20 virgülden sonra en az 12 basamak hassasiyetle birebir ÖZDEŞTİR:
    |Alpha - KR-20| < 1e-12
    """
    # Deterministik sentetik veri seti
    rng = np.random.default_rng(12345)
    n = 200
    k = 20
    # 0 ve 1 yanıtları
    data = rng.binomial(n=1, p=0.6, size=(n, k))
    cols = [f"Item_{i+1:02d}" for i in range(k)]
    df = pd.DataFrame(data, columns=cols)

    rel_res = compute_reliability_indices(df, cols, is_dichotomous=True)

    alpha = rel_res["cronbach_alpha"]
    kr20 = rel_res["kr20"]
    diff = rel_res["alpha_kr20_difference"]

    # Hem yuvarlatılmış değerlerin hem de ham farkın tolerans içinde olduğunu test et
    assert abs(alpha - kr20) < 1e-6
    assert diff < 1e-12, f"KR-20 ve Alfa sayısal olarak eşdeğer değil! Fark: {diff}"


def test_kr21_less_than_or_equal_to_kr20():
    """KR-21'in maddeler farklı güçlüklere sahipken KR-20'den küçük veya eşit olmasının doğrulanması."""
    rng = np.random.default_rng(999)
    # Farklı güçlüklerde maddeler oluştur
    p_values = np.linspace(0.2, 0.8, 15)
    data = []
    for p in p_values:
        data.append(rng.binomial(n=1, p=p, size=150))
    df = pd.DataFrame(np.array(data).T, columns=[f"M_{i+1:02d}" for i in range(15)])

    rel_res = compute_reliability_indices(df, list(df.columns), is_dichotomous=True)

    assert rel_res["kr21"] <= rel_res["kr20"] + 1e-6


def test_split_half_spearman_brown_and_guttman():
    """İki-yarı güvenirlik, Spearman-Brown kehanet formülü ve Guttman L4 testi."""
    rng = np.random.default_rng(42)
    # Yüksek korelasyonlu veri üret
    base = rng.normal(0, 1, 100)
    data = {}
    for i in range(10):
        data[f"Item_{i+1}"] = base + rng.normal(0, 0.4, 100)
    df = pd.DataFrame(data)

    rel_res = compute_reliability_indices(df, list(df.columns), is_dichotomous=False)

    sb = rel_res["spearman_brown_odd_even"]
    guttman_l4 = rel_res["guttman_lambda4"]
    alpha = rel_res["cronbach_alpha"]

    # Güvenirlik katsayıları makul pozitif aralıkta olmalıdır [0.70, 1.0]
    assert 0.70 <= sb <= 1.0
    assert 0.70 <= guttman_l4 <= 1.0
    assert 0.70 <= alpha <= 1.0


def test_sem_and_confidence_intervals():
    """Ölçmenin standart hatası (SEM) ve Feldt güven aralığı testi."""
    rng = np.random.default_rng(77)
    thetas = rng.normal(0, 1, 150)
    data = {}
    for i in range(10):
        probs = 1.0 / (1.0 + np.exp(-1.5 * thetas))
        data[f"I_{i}"] = (rng.uniform(size=150) < probs).astype(int)
    df = pd.DataFrame(data)

    rel_res = compute_reliability_indices(df, list(df.columns), is_dichotomous=True)

    sem = rel_res["sem"]
    sd = rel_res["total_std"]
    alpha = rel_res["cronbach_alpha"]

    # SEM = SD * sqrt(1 - alpha)
    expected_sem = sd * np.sqrt(max(0.0, 1.0 - alpha))
    assert pytest.approx(sem, rel=1e-3) == expected_sem

    # Güven aralıkları sınır kontrolü
    assert rel_res["alpha_ci_lower"] <= alpha <= rel_res["alpha_ci_upper"]
    assert 0.0 <= rel_res["alpha_ci_lower"]
    assert rel_res["alpha_ci_upper"] <= 1.0
