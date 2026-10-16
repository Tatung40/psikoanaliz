"""
PsikoAnaliz — Sentetik Örnek Veri Setleri Üretici (sample_data/generate_samples.py)
Deterministik tohumla (seed) bilimsel geçerliliği olan sentetik psikometrik veri setleri üretir.
Geliştirici: Doç. Dr. Mehmet Taha Eser
"""

import os
import numpy as np
import pandas as pd


def generate_all_samples(output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    # 1. ÇOKTAN SEÇMELİ HAM VERİ SETİ (30 Madde, 350 Katılımcı)
    n_examinees = 350
    k_mc = 30
    options = ["A", "B", "C", "D", "E"]
    
    # Cevap anahtarı
    keys = [options[rng.integers(0, 5)] for _ in range(k_mc)]
    
    # Katılımcı yetenekleri theta ~ N(0, 1)
    thetas = rng.normal(0, 1, n_examinees)
    # Madde güçlükleri b ~ N(0, 0.9)
    item_b = rng.normal(0, 0.9, k_mc)
    
    mc_rows = []
    genders = rng.choice(["Kadın", "Erkek"], size=n_examinees, p=[0.55, 0.45])
    branches = rng.choice(["Şube A", "Şube B", "Şube C"], size=n_examinees)

    for i in range(n_examinees):
        theta = thetas[i]
        row_responses = {
            "Ogrenci_ID": f"OGR_{i+1:04d}",
            "Cinsiyet": genders[i],
            "Sube": branches[i]
        }
        for j in range(k_mc):
            # 2PL lojistik başarı olasılığı
            p_correct = 1.0 / (1.0 + np.exp(-1.5 * (theta - item_b[j])))
            # Rastgele tahmin payı
            p_correct = 0.20 + 0.80 * p_correct

            if rng.uniform() < p_correct:
                ans = keys[j]
            else:
                distractors = [opt for opt in options if opt != keys[j]]
                ans = rng.choice(distractors)
            row_responses[f"M_{j+1:02d}"] = ans
        mc_rows.append(row_responses)

    df_mc = pd.DataFrame(mc_rows)
    df_mc.to_csv(os.path.join(output_dir, "coktan_secmeli_ham.csv"), index=False, encoding="utf-8-sig")

    # Cevap anahtarını ayrı bir CSV olarak da kaydet
    df_key = pd.DataFrame({
        "Madde": [f"M_{j+1:02d}" for j in range(k_mc)],
        "Dogru_Cevap": keys
    })
    df_key.to_csv(os.path.join(output_dir, "cevap_anahtari.csv"), index=False, encoding="utf-8-sig")

    # 2. ÖNCEDEN PUANLANMIŞ İKİLİ (0/1) VERİ SETİ (25 Madde, 300 Katılımcı)
    k_bin = 25
    n_bin = 300
    thetas_bin = rng.normal(0, 1, n_bin)
    item_b_bin = np.linspace(-1.8, 1.8, k_bin)
    
    bin_rows = []
    groups_bin = rng.choice(["Grup 1", "Grup 2"], size=n_bin, p=[0.5, 0.5])
    for i in range(n_bin):
        th = thetas_bin[i]
        row = {"Katilimci_ID": f"KAT_{i+1:03d}", "Grup": groups_bin[i]}
        for j in range(k_bin):
            prob = 1.0 / (1.0 + np.exp(-1.7 * (th - item_b_bin[j])))
            val = 1 if rng.uniform() < prob else 0
            row[f"Item_{j+1:02d}"] = val
        bin_rows.append(row)

    df_bin = pd.DataFrame(bin_rows)
    df_bin.to_csv(os.path.join(output_dir, "ikili_puanlanmis.csv"), index=False, encoding="utf-8-sig")

    # 3. LIKERT TİPİ ÇOK KATEGORİLİ VERİ SETİ (20 Madde, 5'li Likert [1-5], 250 Katılımcı)
    k_likert = 20
    n_likert = 250
    thetas_likert = rng.normal(3.2, 0.7, n_likert)
    
    # 4., 8., 12. ve 16. maddeler ters kodlanacak maddeler
    reverse_items = [f"Likert_{idx:02d}" for idx in [4, 8, 12, 16]]
    
    likert_rows = []
    departments = rng.choice(["Eğitim", "Fen-Edebiyat", "Mühendislik"], size=n_likert)
    for i in range(n_likert):
        th = thetas_likert[i]
        row = {"ID": f"KAT_{i+1:03d}", "Fakulte": departments[i]}
        for j in range(k_likert):
            col_name = f"Likert_{j+1:02d}"
            base_score = rng.normal(th, 0.8)
            score = int(np.clip(np.round(base_score), 1, 5))
            # Eğer ters madde ise katılımcı aslında ters yanıt verir
            if col_name in reverse_items:
                score = 6 - score
            row[col_name] = score
        likert_rows.append(row)

    df_likert = pd.DataFrame(likert_rows)
    df_likert.to_csv(os.path.join(output_dir, "likert_olcek.csv"), index=False, encoding="utf-8-sig")

    # 4. UÇ DURUMLAR VE TEST VERİ SETİ (Sabit maddeler, negatif ayırt edicilik, kayıp veri)
    df_edge = df_bin.copy().iloc[:100].copy()
    # M_01 sabit (herkes 1 almış)
    df_edge["Item_01"] = 1
    # M_02 sabit (herkes 0 almış)
    df_edge["Item_02"] = 0
    # M_03 negatif ayırt edicilik (yetenekli olanlar yanlış yapmış)
    # ters sırala
    totals = df_edge[[c for c in df_edge.columns if c.startswith("Item_")]].sum(axis=1)
    df_edge["Item_03"] = (totals < totals.median()).astype(int)
    # Bazı kayıp veriler ekle
    for c in ["Item_05", "Item_10"]:
        df_edge.loc[rng.choice(100, size=15, replace=False), c] = np.nan

    df_edge.to_csv(os.path.join(output_dir, "uc_durumlar_testi.csv"), index=False, encoding="utf-8-sig")
    print("Örnek veri setleri başarıyla üretildi.")


if __name__ == "__main__":
    generate_all_samples(".")
