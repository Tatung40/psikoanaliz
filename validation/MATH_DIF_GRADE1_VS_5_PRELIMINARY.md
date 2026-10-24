# Matematik verisi — keşifsel DIF taraması (2026-10-23)

Veri: math_cross_competency_anonymized_v1.csv; ham katılımcı verisi depoya aktarılmadı.

Karşılaştırma: grade_level=1 (n=407) referans, grade_level=5 (n=285) odak. 28 ikili madde.

Yöntem: her madde için madde-kalan toplam puan ile koşullandırılmış binom GLM (uniform DIF: grup ana etkisi, nonuniform DIF: grup×puan etkileşimi); ayrıca tam sayı madde-kalan toplam puan katmanlarında Mantel–Haenszel ortak odds oranı. Her test ailesinde 28 maddelik Benjamini–Hochberg FDR düzeltmesi (q<.05).

- Uniform lojistik regresyon: 17/28 sinyal.
- Nonuniform lojistik etkileşim: 10/28 sinyal.
- Mantel–Haenszel: 15/28 sinyal.
- Her iki uniform yöntemce işaretlenen: Q01 Q02 Q04 Q06 Q10 Q11 Q15 Q16 Q17 Q19 Q20 Q21 Q24 Q25 Q27 (15 madde).
- Uniform yalnızca lojistik: Q08 Q22.
- Güçlü odak grubu aleyhine OR: Q10 0.2455; Q21 0.2364; Q27 0.2076.
- Güçlü odak grubu lehine OR: Q01 2.9193; Q04 2.3166; Q06 2.2494.

Dikkat: 1. ve 5. sınıf düzeyleri arasında ölçme yapısı, öğretim kapsamı ve yetenek dağılımı değişebilir. Ortak ölçek/ankor, tek boyutluluk, yerel bağımsızlık, çok düzeyli grup bağımlılığı ve eksiklerin 0 kodlanma ihtimali henüz incelenmedi. DIF sinyali yanlılık kanıtı değildir. Bu analiz platformun kendi DIF motorunu doğrulamaz.

Tekrarlanabilir kaynak betik: kullanıcıya sunulan dif_validation.py; ham CSV depoya konmadığından betik yerel girdi gerektirir.
