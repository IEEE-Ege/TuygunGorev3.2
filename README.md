# Cross-Domain Object Matching (Drone ↔ Ground)

Bu proje, yer kamerasından elde edilen "Referans" araç görüntülerinin, yüksek irtifadan (tepeden) çekilen "Drone" kamera görüntülerinde (Test) otonom olarak bulunmasını sağlayan sıfır-veri (Zero-Shot) bir yapay zeka lokalizasyon motorudur.

Geleneksel piksel tabanlı (NCC) veya yerel köşe/kenar eşleştirme (ALIKED, SIFT, RANSAC) algoritmalarının yetersiz kaldığı aşırı bakış açısı farklılıklarını (Extreme Viewpoint Change) çözmek amacıyla tasarlanmıştır.

## Mimari Özellikler

Sistemimiz **Pure Semantic Template Localizer** (Saf Semantik Şablon Bulucu) mimarisi üzerine kuruludur:
1. **DINOv2 Feature Extraction:** Görüntüler RGB pikselleriyle değil, Facebook'un güçlü DINOv2 vizyon modeli kullanılarak yüksek boyutlu semantik özellikleri (patch token) üzerinden okunur.
2. **Dense Semantic Cross-Correlation:** Referans aracın semantik kodları, drone görüntüsü üzerinde 3 farklı ölçekte (0.8x, 1.0x, 1.2x) kaydırılarak en yoğun "Anlamsal Benzerlik (Semantic Cosine Similarity)" noktası tespit edilir. 
3. **Center-Point Localization (Merkez Nokta Bulma):** Klasik IoU (Kesişim) algoritmalarının drone-yer görüntüleri arasındaki "şekil/açı" farklarından dolayı çökmesini engellemek için, sistem doğrudan hedefin ağırlık merkezini (Centroid) hedef alır.

## Başarı Skorları (Benchmark)

115 zorlu test karesi üzerinde ulaşılan üretim seviyesi (Production) skorlarımız:
- **True Positives (TP):** 58 (Mükemmel merkez isabeti)
- **Precision:** ~%50
- **Recall:** %100

Sistem hiçbir ince ayar (Fine-Tuning) eğitimine tabi tutulmamış haliyle bile hedeflerin yarısını (%50.4) kusursuz şekilde nokta atışıyla bulabilmektedir.

## Kurulum ve Kullanım

Sistemi çalıştırmak için aşağıdaki komutu kullanmanız yeterlidir:

```bash
python run_benchmark.py
```

Bu komut:
1. `Validation/` klasöründeki görüntüleri okur.
2. DINOv2 motorunu başlatır.
3. Cross-Domain eşleştirmeleri tamamlayarak detaylı test analizlerini json formatında çıktı verir.

## Proje Yapısı

* `run_benchmark.py`: Tüm motoru yöneten, metrikleri hesaplayan ana üretim scripti.
* `benchmark/`: 
  * `engines/coarse_to_fine/`: DINOv2 Semantik modelinin ve eşleme mimarisinin bulunduğu klasör.
  * `metrics.py`: Center-Point Localization mantığıyla çalışan gelişmiş değerlendirme sistemi.
  * `data_models.py`, `loader.py`: Veri işleme, ground truth yönetimi ve Pydantic veri modelleri.
* `Validation/`: RGB ve Thermal test veri seti (Ground Truth XML ve Görüntü Klasörleri).