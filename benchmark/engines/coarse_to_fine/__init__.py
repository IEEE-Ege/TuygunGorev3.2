"""benchmark.engines.coarse_to_fine — DINOv2 + LightGlue pipeline.

Iki asamali esleme motoru:
    1. Coarse: DINOv2 patch embedding'leri ile aday bolge tespiti.
    2. Fine:   LightGlue ile hassas local feature matching.

Moduller:
    engine            : CoarseToFineEngine (MatchingEngine impl.)
    feature_extractor : DINOv2 feature extraction wrapper
    localizer         : Heatmap tabanli coarse lokalizasyon
    matcher           : LightGlue local feature matching
    verifier          : Geometrik dogrulama ve bbox hesaplama

.. note::
    Bu paket henuz iskelet asamasindadir.
    Implementasyon gelecek surumde eklenecektir.
"""
