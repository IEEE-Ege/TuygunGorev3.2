"""Benchmark - Nesne Tespiti Dogrulama Altyapisi.

Bu paket, validation setleri uzerinde nesne tespiti pipeline'larini
degerlendirmek icin kullanilan benchmark araclarini icerir.

Moduller:
    data_models       : Ground truth veri yapilari (dataclass tanimlari)
    parsers           : CVAT XML 1.1 anotasyon dosyasi okuyucu
    loader            : Validation setlerini tarama ve yukleme
    prediction_models : Algoritma ciktisi veri yapilari
    engine            : Soyut MatchingEngine arayuzu (Strategy Pattern)
    runner            : Benchmark calistirma orkestratoru
    metrics           : Metrik hesaplama iskeleti
"""

from benchmark.data_models import (
    BoundingBox,
    ImageAnnotation,
    ReferenceSet,
    ValidationDataset,
)
from benchmark.engine import MatchingEngine
from benchmark.loader import load_validation_dataset
from benchmark.metrics import MetricsCalculator
from benchmark.prediction_models import (
    BenchmarkPredictions,
    DetectionPrediction,
    MatchResult,
    SetPredictions,
)
from benchmark.runner import BenchmarkRunner

__all__ = [
    # data_models
    "BoundingBox",
    "ImageAnnotation",
    "ReferenceSet",
    "ValidationDataset",
    # prediction_models
    "MatchResult",
    "DetectionPrediction",
    "SetPredictions",
    "BenchmarkPredictions",
    # engine
    "MatchingEngine",
    # runner
    "BenchmarkRunner",
    # loader
    "load_validation_dataset",
    # metrics
    "MetricsCalculator",
]
