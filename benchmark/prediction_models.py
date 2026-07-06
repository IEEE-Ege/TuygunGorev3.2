"""benchmark.prediction_models — Algoritma cikti veri yapilari.

Ground truth modellerinden (data_models) tamamen bagimsizdir.
Tek ortak nokta ``BoundingBox`` sinifidir — bir kutunun koordinat
yapisi evrenseldir ve yeniden tanimlamak gereksiz tekrara yol acar.

Hiyerarsi::

    MatchResult          ->  algoritma sonuc durumu (enum)
    DetectionPrediction  ->  tek bir goruntu icin tahmin
    SetPredictions       ->  bir refXX seti icin tum tahminler
    BenchmarkPredictions ->  tum setler icin tum tahminler
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from benchmark.data_models import BoundingBox


class MatchResult(Enum):
    """Esleme algoritmasinin bir test gorseli icin dondurdugu sonuc durumu.

    Attributes:
        SUCCESS:  Algoritma bir esleme buldu, bbox ve confidence dolu.
        NO_MATCH: Algoritma calisti ama nesneyi bulamadi.
        ERROR:    Calisma sirasinda beklenmeyen bir hata olustu.
    """

    SUCCESS = auto()
    NO_MATCH = auto()
    ERROR = auto()


@dataclass
class DetectionPrediction:
    """Tek bir test gorseli icin algoritmanin urettigi tahmin.

    Attributes:
        image_filename: Test gorselinin dosya adi (or. ``frame_024_00_23.jpg``).
        reference_set: Bu tahminin ait oldugu referans set adi (or. ``ref01``).
        result: Esleme sonucunun durumu.
        predicted_box: Tahmin edilen bounding box.
            Yalnizca ``SUCCESS`` durumunda dolu olmalidir.
        confidence: Algoritmanin guven skoru ``[0.0, 1.0]``.
            ``NO_MATCH`` ve ``ERROR`` durumlarinda ``None`` olabilir.
        processing_time_ms: Bu tahmin icin harcanan sure (milisaniye).
        metadata: Algoritma-spesifik ek bilgiler.
            Ornegin feature extraction suresi, esleme sayisi vb.
    """

    image_filename: str
    reference_set: str
    result: MatchResult
    predicted_box: Optional[BoundingBox] = None
    confidence: Optional[float] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Tahmin basarili mi?"""
        return self.result is MatchResult.SUCCESS

    @property
    def is_no_match(self) -> bool:
        """Algoritma nesneyi bulamadi mi?"""
        return self.result is MatchResult.NO_MATCH

    @property
    def is_error(self) -> bool:
        """Calisma sirasinda hata olustu mu?"""
        return self.result is MatchResult.ERROR

    def __repr__(self) -> str:
        parts = [
            f"image={self.image_filename!r}",
            f"set={self.reference_set!r}",
            f"result={self.result.name}",
        ]
        if self.is_success and self.predicted_box is not None:
            parts.append(
                f"box=({self.predicted_box.xtl:.1f},{self.predicted_box.ytl:.1f})"
                f"->({self.predicted_box.xbr:.1f},{self.predicted_box.ybr:.1f})"
            )
        if self.confidence is not None:
            parts.append(f"conf={self.confidence:.3f}")
        parts.append(f"time={self.processing_time_ms:.1f}ms")
        return f"DetectionPrediction({', '.join(parts)})"


@dataclass
class SetPredictions:
    """Bir referans seti (refXX) icin tum tahminleri tutan konteyner.

    Attributes:
        set_name: Referans set adi (or. ``ref01``).
        engine_name: Tahminleri ureten algoritmanin adi.
        predictions: Bu setteki her test gorseli icin tahmin listesi.
    """

    set_name: str
    engine_name: str
    predictions: List[DetectionPrediction] = field(default_factory=list)

    @property
    def num_predictions(self) -> int:
        """Toplam tahmin sayisi."""
        return len(self.predictions)

    @property
    def num_successful(self) -> int:
        """Basarili tahmin sayisi."""
        return sum(1 for p in self.predictions if p.is_success)

    @property
    def num_no_match(self) -> int:
        """Esleme bulunamayan tahmin sayisi."""
        return sum(1 for p in self.predictions if p.is_no_match)

    @property
    def num_errors(self) -> int:
        """Hata durumundaki tahmin sayisi."""
        return sum(1 for p in self.predictions if p.is_error)

    @property
    def total_processing_time_ms(self) -> float:
        """Tum tahminler icin toplam isleme suresi (ms)."""
        return sum(p.processing_time_ms for p in self.predictions)

    def summary(self) -> str:
        """Set tahminlerinin ozet istatistikleri."""
        return (
            f"{self.set_name}: {self.num_predictions} tahmin "
            f"(basarili={self.num_successful}, "
            f"bulunamadi={self.num_no_match}, "
            f"hata={self.num_errors}, "
            f"sure={self.total_processing_time_ms:.1f}ms)"
        )

    def __repr__(self) -> str:
        return (
            f"SetPredictions(set={self.set_name!r}, "
            f"engine={self.engine_name!r}, "
            f"n={self.num_predictions})"
        )


@dataclass
class BenchmarkPredictions:
    """Tum setler icin tum tahminleri tutan ust duzey konteyner.

    Bir benchmark calistirmasinin tam ciktisidir.

    Attributes:
        engine_name: Tahminleri ureten algoritmanin adi.
        set_predictions: Her referans seti icin tahmin koleksiyonu.
    """

    engine_name: str
    set_predictions: List[SetPredictions] = field(default_factory=list)

    @property
    def num_sets(self) -> int:
        """Toplam set sayisi."""
        return len(self.set_predictions)

    @property
    def total_predictions(self) -> int:
        """Tum setlerdeki toplam tahmin sayisi."""
        return sum(sp.num_predictions for sp in self.set_predictions)

    @property
    def total_successful(self) -> int:
        """Tum setlerdeki basarili tahmin sayisi."""
        return sum(sp.num_successful for sp in self.set_predictions)

    @property
    def total_processing_time_ms(self) -> float:
        """Tum tahminler icin toplam isleme suresi (ms)."""
        return sum(sp.total_processing_time_ms for sp in self.set_predictions)

    def get_set(self, set_name: str) -> Optional[SetPredictions]:
        """Isme gore set tahminlerini dondurur.

        Args:
            set_name: Aranacak set adi (or. ``ref01``).

        Returns:
            Bulunan ``SetPredictions`` veya ``None``.
        """
        for sp in self.set_predictions:
            if sp.set_name == set_name:
                return sp
        return None

    def summary(self) -> str:
        """Benchmark ciktisinin ozet istatistikleri."""
        lines = [
            f"BenchmarkPredictions: engine={self.engine_name!r}",
            f"  Toplam set       : {self.num_sets}",
            f"  Toplam tahmin    : {self.total_predictions}",
            f"  Basarili         : {self.total_successful}",
            f"  Toplam sure      : {self.total_processing_time_ms:.1f}ms",
            "  -----------------------------------",
        ]
        for sp in self.set_predictions:
            lines.append(f"  {sp.summary()}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BenchmarkPredictions(engine={self.engine_name!r}, "
            f"sets={self.num_sets}, n={self.total_predictions})"
        )
