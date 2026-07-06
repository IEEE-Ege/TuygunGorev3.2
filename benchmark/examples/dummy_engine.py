"""benchmark.examples.dummy_engine — Pipeline dogrulama motoru.

Gercek goruntu isleme yapmaz.  Benchmark akisinin
(loader -> runner -> engine -> prediction) bastan sona dogru
calistigini kanitlamak icin kullanilir.

``detect()`` her zaman ``MatchResult.NO_MATCH`` dondurur.

Kullanim::

    from benchmark.examples.dummy_engine import DummyEngine

    engine = DummyEngine()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from benchmark.engine import MatchingEngine
from benchmark.prediction_models import DetectionPrediction, MatchResult

logger = logging.getLogger(__name__)


class DummyEngine(MatchingEngine):
    """Pipeline dogrulama amacli sahte esleme motoru.

    Gercek bir esleme algoritmasi degildir.  Benchmark altyapisinin
    uctan uca calistigini dogrulamak icin kullanilir.

    Davranis:
        - ``initialize``: Durumu hazir olarak isaretler.
        - ``set_reference``: Dosya varligini kontrol eder, baska islem yapmaz.
        - ``detect``: Her zaman ``MatchResult.NO_MATCH`` dondurur.
        - ``cleanup``: Dahili durumu sifirlar.
    """

    def __init__(self) -> None:
        self._initialized: bool = False
        self._reference_path: Optional[Path] = None

    @property
    def name(self) -> str:
        """Algoritma adi."""
        return "DummyEngine"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Engine'i hazirlar.

        Args:
            config: Kullanilmaz, uyumluluk icin kabul edilir.
        """
        self._initialized = True
        logger.info("%s: Baslatildi.", self.name)

    def set_reference(self, reference_image_path: Path) -> None:
        """Referans gorselini kaydeder (isleme yapmaz).

        Args:
            reference_image_path: Referans gorselinin dosya yolu.

        Raises:
            RuntimeError: Engine initialize edilmediyse.
            FileNotFoundError: Dosya bulunamazsa.
        """
        if not self._initialized:
            raise RuntimeError("Engine henuz initialize edilmedi.")

        reference_image_path = Path(reference_image_path)
        if not reference_image_path.is_file():
            raise FileNotFoundError(
                f"Referans gorseli bulunamadi: {reference_image_path}"
            )

        self._reference_path = reference_image_path
        logger.info(
            "%s: Referans ayarlandi: %s",
            self.name,
            reference_image_path.name,
        )

    def detect(self, test_image_path: Path) -> DetectionPrediction:
        """Her zaman ``NO_MATCH`` dondurur.

        Args:
            test_image_path: Test gorselinin dosya yolu.

        Returns:
            ``MatchResult.NO_MATCH`` sonuclu ``DetectionPrediction``.
        """
        start = time.perf_counter()

        test_image_path = Path(test_image_path)
        filename = test_image_path.name

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return DetectionPrediction(
            image_filename=filename,
            reference_set="",  # Runner tarafından set edilmez, engine bilmez
            result=MatchResult.NO_MATCH,
            predicted_box=None,
            confidence=None,
            processing_time_ms=elapsed_ms,
            metadata={"engine": "dummy"},
        )

    def cleanup(self) -> None:
        """Dahili durumu sifirlar."""
        self._initialized = False
        self._reference_path = None
        logger.info("%s: Temizlendi.", self.name)
