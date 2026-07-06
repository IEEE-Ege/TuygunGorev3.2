"""benchmark.runner — Benchmark calistirma orkestratoru.

Veri yukleme katmani ile esleme motoru arasindaki kopruyu kurar.
Kendisi hicbir algoritmayi bilmez ve hicbir metrik hesaplamaz.

Sorumluluklar:
    1. Validation setlerini dolasmak.
    2. ``MatchingEngine``'i dogru sirayla cagirmak.
    3. ``DetectionPrediction`` nesnelerini toplamak.
    4. ``BenchmarkPredictions`` dondurmek.

Kullanim::

    dataset = load_validation_dataset("Validation")
    engine  = SomeMatchingEngine()
    runner  = BenchmarkRunner(engine=engine, dataset=dataset)
    results = runner.run()
    print(results.summary())
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.data_models import ReferenceSet, ValidationDataset
from benchmark.engine import MatchingEngine
from benchmark.prediction_models import (
    BenchmarkPredictions,
    DetectionPrediction,
    MatchResult,
    SetPredictions,
)

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Benchmark calistirma orkestratoru.

    Dependency Injection prensibiyle calisir: engine ve dataset
    disaridan verilir, runner yalnizca akisi yonetir.

    Args:
        engine: Kullanilacak esleme motoru.
        dataset: Uzerinde benchmark calistirilacak validation seti.

    Raises:
        TypeError: ``engine`` bir ``MatchingEngine`` instance'i degilse.
        ValueError: ``dataset`` bossa.
    """

    def __init__(
        self,
        engine: MatchingEngine,
        dataset: ValidationDataset,
    ) -> None:
        if not isinstance(engine, MatchingEngine):
            raise TypeError(
                f"engine bir MatchingEngine instance'i olmali, "
                f"verilen: {type(engine).__name__}"
            )
        if not dataset.reference_sets:
            raise ValueError(
                "dataset bos — en az bir referans seti gerekli."
            )

        self._engine = engine
        self._dataset = dataset

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def engine(self) -> MatchingEngine:
        """Kullanilan esleme motoru."""
        return self._engine

    @property
    def dataset(self) -> ValidationDataset:
        """Uzerinde calisilan validation seti."""
        return self._dataset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self, config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkPredictions:
        """Tum referans setleri uzerinde benchmark'i calistirir.

        Akis::

            engine.initialize(config)
            try:
                for ref_set in dataset:
                    engine.set_reference(...)
                    for image in ref_set:
                        engine.detect(...)
            finally:
                engine.cleanup()

        Args:
            config: Engine'e iletilecek yapilandirma parametreleri.

        Returns:
            Tum setlerin tahminlerini iceren ``BenchmarkPredictions``.
        """
        engine_name = self._engine.name

        logger.info(
            "Benchmark baslatiliyor: engine=%r, sets=%d, images=%d",
            engine_name,
            self._dataset.num_sets,
            self._dataset.total_images,
        )

        total_start = time.perf_counter()

        # Akis: initialize -> try/finally -> cleanup
        self._engine.initialize(config)

        all_set_predictions: List[SetPredictions] = []
        try:
            for ref_set in self._dataset.reference_sets:
                set_preds = self._run_single_set(ref_set)
                all_set_predictions.append(set_preds)
        finally:
            logger.info("Engine temizleniyor: %r", engine_name)
            self._engine.cleanup()

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        results = BenchmarkPredictions(
            engine_name=engine_name,
            set_predictions=all_set_predictions,
        )

        logger.info(
            "Benchmark tamamlandi: %d set, %d tahmin, %.1f ms",
            results.num_sets,
            results.total_predictions,
            total_elapsed_ms,
        )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_single_set(self, ref_set: ReferenceSet) -> SetPredictions:
        """Tek bir referans seti uzerinde engine'i calistirir.

        Args:
            ref_set: Islenecek referans seti.

        Returns:
            Bu setin tum tahminlerini iceren ``SetPredictions``.
        """
        engine_name = self._engine.name

        logger.info(
            "Set calistiriliyor: %s (%d goruntu)",
            ref_set.name,
            ref_set.num_images,
        )

        # Referans gorseli yukle
        if ref_set.reference_image_path is None:
            logger.warning(
                "%s: Referans gorseli bulunamadi, set atlaniyor.",
                ref_set.name,
            )
            return self._create_skipped_set(ref_set, engine_name)

        self._engine.set_reference(ref_set.reference_image_path)

        # Her test gorseli icin detect cagir
        predictions: List[DetectionPrediction] = []
        for annotation in ref_set.annotations:
            prediction = self._detect_safe(
                ref_set.name,
                annotation.filepath,
                annotation.filename,
            )
            predictions.append(prediction)

        set_preds = SetPredictions(
            set_name=ref_set.name,
            engine_name=engine_name,
            predictions=predictions,
        )

        logger.info(
            "%s tamamlandi: %s",
            ref_set.name,
            set_preds.summary(),
        )

        return set_preds

    def _detect_safe(
        self,
        set_name: str,
        image_path: Path,
        image_filename: str,
    ) -> DetectionPrediction:
        """Tek bir test gorseli icin detect cagrisini guvenli yapar.

        ``engine.detect()`` hata firlatmamalidir ama savunmaci
        programlama olarak yakalanmayan exception'lari burada
        ele aliyoruz.

        Args:
            set_name: Log mesajlari icin set adi.
            image_path: Test gorselinin tam yolu.
            image_filename: Test gorselinin dosya adi.

        Returns:
            ``DetectionPrediction`` nesnesi.
        """
        try:
            return self._engine.detect(image_path)
        except Exception:
            logger.exception(
                "%s/%s: Engine exception yakalandi",
                set_name,
                image_filename,
            )
            return DetectionPrediction(
                image_filename=image_filename,
                reference_set=set_name,
                result=MatchResult.ERROR,
                confidence=None,
                processing_time_ms=0.0,
                metadata={"error": "Unhandled engine exception"},
            )

    @staticmethod
    def _create_skipped_set(
        ref_set: ReferenceSet, engine_name: str,
    ) -> SetPredictions:
        """Referans gorseli olmayan set icin ERROR prediction uretir.

        Args:
            ref_set: Atlanan referans seti.
            engine_name: Algoritma adi.

        Returns:
            Tum tahminleri ``ERROR`` olan ``SetPredictions``.
        """
        predictions = [
            DetectionPrediction(
                image_filename=ann.filename,
                reference_set=ref_set.name,
                result=MatchResult.ERROR,
                confidence=None,
                processing_time_ms=0.0,
                metadata={"error": "Referans gorseli bulunamadi"},
            )
            for ann in ref_set.annotations
        ]
        return SetPredictions(
            set_name=ref_set.name,
            engine_name=engine_name,
            predictions=predictions,
        )
