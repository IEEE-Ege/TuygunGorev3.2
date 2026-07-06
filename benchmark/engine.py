"""benchmark.engine — Soyut esleme motoru arayuzu.

Strategy Pattern uygulayarak tum esleme algoritmalarinin uyacagi
kontrati tanimlar.  Benchmark sistemi hicbir zaman somut bir
algoritmay bilmez — yalnizca bu arayuz uzerinden iletisim kurar.

Yasam dongusu::

    initialize(config) -> N x [set_reference() -> M x detect()] -> cleanup()

Context manager protokolu de desteklenir::

    with engine:
        engine.initialize(config)
        engine.set_reference(...)
        engine.detect(...)
    # cleanup() otomatik cagrilir
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from benchmark.prediction_models import DetectionPrediction


class MatchingEngine(ABC):
    """Nesne esleme algoritmalarinin soyut temel sinifi.

    Her somut engine bu sinifi miras alarak tum abstract metotlari
    implemente etmelidir.  ``BenchmarkRunner`` yalnizca bu arayuzu bilir.

    Example::

        class MyEngine(MatchingEngine):
            @property
            def name(self) -> str:
                return "MyAlgorithm"
            ...

        engine = MyEngine()
        runner = BenchmarkRunner(engine=engine, dataset=dataset)
        results = runner.run()
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Algoritmanin insan-tarafindan okunabilir adi.

        Returns:
            Algoritma adi (or. ``"DINOv2+LightGlue"``).
        """

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Algorithmayi calismaya hazirlar.

        Model yukleme, GPU bellek tahsisi, parametre ayarlama gibi
        tek seferlik hazirlak islemlerini burada yapin.

        Args:
            config: Algoritma-spesifik yapilandirma parametreleri.
                ``None`` ise varsayilan ayarlar kullanilmalidir.
        """

    @abstractmethod
    def set_reference(self, reference_image_path: Path) -> None:
        """Referans gorselini isler ve bellege alir.

        Her yeni referans nesnesi icin bir kez cagrilir.  Referans
        feature'larinin cikarilmasi ve saklanmasi burada yapilir.

        Args:
            reference_image_path: Referans gorselinin dosya yolu.

        Raises:
            FileNotFoundError: Gorsel dosyasi bulunamazsa.
        """

    @abstractmethod
    def detect(self, test_image_path: Path) -> DetectionPrediction:
        """Test gorselinde referans nesnesini arar.

        Daha once ``set_reference()`` ile yuklenen referans nesnesini
        test gorselinde bulmaya calisir.

        Args:
            test_image_path: Test gorselinin dosya yolu.

        Returns:
            ``DetectionPrediction`` nesnesi.  Uc olasi durum:

            - ``SUCCESS``  : Nesne bulundu.
            - ``NO_MATCH`` : Nesne bulunamadi.
            - ``ERROR``    : Beklenmeyen hata olustu.

        Note:
            Bu metot hicbir zaman exception firlatmamalidir.
            Hatalar ``MatchResult.ERROR`` ile raporlanmalidir.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Kaynaklari serbest birakir.

        GPU bellegi, gecici dosyalar, acik baglantilar vb.
        temizleme islemlerini burada yapin.

        Bu metot idempotent olmalidir — birden fazla cagrilabilir.
        """

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> MatchingEngine:
        """Context manager girisi.  Engine'i dondurur."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager cikisi.  ``cleanup()`` cagrilir."""
        self.cleanup()

    # ------------------------------------------------------------------
    # Yardimci
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
