#!/usr/bin/env python3
"""Dummy Engine ile uctan uca benchmark testi.

Bu betik, benchmark altyapisinin tamamen calistigini dogrular.
DummyEngine her zaman NO_MATCH dondurur — gercek esleme yapmaz.

Kullanim::

    python -m benchmark.examples.run_dummy_benchmark

Veya dogrudan::

    python benchmark/examples/run_dummy_benchmark.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Proje kokunu PYTHONPATH'e ekle
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark.examples.dummy_engine import DummyEngine
from benchmark.loader import load_validation_dataset
from benchmark.runner import BenchmarkRunner


def main() -> None:
    """Benchmark pipeline'ini DummyEngine ile calistirir."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)-35s | %(levelname)-7s | %(message)s",
    )

    # 1. Validation setini yukle
    validation_dir = _PROJECT_ROOT / "Validation"
    print(f"\n[1/4] Validation seti yukleniyor: {validation_dir}")
    dataset = load_validation_dataset(validation_dir)

    # 2. Dummy engine olustur
    print("[2/4] DummyEngine olusturuluyor...")
    engine = DummyEngine()

    # 3. Runner ile benchmark calistir
    print("[3/4] Benchmark calistiriliyor...")
    runner = BenchmarkRunner(engine=engine, dataset=dataset)
    results = runner.run()

    # 4. Sonuclari goster
    print("\n[4/4] Sonuclar:")
    print("=" * 60)
    print(results.summary())
    print("=" * 60)

    # Detayli inceleme: ilk setin ilk 3 tahmini
    if results.set_predictions:
        first_set = results.set_predictions[0]
        print(f"\n--- {first_set.set_name} detay (ilk 3 tahmin) ---")
        for pred in first_set.predictions[:3]:
            print(f"  {pred}")

    print("\nPipeline dogrulamasi basarili!")


if __name__ == "__main__":
    main()
