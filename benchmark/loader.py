"""
benchmark.loader
=================

Validation klasörünü otomatik tarayarak tüm referans setlerini
bulan ve yükleyen modül.

Kullanım:
    from benchmark.loader import load_validation_dataset

    dataset = load_validation_dataset("path/to/Validation")
    print(dataset.summary())

Bu modül dosya sistemi tarama, referans görsel tespiti ve
parsers modülü aracılığıyla XML okuma işlemlerini koordine eder.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from benchmark.data_models import ReferenceSet, ValidationDataset
from benchmark.parsers import parse_cvat_xml

logger = logging.getLogger(__name__)

# ref klasörlerini eşleştirmek için regex deseni
_REF_DIR_PATTERN = re.compile(r"^ref\d+$", re.IGNORECASE)

# Referans görsel dosya adı deseni (Referans_Nesne_XX.JPG/jpg)
_REF_IMAGE_PATTERN = re.compile(
    r"^Referans_Nesne_\d+\.(jpg|jpeg|png|bmp)$", re.IGNORECASE
)

# Standart alt-klasör ve dosya isimleri
_IMAGES_DIR_NAME = "Images"
_ANNOTATIONS_FILENAME = "annotations.xml"


def load_validation_dataset(validation_dir: str | Path) -> ValidationDataset:
    """Validation klasörünü tarayarak tüm referans setlerini yükler.

    Args:
        validation_dir: Validation ana klasörünün yolu.
                        İçinde ref01/, ref02/ vb. alt klasörler beklenir.

    Returns:
        Tüm yüklenen setleri içeren ValidationDataset nesnesi.

    Raises:
        FileNotFoundError: validation_dir mevcut değilse.
        NotADirectoryError: validation_dir bir klasör değilse.
    """
    validation_dir = Path(validation_dir)

    if not validation_dir.exists():
        raise FileNotFoundError(
            f"Validation klasörü bulunamadı: {validation_dir}"
        )
    if not validation_dir.is_dir():
        raise NotADirectoryError(
            f"Belirtilen yol bir klasör değil: {validation_dir}"
        )

    logger.info("Validation klasörü taranıyor: %s", validation_dir)

    ref_dirs = discover_reference_dirs(validation_dir)
    logger.info("%d referans klasörü bulundu.", len(ref_dirs))

    reference_sets: List[ReferenceSet] = []
    for ref_dir in ref_dirs:
        ref_set = load_reference_set(ref_dir)
        if ref_set is not None:
            reference_sets.append(ref_set)

    dataset = ValidationDataset(
        root_dir=validation_dir,
        reference_sets=reference_sets,
    )

    logger.info(
        "Yükleme tamamlandı: %d set, %d toplam görüntü.",
        dataset.num_sets,
        dataset.total_images,
    )

    return dataset


def discover_reference_dirs(validation_dir: Path) -> List[Path]:
    """Validation klasöründeki tüm refXX klasörlerini bulur ve sıralı döndürür.

    Args:
        validation_dir: Validation ana klasörünün yolu.

    Returns:
        refXX klasörlerinin sıralı listesi (doğal sıralama).
    """
    ref_dirs = sorted(
        d
        for d in validation_dir.iterdir()
        if d.is_dir() and _REF_DIR_PATTERN.match(d.name)
    )

    if not ref_dirs:
        logger.warning(
            "Hiçbir refXX klasörü bulunamadı: %s", validation_dir
        )

    return ref_dirs


def load_reference_set(ref_dir: Path) -> Optional[ReferenceSet]:
    """Tek bir refXX klasörünü yükler.

    Beklenen yapı:
        refXX/
        ├── Images/          (test görselleri)
        ├── Referans_Nesne_XX.JPG  (referans görsel)
        └── annotations.xml  (CVAT anotasyonları)

    Args:
        ref_dir: ref klasörünün tam yolu.

    Returns:
        ReferenceSet nesnesi veya kritik dosya eksikse None.
    """
    set_name = ref_dir.name
    logger.info("Referans seti yükleniyor: %s", set_name)

    # Images klasörünü bul
    images_dir = ref_dir / _IMAGES_DIR_NAME
    if not images_dir.is_dir():
        logger.warning(
            "%s: '%s' klasörü bulunamadı, set atlanıyor.",
            set_name,
            _IMAGES_DIR_NAME,
        )
        return None

    # Anotasyon dosyasını bul
    annotations_path = ref_dir / _ANNOTATIONS_FILENAME
    if not annotations_path.is_file():
        logger.warning(
            "%s: '%s' dosyası bulunamadı, set atlanıyor.",
            set_name,
            _ANNOTATIONS_FILENAME,
        )
        return None

    # Referans görseli bul (opsiyonel — yoksa yine de devam et)
    reference_image_path = _find_reference_image(ref_dir)
    if reference_image_path is None:
        logger.info(
            "%s: Referans görseli bulunamadı (opsiyonel).", set_name
        )

    # XML'i parse et
    annotations = parse_cvat_xml(annotations_path, images_dir)

    ref_set = ReferenceSet(
        name=set_name,
        root_dir=ref_dir,
        reference_image_path=reference_image_path,
        images_dir=images_dir,
        annotations=annotations,
    )

    logger.info(
        "%s yüklendi: %d görüntü, ref_img=%s",
        set_name,
        ref_set.num_images,
        "✓" if reference_image_path else "✗",
    )

    return ref_set


def _find_reference_image(ref_dir: Path) -> Optional[Path]:
    """Referans görselini bulur (Referans_Nesne_XX.JPG formatında).

    ref klasörünün kökünde, Referans_Nesne deseniyle eşleşen ilk
    görsel dosyasını döndürür.

    Args:
        ref_dir: ref klasörünün tam yolu.

    Returns:
        Referans görselinin yolu veya bulunamazsa None.
    """
    for item in ref_dir.iterdir():
        if item.is_file() and _REF_IMAGE_PATTERN.match(item.name):
            return item
    return None
