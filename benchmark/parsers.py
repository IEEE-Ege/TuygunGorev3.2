"""
benchmark.parsers
=================

CVAT XML 1.1 formatındaki anotasyon dosyalarını parse eden modül.

Bu modül yalnızca XML okuma ve veri yapılarına dönüştürme işlemlerinden
sorumludur. Dosya sistemi tarama veya görüntü yükleme işlemleri
`loader` modülünde gerçekleştirilir.

Desteklenen Format:
    CVAT for Images 1.1 — XML tabanlı export formatı.
    Her <image> elementi bir görüntüye, her <box> elementi bir
    bounding box anotasyonuna karşılık gelir.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from benchmark.data_models import BoundingBox, ImageAnnotation

logger = logging.getLogger(__name__)


def parse_cvat_xml(xml_path: Path, images_dir: Path) -> List[ImageAnnotation]:
    """CVAT XML 1.1 anotasyon dosyasını parse eder.

    Args:
        xml_path: annotations.xml dosyasının tam yolu.
        images_dir: Anotasyonlardaki görsellerin bulunduğu klasör.
                    Dosya yolları bu klasöre göre çözülür.

    Returns:
        Her bir <image> elementi için oluşturulmuş ImageAnnotation
        nesnelerinin listesi.

    Raises:
        FileNotFoundError: XML dosyası bulunamazsa.
        ET.ParseError: XML formatı geçersizse.
    """
    xml_path = Path(xml_path)
    images_dir = Path(images_dir)

    if not xml_path.is_file():
        raise FileNotFoundError(f"Anotasyon dosyası bulunamadı: {xml_path}")

    logger.info("CVAT XML parse ediliyor: %s", xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Versiyon kontrolü (opsiyonel ama bilgilendirici)
    version_elem = root.find("version")
    if version_elem is not None:
        version = version_elem.text
        if version != "1.1":
            logger.warning(
                "Beklenen CVAT XML versiyonu 1.1, bulunan: %s (%s)",
                version,
                xml_path,
            )

    annotations: List[ImageAnnotation] = []

    for image_elem in root.findall("image"):
        annotation = _parse_image_element(image_elem, images_dir)
        if annotation is not None:
            annotations.append(annotation)

    logger.info(
        "Parse tamamlandı: %d görüntü, %d toplam kutu (%s)",
        len(annotations),
        sum(a.num_boxes for a in annotations),
        xml_path.name,
    )

    return annotations


def _parse_image_element(
    image_elem: ET.Element, images_dir: Path
) -> ImageAnnotation | None:
    """Tek bir <image> XML elementini ImageAnnotation'a dönüştürür.

    Args:
        image_elem: <image> XML elementi.
        images_dir: Görsellerin bulunduğu klasör.

    Returns:
        ImageAnnotation nesnesi veya geçersiz veriyse None.
    """
    filename = image_elem.get("name")
    width_str = image_elem.get("width")
    height_str = image_elem.get("height")

    if not all([filename, width_str, height_str]):
        logger.warning(
            "Eksik image attribute'u, element atlanıyor: %s",
            ET.tostring(image_elem, encoding="unicode")[:200],
        )
        return None

    try:
        width = int(width_str)
        height = int(height_str)
    except (ValueError, TypeError) as e:
        logger.warning(
            "Geçersiz genişlik/yükseklik değeri '%s'/'%s' (%s): %s",
            width_str,
            height_str,
            filename,
            e,
        )
        return None

    filepath = images_dir / filename

    boxes = _parse_boxes(image_elem, filename)

    return ImageAnnotation(
        filename=filename,
        filepath=filepath,
        width=width,
        height=height,
        boxes=boxes,
    )


def _parse_boxes(image_elem: ET.Element, filename: str) -> List[BoundingBox]:
    """Bir <image> elementindeki tüm <box> elementlerini parse eder.

    Args:
        image_elem: <image> XML elementi.
        filename: Hata mesajları için görüntü dosya adı.

    Returns:
        BoundingBox nesnelerinin listesi.
    """
    boxes: List[BoundingBox] = []

    for box_elem in image_elem.findall("box"):
        box = _parse_single_box(box_elem, filename)
        if box is not None:
            boxes.append(box)

    return boxes


def _parse_single_box(
    box_elem: ET.Element, filename: str
) -> BoundingBox | None:
    """Tek bir <box> XML elementini BoundingBox'a dönüştürür.

    Args:
        box_elem: <box> XML elementi.
        filename: Hata mesajları için görüntü dosya adı.

    Returns:
        BoundingBox nesnesi veya geçersiz veriyse None.
    """
    label = box_elem.get("label")
    xtl_str = box_elem.get("xtl")
    ytl_str = box_elem.get("ytl")
    xbr_str = box_elem.get("xbr")
    ybr_str = box_elem.get("ybr")

    if not all([label, xtl_str, ytl_str, xbr_str, ybr_str]):
        logger.warning(
            "Eksik box attribute'u, kutu atlanıyor (görüntü: %s): %s",
            filename,
            ET.tostring(box_elem, encoding="unicode")[:200],
        )
        return None

    try:
        xtl = float(xtl_str)
        ytl = float(ytl_str)
        xbr = float(xbr_str)
        ybr = float(ybr_str)
    except (ValueError, TypeError) as e:
        logger.warning(
            "Geçersiz koordinat değeri (görüntü: %s): %s",
            filename,
            e,
        )
        return None

    occluded_str = box_elem.get("occluded", "0")
    occluded = occluded_str == "1"

    return BoundingBox(
        xtl=xtl,
        ytl=ytl,
        xbr=xbr,
        ybr=ybr,
        label=label,
        occluded=occluded,
    )
