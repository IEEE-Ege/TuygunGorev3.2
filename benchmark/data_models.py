"""
benchmark.data_models
=====================

Benchmark altyapısında kullanılan tüm veri yapılarını tanımlar.

Hiyerarşi:
    BoundingBox  →  bir anotasyon kutusu (xtl, ytl, xbr, ybr + label)
    ImageAnnotation  →  tek bir görüntünün meta verisi + kutuları
    ReferenceSet  →  bir refXX klasörünün tamamı (referans görsel + test görselleri)
    ValidationDataset  →  tüm ref setlerinin koleksiyonu
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class BoundingBox:
    """Tek bir dikdörtgen anotasyon kutusu.

    CVAT XML formatındaki <box> elementine karşılık gelir.

    Attributes:
        xtl: Sol üst köşe x koordinatı (top-left x).
        ytl: Sol üst köşe y koordinatı (top-left y).
        xbr: Sağ alt köşe x koordinatı (bottom-right x).
        ybr: Sağ alt köşe y koordinatı (bottom-right y).
        label: Anotasyon etiketi (ör. "target").
        occluded: Nesne kısmen kapalı mı? (opsiyonel).
    """

    xtl: float
    ytl: float
    xbr: float
    ybr: float
    label: str
    occluded: bool = False

    @property
    def width(self) -> float:
        """Kutunun piksel genişliği."""
        return self.xbr - self.xtl

    @property
    def height(self) -> float:
        """Kutunun piksel yüksekliği."""
        return self.ybr - self.ytl

    @property
    def area(self) -> float:
        """Kutunun piksel alanı."""
        return self.width * self.height

    def __repr__(self) -> str:
        return (
            f"BoundingBox(label={self.label!r}, "
            f"xtl={self.xtl:.2f}, ytl={self.ytl:.2f}, "
            f"xbr={self.xbr:.2f}, ybr={self.ybr:.2f})"
        )


@dataclass
class ImageAnnotation:
    """Tek bir görüntüye ait anotasyon bilgisi.

    CVAT XML formatındaki <image> elementine karşılık gelir.

    Attributes:
        filename: Görüntü dosya adı (ör. "frame_024_00_23.jpg").
        filepath: Görüntü dosyasının tam yolu.
        width: Görüntü genişliği (piksel).
        height: Görüntü yüksekliği (piksel).
        boxes: Bu görüntüdeki bounding box listesi.
    """

    filename: str
    filepath: Path
    width: int
    height: int
    boxes: List[BoundingBox] = field(default_factory=list)

    @property
    def num_boxes(self) -> int:
        """Bu görüntüdeki toplam kutu sayısı."""
        return len(self.boxes)

    def __repr__(self) -> str:
        return (
            f"ImageAnnotation(filename={self.filename!r}, "
            f"size={self.width}x{self.height}, "
            f"boxes={self.num_boxes})"
        )


@dataclass
class ReferenceSet:
    """Tek bir referans nesne setini temsil eder (ör. ref01/).

    Her ref klasörü bağımsız bir validation setidir ve şunları içerir:
      - Bir referans görsel  (Referans_Nesne_XX.JPG)
      - Test görselleri      (Images/ klasörü)
      - Anotasyon dosyası    (annotations.xml)

    Attributes:
        name: Set adı (ör. "ref01").
        root_dir: Set klasörünün tam yolu.
        reference_image_path: Referans görselin tam yolu (varsa).
        images_dir: Test görsellerinin bulunduğu klasör yolu.
        annotations: İçerdiği görüntü anotasyonlarının listesi.
    """

    name: str
    root_dir: Path
    reference_image_path: Optional[Path]
    images_dir: Path
    annotations: List[ImageAnnotation] = field(default_factory=list)

    @property
    def num_images(self) -> int:
        """Setteki toplam anotasyonlu görüntü sayısı."""
        return len(self.annotations)

    def __repr__(self) -> str:
        ref_status = "✓" if self.reference_image_path else "✗"
        return (
            f"ReferenceSet(name={self.name!r}, "
            f"ref_image={ref_status}, "
            f"images={self.num_images})"
        )


@dataclass
class ValidationDataset:
    """Tüm validation setlerini bir arada tutan üst düzey konteyner.

    Attributes:
        root_dir: Validation ana klasörünün yolu.
        reference_sets: Yüklenen tüm referans set listesi.
    """

    root_dir: Path
    reference_sets: List[ReferenceSet] = field(default_factory=list)

    @property
    def num_sets(self) -> int:
        """Toplam referans set sayısı."""
        return len(self.reference_sets)

    @property
    def total_images(self) -> int:
        """Tüm setlerdeki toplam anotasyonlu görüntü sayısı."""
        return sum(rs.num_images for rs in self.reference_sets)

    def get_set(self, name: str) -> Optional[ReferenceSet]:
        """İsme göre referans seti döndürür.

        Args:
            name: Aranacak set adı (ör. "ref01").

        Returns:
            Bulunan ReferenceSet veya None.
        """
        for rs in self.reference_sets:
            if rs.name == name:
                return rs
        return None

    def summary(self) -> str:
        """Dataset'in özet istatistiklerini döndürür."""
        lines = [
            f"ValidationDataset: {self.root_dir}",
            f"  Toplam set sayısı : {self.num_sets}",
            f"  Toplam görüntü    : {self.total_images}",
            "  ─────────────────────────────────",
        ]
        for rs in self.reference_sets:
            total_boxes = sum(img.num_boxes for img in rs.annotations)
            lines.append(
                f"  {rs.name:8s} │ {rs.num_images:3d} görüntü │ "
                f"{total_boxes:3d} kutu │ "
                f"ref_img={'✓' if rs.reference_image_path else '✗'}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ValidationDataset(root={self.root_dir.name!r}, "
            f"sets={self.num_sets}, "
            f"images={self.total_images})"
        )
