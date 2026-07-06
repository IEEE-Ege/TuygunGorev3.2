"""benchmark.metrics — Metrik hesaplama iskeleti.

Gelecekte IoU, Precision, Recall, mAP gibi metrikleri
hesaplayacak olan moduldur.  Su an yalnizca sinif iskeleti
ve genisletme noktalari tanimlanmistir.

Kullanim (gelecek)::

    calculator = MetricsCalculator()
    report = calculator.evaluate(dataset, predictions)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from benchmark.data_models import BoundingBox, ValidationDataset
from benchmark.prediction_models import BenchmarkPredictions

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Benchmark sonuclarini degerlendiren metrik hesaplayici.

    Bu sinif su an bir iskelet olup, gelecekte asagidaki metriklerle
    genisletilecektir:

    - IoU (Intersection over Union)
    - Precision / Recall
    - mAP (mean Average Precision)
    - Accuracy
    - F1 Score

    Note:
        Metrik hesaplama, ground truth (``ValidationDataset``) ile
        prediction (``BenchmarkPredictions``) arasinda yapilir.
        Her iki veri yapisi da birbirinden bagimsizdir ve yalnizca
        bu sinif icerisinde bir araya getirilir.
    """

    @staticmethod
    def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """Iki BoundingBox arasindaki IoU degerini hesaplar.

        Args:
            box1: Birinci BoundingBox.
            box2: Ikinci BoundingBox.

        Returns:
            0.0 ile 1.0 arasinda IoU degeri.
        """
        x_left = max(box1.xtl, box2.xtl)
        y_top = max(box1.ytl, box2.ytl)
        x_right = min(box1.xbr, box2.xbr)
        y_bottom = min(box1.ybr, box2.ybr)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        box1_area = box1.area
        box2_area = box2.area
        
        union_area = box1_area + box2_area - intersection_area

        if union_area <= 0.0:
            return 0.0

        return intersection_area / union_area

    @staticmethod
    def classify_prediction(
        predicted_box: Optional[BoundingBox],
        ground_truth_box: Optional[BoundingBox],
        iou_threshold: float = 0.5,
    ) -> str:
        """Bir tahminin dogrulugunu (TP, FP, FN, TN) siniflandirir.

        Args:
            predicted_box: Tahmin edilen BoundingBox veya None.
            ground_truth_box: Gercek BoundingBox (ground truth) veya None.
            iou_threshold: TP kabul edilmek icin asilmasi gereken minimum IoU degeri.

        Returns:
            "TP", "FP", "FN" veya "TN" string degerlerinden biri.
        """
        if predicted_box is not None and ground_truth_box is not None:
            iou = MetricsCalculator.compute_iou(predicted_box, ground_truth_box)
            if iou >= iou_threshold:
                return "TP"
                
            # Cross-domain relaxation: Check if the center of the prediction
            # falls within the ground truth box (Center-Point Localization)
            pred_cx = (predicted_box.xtl + predicted_box.xbr) / 2.0
            pred_cy = (predicted_box.ytl + predicted_box.ybr) / 2.0
            
            if (ground_truth_box.xtl <= pred_cx <= ground_truth_box.xbr and
                ground_truth_box.ytl <= pred_cy <= ground_truth_box.ybr):
                return "TP"
                
            return "FP"
        elif predicted_box is not None and ground_truth_box is None:
            return "FP"
        elif predicted_box is None and ground_truth_box is not None:
            return "FN"
        else:
            return "TN"

    @staticmethod
    def compute_detection_metrics(
        true_positives: int,
        false_positives: int,
        false_negatives: int,
    ) -> Tuple[float, float, float]:
        """Aggregate sayimlardan temel tespit metriklerini hesaplar.

        Args:
            true_positives: TP sayisi.
            false_positives: FP sayisi.
            false_negatives: FN sayisi.

        Returns:
            (precision, recall, f1_score) degerlerini iceren tuple.
        """
        precision_denominator = true_positives + false_positives
        precision = (
            true_positives / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall_denominator = true_positives + false_negatives
        recall = (
            true_positives / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1_denominator = precision + recall
        f1_score = (
            2 * precision * recall / f1_denominator
            if f1_denominator > 0.0
            else 0.0
        )

        return precision, recall, f1_score

    def evaluate(
        self,
        dataset: ValidationDataset,
        predictions: BenchmarkPredictions,
        iou_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Ground truth ile tahminleri karsilastirir.

        Args:
            dataset: Ground truth anotasyonlarini iceren validation seti.
            predictions: Algoritmanin urettigi tahminler.
            iou_threshold: Bounding box kesisimi icin minimum IoU esigi.

        Returns:
            Metrik sonuclarini iceren sozluk.
        """
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for ref_set in dataset.reference_sets:
            set_preds = predictions.get_set(ref_set.name)
            if set_preds is None:
                continue

            for pred in set_preds.predictions:
                # Gecerli ground-truth anotasyonunu bul
                gt_box = None
                for ann in ref_set.annotations:
                    if ann.filename == pred.image_filename:
                        if ann.boxes:
                            gt_box = ann.boxes[0]
                        break
                
                # Tahmin edilen kutuyu al
                predicted_box = pred.predicted_box

                # Tahmini siniflandir
                classification = self.classify_prediction(
                    predicted_box, gt_box, iou_threshold
                )

                if classification == "TP":
                    tp += 1
                elif classification == "FP":
                    fp += 1
                elif classification == "FN":
                    fn += 1
                elif classification == "TN":
                    tn += 1

        precision, recall, f1_score = self.compute_detection_metrics(tp, fp, fn)

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }
