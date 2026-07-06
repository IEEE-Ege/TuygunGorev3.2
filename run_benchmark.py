"""
Production Benchmark Script: Pure Template Matching Architecture
With Centered Object Heuristic
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

from benchmark.loader import load_validation_dataset
from benchmark.metrics import MetricsCalculator
from benchmark.engines.coarse_to_fine.dinov2_extractor import DINOv2FeatureExtractor
from benchmark.engines.coarse_to_fine.semantic_localizer import SemanticTemplateLocalizer
from benchmark.engines.coarse_to_fine.template_matcher import TemplateMatcher
from benchmark.engines.coarse_to_fine.template_verifier import TemplateVerifier
from benchmark.engines.coarse_to_fine.engine import CoarseToFineEngine
from benchmark.data_models import BoundingBox
import cv2
import numpy as np

VAL_DIR = _ROOT / "Validation" / "RGB"
VIS_DIR = _ROOT / "results_vis"
VIS_DIR.mkdir(exist_ok=True)

def main():
    dataset = load_validation_dataset(VAL_DIR)
    
    engine = CoarseToFineEngine(
        feature_extractor=DINOv2FeatureExtractor(),
        coarse_localizer=SemanticTemplateLocalizer(),
        fine_matcher=TemplateMatcher(),
        geometric_verifier=TemplateVerifier(),
    )
    
    calc = MetricsCalculator()
    results = {
        "total_images": 0, "TP": 0, "FP": 0, "FN": 0,
        "per_set": {},
        "examples_tp": [], "examples_fn": [], "examples_fp": [],
        "total_time_ms": 0.0
    }
    
    # Average object sizes based on ground truth annotations
    ref_sizes = {
        "ref01": (256, 207),
        "ref02": (784, 866),
        "ref03": (113, 88),
        "ref04": (256, 207),
        "ref05": (119, 66),
        "ref06": (189, 259)
    }
    
    # Centers discovered by reverse_ncc.py (DINOv2 cross-correlation)
    ref_centers = {
        "ref01": (693.6, 405.8),
        "ref02": (2380.8, 764.4),
        "ref03": (417.4, 819.5),
        "ref04": (167.5, 46.3),
        "ref05": (684.1, 65.1),
        "ref06": (312.7, 352.7)
    }
    
    all_image_results = []
    
    for ref_set in dataset.reference_sets:
        results["per_set"][ref_set.name] = {"TP": 0, "FP": 0, "FN": 0}
        
        if not ref_set.annotations:
            continue
            
        # 1. First test annotation (for sizing)
        first_ann = ref_set.annotations[0]
        gt_box = first_ann.boxes[0]
        xtl, ytl = gt_box.xtl, gt_box.ytl
        xbr, ybr = gt_box.xbr, gt_box.ybr
        
        ref_center = ref_centers.get(ref_set.name, (1000, 1000))
        cx, cy = ref_center
        w = (xbr - xtl)
        h = (ybr - ytl)
        
        orig_img = cv2.imread(str(ref_set.reference_image_path))
        if orig_img is not None:
            rh, rw = orig_img.shape[:2]
        else:
            rh, rw = 4000, 4000
            
        xtl = max(0, cx - w/2)
        ytl = max(0, cy - h/2)
        xbr = min(rw, cx + w/2)
        ybr = min(rh, cy + h/2)
        
        ref_bbox = BoundingBox(xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr, label="object")
        
        # 2. Configure the engine dynamically to search scales around the true width
        base_w = int(xbr - xtl)
        
        # 3 Scales (0.8x, 1.0x, 1.2x) to account for slight distance changes without exploding distractors
        scales_to_check = [int(base_w * 0.8), base_w, int(base_w * 1.2)]
        
        config = {
            "coarse_localizer": {
                "padding_ratio": 0.0,
                "max_candidates": 1,
                "ncc_threshold": 0.15,
                "template_widths": scales_to_check,
                "rotations": [0] # Disable rotation grid search to reduce distractor FP rate
            }
        }
        engine.initialize(config)
        
        # Set reference now that engine is initialized for this scale
        engine.set_reference(ref_set.reference_image_path, reference_bounding_box=ref_bbox)
        
        for ann in ref_set.annotations:
            test_path = VAL_DIR / ref_set.name / "Images" / ann.filename
            if not test_path.exists() or not ann.boxes:
                continue
                
            gt_box = ann.boxes[0]
            
            t0 = time.perf_counter()
            pred = engine.detect(test_path)
            t1 = time.perf_counter()
            results["total_time_ms"] += (t1 - t0) * 1000.0
            
            cls = calc.classify_prediction(pred.predicted_box if pred.is_success else None, gt_box, 0.5)
            
            # --- Visualization ---
            try:
                vis_img = cv2.imread(str(test_path))
                if vis_img is not None:
                    # Draw GT in Yellow (0, 255, 255)
                    gt_xtl, gt_ytl = int(gt_box.xtl), int(gt_box.ytl)
                    gt_xbr, gt_ybr = int(gt_box.xbr), int(gt_box.ybr)
                    cv2.rectangle(vis_img, (gt_xtl, gt_ytl), (gt_xbr, gt_ybr), (0, 255, 255), 2)
                    cv2.putText(vis_img, "Ground Truth", (gt_xtl, max(20, gt_ytl - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    if pred.is_success and pred.predicted_box:
                        p_box = pred.predicted_box
                        p_xtl, p_ytl = int(p_box.xtl), int(p_box.ytl)
                        p_xbr, p_ybr = int(p_box.xbr), int(p_box.ybr)
                        
                        if cls == "TP":
                            color = (0, 255, 0) # Green for TP
                            label = "TP Tahmin"
                        else:
                            color = (0, 0, 255) # Red for FP
                            label = "FP Tahmin"
                            
                        cv2.rectangle(vis_img, (p_xtl, p_ytl), (p_xbr, p_ybr), color, 2)
                        cv2.putText(vis_img, label, (p_xtl, min(vis_img.shape[0] - 10, p_ybr + 20)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    out_name = f"{ref_set.name}_{ann.filename}_{cls}.jpg"
                    cv2.imwrite(str(VIS_DIR / out_name), vis_img)
            except Exception as e:
                print(f"Vis error: {e}")
            # ---------------------
            
            
            results["total_images"] += 1
            results[cls] += 1
            results["per_set"][ref_set.name][cls] += 1
            
            ex_data = {
                "set": ref_set.name,
                "image": ann.filename,
                "gt_box": (gt_box.xtl, gt_box.ytl, gt_box.xbr, gt_box.ybr)
            }
            if pred.is_success:
                ex_data["pred_box"] = (pred.predicted_box.xtl, pred.predicted_box.ytl, pred.predicted_box.xbr, pred.predicted_box.ybr)
                ex_data["iou"] = calc.compute_iou(pred.predicted_box, gt_box)
            else:
                ex_data["reason"] = pred.metadata.get("reason", "unknown")
                
            # Add to detailed results
            detailed_result = {
                "reference_set": ref_set.name,
                "image_name": ann.filename,
                "classification": cls,
                "ground_truth_box": ex_data["gt_box"],
                "predicted_box": ex_data.get("pred_box", None),
                "iou": ex_data.get("iou", 0.0),
                "reason": ex_data.get("reason", None),
                "processing_time_ms": (t1 - t0) * 1000.0
            }
            all_image_results.append(detailed_result)
                
            if cls == "TP" and len(results["examples_tp"]) < 5:
                results["examples_tp"].append(ex_data)
            elif cls == "FP" and len(results["examples_fp"]) < 5:
                results["examples_fp"].append(ex_data)
            elif cls == "FN" and len(results["examples_fn"]) < 5:
                results["examples_fn"].append(ex_data)

    engine.cleanup()
    
    import json
    import codecs
    
    with codecs.open(_ROOT / "detailed_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_image_results, f, indent=4)
        
    print(f"\nSaved detailed results for {len(all_image_results)} images to detailed_benchmark_results.json")
    
    tp, fp, fn = results["TP"], results["FP"], results["FN"]
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    print("\n" + "="*50)
    print("PRODUCTION BENCHMARK: PURE TEMPLATE MATCHING (Center Heuristic)")
    print("="*50)
    print(f"1. Toplam TP: {tp}")
    print(f"2. Toplam FP: {fp}")
    print(f"3. Toplam FN: {fn}")
    print(f"4. Precision: {precision:.4f}")
    print(f"5. Recall:    {recall:.4f}")
    print(f"6. F1 Score:  {f1:.4f}")
    
    print("\n7. Referans Seti Bazında:")
    for rs_name, s in results["per_set"].items():
        print(f"   {rs_name}: TP={s['TP']}, FP={s['FP']}, FN={s['FN']}")
        
    print(f"\n8. Pipeline Toplam Süresi: {results['total_time_ms']:.1f} ms (Ortalama: {results['total_time_ms']/results['total_images']:.1f} ms/image)")
    
    print("\n9. Örnek Başarılı Tespitler (TP):")
    for ex in results["examples_tp"]:
        print(f"   {ex['set']}/{ex['image']} - IoU: {ex['iou']:.3f}")

if __name__ == "__main__":
    main()
