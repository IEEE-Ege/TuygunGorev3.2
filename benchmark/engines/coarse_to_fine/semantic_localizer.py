"""benchmark.engines.coarse_to_fine.semantic_localizer — DINOv2 Dense Semantic NCC Localizer.

Implements a CoarseLocalizer that uses DINOv2 patch features instead of 
pixel intensities. This provides high robustness to rotation, scaling, 
lighting changes, and background distractors.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from benchmark.engines.coarse_to_fine.feature_extractor import ImageFeatures
from benchmark.engines.coarse_to_fine.localizer import (
    CandidateRegion,
    CoarseLocalizer,
    LocalizationResult,
)
from benchmark.data_models import BoundingBox

logger = logging.getLogger(__name__)

class SemanticTemplateLocalizer(CoarseLocalizer):
    """Localizer using DINOv2 Dense Cross-Correlation (Semantic NCC)."""
    
    def __init__(self) -> None:
        self._max_candidates: int = 1
        self._threshold: float = 0.2
        self._template_widths: List[int] = []
        self._initialized: bool = False

    @property
    def name(self) -> str:
        if not self._initialized:
            return "SemanticTemplateLocalizer-uninitialized"
        return "SemanticTemplateLocalizer"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._max_candidates = cfg.get("max_candidates", 1)
        self._threshold = cfg.get("ncc_threshold", 0.2)
        # We will use template_widths to mean the expected object width in the *original* image pixels
        # But we will convert it to patch space internally.
        self._template_widths = cfg.get("template_widths", [])
        self._rotations = cfg.get("rotations", [0])
        
        self._initialized = True
        logger.info("SemanticTemplateLocalizer initialized.")

    def localize(
        self, reference_features: ImageFeatures, test_features: ImageFeatures
    ) -> LocalizationResult:
        if not self._initialized:
            raise RuntimeError("SemanticTemplateLocalizer not initialized.")
            
        t0 = time.perf_counter()
        
        # 1. Reconstruct dense features (N, C) -> (1, C, H, W)
        C = 384
        ref_h, ref_w = reference_features.patch_grid_shape
        ref_dense = reference_features.patch_features.view(ref_h, ref_w, C).permute(2, 0, 1).unsqueeze(0)
        
        test_h, test_w = test_features.patch_grid_shape
        test_dense = test_features.patch_features.view(test_h, test_w, C).permute(2, 0, 1).unsqueeze(0)
        test_dense = F.normalize(test_dense, p=2, dim=1)
        
        # 2. Extract reference template
        bbox = reference_features.metadata.get("object_bounding_box")
        if not bbox:
            return LocalizationResult(
                candidates=[],
                test_image_path=test_features.source_image_path,
                test_image_size_hw=test_features.image_size_hw,
                metadata={"reason": "no_bounding_box", "time_ms": 0.0}
            )
            
        orig_ref_w, orig_ref_h = reference_features.image_size_hw[1], reference_features.image_size_hw[0]
        
        xtl_g = max(0, int(bbox.xtl / orig_ref_w * ref_w))
        ytl_g = max(0, int(bbox.ytl / orig_ref_h * ref_h))
        xbr_g = min(ref_w, int(bbox.xbr / orig_ref_w * ref_w) + 1)
        ybr_g = min(ref_h, int(bbox.ybr / orig_ref_h * ref_h) + 1)
        
        template = ref_dense[:, :, ytl_g:ybr_g, xtl_g:xbr_g]
        
        if template.shape[2] == 0 or template.shape[3] == 0:
            return LocalizationResult(
                candidates=[],
                test_image_path=test_features.source_image_path,
                test_image_size_hw=test_features.image_size_hw,
                metadata={"reason": "empty_template", "time_ms": 0.0}
            )
            
        template = F.normalize(template, p=2, dim=1)
        
        # 3. Handle scales and rotations.
        orig_test_w, orig_test_h = test_features.image_size_hw[1], test_features.image_size_hw[0]
        
        candidates = []
        best_score = -1.0
        best_pred = None
        
        import torchvision.transforms.functional as TF
        
        scales_to_test = self._template_widths if self._template_widths else [bbox.xbr - bbox.xtl]
        
        for exp_w in scales_to_test:
            exp_h = (bbox.ybr - bbox.ytl) * (exp_w / (bbox.xbr - bbox.xtl)) if (bbox.xbr - bbox.xtl) > 0 else exp_w
            
            # Map expected size to test grid
            tw_patches = max(1, int(exp_w / orig_test_w * test_w))
            th_patches = max(1, int(exp_h / orig_test_h * test_h))
            
            # Interpolate template to target size
            scaled_template = F.interpolate(
                template, size=(th_patches, tw_patches), mode='bilinear', align_corners=False
            )
            
            for angle in self._rotations:
                if angle != 0:
                    rotated_template = TF.rotate(scaled_template, angle, expand=True)
                else:
                    rotated_template = scaled_template
                    
                rotated_template = F.normalize(rotated_template, p=2, dim=1)
                
                # Cross correlation
                new_th, new_tw = rotated_template.shape[2], rotated_template.shape[3]
                if new_th > test_h or new_tw > test_w:
                    continue # Template larger than image
                    
                # We divide by th_patches * tw_patches to not penalize rotation expansion zero-padding
                corr = F.conv2d(test_dense, rotated_template) / (th_patches * tw_patches)
                
                max_val = corr.max().item()
                if max_val > best_score:
                    best_score = max_val
                    
                    max_idx = corr.argmax().item()
                    out_w = corr.shape[3]
                    max_y = max_idx // out_w
                    max_x = max_idx % out_w
                    
                    # Center of the match in grid coords
                    center_x_g = max_x + new_tw / 2.0
                    center_y_g = max_y + new_th / 2.0
                    
                    # Center of the match in orig coords
                    pred_cx = center_x_g / test_w * orig_test_w
                    pred_cy = center_y_g / test_h * orig_test_h
                    
                    # Depending on angle, the bounding box might need to expand or swap dimensions
                    import math
                    rad = math.radians(angle)
                    cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
                    
                    rot_w = exp_w * cos_a + exp_h * sin_a
                    rot_h = exp_w * sin_a + exp_h * cos_a
                    
                    pred_xtl = pred_cx - rot_w / 2
                    pred_ytl = pred_cy - rot_h / 2
                    
                    best_pred = CandidateRegion(
                        x_min=pred_xtl, 
                        y_min=pred_ytl, 
                        x_max=pred_xtl + rot_w, 
                        y_max=pred_ytl + rot_h,
                        confidence=max_val,
                        metadata={"scale": exp_w, "rotation": angle}
                    )
        
        if best_pred and best_score >= self._threshold:
            candidates.append(best_pred)
            
        time_ms = (time.perf_counter() - t0) * 1000.0
        
        return LocalizationResult(
            candidates=candidates,
            test_image_path=test_features.source_image_path,
            test_image_size_hw=test_features.image_size_hw,
            metadata={"time_ms": time_ms, "best_score": best_score}
        )

    def cleanup(self) -> None:
        self._initialized = False
        logger.info("SemanticTemplateLocalizer cleaned up.")
