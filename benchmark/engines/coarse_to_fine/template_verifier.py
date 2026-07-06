"""benchmark.engines.coarse_to_fine.template_verifier

A passthrough geometric verifier for Pure Template Matching.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from benchmark.engines.coarse_to_fine.verifier import GeometricVerifier, VerificationResult
from benchmark.engines.coarse_to_fine.matcher import MatchingResult
from benchmark.data_models import BoundingBox

logger = logging.getLogger(__name__)


class TemplateVerifier(GeometricVerifier):
    """A passthrough verifier for Pure Template Matching.
    
    This verifier bypasses RANSAC and Homography estimation.
    It simply takes the candidate_region from the MatchingResult
    and converts it directly into a verified BoundingBox, preserving
    the localization confidence as the final verification confidence.
    """

    def __init__(self) -> None:
        self._initialized = False

    @property
    def name(self) -> str:
        return "Template-Passthrough-Verifier"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._initialized = True
        logger.info("TemplateVerifier initialized.")

    def verify(
        self,
        matching_result: MatchingResult,
        reference_bounding_box: Optional[BoundingBox] = None,
    ) -> VerificationResult:
        if not self._initialized:
            raise RuntimeError("TemplateVerifier not initialized.")

        cand = matching_result.candidate_region
        
        # In pure template matching, the candidate region itself is the bounding box
        box = BoundingBox(
            xtl=cand.x_min,
            ytl=cand.y_min,
            xbr=cand.x_max,
            ybr=cand.y_max,
            label="object"
        )
        
        return VerificationResult(
            success=True,
            candidate_region=cand,
            test_image_path=matching_result.test_keypoints.image_path if matching_result.test_keypoints else None,
            test_image_size_hw=matching_result.test_keypoints.image_size_hw if matching_result.test_keypoints else (0,0),
            verified_bounding_box=box,
            homography=None,
            inlier_mask=None,
            num_inliers=1,
            num_outliers=0,
            confidence=matching_result.overall_confidence,
            mean_reprojection_error=0.0,
            metadata={"verifier": "TemplateVerifier"}
        )

    def cleanup(self) -> None:
        self._initialized = False
