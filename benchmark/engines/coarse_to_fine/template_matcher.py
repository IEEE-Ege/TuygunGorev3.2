"""benchmark.engines.coarse_to_fine.template_matcher

A passthrough fine matcher that bypasses local feature matching.
It directly converts the coarse candidate region into a MatchingResult,
allowing Pure Template Matching architectures to work with the existing
pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from benchmark.engines.coarse_to_fine.feature_extractor import ImageFeatures
from benchmark.engines.coarse_to_fine.localizer import CandidateRegion
from benchmark.engines.coarse_to_fine.matcher import FineMatcher, MatchingResult, KeypointSet

logger = logging.getLogger(__name__)


class TemplateMatcher(FineMatcher):
    """A passthrough matcher for Pure Template Matching.
    
    This matcher does not extract or compare local keypoints. It simply
    accepts a CandidateRegion and returns it wrapped in a MatchingResult,
    bypassing the feature-matching bottleneck.
    """

    def __init__(self) -> None:
        self._initialized = False

    @property
    def name(self) -> str:
        return "Template-Passthrough-Matcher"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._initialized = True
        logger.info("TemplateMatcher initialized.")

    def set_reference(self, reference_features: ImageFeatures) -> None:
        if not self._initialized:
            raise RuntimeError("TemplateMatcher not initialized.")
        # Nothing to compute for reference

    def match(
        self,
        test_features: ImageFeatures,
        candidate_region: CandidateRegion,
    ) -> MatchingResult:
        if not self._initialized:
            raise RuntimeError("TemplateMatcher not initialized.")

        # Create an empty keypoint set just to satisfy the data contract
        # (VerificationResult might need test_image_path from test_keypoints).
        dummy_kpts = KeypointSet(
            keypoints=[],
            image_path=test_features.source_image_path,
            image_size_hw=test_features.image_size_hw,
            descriptors=None,
            metadata={}
        )

        return MatchingResult(
            reference_keypoints=None,
            test_keypoints=dummy_kpts,
            reference_indices=[0],  # Dummy indices to pretend there is 1 match
            test_indices=[0],       # so that downstream code doesn't crash
            candidate_region=candidate_region,
            match_confidences=[1.0],
            overall_confidence=candidate_region.confidence if candidate_region.confidence else 1.0,
            metadata={"matcher": "TemplateMatcher"}
        )

    def cleanup(self) -> None:
        self._initialized = False
