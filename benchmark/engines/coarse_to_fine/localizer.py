"""benchmark.engines.coarse_to_fine.localizer — Coarse localization abstraction.

Defines the ``CoarseLocalizer`` interface and the ``LocalizationResult``
data contract that bridges the **FeatureExtractor** → **Localizer** →
**FineMatcher** pipeline.

Responsibility boundary:
    * **Input** : Two ``ImageFeatures`` objects (reference + test)
      produced by a ``FeatureExtractor``.
    * **Output**: A ``LocalizationResult`` describing zero or more
      candidate regions where the reference object may exist in the
      test image.
    * **NOT** responsible for: feature extraction, local keypoint
      matching, geometric verification, or bounding-box regression.

Design rationale:
    * **Single Responsibility** — This class only answers *"where in
      the test image should the fine matcher look?"*.  It does not
      decide *if* the object is truly there (that is the verifier's
      job).
    * **Open/Closed** — Different localization strategies (cosine
      heatmap, cross-attention, sliding-window, …) can be added by
      subclassing without modifying existing code.
    * **Dependency Inversion** — ``CoarseToFineEngine`` depends on
      this abstraction, never on a concrete similarity algorithm.
    * **Framework-agnostic** — All tensor-like fields are typed as
      ``object``.  No deep-learning library is imported.

Lifecycle managed by the owning engine::

    localizer.initialize(config)
    result = localizer.localize(ref_features, test_features)
    localizer.cleanup()

Data flow::

    FeatureExtractor          CoarseLocalizer            FineMatcher
    ┌──────────────┐     ┌─────────────────────┐     ┌─────────────┐
    │ ImageFeatures │────▶│ localize()          │────▶│ match()     │
    │ (reference)   │     │                     │     │             │
    │               │     │  1. compare patches │     │ operates on │
    │ ImageFeatures │────▶│  2. build heatmap   │     │ candidate   │
    │ (test)        │     │  3. extract regions │     │ regions     │
    └──────────────┘     │  4. return result   │     └─────────────┘
                          └─────────────────────┘
                                    │
                                    ▼
                          LocalizationResult
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmark.engines.coarse_to_fine.feature_extractor import ImageFeatures

logger = logging.getLogger(__name__)


# =====================================================================
# Data contracts
# =====================================================================


@dataclass(frozen=True)
class CandidateRegion:
    """A single rectangular region of interest in the test image.

    Coordinates are expressed in the **original** test image space
    (i.e. *before* any resizing applied during feature extraction).
    The ``CoarseLocalizer`` is responsible for performing the
    coordinate mapping from heatmap space back to image space using
    ``ImageFeatures.image_size_hw`` and ``ImageFeatures.patch_grid_shape``.

    Attributes:
        x_min:
            Left edge of the region in pixels (inclusive).
        y_min:
            Top edge of the region in pixels (inclusive).
        x_max:
            Right edge of the region in pixels (exclusive).
        y_max:
            Bottom edge of the region in pixels (exclusive).
        confidence:
            Localization confidence for this region, in ``[0.0, 1.0]``.
            Derived from the aggregated similarity within the region.
            Higher values indicate stronger evidence that the
            reference object is present in this area.

            ``None`` when the localization backend does not produce a
            meaningful per-region score.
        metadata:
            Free-form region-level diagnostics.
            Possible keys (by convention, not enforced):
                * ``"peak_similarity"`` : float — max similarity in region
                * ``"mean_similarity"`` : float — average similarity
                * ``"area_pixels"``     : int   — region area
                * ``"rank"``            : int   — 0-based rank by confidence
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------

    @property
    def width(self) -> float:
        """Region width in pixels."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Region height in pixels."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Region area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Region center as ``(cx, cy)``."""
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )

    def __repr__(self) -> str:
        conf = f"{self.confidence:.3f}" if self.confidence is not None else "?"
        return (
            f"CandidateRegion("
            f"({self.x_min:.0f},{self.y_min:.0f})"
            f"->({self.x_max:.0f},{self.y_max:.0f}), "
            f"conf={conf})"
        )


@dataclass(frozen=True)
class LocalizationResult:
    """Immutable output of a single ``CoarseLocalizer.localize()`` call.

    Encapsulates everything the downstream ``FineMatcher`` needs to
    know about the coarse localization stage without exposing any
    implementation details of the similarity algorithm used.

    Attributes:
        candidates:
            Zero or more candidate regions, sorted by descending
            confidence (best first).

            * **Empty list** signals that no plausible region was
              found — the ``CoarseToFineEngine`` should produce a
              ``MatchResult.NO_MATCH`` prediction without invoking
              the ``FineMatcher``.
            * **One entry** is the common case when the localizer is
              confident about a single location.
            * **Multiple entries** occur when the localizer is
              uncertain and wants the ``FineMatcher`` to evaluate
              several hypotheses (e.g. repeated texture, multiple
              similar objects).

        global_similarity:
            Optional scalar similarity between the reference and test
            image-level descriptors (``ImageFeatures.global_features``).
            In ``[0.0, 1.0]``.

            Useful as a fast pre-filter: if the global similarity is
            below a configured threshold, the engine may skip the
            more expensive patch-level comparison entirely.

            ``None`` when global features are unavailable or the
            localization backend does not compute this score.

        heatmap:
            Optional raw similarity heatmap kept for debugging and
            visualisation.  Typed as ``object`` because its concrete
            type depends on the backend (e.g. ``numpy.ndarray`` of
            shape ``(grid_h, grid_w)``).

            * Implementations that wish to save memory may set this
              to ``None`` in production mode and only populate it
              when a ``"debug"`` flag is set in the config.
            * The ``CoarseToFineEngine`` never reads this field; it
              exists solely for offline analysis tools.

        heatmap_shape:
            Spatial shape ``(height, width)`` of ``heatmap`` when it
            is populated.  Stored separately so that downstream code
            can reason about the heatmap grid even when the raw
            heatmap tensor is not retained.

            ``None`` when ``heatmap`` is ``None``.

        test_image_path:
            Path of the test image that was localized.  Retained for
            traceability and logging.

        test_image_size_hw:
            Original ``(height, width)`` of the test image in pixels.
            Copied from ``ImageFeatures.image_size_hw`` so that
            downstream components do not need a back-reference to
            the raw features.

        metadata:
            Free-form diagnostics for the entire localization call.
            Possible keys (by convention, not enforced):
                * ``"localization_time_ms"`` : float
                * ``"num_patches_compared"`` : int
                * ``"threshold_used"``       : float
                * ``"algorithm"``            : str
    """

    candidates: List[CandidateRegion]
    test_image_path: Path
    test_image_size_hw: Tuple[int, int]
    global_similarity: Optional[float] = None
    heatmap: Optional[object] = None
    heatmap_shape: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------

    @property
    def has_candidates(self) -> bool:
        """Whether at least one candidate region was found."""
        return len(self.candidates) > 0

    @property
    def num_candidates(self) -> int:
        """Number of candidate regions."""
        return len(self.candidates)

    @property
    def best_candidate(self) -> Optional[CandidateRegion]:
        """Highest-confidence candidate, or ``None`` if empty.

        Candidates are expected to be sorted by descending confidence,
        so this is simply the first element.
        """
        return self.candidates[0] if self.candidates else None

    @property
    def has_heatmap(self) -> bool:
        """Whether the raw heatmap tensor was retained."""
        return self.heatmap is not None

    def __repr__(self) -> str:
        gs = f"{self.global_similarity:.3f}" if self.global_similarity is not None else "n/a"
        hm = f"{self.heatmap_shape[0]}x{self.heatmap_shape[1]}" if self.heatmap_shape else "n/a"
        img_h, img_w = self.test_image_size_hw
        return (
            f"LocalizationResult("
            f"candidates={self.num_candidates}, "
            f"global_sim={gs}, "
            f"heatmap={hm}, "
            f"img={img_w}x{img_h}, "
            f"src={self.test_image_path.name!r})"
        )


# =====================================================================
# Abstract localizer
# =====================================================================


class CoarseLocalizer(ABC):
    """Abstract interface for coarse localization.

    Concrete subclasses implement a specific strategy for comparing
    reference and test ``ImageFeatures`` and estimating one or more
    candidate regions in the test image.

    Possible strategies (non-exhaustive):
        * **Patch cosine-similarity heatmap** — compute per-patch
          cosine similarity, threshold, extract connected components.
        * **Cross-attention** — use transformer cross-attention
          between reference and test patch tokens.
        * **Sliding-window embedding distance** — slide a window
          over the test feature grid and score each position.
        * **Hierarchical / multi-scale** — operate at multiple
          resolutions and fuse results.

    Lifecycle::

        localizer.initialize(config)   # configure thresholds, etc.
        ...
        result = localizer.localize(ref_feats, test_feats)
        ...
        localizer.cleanup()            # release resources

    The owning ``CoarseToFineEngine`` is responsible for calling
    these methods in order.

    Example of a future concrete implementation::

        class CosineHeatmapLocalizer(CoarseLocalizer):

            @property
            def name(self) -> str:
                return "CosineHeatmap"

            def initialize(self, config=None):
                self._threshold = (config or {}).get("threshold", 0.4)

            def localize(self, ref_features, test_features):
                # 1. cosine similarity between patch grids
                # 2. reshape to spatial heatmap
                # 3. threshold + connected components
                # 4. extract bounding regions
                # 5. map to original image coordinates
                # 6. return LocalizationResult(...)
                ...

            def cleanup(self):
                pass  # stateless aside from config
    """

    # -----------------------------------------------------------------
    # Abstract interface
    # -----------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this localization strategy.

        Returns:
            Strategy name string, e.g. ``"CosineHeatmap"``.
            Used in logging and diagnostics only.
        """

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Configure the localizer for a benchmark run.

        Called **once** before any ``localize`` call.  Implementations
        should store algorithm-specific parameters here.

        Unlike ``FeatureExtractor.initialize``, this method typically
        does **not** load heavy models — coarse localization often
        uses lightweight numerical operations on pre-extracted features.

        Args:
            config:
                Strategy-specific configuration dictionary.
                Expected keys (by convention, not enforced):

                * ``"similarity_threshold"`` : float
                    Minimum similarity value for a heatmap cell to be
                    considered part of a candidate region.
                    Default depends on the strategy.
                * ``"min_region_area_ratio"`` : float
                    Minimum candidate region area as a fraction of the
                    total image area.  Regions smaller than this are
                    discarded.  Prevents noise from producing tiny
                    false-positive regions.
                * ``"max_candidates"`` : int
                    Maximum number of candidate regions to return.
                    Default ``1`` for single-object scenarios.
                * ``"padding_ratio"`` : float
                    Factor by which to expand each candidate region
                    to give the ``FineMatcher`` more spatial context.
                    E.g. ``0.2`` → expand by 20 % on each side.
                * ``"retain_heatmap"`` : bool
                    If ``True``, store the raw heatmap tensor in
                    ``LocalizationResult.heatmap`` for debugging.
                    Default ``False`` in production.

                ``None`` means use all defaults.
        """

    @abstractmethod
    def localize(
        self,
        reference_features: ImageFeatures,
        test_features: ImageFeatures,
    ) -> LocalizationResult:
        """Estimate candidate regions in the test image.

        This is the core localization method.  Implementations must
        execute the following conceptual pipeline (the exact algorithm
        varies by subclass):

        1. **Compare representations** — Compute a dense similarity
           between ``reference_features.patch_features`` and
           ``test_features.patch_features``.  The specific metric
           (cosine, dot-product, learned distance, …) is a subclass
           decision.

        2. **Build spatial similarity map** — Reshape the raw
           similarity scores into a 2-D grid using
           ``test_features.patch_grid_shape`` to recover spatial
           structure.  Optionally normalise or smooth the map.

        3. **Extract candidate regions** — Identify contiguous
           high-similarity areas.  Typical approaches include
           thresholding + connected-component labelling, peak finding,
           or top-k pooling.  Each area becomes a
           ``CandidateRegion``.

        4. **Map to original image coordinates** — Convert region
           boundaries from heatmap (patch-grid) space to the original
           test image pixel space using
           ``test_features.image_size_hw`` and
           ``test_features.patch_grid_shape``.  Optionally apply
           padding so the ``FineMatcher`` receives surrounding
           context.

        5. **Assemble result** — Package everything into a
           ``LocalizationResult``, optionally including the raw
           heatmap and a global-similarity pre-filter score.

        Optionally, before step 1, an implementation may compute a
        fast **global similarity** between
        ``reference_features.global_features`` and
        ``test_features.global_features`` and short-circuit with an
        empty ``LocalizationResult`` if the score is below a
        configured threshold.

        Args:
            reference_features:
                ``ImageFeatures`` extracted from the reference image.
                Remains constant across all test images in a
                ``ReferenceSet``.
            test_features:
                ``ImageFeatures`` extracted from the current test
                image.

        Returns:
            ``LocalizationResult`` with zero or more candidates.

        Raises:
            RuntimeError:
                If the localizer has not been initialized, or if the
                feature dimensions are incompatible.

        Note:
            This method must be **stateless** with respect to previous
            calls: each invocation depends only on the two
            ``ImageFeatures`` arguments and the configuration set
            during ``initialize``.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Release any resources held by the localizer.

        For most localization strategies this is a no-op because
        the heavy state lives in ``FeatureExtractor``.  However,
        subclasses that cache intermediate structures (e.g. a
        pre-computed reference projection) should free them here.

        Must be **idempotent** — safe to call multiple times.
        """

    # -----------------------------------------------------------------
    # Concrete helpers (shared by all subclasses)
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_features_compatible(
        reference: ImageFeatures,
        test: ImageFeatures,
    ) -> None:
        """Check that reference and test features are dimensionally compatible.

        Concrete subclasses should call this at the top of
        ``localize()`` to fail fast with a clear error message.

        The check verifies that both ``ImageFeatures`` objects
        have the same ``patch_grid_shape[0] * patch_grid_shape[1]``
        grid resolution proportionality — specifically, that they
        were produced by the **same model with the same patch size**.
        The actual grid dimensions may differ (reference and test
        images can have different aspect ratios), but the embedding
        dimensionality must match.

        Args:
            reference: Reference ``ImageFeatures``.
            test: Test ``ImageFeatures``.

        Raises:
            ValueError:
                If a basic compatibility check fails.

        Note:
            This method performs only structural checks that are
            possible without inspecting the actual tensor contents.
            It cannot verify embedding dimensionality without
            importing a tensor library, so that check is deferred
            to the concrete subclass.
        """
        if reference.patch_features is None:
            raise ValueError(
                "Reference ImageFeatures has no patch_features."
            )
        if test.patch_features is None:
            raise ValueError(
                "Test ImageFeatures has no patch_features."
            )
        if reference.patch_grid_shape == (0, 0):
            raise ValueError(
                "Reference patch_grid_shape is (0, 0) — likely uninitialised."
            )
        if test.patch_grid_shape == (0, 0):
            raise ValueError(
                "Test patch_grid_shape is (0, 0) — likely uninitialised."
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
