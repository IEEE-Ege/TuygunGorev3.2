"""benchmark.engines.coarse_to_fine.matcher — Fine matching abstraction.

Defines the ``FineMatcher`` interface and its associated data contracts
that bridge **CoarseLocalizer** → **FineMatcher** → **GeometricVerifier**
in the coarse-to-fine pipeline.

Responsibility boundary:
    * **Input** : Reference ``ImageFeatures``, test ``ImageFeatures``,
      and one or more ``CandidateRegion`` objects produced by the
      ``CoarseLocalizer``.
    * **Output**: A ``MatchingResult`` per candidate containing the
      raw keypoint correspondences, confidence scores, and all
      information the downstream ``GeometricVerifier`` needs to
      estimate a geometric transform and produce a bounding box.
    * **NOT** responsible for: feature extraction, coarse
      localization, geometric verification, or bounding-box
      regression.

Design rationale:
    * **Single Responsibility** — This class only answers *"which
      keypoints in the reference correspond to which keypoints in
      the candidate region?"*.  Geometric consistency is the
      verifier's job.
    * **Open/Closed** — Different matcher backends
      (ALIKED + LightGlue, SuperPoint + LightGlue,
       XFeat + LightGlue, OmniGlue, EfficientLoFTR, …) can be
      added by subclassing without modifying existing code or this
      interface.
    * **Dependency Inversion** — ``CoarseToFineEngine`` depends on
      this abstraction, never on a concrete matcher library.
    * **Framework-agnostic** — All tensor-like fields are typed as
      ``object``.  No deep-learning library is imported.

Lifecycle managed by the owning engine::

    matcher.initialize(config)          # load detector + matcher models
    matcher.set_reference(ref_feats)    # pre-compute reference keypoints
    result = matcher.match(             # match against a candidate region
        test_features, candidate,
    )
    matcher.cleanup()                   # release models & device memory

Data flow::

    CoarseLocalizer            FineMatcher                GeometricVerifier
    ┌───────────────┐     ┌──────────────────────┐     ┌───────────────────┐
    │ Localization   │     │                      │     │                   │
    │ Result         │     │  1. crop candidate   │     │ verify()          │
    │                │     │  2. detect keypoints │     │   - RANSAC        │
    │  candidates ───┼────▶│  3. match ref ↔ test │────▶│   - inlier filter │
    │                │     │  4. score matches    │     │   - bbox estimate │
    └───────────────┘     │  5. return result    │     └───────────────────┘
                           └──────────────────────┘
                                     │
                                     ▼
                              MatchingResult
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmark.engines.coarse_to_fine.feature_extractor import ImageFeatures
from benchmark.engines.coarse_to_fine.localizer import CandidateRegion

logger = logging.getLogger(__name__)


# =====================================================================
# Data contracts
# =====================================================================


@dataclass(frozen=True)
class Keypoint:
    """A single detected interest point in an image.

    Coordinates are expressed in the **original** image pixel space
    (i.e. *before* any resizing or cropping applied by the matcher).
    The concrete ``FineMatcher`` is responsible for mapping
    detector-space coordinates back to original image space.

    This dataclass is intentionally minimal: it holds only the
    information that is universally available across all keypoint
    detectors (SuperPoint, ALIKED, XFeat, SIFT, …).  Backend-specific
    extras (e.g. scale, orientation, response) belong in ``metadata``.

    Attributes:
        x:
            Horizontal position in pixels (column index).
        y:
            Vertical position in pixels (row index).
        confidence:
            Detector confidence / response score, in ``[0.0, 1.0]``.
            ``None`` when the detector does not produce per-keypoint
            scores (rare but possible).
        descriptor:
            Feature descriptor associated with this keypoint.
            Typed as ``object`` because the concrete type and
            dimensionality depend on the detector backend.

            Expected semantics (examples):
                * SuperPoint : ``torch.Tensor`` of shape ``(256,)``
                * ALIKED     : ``torch.Tensor`` of shape ``(128,)``
                * SIFT       : ``numpy.ndarray`` of shape ``(128,)``

            ``None`` for matchers that operate without explicit
            descriptors (e.g. detector-free approaches like
            EfficientLoFTR that produce matches directly).
        metadata:
            Free-form per-keypoint diagnostics.
            Possible keys (by convention, not enforced):
                * ``"scale"``       : float — detected scale
                * ``"orientation"`` : float — dominant orientation (rad)
                * ``"response"``    : float — raw detector response
                * ``"octave"``      : int   — scale-space octave
    """

    x: float
    y: float
    confidence: Optional[float] = None
    descriptor: Optional[object] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        conf = f"{self.confidence:.3f}" if self.confidence is not None else "?"
        desc = "yes" if self.descriptor is not None else "no"
        return (
            f"Keypoint(({self.x:.1f},{self.y:.1f}), "
            f"conf={conf}, desc={desc})"
        )


@dataclass(frozen=True)
class KeypointSet:
    """An ordered collection of keypoints detected in a single image.

    Bundles the keypoints with their source image metadata so that
    downstream components (``GeometricVerifier``) can reason about
    coordinate systems without back-references to ``ImageFeatures``.

    The ``descriptors`` field provides an optional *batched*
    representation of all descriptors as a single tensor-like object.
    This is separate from the per-``Keypoint`` descriptors to support
    matchers that operate on batched descriptor matrices for
    performance.  Concrete implementations should populate **both**
    the per-keypoint and the batched representations, or document
    which one they use.

    Attributes:
        keypoints:
            Ordered list of detected keypoints.  The ordering is
            preserved across the ``reference_indices`` /
            ``test_indices`` in ``MatchingResult`` so that matches
            can be traced back unambiguously.
        image_path:
            Source image path for traceability.
        image_size_hw:
            Original ``(height, width)`` of the source image in
            pixels.  Used by the ``GeometricVerifier`` to normalise
            coordinates and clip bounding boxes.
        descriptors:
            Optional batched descriptor matrix.  Typed as ``object``
            because its concrete type depends on the backend.

            Expected semantics:
                * Type  : ``torch.Tensor`` or ``numpy.ndarray``
                * Shape : ``(num_keypoints, descriptor_dim)``

            ``None`` for detector-free matchers.
        metadata:
            Free-form diagnostics.
            Possible keys:
                * ``"detector_name"``       : str
                * ``"num_raw_detections"``   : int — before NMS / top-k
                * ``"detection_time_ms"``    : float
    """

    keypoints: List[Keypoint]
    image_path: Path
    image_size_hw: Tuple[int, int]
    descriptors: Optional[object] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------

    @property
    def num_keypoints(self) -> int:
        """Total number of detected keypoints."""
        return len(self.keypoints)

    @property
    def is_empty(self) -> bool:
        """Whether no keypoints were detected."""
        return len(self.keypoints) == 0

    @property
    def coordinates(self) -> List[Tuple[float, float]]:
        """All keypoint ``(x, y)`` pairs as a plain list.

        Useful for lightweight iteration without inspecting individual
        ``Keypoint`` objects.
        """
        return [(kp.x, kp.y) for kp in self.keypoints]

    def __repr__(self) -> str:
        img_h, img_w = self.image_size_hw
        desc = "yes" if self.descriptors is not None else "no"
        return (
            f"KeypointSet(n={self.num_keypoints}, "
            f"desc={desc}, "
            f"img={img_w}x{img_h}, "
            f"src={self.image_path.name!r})"
        )


@dataclass(frozen=True)
class MatchingResult:
    """Immutable output of a single ``FineMatcher.match()`` call.

    Encapsulates *everything* the downstream ``GeometricVerifier``
    needs to:

    1. Retrieve the matched keypoint coordinate pairs.
    2. Run RANSAC (or another robust estimator) to find inliers.
    3. Estimate a geometric transform (homography / affine).
    4. Project the reference bounding box into the test image.
    5. Compute a confidence score.

    The ``reference_indices`` and ``test_indices`` lists are
    parallel arrays of the same length.  The *i*-th match pair is::

        ref_kp = reference_keypoints.keypoints[reference_indices[i]]
        tst_kp = test_keypoints.keypoints[test_indices[i]]

    This index-based design (rather than duplicating ``Keypoint``
    objects) avoids data redundancy and allows the verifier to
    efficiently look up descriptor information when needed.

    Attributes:
        reference_keypoints:
            ``KeypointSet`` detected in the reference image (or the
            relevant crop of it).  Coordinates are in the **original**
            reference image pixel space.
        test_keypoints:
            ``KeypointSet`` detected in the candidate region of the
            test image.  Coordinates are in the **original** test
            image pixel space (not the crop-local space).
        reference_indices:
            Indices into ``reference_keypoints.keypoints`` for each
            match.  Length equals the number of matches.
        test_indices:
            Indices into ``test_keypoints.keypoints`` for each match.
            Same length as ``reference_indices``.  The *i*-th element
            corresponds to the *i*-th element in ``reference_indices``.
        match_confidences:
            Per-match confidence scores in ``[0.0, 1.0]``, parallel to
            the index arrays.  Higher means the matcher is more certain
            that the correspondence is correct.

            ``None`` when the matcher backend does not produce
            per-match scores (e.g. some mutual-nearest-neighbour
            approaches).
        candidate_region:
            The ``CandidateRegion`` that this matching was performed
            on.  Retained so the ``GeometricVerifier`` knows which
            spatial context to consider.
        overall_confidence:
            Aggregate confidence for this candidate, in ``[0.0, 1.0]``.
            Typically derived from match count, mean match confidence,
            and/or mutual-nearest-neighbour ratio.

            ``None`` when the matcher does not compute an aggregate.
        metadata:
            Free-form diagnostics for the matching call.
            Possible keys (by convention, not enforced):
                * ``"matcher_name"``       : str
                * ``"matching_time_ms"``   : float
                * ``"num_raw_matches"``    : int — before filtering
                * ``"num_filtered"``       : int — removed by ratio test
                * ``"crop_size_hw"``       : Tuple[int, int]
                * ``"detector_name"``      : str
    """

    reference_keypoints: KeypointSet
    test_keypoints: KeypointSet
    reference_indices: List[int]
    test_indices: List[int]
    candidate_region: CandidateRegion
    match_confidences: Optional[List[float]] = None
    overall_confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------

    @property
    def num_matches(self) -> int:
        """Number of putative keypoint correspondences."""
        return len(self.reference_indices)

    @property
    def has_matches(self) -> bool:
        """Whether at least one correspondence was found."""
        return self.num_matches > 0

    @property
    def matched_ref_coordinates(self) -> List[Tuple[float, float]]:
        """``(x, y)`` coordinates of matched reference keypoints.

        Convenience accessor for the ``GeometricVerifier``, which
        needs coordinate arrays to run RANSAC.
        """
        kps = self.reference_keypoints.keypoints
        return [(kps[i].x, kps[i].y) for i in self.reference_indices]

    @property
    def matched_test_coordinates(self) -> List[Tuple[float, float]]:
        """``(x, y)`` coordinates of matched test keypoints.

        Parallel to ``matched_ref_coordinates``.
        """
        kps = self.test_keypoints.keypoints
        return [(kps[i].x, kps[i].y) for i in self.test_indices]

    def __repr__(self) -> str:
        conf = (
            f"{self.overall_confidence:.3f}"
            if self.overall_confidence is not None
            else "?"
        )
        return (
            f"MatchingResult("
            f"matches={self.num_matches}, "
            f"ref_kps={self.reference_keypoints.num_keypoints}, "
            f"test_kps={self.test_keypoints.num_keypoints}, "
            f"conf={conf}, "
            f"region={self.candidate_region!r})"
        )


# =====================================================================
# Abstract matcher
# =====================================================================


class FineMatcher(ABC):
    """Abstract interface for local feature matching.

    Concrete subclasses wrap a specific keypoint detector + matcher
    combination and implement the four lifecycle methods.  The
    interface is designed to accommodate both **detect-then-match**
    pipelines (SuperPoint + LightGlue, ALIKED + LightGlue,
    XFeat + LightGlue) and **detector-free** pipelines
    (EfficientLoFTR, OmniGlue) without API changes.

    Detect-then-match pipeline::

        1. Detect keypoints + descriptors in reference image.
        2. Detect keypoints + descriptors in test crop.
        3. Match descriptors (e.g. LightGlue graph neural network).
        4. Return ``MatchingResult``.

    Detector-free pipeline::

        1. Feed reference + test crop into a monolithic model.
        2. Model directly outputs matched coordinate pairs.
        3. Wrap outputs as ``Keypoint`` / ``KeypointSet`` objects
           (descriptors may be ``None``).
        4. Return ``MatchingResult``.

    Both pipelines produce an identical ``MatchingResult`` that the
    ``GeometricVerifier`` can consume uniformly.

    Lifecycle::

        matcher.initialize(config)            # load models
        matcher.set_reference(ref_features)   # pre-compute ref keypoints
        ...
        result = matcher.match(               # per candidate region
            test_features, candidate,
        )
        ...
        matcher.cleanup()                     # release resources

    The ``set_reference`` / ``match`` split exists because reference
    keypoints are computed **once** and reused across all test images
    and all candidate regions within a ``ReferenceSet``.

    Example of a future concrete implementation::

        class LightGlueMatcher(FineMatcher):

            @property
            def name(self) -> str:
                return "ALIKED+LightGlue"

            def initialize(self, config=None):
                # load ALIKED detector, load LightGlue matcher
                ...

            def set_reference(self, reference_features):
                # detect ALIKED keypoints in reference image
                # cache for reuse across test images
                ...

            def match(self, test_features, candidate_region):
                # crop test image to candidate region
                # detect ALIKED keypoints in crop
                # run LightGlue between ref and crop keypoints
                # map crop-local coords to original image space
                # return MatchingResult(...)
                ...

            def cleanup(self):
                # del models; torch.cuda.empty_cache()
                ...
    """

    # -----------------------------------------------------------------
    # Abstract interface
    # -----------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this matcher backend.

        Should encode both the detector and the matcher when
        applicable.

        Returns:
            Backend name string, e.g. ``"ALIKED+LightGlue"``,
            ``"SuperPoint+LightGlue"``, ``"EfficientLoFTR"``.
        """

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Load detector and matcher models, prepare for inference.

        Called **once** before any ``set_reference`` or ``match`` call.
        Implementations must perform all heavy one-time setup here.

        For detect-then-match pipelines:
            * Load the keypoint detector model (e.g. ALIKED, SuperPoint).
            * Load the matcher model (e.g. LightGlue).
            * Move both to the target device.
            * Set both to evaluation mode.

        For detector-free pipelines:
            * Load the monolithic model (e.g. EfficientLoFTR).
            * Move to device and set to eval mode.

        Args:
            config:
                Backend-specific configuration dictionary.
                Expected keys (by convention, not enforced):

                * ``"detector_name"`` : str
                    Keypoint detector variant.
                    E.g. ``"aliked-n16"``, ``"superpoint"``.
                * ``"matcher_name"`` : str
                    Matcher variant.  E.g. ``"lightglue"``.
                * ``"device"`` : str
                    Target device.  E.g. ``"cuda"`` or ``"cpu"``.
                * ``"max_keypoints"`` : int
                    Maximum number of keypoints to detect per image.
                    Controls speed-accuracy trade-off.
                * ``"match_threshold"`` : float
                    Minimum confidence for a match to be retained.
                * ``"resize_max"`` : int
                    Maximum dimension to resize inputs to before
                    detection.  Larger values yield more keypoints
                    but are slower.

                ``None`` means use all defaults.

        Raises:
            RuntimeError:
                If model loading fails (weights not found, OOM, …).
        """

    @abstractmethod
    def set_reference(self, reference_features: ImageFeatures) -> None:
        """Pre-compute and cache reference keypoints.

        Called **once per reference set**, before any ``match`` calls
        for that set.  The cached reference representation is reused
        across all test images and all candidate regions.

        For detect-then-match pipelines:
            * Load the reference image from
              ``reference_features.source_image_path``.
            * Detect keypoints and extract descriptors.
            * Store the ``KeypointSet`` internally for reuse.

        For detector-free pipelines:
            * Pre-process the reference image into the format
              expected by the model.
            * Cache the pre-processed representation.

        Args:
            reference_features:
                ``ImageFeatures`` extracted from the reference image.
                The ``source_image_path`` and ``image_size_hw`` fields
                are guaranteed to be populated.

        Raises:
            FileNotFoundError:
                If the reference image file does not exist.
            RuntimeError:
                If the matcher has not been initialized.
        """

    @abstractmethod
    def match(
        self,
        test_features: ImageFeatures,
        candidate_region: CandidateRegion,
    ) -> MatchingResult:
        """Match reference keypoints against a candidate region.

        This is the core matching method.  Implementations must
        execute the following conceptual pipeline:

        1. **Crop** the test image to the ``candidate_region``
           boundaries (with optional padding).  Use
           ``test_features.source_image_path`` to load the image
           and ``test_features.image_size_hw`` for bounds checking.

        2. **Detect** keypoints in the cropped region.
           For detector-free pipelines, feed the reference + crop
           directly into the monolithic model.

        3. **Match** reference keypoints (cached by
           ``set_reference``) against the crop keypoints.

        4. **Map coordinates** from crop-local space back to the
           original test image pixel space using the crop offset
           derived from ``candidate_region``.  This ensures all
           coordinates in the returned ``MatchingResult`` are in
           the same space as the ground-truth annotations.

        5. **Assemble** a ``MatchingResult`` with the keypoint sets,
           index-based correspondences, and confidence scores.

        Args:
            test_features:
                ``ImageFeatures`` of the test image.  The
                ``source_image_path`` field points to the full test
                image (not a crop).
            candidate_region:
                ``CandidateRegion`` from the ``CoarseLocalizer``
                defining the spatial extent to search within.

        Returns:
            ``MatchingResult`` containing the putative correspondences.
            If no matches are found, ``MatchingResult`` will have
            empty index lists (``num_matches == 0``).

        Raises:
            RuntimeError:
                If ``set_reference`` has not been called, or if the
                matcher has not been initialized.

        Note:
            This method must be safe to call multiple times with
            different ``test_features`` / ``candidate_region`` pairs
            between a single ``set_reference`` / ``cleanup`` pair.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Release all models, caches, and device resources.

        Implementations must:

        * Delete detector and matcher model references.
        * Clear any cached reference keypoints.
        * Release device-side memory
          (e.g. ``torch.cuda.empty_cache()``).
        * Reset internal state so that ``initialize`` could
          theoretically be called again.

        Must be **idempotent** — safe to call multiple times.
        """

    # -----------------------------------------------------------------
    # Concrete helpers (shared by all subclasses)
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_indices_consistent(
        reference_indices: List[int],
        test_indices: List[int],
        ref_keypoints: KeypointSet,
        test_keypoints: KeypointSet,
    ) -> None:
        """Validate that match index arrays are well-formed.

        Concrete subclasses should call this before constructing a
        ``MatchingResult`` to catch indexing bugs early.

        Checks:
            * ``reference_indices`` and ``test_indices`` have equal
              length.
            * All indices are within the valid range of their
              respective ``KeypointSet``.

        Args:
            reference_indices: Indices into reference keypoints.
            test_indices: Indices into test keypoints.
            ref_keypoints: Reference ``KeypointSet``.
            test_keypoints: Test ``KeypointSet``.

        Raises:
            ValueError:
                If any consistency check fails.
        """
        if len(reference_indices) != len(test_indices):
            raise ValueError(
                f"Index arrays have different lengths: "
                f"reference={len(reference_indices)}, "
                f"test={len(test_indices)}."
            )

        n_ref = ref_keypoints.num_keypoints
        n_test = test_keypoints.num_keypoints

        for i, idx in enumerate(reference_indices):
            if not (0 <= idx < n_ref):
                raise ValueError(
                    f"reference_indices[{i}]={idx} is out of range "
                    f"[0, {n_ref})."
                )

        for i, idx in enumerate(test_indices):
            if not (0 <= idx < n_test):
                raise ValueError(
                    f"test_indices[{i}]={idx} is out of range "
                    f"[0, {n_test})."
                )

    @staticmethod
    def _validate_reference_set(reference_features: ImageFeatures) -> None:
        """Validate that reference features are usable.

        Concrete subclasses should call this at the top of
        ``set_reference()`` to fail fast.

        Args:
            reference_features: Reference ``ImageFeatures`` to check.

        Raises:
            FileNotFoundError:
                If the reference image file does not exist.
        """
        path = reference_features.source_image_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Reference image not found: {path}"
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
