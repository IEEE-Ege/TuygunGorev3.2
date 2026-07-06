"""benchmark.engines.coarse_to_fine.verifier — Geometric verification abstraction.

Defines the ``GeometricVerifier`` interface and the ``VerificationResult``
data contract that terminates the **FeatureExtractor** -> **CoarseLocalizer**
-> **FineMatcher** -> **GeometricVerifier** pipeline.

Responsibility boundary:
    * **Input** : A single ``MatchingResult`` produced by a ``FineMatcher``
      — i.e. a set of putative keypoint correspondences between the
      reference image and one candidate region of the test image.
    * **Output**: A ``VerificationResult`` describing whether those
      correspondences are geometrically consistent, together with the
      estimated transform, the inlier/outlier split, and — when
      verification succeeds — the reference bounding box projected into
      test-image coordinates.
    * **NOT** responsible for: feature extraction, coarse localization,
      keypoint detection/matching, or computing evaluation metrics
      (IoU, precision, recall, ...). Those belong to ``FeatureExtractor``,
      ``CoarseLocalizer``, ``FineMatcher``, and ``benchmark.metrics``
      respectively.

Why geometric verification is its own pipeline stage
------------------------------------------------------
A ``FineMatcher`` (e.g. ALIKED + LightGlue) produces *putative*
correspondences: pairs of keypoints that look similar in descriptor
space. Descriptor similarity alone is not proof that two points
correspond to the same physical location — repetitive textures,
symmetric objects, and background clutter routinely produce confident
but *geometrically inconsistent* matches (outliers).

Geometric verification asks a different, stricter question: *"is there
a single, consistent geometric transform that explains most of these
correspondences simultaneously?"* This is precisely the separation of
concerns that keeps ``FineMatcher`` and ``GeometricVerifier`` as
distinct abstractions:

    * ``FineMatcher``       -> appearance-based correspondence search.
    * ``GeometricVerifier`` -> structure-based correspondence filtering
                                 and pose/transform recovery.

Mixing the two into a single component would violate the Single
Responsibility Principle and would make it impossible to swap, say,
LightGlue for OmniGlue without also touching the RANSAC logic, or to
swap OpenCV RANSAC for MAGSAC++ without touching the matcher.

Background concepts (for future implementors)
------------------------------------------------
RANSAC (Random Sample Consensus)
    A robust estimation algorithm for fitting a model (here, a
    geometric transform) to data that contains outliers. It repeatedly:

    1. Samples a minimal random subset of correspondences needed to fit
       the model (e.g. 4 point pairs for a homography).
    2. Fits the model to that subset.
    3. Counts how many *other* correspondences agree with the fitted
       model within a tolerance (the *inliers*).
    4. Keeps the model with the largest inlier set after many trials
       (or the first one that exceeds a confidence threshold).

    The correspondences that agree with the winning model are the
    *inliers*; everything else is discarded as an *outlier*. RANSAC is
    a strategy, not a specific transform — the same procedure is used
    to fit homographies, affine transforms, fundamental matrices, etc.

Homography estimation
    A homography is a 3x3 projective transform that maps points on one
    plane to points on another plane (up to scale). It is the standard
    model used here because the reference object is treated as a
    roughly planar patch being viewed from a possibly different
    viewpoint/scale/rotation in the test frame. Given >= 4 inlier
    correspondences, a homography can be estimated (typically via
    Direct Linear Transform, refined with RANSAC to reject outliers).
    Concrete implementations may fall back to a simpler affine or
    similarity transform when the correspondence count or geometry is
    insufficient for a stable homography.

Geometric consistency
    The general property that a set of correspondences agrees with a
    single, physically-plausible transform. This is what separates a
    true detection (many correspondences agreeing on one consistent
    mapping) from a coincidental cluster of descriptor-similar points
    that do not actually share a spatial relationship. Reprojection
    error — the pixel distance between a transformed reference point
    and its matched test point — is the usual per-correspondence
    measure of geometric consistency, and its aggregate (e.g. mean
    inlier reprojection error) is a common confidence signal.

Framework independence
    Exactly as in ``feature_extractor.py``, ``localizer.py``, and
    ``matcher.py``, this module must remain importable without any
    deep-learning or computer-vision library installed. All
    transform/matrix/mask-like fields are therefore typed as
    ``object``; concrete subclasses document their real types
    (e.g. ``numpy.ndarray`` for an OpenCV-based implementation).

Lifecycle managed by the owning engine::

    verifier.initialize(config)
    result = verifier.verify(matching_result, reference_bbox)
    verifier.cleanup()

Data flow::

    FineMatcher                 GeometricVerifier
    ┌──────────────────┐   ┌───────────────────────────┐
    │ MatchingResult    │   │ verify()                  │
    │                    │──▶│  1. RANSAC model fit      │
    │  correspondences   │   │  2. inlier / outlier split│
    │  + candidate_region│   │  3. bbox projection       │
    └──────────────────┘   │  4. confidence scoring    │
                            └───────────────────────────┘
                                        │
                                        ▼
                              VerificationResult
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from benchmark.data_models import BoundingBox
from benchmark.engines.coarse_to_fine.localizer import CandidateRegion
from benchmark.engines.coarse_to_fine.matcher import MatchingResult

logger = logging.getLogger(__name__)


# =====================================================================
# Data contracts
# =====================================================================


@dataclass(frozen=True)
class VerificationResult:
    """Immutable output of a single ``GeometricVerifier.verify()`` call.

    Encapsulates everything the ``CoarseToFineEngine`` needs to turn a
    (possibly noisy) set of keypoint correspondences into a final
    ``DetectionPrediction`` — without the verifier needing to know
    anything about ``BenchmarkRunner`` or ``MetricsCalculator``.

    Attributes:
        success:
            Whether geometric verification succeeded. ``True`` means a
            sufficiently consistent transform was found and
            ``verified_bounding_box`` is populated. ``False`` means the
            correspondences were rejected as geometrically inconsistent
            (e.g. too few inliers, degenerate point configuration, or
            RANSAC failed to converge). Callers should treat a
            ``False`` result the same way as ``MatchResult.NO_MATCH`` —
            not as an error.

        verified_bounding_box:
            The reference object's bounding box, projected into test
            image pixel space via the estimated transform, and clipped
            to the test image bounds. ``None`` when ``success`` is
            ``False``.

        homography:
            The estimated geometric transform (typically a 3x3
            homography, but a concrete implementation may substitute
            an affine or similarity matrix for degenerate cases).
            Typed as ``object`` because this module must not import a
            linear algebra library.

            Expected semantics (OpenCV example):
                * Type  : ``numpy.ndarray``
                * Shape : ``(3, 3)``

            ``None`` when ``success`` is ``False``.

        inlier_mask:
            A boolean mask, parallel to
            ``matching_result.reference_indices`` /
            ``matching_result.test_indices``, indicating which
            putative correspondences were classified as inliers by the
            robust estimator. Typed as ``object`` for the same reason
            as ``homography``.

            Expected semantics (OpenCV example):
                * Type  : ``numpy.ndarray``
                * Shape : ``(num_matches,)``
                * Dtype : ``bool`` (or ``uint8`` mapped to bool)

            ``None`` when verification could not run at all (e.g. zero
            input correspondences).

        num_inliers:
            Number of correspondences classified as inliers. ``0`` when
            ``inlier_mask`` is ``None`` or ``success`` is ``False``.

        num_outliers:
            Number of correspondences classified as outliers. Equal to
            ``matching_result.num_matches - num_inliers`` whenever
            ``inlier_mask`` is populated.

        confidence:
            Aggregate verification confidence in ``[0.0, 1.0]``.
            Concrete implementations typically derive this from
            ``inlier_ratio``, ``num_inliers``, and
            ``mean_reprojection_error`` combined into a single score.
            ``None`` when it cannot be meaningfully computed.

        mean_reprojection_error:
            Average pixel distance between transformed reference
            inlier points and their matched test points, under the
            estimated transform. Lower is better. ``None`` when
            verification failed or the concrete backend does not
            compute it.

        candidate_region:
            The ``CandidateRegion`` this verification was performed
            on. Carried through from the input ``MatchingResult`` so
            that downstream code does not need a back-reference.

        test_image_path:
            Path of the test image that was verified. Retained for
            traceability and logging, mirroring the convention used by
            ``LocalizationResult`` and ``MatchingResult``.

        test_image_size_hw:
            Original ``(height, width)`` of the test image in pixels.
            Used to confirm that ``verified_bounding_box`` was clipped
            to valid image bounds.

        metadata:
            Free-form diagnostics for the verification call.
            Possible keys (by convention, not enforced):
                * ``"verifier_name"``         : str
                * ``"verification_time_ms"``   : float
                * ``"model_type"``             : str — e.g. ``"homography"``, ``"affine"``
                * ``"ransac_reproj_threshold"``: float
                * ``"ransac_confidence"``      : float
                * ``"ransac_max_iters"``       : int
                * ``"failure_reason"``         : str — populated when ``success`` is ``False``
    """

    success: bool
    candidate_region: CandidateRegion
    test_image_path: Path
    test_image_size_hw: Tuple[int, int]
    verified_bounding_box: Optional[BoundingBox] = None
    homography: Optional[object] = None
    inlier_mask: Optional[object] = None
    num_inliers: int = 0
    num_outliers: int = 0
    confidence: Optional[float] = None
    mean_reprojection_error: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------

    @property
    def num_correspondences_considered(self) -> int:
        """Total correspondences evaluated (inliers + outliers)."""
        return self.num_inliers + self.num_outliers

    @property
    def inlier_ratio(self) -> Optional[float]:
        """Fraction of considered correspondences classified as inliers.

        Returns:
            ``num_inliers / num_correspondences_considered``, or
            ``None`` when no correspondences were considered (avoids a
            division by zero for empty-match verification attempts).
        """
        total = self.num_correspondences_considered
        if total == 0:
            return None
        return self.num_inliers / total

    @property
    def has_verified_box(self) -> bool:
        """Whether a projected bounding box is available."""
        return self.verified_bounding_box is not None

    def __repr__(self) -> str:
        conf = f"{self.confidence:.3f}" if self.confidence is not None else "?"
        ratio = self.inlier_ratio
        ratio_str = f"{ratio:.2f}" if ratio is not None else "?"
        return (
            f"VerificationResult("
            f"success={self.success}, "
            f"inliers={self.num_inliers}/{self.num_correspondences_considered} "
            f"({ratio_str}), "
            f"conf={conf}, "
            f"src={self.test_image_path.name!r})"
        )


# =====================================================================
# Abstract verifier
# =====================================================================


class GeometricVerifier(ABC):
    """Abstract interface for geometric verification.

    Concrete subclasses implement a specific robust-estimation strategy
    (OpenCV RANSAC homography, MAGSAC++, LO-RANSAC, an affine-only
    fallback for low-correspondence cases, ...) and turn a
    ``MatchingResult`` into a ``VerificationResult``.

    Possible strategies (non-exhaustive):
        * **RANSAC + homography** — the default target strategy for
          this project (see ``pipeline.md``); assumes a roughly planar
          reference object.
        * **RANSAC + affine/similarity** — a lighter-weight fallback
          when too few correspondences are available for a stable
          homography (a homography needs >= 4 non-degenerate point
          pairs; an affine transform needs >= 3).
        * **MAGSAC++ / USAC** — alternative robust estimators with
          different inlier-threshold sensitivity, usable as drop-in
          replacements without touching this interface.

    Lifecycle::

        verifier.initialize(config)   # configure thresholds, RANSAC params
        ...
        result = verifier.verify(matching_result, reference_bbox)
        ...
        verifier.cleanup()            # release resources

    The owning ``CoarseToFineEngine`` is responsible for calling these
    methods in order, exactly as it does for ``FeatureExtractor``,
    ``CoarseLocalizer``, and ``FineMatcher``.

    Note:
        A ``GeometricVerifier`` must never import ``BenchmarkRunner`` or
        ``MetricsCalculator``, and must never compute evaluation
        metrics (IoU, precision, recall, mAP, ...). Its sole
        responsibility ends at producing a verified bounding box (or a
        failure) plus the diagnostics needed to explain that outcome.
        Comparing that box against ground truth is exclusively
        ``MetricsCalculator``'s job.

    Example of a future concrete implementation::

        class RansacHomographyVerifier(GeometricVerifier):

            @property
            def name(self) -> str:
                return "RANSAC-Homography"

            def initialize(self, config=None):
                cfg = config or {}
                self._reproj_threshold = cfg.get("ransac_reproj_threshold", 5.0)
                self._min_inliers = cfg.get("min_inliers", 8)
                self._ransac_confidence = cfg.get("ransac_confidence", 0.99)

            def verify(self, matching_result, reference_bbox):
                # 1. cv2.findHomography(ref_pts, test_pts, cv2.RANSAC, ...)
                # 2. split inliers/outliers via the returned mask
                # 3. reject if num_inliers < self._min_inliers
                # 4. cv2.perspectiveTransform(reference_bbox corners, H)
                # 5. clip to test_image_size_hw
                # 6. return VerificationResult(...)
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
        """Human-readable identifier for this verification strategy.

        Returns:
            Strategy name string, e.g. ``"RANSAC-Homography"``.
            Used in logging and diagnostics only.
        """

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Configure the verifier for a benchmark run.

        Called **once** before any ``verify`` call. Implementations
        should store algorithm-specific parameters here. Like
        ``CoarseLocalizer.initialize``, this typically does **not**
        load heavy models — geometric verification is a numerical
        optimisation over already-detected keypoints, not a learned
        inference step.

        Args:
            config:
                Strategy-specific configuration dictionary.
                Expected keys (by convention, not enforced):

                * ``"model_type"`` : str
                    Which geometric model to fit.
                    E.g. ``"homography"``, ``"affine"``, ``"similarity"``.
                    Default should be ``"homography"``.
                * ``"ransac_reproj_threshold"`` : float
                    Maximum reprojection error (in pixels) for a
                    correspondence to be counted as an inlier.
                * ``"ransac_confidence"`` : float
                    Desired probability, in ``(0.0, 1.0)``, that the
                    estimated model is correct. Controls the number of
                    RANSAC iterations internally.
                * ``"ransac_max_iters"`` : int
                    Upper bound on RANSAC iterations, to bound
                    worst-case latency.
                * ``"min_inliers"`` : int
                    Minimum inlier count required for ``success`` to be
                    ``True``. Below this, the correspondences are
                    considered geometrically inconsistent.
                * ``"min_inlier_ratio"`` : float
                    Minimum ``inlier_ratio`` required for success, used
                    in addition to (or instead of) ``min_inliers``.

                ``None`` means use all defaults.

        Raises:
            RuntimeError:
                If the verifier's internal setup fails (e.g. invalid
                configuration combination).
        """

    @abstractmethod
    def verify(
        self,
        matching_result: MatchingResult,
        reference_bounding_box: BoundingBox,
    ) -> VerificationResult:
        """Geometrically verify a set of putative correspondences.

        This is the core verification method. Implementations must
        execute the following conceptual pipeline (the exact estimator
        varies by subclass):

        1. **Gather correspondences** — Read
           ``matching_result.matched_ref_coordinates`` and
           ``matching_result.matched_test_coordinates`` as the input
           point pairs for the robust estimator.

        2. **Handle degenerate input** — If there are too few
           correspondences to fit the configured model (e.g. fewer
           than 4 for a homography), short-circuit and return a
           ``VerificationResult`` with ``success=False`` and a
           ``"failure_reason"`` in ``metadata`` rather than raising.

        3. **Robust model fitting (RANSAC or equivalent)** — Fit the
           configured geometric model while simultaneously identifying
           the inlier subset. See the module docstring for a
           conceptual explanation of RANSAC and homography estimation.

        4. **Apply acceptance thresholds** — Compare the resulting
           inlier count / ratio against ``min_inliers`` /
           ``min_inlier_ratio`` from ``initialize``. If the model is
           too weakly supported, set ``success=False``.

        5. **Project the bounding box** — On success, transform the
           corners of ``reference_bounding_box`` through the estimated
           model into test-image coordinates, form the axis-aligned
           ``verified_bounding_box``, and clip it to
           ``test_features.image_size_hw`` bounds (available via
           ``matching_result.test_keypoints.image_size_hw``).

        6. **Score confidence** — Combine inlier ratio, inlier count,
           and mean reprojection error into a single ``confidence``
           value in ``[0.0, 1.0]``.

        7. **Assemble result** — Package everything into a
           ``VerificationResult``.

        Args:
            matching_result:
                ``MatchingResult`` produced by a ``FineMatcher`` for a
                single candidate region. May have zero matches, in
                which case verification must fail gracefully rather
                than raising.
            reference_bounding_box:
                The reference object's bounding box in reference-image
                pixel space, to be projected into the test image on
                success.

        Returns:
            ``VerificationResult`` describing the outcome. A failed
            geometric verification (insufficient or inconsistent
            correspondences) is a normal, expected return value — it
            is signalled via ``success=False``, not an exception.

        Raises:
            RuntimeError:
                If the verifier has not been initialized.

        Note:
            This method must be **stateless** with respect to previous
            calls: each invocation depends only on ``matching_result``,
            ``reference_bounding_box``, and the configuration set
            during ``initialize`` — mirroring the statelessness
            contract of ``CoarseLocalizer.localize``.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Release any resources held by the verifier.

        For most verification strategies this is a no-op because
        robust estimation is a lightweight numerical procedure with no
        persistent model state. Subclasses that cache intermediate
        structures should free them here.

        Must be **idempotent** — safe to call multiple times.
        """

    # -----------------------------------------------------------------
    # Concrete helpers (shared by all subclasses)
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_has_matches(matching_result: MatchingResult) -> None:
        """Check that the matching result contains at least one correspondence.

        Concrete subclasses should call this at the top of ``verify()``
        to distinguish "nothing to verify" from a genuine estimator
        failure, and to decide whether to short-circuit with
        ``success=False`` instead of invoking the robust estimator.

        Args:
            matching_result: ``MatchingResult`` to check.

        Raises:
            ValueError:
                If ``matching_result`` is ``None``.
        """
        if matching_result is None:
            raise ValueError(
                "matching_result must not be None."
            )

    @staticmethod
    def _validate_minimum_correspondences(
        matching_result: MatchingResult,
        minimum_required: int,
    ) -> bool:
        """Check whether enough correspondences exist to fit the model.

        This does not raise — insufficient correspondences are an
        expected, common outcome (e.g. a small or textureless
        candidate region) and should be handled by returning a
        ``VerificationResult`` with ``success=False``, not by raising.

        Args:
            matching_result: ``MatchingResult`` to check.
            minimum_required: Minimum number of correspondences needed
                by the configured geometric model (e.g. ``4`` for a
                homography, ``3`` for an affine transform).

        Returns:
            ``True`` if ``matching_result.num_matches`` meets or
            exceeds ``minimum_required``, ``False`` otherwise.
        """
        return matching_result.num_matches >= minimum_required

    @staticmethod
    def _reference_bbox_corners(
        reference_bounding_box: BoundingBox,
    ) -> Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ]:
        """Return the four corners of a reference bounding box.

        Corners are returned in a fixed, consistent winding order
        (top-left, top-right, bottom-right, bottom-left) so that
        concrete subclasses can feed them directly into a
        perspective-transform call without re-deriving the order.

        Args:
            reference_bounding_box: Reference-image-space bounding box.

        Returns:
            A 4-tuple of ``(x, y)`` pixel coordinates in
            top-left / top-right / bottom-right / bottom-left order.
        """
        b = reference_bounding_box
        return (
            (b.xtl, b.ytl),
            (b.xbr, b.ytl),
            (b.xbr, b.ybr),
            (b.xtl, b.ybr),
        )

    @staticmethod
    def _clip_bounding_box_to_image(
        candidate_box: BoundingBox,
        image_size_hw: Tuple[int, int],
    ) -> BoundingBox:
        """Clip a projected bounding box to valid image pixel bounds.

        Robust estimation can legitimately project corners slightly
        outside the test image (e.g. when the reference object is
        partially out of frame). Concrete subclasses should call this
        after projecting corners and before constructing the final
        ``VerificationResult.verified_bounding_box``, so that
        downstream consumers never receive out-of-bounds coordinates.

        Args:
            candidate_box: Axis-aligned box in test-image pixel space,
                built from the (possibly out-of-bounds) transformed
                corners.
            image_size_hw: ``(height, width)`` of the test image.

        Returns:
            A new ``BoundingBox`` with coordinates clamped to
            ``[0, width]`` horizontally and ``[0, height]`` vertically.
        """
        img_h, img_w = image_size_hw
        xtl = min(max(candidate_box.xtl, 0.0), float(img_w))
        ytl = min(max(candidate_box.ytl, 0.0), float(img_h))
        xbr = min(max(candidate_box.xbr, 0.0), float(img_w))
        ybr = min(max(candidate_box.ybr, 0.0), float(img_h))
        return BoundingBox(xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
