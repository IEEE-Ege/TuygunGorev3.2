"""benchmark.engines.coarse_to_fine.engine — CoarseToFineEngine orchestrator.

Wires together the four pipeline abstractions —
``FeatureExtractor``, ``CoarseLocalizer``, ``FineMatcher``, and
``GeometricVerifier`` — into the concrete ``MatchingEngine`` implementation
targeted by this project (see ``pipeline.md`` and ``AI_HANDOFF.md``).

Scope of this module
------------------------
This module contains **orchestration logic only**. It calls the four
abstractions in the correct order, interprets their return values, and
translates the outcome into a ``DetectionPrediction`` for
``BenchmarkRunner``. It does not, and must not, contain:

    * Any vision algorithm (no DINOv2, no ALIKED, no LightGlue, no RANSAC).
    * Any framework import (no PyTorch, no OpenCV, no NumPy).
    * Any evaluation logic (no IoU, no precision/recall — that is
      ``MetricsCalculator``'s job).
    * Any dataset parsing (no CVAT XML — that is ``benchmark.parsers``'
      job; this engine only ever receives already-resolved ``Path``
      objects from ``BenchmarkRunner``).

Because every collaborator is injected as an already-constructed
instance (Dependency Injection — see ``__init__``), this file remains
fully testable and fully framework-independent even though the
components it orchestrates will eventually wrap PyTorch/OpenCV models.
Swapping DINOv2 for a different backbone, or RANSAC for MAGSAC++, never
requires touching this file.

Pipeline overview (see ``pipeline.md`` for the full diagram)::

    set_reference(reference_image_path):
        FeatureExtractor.extract(reference_image)
            -> cached ImageFeatures
        FineMatcher.set_reference(cached ImageFeatures)
            -> matcher pre-computes/caches reference keypoints

    detect(test_image_path):
        1. FeatureExtractor.extract(test_image)
        2. CoarseLocalizer.localize(reference_features, test_features)
             -> zero or more CandidateRegion, best-first
        3. for each candidate (best-first):
               FineMatcher.match(test_features, candidate)
               if putative correspondences found:
                   GeometricVerifier.verify(matching_result, reference_bbox)
                   if verification succeeds:
                       -> DetectionPrediction(SUCCESS, verified bbox, ...)
        4. if no candidate survives matching + verification:
               -> DetectionPrediction(NO_MATCH, ...)

Every stage boundary is a plain Python object exchange
(``ImageFeatures`` -> ``LocalizationResult`` -> ``MatchingResult`` ->
``VerificationResult``); this engine never inspects the internals of
those objects beyond the fields their own modules document as public
contracts.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmark.data_models import BoundingBox
from benchmark.engine import MatchingEngine
from benchmark.engines.coarse_to_fine.feature_extractor import (
    FeatureExtractor,
    ImageFeatures,
)
from benchmark.engines.coarse_to_fine.localizer import (
    CandidateRegion,
    CoarseLocalizer,
    LocalizationResult,
)
from benchmark.engines.coarse_to_fine.matcher import FineMatcher, MatchingResult
from benchmark.engines.coarse_to_fine.verifier import (
    GeometricVerifier,
    VerificationResult,
)
from benchmark.prediction_models import DetectionPrediction, MatchResult

logger = logging.getLogger(__name__)


class CoarseToFineEngine(MatchingEngine):
    """Concrete ``MatchingEngine`` that orchestrates the coarse-to-fine pipeline.

    ``CoarseToFineEngine`` owns exactly four collaborators — one per
    pipeline stage — and is otherwise stateless with respect to any
    concrete vision algorithm. All four collaborators are supplied by
    the caller via the constructor (Dependency Injection); this class
    never instantiates a concrete ``FeatureExtractor``,
    ``CoarseLocalizer``, ``FineMatcher``, or ``GeometricVerifier``
    itself. This keeps the engine's own code testable with dummy/mock
    components and keeps the choice of backbone (DINOv2 vs. a future
    alternative), matcher (ALIKED+LightGlue vs. a future alternative),
    and verifier (RANSAC-homography vs. a future alternative) entirely
    external to this class.

    Attributes (private):
        _feature_extractor: Injected ``FeatureExtractor``.
        _coarse_localizer: Injected ``CoarseLocalizer``.
        _fine_matcher: Injected ``FineMatcher``.
        _geometric_verifier: Injected ``GeometricVerifier``.
        _initialized: Whether ``initialize()`` has been called
            successfully and ``cleanup()`` has not since undone it.
        _reference_features: Cached ``ImageFeatures`` for the current
            reference image, populated by ``set_reference()``.
        _reference_bounding_box: Cached reference-image-space
            ``BoundingBox`` of the object to search for, used by the
            ``GeometricVerifier`` to project a verified box into the
            test image. See ``set_reference()`` for how this is
            obtained when the caller does not supply one explicitly.
        _reference_bbox_is_fallback: Whether
            ``_reference_bounding_box`` was derived from a fallback
            (the full reference image) rather than supplied
            explicitly. Surfaced in ``detect()`` metadata for
            transparency.
        _reference_set_name: Best-effort identifier for the current
            reference set, used to populate
            ``DetectionPrediction.reference_set``. See
            ``set_reference()`` for the derivation rule and its
            rationale.

    Example::

        engine = CoarseToFineEngine(
            feature_extractor=DINOv2Extractor(),
            coarse_localizer=CosineHeatmapLocalizer(),
            fine_matcher=LightGlueMatcher(),
            geometric_verifier=RansacHomographyVerifier(),
        )
        engine.initialize(config)
        engine.set_reference(ref_set.reference_image_path)
        prediction = engine.detect(test_image_path)
        engine.cleanup()

    Note:
        Concrete component classes above are illustrative names taken
        from ``pipeline.md`` / ``AI_HANDOFF.md``'s future milestones.
        None of them are implemented or imported by this module.
    """

    # -----------------------------------------------------------------
    # Construction (Dependency Injection)
    # -----------------------------------------------------------------

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        coarse_localizer: CoarseLocalizer,
        fine_matcher: FineMatcher,
        geometric_verifier: GeometricVerifier,
    ) -> None:
        """Wire the four pipeline stages into this engine.

        All four arguments must already be fully constructed instances
        of their respective abstractions. This constructor never
        instantiates a concrete implementation on the caller's behalf
        — that decision belongs to whatever assembles the benchmark
        run (a config-driven factory, a test fixture, a notebook, ...).

        Args:
            feature_extractor: Stage 1 collaborator — converts an
                image path into an ``ImageFeatures`` representation.
            coarse_localizer: Stage 2 collaborator — compares
                reference and test ``ImageFeatures`` and proposes
                candidate regions.
            fine_matcher: Stage 3 collaborator — finds local keypoint
                correspondences within a candidate region.
            geometric_verifier: Stage 4 collaborator — robustly fits a
                geometric transform to the correspondences and
                projects the reference bounding box.

        Raises:
            TypeError: If any argument is not an instance of the
                expected abstraction. Fails fast rather than deferring
                to a confusing error deep inside ``detect()``.
        """
        self._validate_component_type(feature_extractor, FeatureExtractor, "feature_extractor")
        self._validate_component_type(coarse_localizer, CoarseLocalizer, "coarse_localizer")
        self._validate_component_type(fine_matcher, FineMatcher, "fine_matcher")
        self._validate_component_type(geometric_verifier, GeometricVerifier, "geometric_verifier")

        self._feature_extractor = feature_extractor
        self._coarse_localizer = coarse_localizer
        self._fine_matcher = fine_matcher
        self._geometric_verifier = geometric_verifier

        # Lifecycle state.
        self._initialized: bool = False

        # Reference-set cache, populated by set_reference().
        self._reference_features: Optional[ImageFeatures] = None
        self._reference_bounding_box: Optional[BoundingBox] = None
        self._reference_bbox_is_fallback: bool = False
        self._reference_set_name: Optional[str] = None

    # -----------------------------------------------------------------
    # MatchingEngine interface
    # -----------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable identifier composed from the injected components.

        Rather than hardcoding a backbone-specific name (e.g.
        ``"CoarseToFine(DINOv2+LightGlue)"``), the name is derived from
        whatever components were actually injected. This keeps the
        name accurate regardless of which concrete implementations are
        wired in, and avoids this file needing to change when a new
        backbone is swapped in.

        Returns:
            A string of the form
            ``"CoarseToFine(<extractor>+<localizer>+<matcher>+<verifier>)"``.
        """
        return (
            "CoarseToFine("
            f"{self._feature_extractor.name}+"
            f"{self._coarse_localizer.name}+"
            f"{self._fine_matcher.name}+"
            f"{self._geometric_verifier.name})"
        )

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize every owned component, in pipeline order.

        Initialization order is deliberately **forward** through the
        pipeline (extractor -> localizer -> matcher -> verifier). This
        mirrors the order in which the components are actually used
        during ``detect()`` and gives the most useful failure signal:
        if, say, ``FineMatcher.initialize()`` fails, the caller already
        knows ``FeatureExtractor`` and ``CoarseLocalizer`` initialized
        successfully.

        This method does not perform any partial-failure rollback. If
        a later component fails to initialize, earlier components
        remain initialized; the caller should treat any exception from
        this method as fatal for the whole engine and call
        ``cleanup()`` before discarding it, so that any components
        that *did* acquire resources release them (``cleanup()`` is
        safe to call on components regardless of whether they were
        successfully initialized, per their own idempotency contract).

        Args:
            config:
                Optional nested configuration dictionary. Recognised
                top-level keys (all optional):

                * ``"feature_extractor"``  : ``Dict[str, Any]`` forwarded
                  verbatim to ``FeatureExtractor.initialize()``.
                * ``"coarse_localizer"``   : ``Dict[str, Any]`` forwarded
                  verbatim to ``CoarseLocalizer.initialize()``.
                * ``"fine_matcher"``       : ``Dict[str, Any]`` forwarded
                  verbatim to ``FineMatcher.initialize()``.
                * ``"geometric_verifier"`` : ``Dict[str, Any]`` forwarded
                  verbatim to ``GeometricVerifier.initialize()``.

                Any key that is absent results in that component
                receiving ``None`` (i.e. "use component defaults").
                ``None`` for the whole argument means every component
                receives ``None``.

        Raises:
            RuntimeError:
                If any owned component raises during its own
                ``initialize()`` call. The original exception is
                re-raised after being logged with the offending
                component's name for diagnosability.
        """
        cfg = config or {}
        logger.info("CoarseToFineEngine initializing: %r", self.name)

        # Forward pipeline order: extractor -> localizer -> matcher -> verifier.
        self._initialize_component(
            self._feature_extractor, cfg.get("feature_extractor"), "feature_extractor",
        )
        self._initialize_component(
            self._coarse_localizer, cfg.get("coarse_localizer"), "coarse_localizer",
        )
        self._initialize_component(
            self._fine_matcher, cfg.get("fine_matcher"), "fine_matcher",
        )
        self._initialize_component(
            self._geometric_verifier, cfg.get("geometric_verifier"), "geometric_verifier",
        )

        self._initialized = True
        logger.info("CoarseToFineEngine initialized: %r", self.name)

    def set_reference(
        self,
        reference_image_path: Path,
        reference_bounding_box: Optional[BoundingBox] = None,
    ) -> None:
        """Extract and cache reference-side state for future ``detect()`` calls.

        This method performs **only** the work that is shared across
        every test image in a reference set: feature extraction on the
        reference image, and letting the ``FineMatcher`` pre-compute
        whatever it needs from those features. It deliberately never
        calls ``CoarseLocalizer.localize()`` or
        ``GeometricVerifier.verify()`` — those operate on a
        *(reference, test)* pair and belong exclusively to
        ``detect()``.

        Args:
            reference_image_path:
                Path to the reference image for the current reference
                set (``ReferenceSet.reference_image_path`` in
                ``BenchmarkRunner``'s terms).
            reference_bounding_box:
                The ground-truth bounding box of the object of
                interest, **in reference-image pixel space**. This is
                what ``GeometricVerifier.verify()`` projects into each
                test image on a successful match.

                ``BenchmarkRunner`` currently calls
                ``engine.set_reference(ref_set.reference_image_path)``
                with a single positional argument, so this parameter
                is optional to preserve that call site unmodified (see
                "Forbidden Modifications" in ``CLAUDE.md`` — this
                engine must not require changes to ``runner.py``).
                When omitted, a fallback bounding box spanning the
                **entire reference image** is used instead, and
                ``_reference_bbox_is_fallback`` is set so that
                ``detect()`` can surface this in prediction metadata.
                This fallback keeps the pipeline fully runnable
                end-to-end today; wiring the true per-reference-set
                object bounding box through from the validation
                dataset is future work tracked outside this file
                (extending ``ReferenceSet`` / ``BenchmarkRunner``,
                which this task does not authorize touching).

        Raises:
            RuntimeError:
                If ``initialize()`` has not been called yet.
        """
        self._require_initialized()

        logger.info("Setting reference image: %s", reference_image_path)

        # Stage 1 (reference side only): feature extraction.
        self._reference_features = self._feature_extractor.extract(reference_image_path)

        # Resolve the reference bounding box, falling back to the full
        # reference image when the caller did not supply one.
        if reference_bounding_box is not None:
            self._reference_bounding_box = reference_bounding_box
            self._reference_bbox_is_fallback = False
            self._reference_features.metadata["object_bounding_box"] = reference_bounding_box
        else:
            img_h, img_w = self._reference_features.image_size_hw
            self._reference_bounding_box = BoundingBox(
                xtl=0.0, ytl=0.0, xbr=float(img_w), ybr=float(img_h),
                label="reference",
            )
            self._reference_bbox_is_fallback = True
            self._reference_features.metadata["object_bounding_box"] = self._reference_bounding_box
            logger.warning(
                "No reference_bounding_box supplied for %s; falling back to "
                "the full reference image (%dx%d). Verification confidence "
                "and projected boxes will be less meaningful until the true "
                "object box is wired through.",
                reference_image_path,
                img_w,
                img_h,
            )

        # Let the matcher pre-compute/cache whatever it needs (e.g.
        # reference keypoints) exactly once per reference set.
        self._fine_matcher.set_reference(self._reference_features)

        # Best-effort reference-set identifier for DetectionPrediction.
        # BenchmarkRunner does not currently pass ref_set.name through
        # to the engine (it only passes reference_image_path), so the
        # parent directory name is used as a conventional stand-in
        # (validation sets are laid out as "<refXX>/<reference_image>").
        self._reference_set_name = reference_image_path.parent.name or "unknown"

        logger.info(
            "Reference ready: set=%r, bbox_fallback=%s",
            self._reference_set_name,
            self._reference_bbox_is_fallback,
        )

    def detect(self, test_image_path: Path) -> DetectionPrediction:
        """Run the full coarse-to-fine pipeline on a single test image.

        Orchestration flow:

            1. **Feature extraction** — ``FeatureExtractor.extract()`` on
               the test image.
            2. **Coarse localization** — ``CoarseLocalizer.localize()``
               compares reference and test features and proposes
               candidate regions, best-first. Zero candidates
               short-circuits directly to ``MatchResult.NO_MATCH``.
            3. **Per-candidate fine matching + verification** — for each
               candidate region, in the order supplied by the
               localizer:

               a. ``FineMatcher.match()`` searches for local keypoint
                  correspondences within the candidate region. A
                  candidate with no correspondences is skipped.
               b. ``GeometricVerifier.verify()`` checks whether those
                  correspondences are geometrically consistent and, if
                  so, projects the reference bounding box into test
                  image space. A candidate that fails verification is
                  skipped.
               c. The **first** candidate that survives both stages
                  wins; its verified box and confidence become the
                  prediction. This is a reasonable default policy
                  because ``LocalizationResult.candidates`` is
                  documented as sorted by descending confidence — later
                  candidates are strictly less likely a priori.

            4. If every candidate is exhausted without a successful
               verification, the result is ``MatchResult.NO_MATCH`` —
               not an error. A geometrically-unverifiable or
               unmatchable image is an expected outcome for a one-shot
               matching benchmark, not a pipeline failure.

        Args:
            test_image_path:
                Path to the test frame to search for the reference
                object.

        Returns:
            A fully-populated ``DetectionPrediction``:

            * ``image_filename`` — ``test_image_path.name``.
            * ``reference_set`` — the identifier cached by the most
              recent ``set_reference()`` call.
            * ``result`` — ``MatchResult.SUCCESS`` or
              ``MatchResult.NO_MATCH`` (this method never returns
              ``MatchResult.ERROR``; unexpected exceptions from a
              component propagate to the caller, matching the
              contract documented by ``BenchmarkRunner._detect_safe``,
              which is responsible for converting unhandled exceptions
              into ``MatchResult.ERROR`` predictions).
            * ``predicted_box`` — the verified bounding box on success,
              ``None`` otherwise.
            * ``confidence`` — the verifier's confidence on success,
              ``None`` otherwise.
            * ``processing_time_ms`` — total wall-clock time for this
              call, covering all four stages.
            * ``metadata`` — diagnostics: which pipeline stage produced
              the final outcome, how many candidates were evaluated,
              per-stage component names, and whether the reference
              bounding box was a fallback.

        Raises:
            RuntimeError:
                If ``initialize()`` or ``set_reference()`` has not been
                called yet.

        Note:
            This method deliberately does not catch exceptions raised
            by the owned components. ``BenchmarkRunner._detect_safe()``
            already wraps engine.detect() calls and converts unhandled
            exceptions into ``DetectionPrediction(result=MatchResult.ERROR)``
            — duplicating that handling here would hide genuine bugs
            behind a second, engine-local safety net.
        """
        self._require_initialized()
        self._require_reference_set()

        start = time.perf_counter()

        # ---------------------------------------------------------
        # Stage 1: feature extraction (test side).
        # ---------------------------------------------------------
        test_features = self._feature_extractor.extract(test_image_path)

        # ---------------------------------------------------------
        # Stage 2: coarse localization.
        # ---------------------------------------------------------
        localization_result = self._coarse_localizer.localize(
            self._reference_features, test_features,
        )

        if not localization_result.has_candidates:
            return self._build_no_match_prediction(
                test_image_path=test_image_path,
                start_time=start,
                stage="coarse_localization",
                reason="no_candidate_regions",
                extra_metadata={
                    "global_similarity": localization_result.global_similarity,
                },
            )

        # ---------------------------------------------------------
        # Stage 3 + 4: fine matching and geometric verification,
        # attempted per candidate (best confidence first) until one
        # succeeds or all are exhausted.
        # ---------------------------------------------------------
        candidates_evaluated = 0
        last_matching_result: Optional[MatchingResult] = None
        last_verification_result: Optional[VerificationResult] = None

        for candidate in localization_result.candidates:
            candidates_evaluated += 1
            matching_result, verification_result = self._attempt_candidate(
                test_features=test_features,
                candidate=candidate,
            )
            last_matching_result = matching_result
            last_verification_result = verification_result

            if verification_result is not None and verification_result.success:
                return self._build_success_prediction(
                    test_image_path=test_image_path,
                    start_time=start,
                    candidates_evaluated=candidates_evaluated,
                    matching_result=matching_result,
                    verification_result=verification_result,
                )

        # No candidate survived matching + verification.
        stage, reason = self._diagnose_failure(
            last_matching_result, last_verification_result,
        )
        return self._build_no_match_prediction(
            test_image_path=test_image_path,
            start_time=start,
            stage=stage,
            reason=reason,
            extra_metadata={"candidates_evaluated": candidates_evaluated},
        )

    def cleanup(self) -> None:
        """Release all owned components' resources.

        Cleanup order is the **reverse** of initialization order
        (verifier -> matcher -> localizer -> extractor). Although the
        four components do not hold references to each other, tearing
        down in reverse order is a conservative convention that
        generalises safely if a future component's ``cleanup()`` ever
        needs to assume a downstream stage has already released shared
        resources (e.g. a shared device context acquired last should
        be released first).

        This method is safe to call multiple times: each component's
        own ``cleanup()`` is documented as idempotent, and this method
        additionally guards against re-entering an already-clean state
        by simply resetting the engine's own cached reference state
        every time it runs, which is itself an idempotent operation.

        A failure in one component's ``cleanup()`` does not prevent the
        remaining components from being cleaned up: each call is
        individually wrapped so that, e.g., a ``GeometricVerifier``
        that fails to clean up does not leak a ``FeatureExtractor``'s
        GPU memory. All exceptions are logged; the first one
        encountered is re-raised after every component has had a
        chance to clean up, so callers still learn that something went
        wrong.
        """
        logger.info("CoarseToFineEngine cleaning up: %r", self.name)

        # Reverse pipeline order: verifier -> matcher -> localizer -> extractor.
        components_in_cleanup_order: List[Tuple[str, Any]] = [
            ("geometric_verifier", self._geometric_verifier),
            ("fine_matcher", self._fine_matcher),
            ("coarse_localizer", self._coarse_localizer),
            ("feature_extractor", self._feature_extractor),
        ]

        first_error: Optional[Exception] = None
        for component_label, component in components_in_cleanup_order:
            try:
                component.cleanup()
            except Exception as exc:  # noqa: BLE001 - must not abort remaining cleanups
                logger.exception("Error cleaning up %s: %r", component_label, component)
                if first_error is None:
                    first_error = exc

        # Reset engine-owned cached state regardless of component outcomes.
        self._reference_features = None
        self._reference_bounding_box = None
        self._reference_bbox_is_fallback = False
        self._reference_set_name = None
        self._initialized = False

        logger.info("CoarseToFineEngine cleanup complete: %r", self.name)

        if first_error is not None:
            raise first_error

    # -----------------------------------------------------------------
    # Private orchestration helpers
    # -----------------------------------------------------------------

    def _attempt_candidate(
        self,
        test_features: ImageFeatures,
        candidate: CandidateRegion,
    ) -> Tuple[Optional[MatchingResult], Optional[VerificationResult]]:
        """Run fine matching + geometric verification for one candidate.

        Isolated as its own method so that ``detect()``'s per-candidate
        loop stays readable, and so unit tests can exercise a single
        candidate attempt without running the whole pipeline.

        Args:
            test_features: ``ImageFeatures`` of the test image (shared
                across all candidates for this ``detect()`` call).
            candidate: The ``CandidateRegion`` to attempt.

        Returns:
            A ``(matching_result, verification_result)`` tuple.
            ``matching_result`` is ``None`` only if the matcher itself
            raises (which propagates — see ``detect()``'s exception
            policy); it is a valid, possibly-empty ``MatchingResult``
            otherwise. ``verification_result`` is ``None`` when
            matching produced no correspondences at all (verification
            is skipped in that case), and a ``VerificationResult``
            otherwise.
        """
        matching_result = self._fine_matcher.match(test_features, candidate)

        if not matching_result.has_matches:
            logger.debug(
                "Candidate %r produced no correspondences; skipping verification.",
                candidate,
            )
            return matching_result, None

        verification_result = self._geometric_verifier.verify(
            matching_result, self._reference_bounding_box,
        )
        return matching_result, verification_result

    @staticmethod
    def _diagnose_failure(
        matching_result: Optional[MatchingResult],
        verification_result: Optional[VerificationResult],
    ) -> Tuple[str, str]:
        """Classify why the last-attempted candidate did not succeed.

        Used only to populate human-readable ``NO_MATCH`` metadata; it
        does not affect control flow.

        Args:
            matching_result: Result of the last matching attempt, or
                ``None`` if no candidate was attempted.
            verification_result: Result of the last verification
                attempt, or ``None`` if matching produced no
                correspondences (or no candidate was attempted).

        Returns:
            A ``(stage, reason)`` pair suitable for
            ``DetectionPrediction.metadata``.
        """
        if matching_result is None:
            return "fine_matching", "no_candidates_attempted"
        if not matching_result.has_matches:
            return "fine_matching", "no_correspondences_found"
        if verification_result is None:
            return "geometric_verification", "verification_not_attempted"
        return "geometric_verification", "geometric_verification_failed"

    def _build_success_prediction(
        self,
        test_image_path: Path,
        start_time: float,
        candidates_evaluated: int,
        matching_result: MatchingResult,
        verification_result: VerificationResult,
    ) -> DetectionPrediction:
        """Assemble a ``MatchResult.SUCCESS`` prediction.

        Args:
            test_image_path: Test image that was detected on.
            start_time: ``time.perf_counter()`` value captured at the
                start of ``detect()``.
            candidates_evaluated: Number of candidates tried before the
                winning one.
            matching_result: The ``MatchingResult`` for the winning
                candidate.
            verification_result: The successful ``VerificationResult``
                for the winning candidate.

        Returns:
            A fully-populated successful ``DetectionPrediction``.
        """
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return DetectionPrediction(
            image_filename=test_image_path.name,
            reference_set=self._reference_set_name or "unknown",
            result=MatchResult.SUCCESS,
            predicted_box=verification_result.verified_bounding_box,
            confidence=verification_result.confidence,
            processing_time_ms=elapsed_ms,
            metadata={
                "stage": "geometric_verification",
                "candidates_evaluated": candidates_evaluated,
                "num_matches": matching_result.num_matches,
                "num_inliers": verification_result.num_inliers,
                "num_outliers": verification_result.num_outliers,
                "inlier_ratio": verification_result.inlier_ratio,
                "reference_bbox_is_fallback": self._reference_bbox_is_fallback,
                "feature_extractor": self._feature_extractor.name,
                "coarse_localizer": self._coarse_localizer.name,
                "fine_matcher": self._fine_matcher.name,
                "geometric_verifier": self._geometric_verifier.name,
            },
        )

    def _build_no_match_prediction(
        self,
        test_image_path: Path,
        start_time: float,
        stage: str,
        reason: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> DetectionPrediction:
        """Assemble a ``MatchResult.NO_MATCH`` prediction.

        Centralises the ``NO_MATCH`` construction so every early-return
        path in ``detect()`` (no candidates, no correspondences, failed
        verification) produces a consistently-shaped prediction.

        Args:
            test_image_path: Test image that was detected on.
            start_time: ``time.perf_counter()`` value captured at the
                start of ``detect()``.
            stage: Which pipeline stage produced the ``NO_MATCH``
                outcome (e.g. ``"coarse_localization"``,
                ``"fine_matching"``, ``"geometric_verification"``).
            reason: Short machine-readable reason code.
            extra_metadata: Additional stage-specific diagnostics to
                merge into ``metadata``.

        Returns:
            A fully-populated ``NO_MATCH`` ``DetectionPrediction``.
        """
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        metadata: Dict[str, Any] = {
            "stage": stage,
            "reason": reason,
            "reference_bbox_is_fallback": self._reference_bbox_is_fallback,
            "feature_extractor": self._feature_extractor.name,
            "coarse_localizer": self._coarse_localizer.name,
            "fine_matcher": self._fine_matcher.name,
            "geometric_verifier": self._geometric_verifier.name,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return DetectionPrediction(
            image_filename=test_image_path.name,
            reference_set=self._reference_set_name or "unknown",
            result=MatchResult.NO_MATCH,
            predicted_box=None,
            confidence=None,
            processing_time_ms=elapsed_ms,
            metadata=metadata,
        )

    def _initialize_component(
        self,
        component: Any,
        component_config: Optional[Dict[str, Any]],
        component_label: str,
    ) -> None:
        """Initialize a single owned component with clear error context.

        Args:
            component: The component instance to initialize.
            component_config: The sub-config to forward, or ``None``.
            component_label: Human-readable label used in log messages
                and the re-raised error, e.g. ``"feature_extractor"``.

        Raises:
            RuntimeError:
                Wrapping any exception raised by
                ``component.initialize()``, annotated with
                ``component_label`` for diagnosability.
        """
        logger.debug("Initializing %s: %r", component_label, component)
        try:
            component.initialize(component_config)
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise RuntimeError(
                f"CoarseToFineEngine failed to initialize component "
                f"{component_label!r} ({component!r}): {exc}"
            ) from exc

    def _require_initialized(self) -> None:
        """Guard that raises unless ``initialize()`` has succeeded.

        Raises:
            RuntimeError:
                If ``initialize()`` has not been called, or if
                ``cleanup()`` has been called since the last
                ``initialize()``.
        """
        if not self._initialized:
            raise RuntimeError(
                "CoarseToFineEngine.initialize() must be called before use."
            )

    def _require_reference_set(self) -> None:
        """Guard that raises unless ``set_reference()`` has succeeded.

        Raises:
            RuntimeError:
                If ``set_reference()`` has not been called since the
                last ``initialize()`` / ``cleanup()`` cycle.
        """
        if self._reference_features is None or self._reference_bounding_box is None:
            raise RuntimeError(
                "CoarseToFineEngine.set_reference() must be called before "
                "detect()."
            )

    @staticmethod
    def _validate_component_type(
        component: Any, expected_type: type, component_label: str,
    ) -> None:
        """Validate a constructor argument's type, failing fast.

        Args:
            component: The value passed to ``__init__``.
            expected_type: The abstract base class it must be an
                instance of.
            component_label: Human-readable label for the error
                message, e.g. ``"feature_extractor"``.

        Raises:
            TypeError:
                If ``component`` is not an instance of
                ``expected_type``.
        """
        if not isinstance(component, expected_type):
            raise TypeError(
                f"{component_label} must be an instance of "
                f"{expected_type.__name__}, got {type(component).__name__}."
            )

    def __repr__(self) -> str:
        return f"CoarseToFineEngine(name={self.name!r}, initialized={self._initialized})"
