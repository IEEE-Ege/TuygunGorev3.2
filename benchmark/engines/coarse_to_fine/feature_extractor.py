"""benchmark.engines.coarse_to_fine.feature_extractor — Feature extraction abstraction.

Defines the ``FeatureExtractor`` interface that decouples the
coarse-to-fine pipeline from any specific vision backbone.

The current design targets **DINOv2 ViT** as the first concrete
implementation, but the abstraction intentionally avoids binding
to any framework, model, or tensor library.  A future implementor
must subclass ``FeatureExtractor`` and fill in the model-specific
logic at clearly marked extension points.

Design rationale:
    * **Single Responsibility** — This class is responsible *only*
      for converting a raw image path into a feature representation.
      It does not localise, match, or verify anything.
    * **Open/Closed** — New backbones (SigLIP, SAM, MAE, …) can be
      added by subclassing without modifying existing code.
    * **Dependency Inversion** — ``CoarseToFineEngine`` depends on
      this abstraction, never on a concrete backbone.

Lifecycle managed by the owning engine::

    extractor.initialize(config)   # load model, allocate device
    extractor.extract(ref_path)    # extract features for reference
    extractor.extract(test_path)   # extract features for each test image
    ...
    extractor.cleanup()            # release GPU / model memory

Type contract for ``extract()``:
    The method returns an ``ImageFeatures`` dataclass that bundles
    every representation the downstream ``CoarseLocalizer`` may need
    (patch embeddings, global descriptor, spatial grid shape).
    The actual tensor types are intentionally typed as ``object``
    because this module must not import any deep-learning framework.
    When DINOv2 is implemented, the concrete subclass documentation
    will specify the real types (e.g. ``torch.Tensor``).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# Feature representation
# =====================================================================


@dataclass(frozen=True)
class ImageFeatures:
    """Immutable container for extracted image features.

    This is the *only* data object that crosses the boundary between
    ``FeatureExtractor`` and ``CoarseLocalizer``.  It is intentionally
    framework-agnostic: all tensor-like fields are typed as ``object``
    so that no deep-learning import is required at the interface level.

    A concrete ``FeatureExtractor`` implementation is expected to
    populate these fields with its native tensor types and document
    them in the subclass.

    Attributes:
        patch_features:
            Dense, spatially-arranged feature map produced by the
            vision backbone.  For a ViT model this corresponds to
            the patch token outputs *excluding* the CLS token.

            Expected semantics (DINOv2 example):
                * Type  : ``torch.Tensor``
                * Shape : ``(num_patches, embed_dim)``
                          e.g. ``(3600, 384)`` for ViT-S/14 at
                          840 × 840 input (60 × 60 grid).
                * Dtype : ``float32``

            The ``CoarseLocalizer`` reshapes this into a 2-D spatial
            grid using ``patch_grid_shape`` and computes a similarity
            heatmap against the reference patch features.

        global_features:
            A single, image-level descriptor vector.  For a ViT model
            this is typically the CLS token output.

            Expected semantics (DINOv2 example):
                * Type  : ``torch.Tensor``
                * Shape : ``(embed_dim,)``  e.g. ``(384,)``
                * Dtype : ``float32``

            Used for fast retrieval-style filtering before the more
            expensive patch-level comparison.  May be ``None`` when
            global descriptors are not available or not needed.

        patch_grid_shape:
            The (height, width) of the spatial grid that the
            ``patch_features`` can be reshaped into.

            For a ViT with patch size *p* and an input image resized
            to *(H, W)*::

                patch_grid_shape = (H // p, W // p)

            Example: 840 × 840 image, patch size 14 → ``(60, 60)``.

        source_image_path:
            Absolute path of the image from which these features were
            extracted.  Retained for traceability and debugging.

        image_size_hw:
            Original image dimensions as ``(height, width)`` in pixels
            *before* any resizing.  Required by the ``Localizer`` to
            map heatmap coordinates back to original image space.

        metadata:
            Free-form dictionary for backend-specific diagnostics.
            Possible keys (by convention, not enforced):
                * ``"model_name"``       : str — e.g. ``"dinov2_vits14"``
                * ``"input_size_hw"``     : Tuple[int, int] — model input dims
                * ``"extraction_time_ms"`` : float — wall-clock time
                * ``"device"``           : str — e.g. ``"cuda:0"``
    """

    patch_features: object
    patch_grid_shape: Tuple[int, int]
    source_image_path: Path
    image_size_hw: Tuple[int, int]
    global_features: Optional[object] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------

    @property
    def num_patches(self) -> int:
        """Total number of spatial patches.

        Computed from ``patch_grid_shape`` so that callers do not need
        to inspect the raw tensor.
        """
        h, w = self.patch_grid_shape
        return h * w

    @property
    def has_global(self) -> bool:
        """Whether a global descriptor is available."""
        return self.global_features is not None

    def __repr__(self) -> str:
        h, w = self.patch_grid_shape
        img_h, img_w = self.image_size_hw
        global_tag = "yes" if self.has_global else "no"
        return (
            f"ImageFeatures("
            f"grid={h}x{w}, "
            f"patches={self.num_patches}, "
            f"global={global_tag}, "
            f"img={img_w}x{img_h}, "
            f"src={self.source_image_path.name!r})"
        )


# =====================================================================
# Abstract extractor
# =====================================================================


class FeatureExtractor(ABC):
    """Abstract interface for image feature extraction.

    Concrete subclasses wrap a specific vision backbone (DINOv2, SigLIP,
    MAE, …) and implement the three lifecycle methods:

    1. ``initialize`` — Load the model, allocate device memory.
    2. ``extract``    — Run inference on a single image, return
       an ``ImageFeatures`` instance.
    3. ``cleanup``    — Release model and device resources.

    The owning ``CoarseToFineEngine`` calls these methods in order and
    is responsible for lifecycle management.  ``FeatureExtractor``
    itself is stateless between ``extract`` calls (aside from the
    loaded model kept in memory).

    Example of a future concrete implementation::

        class DINOv2Extractor(FeatureExtractor):

            @property
            def name(self) -> str:
                return "DINOv2-ViT-S/14"

            def initialize(self, config=None):
                # torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                # move to device, set eval mode
                ...

            def extract(self, image_path):
                # load image, preprocess, forward pass
                # return ImageFeatures(...)
                ...

            def cleanup(self):
                # del self._model; torch.cuda.empty_cache()
                ...

    Extension points are marked with ``NotImplementedError`` and
    detailed inline documentation describing exactly what the
    implementation must do.
    """

    # -----------------------------------------------------------------
    # Abstract interface
    # -----------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this extractor backend.

        Returns:
            Backend name string, e.g. ``"DINOv2-ViT-S/14"``.
            Used in logging and diagnostics only.
        """

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Load the vision model and prepare for inference.

        This method is called **once** before any ``extract`` call.
        Implementations must perform all heavy one-time setup here:

        * Load pre-trained weights (e.g. ``torch.hub.load``).
        * Move the model to the target device (CPU / CUDA).
        * Set the model to evaluation mode.
        * Store input-size, patch-size, and normalisation parameters
          derived from ``config`` or model defaults.

        Args:
            config:
                Backend-specific configuration dictionary.
                Expected keys (by convention, not enforced):

                * ``"model_name"`` : str
                    Which model variant to load.
                    Default should be the smallest variant suitable
                    for edge deployment (e.g. ``"dinov2_vits14"``).
                * ``"device"`` : str
                    Target device, e.g. ``"cuda"`` or ``"cpu"``.
                    Default should auto-detect CUDA availability.
                * ``"input_size"`` : int | Tuple[int, int]
                    Spatial resolution to resize images to before
                    inference.  Must be divisible by the patch size.
                    Default should balance accuracy and speed
                    (e.g. ``518`` for DINOv2 with patch size 14).

                ``None`` means use all defaults.

        Raises:
            RuntimeError:
                If model loading fails (e.g. missing weights, OOM).
        """

    @abstractmethod
    def extract(self, image_path: Path) -> ImageFeatures:
        """Extract features from a single image.

        This is the core inference method.  Implementations must:

        1. **Load** the image from ``image_path`` and record its
           original ``(height, width)`` before any resizing.
        2. **Preprocess** the image to match the backbone's expected
           input format (resize, centre-crop, normalise, convert to
           tensor, move to device).
        3. **Run forward pass** through the vision model with
           gradients disabled.
        4. **Collect outputs**:

           * *Patch features* — the dense spatial token outputs
             (e.g. ViT patch tokens, excluding CLS).
           * *Global features* — the image-level descriptor
             (e.g. ViT CLS token).  May be ``None`` if the
             backbone does not produce one.

        5. **Package** the results into an ``ImageFeatures`` instance,
           including the ``patch_grid_shape`` so the ``Localizer``
           can reshape the flat patch array into a spatial map.

        Args:
            image_path:
                Absolute path to the image file.  The file format
                must be readable by the imaging library used in the
                concrete implementation (JPEG, PNG at minimum).

        Returns:
            ``ImageFeatures`` containing patch-level and (optionally)
            global-level representations of the image.

        Raises:
            FileNotFoundError:
                If ``image_path`` does not exist.
            RuntimeError:
                If inference fails (e.g. corrupt image, device error).

        Note:
            This method must be **re-entrant**: it may be called many
            times with different images between a single
            ``initialize`` / ``cleanup`` pair, and each call must be
            independent of previous calls.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Release the model and all associated resources.

        Implementations must:

        * Delete the model reference so that Python's GC can free it.
        * Explicitly clear any device-side caches
          (e.g. ``torch.cuda.empty_cache()``).
        * Reset internal state so that ``initialize`` could
          theoretically be called again.

        This method must be **idempotent** — calling it multiple
        times must not raise an error.
        """

    # -----------------------------------------------------------------
    # Concrete helpers (shared by all subclasses)
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_image_path(image_path: Path) -> Path:
        """Validate that the image file exists and return a resolved Path.

        Concrete subclasses should call this at the top of
        ``extract()`` to fail fast with a clear error message.

        Args:
            image_path: Path to validate.

        Returns:
            Resolved absolute ``Path``.

        Raises:
            FileNotFoundError:
                If the path does not point to an existing file.
        """
        resolved = Path(image_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Image file not found: {resolved}"
            )
        return resolved

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
