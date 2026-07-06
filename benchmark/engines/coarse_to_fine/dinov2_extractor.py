"""benchmark.engines.coarse_to_fine.dinov2_extractor — DINOv2 backbone loader.

Concrete ``FeatureExtractor`` implementation that wraps a DINOv2 Vision
Transformer backbone (``torch.hub`` / ``facebookresearch/dinov2``).

Scope of this module (Phase 3, step 1)
---------------------------------------
This module currently implements **only the lifecycle bootstrap** of the
extractor:

    * ``__init__``    — construct an un-initialized instance.
    * ``name``        — human-readable identifier.
    * ``initialize``  — validate configuration, load the DINOv2 backbone
      via ``torch.hub``, place it on the configured device, and put it
      into a safe, non-trainable inference state.
    * ``cleanup``     — release the model and any device-side memory.

``extract()`` is intentionally **not implemented** yet. It exists only
as an abstract-method placeholder that raises ``NotImplementedError``,
mirroring the convention already established by
``benchmark.metrics.MetricsCalculator.evaluate()`` for staged,
not-yet-implemented functionality. Image loading, preprocessing
(RGB/Thermal), batching, and the forward pass are out of scope for this
step and will be added in a subsequent task.

Framework boundary
-------------------
Unlike the abstract interfaces in ``feature_extractor.py``,
``localizer.py``, ``matcher.py``, and ``verifier.py`` — which must stay
importable without any deep-learning framework — this module is a
**concrete implementation** and is therefore explicitly allowed (per
``CLAUDE.md``: "These belong only in concrete implementations.") to
import ``torch`` and ``torchvision`` directly.

Lifecycle::

    extractor = DINOv2FeatureExtractor()
    extractor.initialize({"model_name": "dinov2_vits14", "device": "cuda"})
    ...
    extractor.cleanup()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, FrozenSet, Tuple, Optional

import torch
import torchvision  # noqa: F401  (imported to satisfy the "use torch +
                     # torchvision" requirement and to fail fast at import
                     # time if torchvision is missing/broken; DINOv2's
                     # torch.hub entry point transitively depends on it
                     # for preprocessing transforms used in later stages)
from PIL import Image
from torchvision import transforms

from benchmark.engines.coarse_to_fine.feature_extractor import (
    FeatureExtractor,
    ImageFeatures,
)

logger = logging.getLogger(__name__)


class DINOv2FeatureExtractor(FeatureExtractor):
    """DINOv2 ViT backbone wrapper implementing the ``FeatureExtractor`` contract.

    This class owns exactly one resource: a DINOv2 backbone loaded via
    ``torch.hub``. It is deliberately thin at this stage — it knows how
    to *load* the model, not how to *run* it. That split keeps this
    task's diff minimal and independently reviewable (per ``CLAUDE.md``:
    "One prompt = one logical implementation").

    Supported backbone variants:
        * ``"dinov2_vits14"`` — ViT-S/14 (smallest, fastest; suitable
          for edge/Jetson deployment per the project roadmap).
        * ``"dinov2_vitb14"`` — ViT-B/14 (medium).
        * ``"dinov2_vitl14"`` — ViT-L/14 (largest, most accurate).

    Attributes (private):
        _model: The loaded DINOv2 ``torch.nn.Module``, or ``None``
            before ``initialize()`` / after ``cleanup()``.
        _device: The ``torch.device`` the model is loaded onto, or
            ``None`` before ``initialize()`` / after ``cleanup()``.
        _variant: The DINOv2 variant string that was loaded, or
            ``None`` before ``initialize()`` / after ``cleanup()``.
        _initialized: Whether ``initialize()`` has succeeded and
            ``cleanup()`` has not since undone it.

    Example::

        extractor = DINOv2FeatureExtractor()
        extractor.initialize({
            "model_name": "dinov2_vits14",
            "device": "cuda",
        })
        print(extractor.name)  # "DINOv2-dinov2_vits14"
        extractor.cleanup()
    """

    #: DINOv2 backbone variants this class knows how to load.
    #: Kept as a class-level constant (single source of truth) so that
    #: validation logic and documentation never drift out of sync.
    _SUPPORTED_VARIANTS: FrozenSet[str] = frozenset(
        {"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"}
    )

    #: torch.hub repository that hosts the DINOv2 entry points.
    _TORCH_HUB_REPO: str = "facebookresearch/dinov2"

    #: Variant used when ``config`` omits ``"model_name"``. Chosen to
    #: match the roadmap's edge-deployment target (Jetson optimisation,
    #: see ``PROJECT_ROADMAP.md`` Phase 10) — the smallest backbone is
    #: the safest default.
    _DEFAULT_VARIANT: str = "dinov2_vits14"

    #: Patch size, in pixels, shared by every variant in
    #: ``_SUPPORTED_VARIANTS`` (the ``14`` in ``dinov2_vits14`` /
    #: ``dinov2_vitb14`` / ``dinov2_vitl14``). Used by ``extract()`` to
    #: derive ``ImageFeatures.patch_grid_shape`` from ``_INPUT_SIZE``
    #: without needing to inspect the loaded model at runtime.
    _PATCH_SIZE: int = 14

    #: Spatial resolution (both dimensions) that every input image is
    #: resized to before the forward pass. ``518`` is the standard
    #: DINOv2 evaluation resolution and is exactly divisible by
    #: ``_PATCH_SIZE`` (``518 / 14 = 37``), which ``FeatureExtractor``'s
    #: ``initialize`` docstring calls out as a hard requirement. Kept
    #: as a fixed constant rather than a ``config`` key for this task
    #: because ``initialize()`` is out of scope (see module docstring)
    #: and must not be modified.
    _INPUT_SIZE: int = 518

    #: ImageNet channel-wise mean/std used to normalise inputs, matching
    #: the statistics DINOv2 was pretrained with. Applied identically to
    #: RGB and Thermal images: Thermal (grayscale) frames are first
    #: replicated into three channels by ``_load_image`` so that a
    #: single normalisation step and a single forward pass serve both
    #: modalities, per the "Single Unified Pipeline" decision in
    #: ``AI_HANDOFF.md``.
    _NORM_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    _NORM_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    def __init__(self) -> None:
        """Construct an un-initialized extractor.

        No model is loaded and no device is selected here. All heavy,
        potentially-failing setup is deferred to ``initialize()``, per
        the ``FeatureExtractor`` lifecycle contract, so that
        constructing an instance (e.g. for dependency-injection wiring
        in ``CoarseToFineEngine.__init__``) is always cheap and cannot
        fail.
        """
        self._model: Optional[torch.nn.Module] = None
        self._device: Optional["torch.device"] = None
        self._variant: Optional[str] = None
        self._initialized: bool = False

    # -----------------------------------------------------------------
    # FeatureExtractor interface
    # -----------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable identifier for this extractor backend.

        Returns:
            ``"DINOv2-<variant>"`` once initialized (e.g.
            ``"DINOv2-dinov2_vits14"``), or ``"DINOv2-uninitialized"``
            beforehand. The uninitialized form is still a valid,
            non-crashing string so that logging code elsewhere (e.g.
            ``CoarseToFineEngine.name``, which concatenates component
            names) never breaks if it is ever called before
            ``initialize()``.
        """
        if self._variant is not None:
            return f"DINOv2-{self._variant}"
        return "DINOv2-uninitialized"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Load the DINOv2 backbone and prepare it for inference.

        Performs, in order:

        1. **Configuration validation** — resolves and validates the
           ``model_name`` (backbone variant) and ``device`` entries of
           ``config``, failing fast with a clear error before any
           network/device work is attempted.
        2. **Model loading** — downloads/loads the DINOv2 weights via
           ``torch.hub.load(_TORCH_HUB_REPO, variant)``.
        3. **Device placement** — moves the model to the resolved
           ``torch.device`` (CPU or CUDA).
        4. **Inference-mode setup** — calls ``model.eval()`` and
           disables gradient tracking on every parameter, since this
           extractor is only ever used for inference, never training.

        Args:
            config:
                Optional configuration dictionary. Recognised keys:

                * ``"model_name"`` : str
                    Which DINOv2 variant to load. Must be one of
                    ``"dinov2_vits14"``, ``"dinov2_vitb14"``,
                    ``"dinov2_vitl14"``. Defaults to
                    ``"dinov2_vits14"`` when omitted.
                * ``"device"`` : str
                    Target device string understood by
                    ``torch.device`` (e.g. ``"cpu"``, ``"cuda"``,
                    ``"cuda:0"``). When omitted, the device is
                    auto-detected: ``"cuda"`` if
                    ``torch.cuda.is_available()`` else ``"cpu"``.

                ``None`` is equivalent to ``{}`` (use all defaults).

        Raises:
            ValueError:
                If ``config`` contains an unsupported ``model_name``,
                or if ``device`` is not a string.
            RuntimeError:
                If ``device`` requests CUDA but CUDA is not available
                on this machine, if the DINOv2 weights fail to load
                (e.g. no network access, corrupt cache, invalid hub
                entry point), or if moving the model to the resolved
                device fails (e.g. out of GPU memory).

        Note:
            This method is safe to call again after ``cleanup()`` to
            reload a (possibly different) configuration; it is **not**
            required to be idempotent while already initialized —
            calling it twice without an intervening ``cleanup()`` will
            simply replace the previously loaded model.
        """
        cfg = config or {}

        variant = self._validate_variant(cfg.get("model_name", self._DEFAULT_VARIANT))
        device = self._resolve_device(cfg.get("device"))

        logger.info(
            "DINOv2FeatureExtractor initializing: variant=%s, device=%s",
            variant,
            device,
        )

        model = self._load_backbone(variant)
        model = self._place_on_device(model, device)
        self._set_inference_mode(model)

        self._model = model
        self._device = device
        self._variant = variant
        self._initialized = True

        logger.info("DINOv2FeatureExtractor initialized: %s", self.name)

    def extract(self, image_path: Path) -> ImageFeatures:
        """Extract DINOv2 patch and global features from a single image.

        Implements the five steps documented by
        ``FeatureExtractor.extract()``:

        1. **Load** — ``_load_image`` decodes ``image_path`` via PIL and
           records the original ``(height, width)`` *before* resizing.
           RGB and Thermal images are handled by the same call: PIL's
           ``convert("RGB")`` replicates a single luminance channel
           into three identical channels for grayscale Thermal frames,
           and is a harmless no-op for already-RGB frames. No separate
           Thermal code path exists, per the "Single Unified Pipeline"
           decision in ``AI_HANDOFF.md``.
        2. **Preprocess** — ``_preprocess`` resizes to
           ``(_INPUT_SIZE, _INPUT_SIZE)``, converts to a tensor,
           normalises with the ImageNet statistics DINOv2 was
           pretrained with, batches (``unsqueeze(0)``), and moves the
           result to ``self._device``.
        3. **Forward pass** — ``self._model.forward_features(...)`` is
           called under ``torch.no_grad()`` (the model's parameters
           already have ``requires_grad_(False)`` from ``initialize()``,
           but ``no_grad()`` additionally avoids building an autograd
           graph for the activations themselves).
        4. **Collect outputs** — the patch tokens
           (``"x_norm_patchtokens"``) become ``patch_features`` and the
           CLS token (``"x_norm_clstoken"``) becomes ``global_features``,
           both with the batch dimension of size 1 squeezed out so that
           their shapes match the ``(num_patches, embed_dim)`` /
           ``(embed_dim,)`` contract documented on ``ImageFeatures``.
        5. **Package** — ``patch_grid_shape`` is derived from the fixed
           ``_INPUT_SIZE`` / ``_PATCH_SIZE`` ratio (identical for every
           call, since every image is resized to the same square
           resolution), and everything is wrapped in an
           ``ImageFeatures`` instance together with traceability
           metadata.

        Args:
            image_path:
                Absolute path to the image file (RGB or Thermal; JPEG
                and PNG at minimum, per whatever formats PIL's
                installed codecs support).

        Returns:
            ``ImageFeatures`` populated with:

            * ``patch_features`` — ``torch.Tensor`` of shape
              ``(num_patches, embed_dim)``.
            * ``global_features`` — ``torch.Tensor`` of shape
              ``(embed_dim,)``.
            * ``patch_grid_shape`` — ``(_INPUT_SIZE // _PATCH_SIZE,) * 2``,
              e.g. ``(37, 37)`` at the default resolution.
            * ``source_image_path`` — the resolved, validated path.
            * ``image_size_hw`` — the *original* image dimensions,
              captured before resizing.
            * ``metadata`` — ``model_name``, ``input_size_hw``,
              ``extraction_time_ms``, and ``device``.

        Raises:
            FileNotFoundError:
                If ``image_path`` does not point to an existing file.
            RuntimeError:
                If ``initialize()`` has not been called yet, if the
                image file exists but cannot be decoded, or if the
                forward pass itself fails (e.g. device error).

        Note:
            This method is re-entrant: it holds no state between calls
            beyond the already-loaded, read-only model and device set
            up by ``initialize()``, matching the contract documented by
            ``FeatureExtractor.extract()``.
        """
        self._require_initialized()
        resolved_path = self._validate_image_path(image_path)

        pil_image, image_size_hw = self._load_image(resolved_path)
        input_tensor = self._preprocess(pil_image)

        try:
            forward_start = time.perf_counter()
            with torch.no_grad():
                outputs = self._model.forward_features(input_tensor)
            extraction_time_ms = (time.perf_counter() - forward_start) * 1000.0
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise RuntimeError(
                f"DINOv2 forward pass failed for {resolved_path}: {exc}"
            ) from exc

        patch_features = outputs["x_norm_patchtokens"].squeeze(0)
        global_features = outputs["x_norm_clstoken"].squeeze(0)

        return ImageFeatures(
            patch_features=patch_features,
            patch_grid_shape=self._compute_patch_grid_shape(),
            source_image_path=resolved_path,
            image_size_hw=image_size_hw,
            global_features=global_features,
            metadata={
                "model_name": self._variant,
                "input_size_hw": (self._INPUT_SIZE, self._INPUT_SIZE),
                "extraction_time_ms": extraction_time_ms,
                "device": str(self._device),
            },
        )

    def cleanup(self) -> None:
        """Release the DINOv2 model and any device-side memory.

        Deletes the model reference (allowing Python's garbage
        collector to free host memory) and, when the model was placed
        on a CUDA device, explicitly clears the CUDA caching allocator
        via ``torch.cuda.empty_cache()`` so that GPU memory is returned
        promptly rather than waiting for the next allocation to trigger
        it.

        This method is **idempotent**: calling it when the extractor
        was never initialized, or calling it multiple times in a row,
        is always safe and never raises.
        """
        was_on_cuda = self._device is not None and self._device.type == "cuda"

        if self._model is not None:
            del self._model

        self._model = None
        self._device = None
        self._variant = None
        self._initialized = False

        if was_on_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("DINOv2FeatureExtractor cleaned up.")

    # -----------------------------------------------------------------
    # Private helpers — lifecycle guard
    # -----------------------------------------------------------------

    def _require_initialized(self) -> None:
        """Guard that raises unless ``initialize()`` has succeeded.

        Mirrors the guard pattern used by ``CoarseToFineEngine`` (see
        ``engine.py``) so that calling ``extract()`` before
        ``initialize()``, or after ``cleanup()``, fails fast with a
        clear, actionable message instead of an opaque
        ``AttributeError`` on ``self._model``.

        Raises:
            RuntimeError:
                If ``initialize()`` has not been called, or if
                ``cleanup()`` has been called since the last
                ``initialize()``.
        """
        if not self._initialized:
            raise RuntimeError(
                "DINOv2FeatureExtractor.initialize() must be called "
                "before extract()."
            )

    # -----------------------------------------------------------------
    # Private helpers — extraction (image loading, preprocessing, grid)
    # -----------------------------------------------------------------

    @staticmethod
    def _load_image(image_path: Path) -> "Tuple[Image.Image, Tuple[int, int]]":
        """Load an image file and record its original pixel dimensions.

        Handles both RGB and Thermal (grayscale / single-channel)
        source images through the same code path:
        ``Image.convert("RGB")`` replicates a single luminance channel
        into three identical channels when the source is grayscale,
        and is effectively a no-op (aside from dropping any alpha or
        palette info) when the source is already RGB. This is the
        entire mechanism by which this extractor supports both
        modalities without a separate Thermal-specific pipeline, per
        the "Single Unified Pipeline" decision recorded in
        ``AI_HANDOFF.md`` and ``CLAUDE.md``.

        Args:
            image_path: Resolved, existing path to the image file (as
                returned by ``FeatureExtractor._validate_image_path``).

        Returns:
            A ``(pil_image, image_size_hw)`` tuple, where ``pil_image``
            is the loaded image in RGB mode and ``image_size_hw`` is
            its original ``(height, width)`` in pixels, captured
            *before* the resizing performed by ``_preprocess``.

        Raises:
            RuntimeError:
                If the file exists but cannot be decoded as an image
                (corrupt file, unsupported/unregistered format, ...).
        """
        try:
            with Image.open(image_path) as raw_image:
                rgb_image = raw_image.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise RuntimeError(
                f"Failed to decode image at {image_path}: {exc}"
            ) from exc

        width, height = rgb_image.size
        return rgb_image, (height, width)

    def _preprocess(self, pil_image: "Image.Image") -> "torch.Tensor":
        """Resize, normalise, and tensorise an image for DINOv2 inference.

        Builds and applies a ``torchvision.transforms`` pipeline that
        matches the standard DINOv2 evaluation preprocessing: a square
        resize to ``_INPUT_SIZE`` (divisible by ``_PATCH_SIZE``, as
        required by the model's patch embedding), conversion to a
        ``[0, 1]``-range float tensor, and per-channel normalisation
        with the ImageNet statistics DINOv2 was pretrained with. The
        same pipeline is used for both RGB and Thermal inputs — by the
        time this method runs, ``_load_image`` has already converted
        Thermal frames into 3-channel form, so there is nothing
        modality-specific left to branch on here.

        Args:
            pil_image: RGB ``PIL.Image`` as returned by ``_load_image``.

        Returns:
            A batched ``torch.Tensor`` of shape
            ``(1, 3, _INPUT_SIZE, _INPUT_SIZE)``, moved to
            ``self._device``.
        """
        preprocess = transforms.Compose(
            [
                transforms.Resize((self._INPUT_SIZE, self._INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self._NORM_MEAN, std=self._NORM_STD),
            ]
        )
        tensor = preprocess(pil_image).unsqueeze(0)
        return tensor.to(self._device)

    @classmethod
    def _compute_patch_grid_shape(cls) -> Tuple[int, int]:
        """Compute the spatial patch grid shape for the fixed input size.

        Every call to ``extract()`` resizes its input to the same
        square ``_INPUT_SIZE``, so the resulting patch grid shape is
        identical for every image and can be computed without
        inspecting the model output.

        Returns:
            ``(_INPUT_SIZE // _PATCH_SIZE, _INPUT_SIZE // _PATCH_SIZE)``,
            e.g. ``(37, 37)`` for the default 518x518 input with a
            patch size of 14.
        """
        side = cls._INPUT_SIZE // cls._PATCH_SIZE
        return (side, side)

    # -----------------------------------------------------------------
    # Private helpers — configuration validation
    # -----------------------------------------------------------------

    @classmethod
    def _validate_variant(cls, model_name: Any) -> str:
        """Validate the requested DINOv2 variant string.

        Args:
            model_name: The raw ``config["model_name"]`` value (or the
                class default), expected to be a string.

        Returns:
            The validated variant string, unchanged.

        Raises:
            ValueError:
                If ``model_name`` is not a string, or is a string that
                is not one of ``_SUPPORTED_VARIANTS``.
        """
        if not isinstance(model_name, str):
            raise ValueError(
                f"'model_name' must be a string, got "
                f"{type(model_name).__name__}: {model_name!r}."
            )
        if model_name not in cls._SUPPORTED_VARIANTS:
            supported = ", ".join(sorted(cls._SUPPORTED_VARIANTS))
            raise ValueError(
                f"Unsupported DINOv2 variant {model_name!r}. "
                f"Supported variants: {supported}."
            )
        return model_name

    @staticmethod
    def _resolve_device(device_config: Optional[Any]) -> "torch.device":
        """Resolve and validate the target ``torch.device``.

        Args:
            device_config: The raw ``config["device"]`` value. ``None``
                triggers auto-detection.

        Returns:
            A validated ``torch.device`` instance. When
            ``device_config`` is ``None``, this is ``cuda`` if
            ``torch.cuda.is_available()`` else ``cpu``.

        Raises:
            ValueError:
                If ``device_config`` is neither ``None`` nor a string,
                or if it is a string that ``torch.device`` cannot
                parse.
            RuntimeError:
                If ``device_config`` resolves to a CUDA device but CUDA
                is not available in this environment.
        """
        if device_config is None:
            auto_device = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.device(auto_device)

        if not isinstance(device_config, str):
            raise ValueError(
                f"'device' must be a string (e.g. 'cpu', 'cuda', "
                f"'cuda:0'), got {type(device_config).__name__}: "
                f"{device_config!r}."
            )

        try:
            device = torch.device(device_config)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Invalid 'device' string {device_config!r}: {exc}"
            ) from exc

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Device {device_config!r} requests CUDA, but CUDA is "
                f"not available on this machine. Use 'cpu' or omit "
                f"'device' to auto-detect."
            )

        return device

    # -----------------------------------------------------------------
    # Private helpers — model loading
    # -----------------------------------------------------------------

    @classmethod
    def _load_backbone(cls, variant: str) -> "torch.nn.Module":
        """Load a DINOv2 backbone via ``torch.hub``.

        Args:
            variant: A pre-validated DINOv2 variant string (one of
                ``_SUPPORTED_VARIANTS``).

        Returns:
            The loaded backbone as a ``torch.nn.Module``, still on its
            default device (typically CPU) and in its default
            (training) mode — placement and mode are handled by
            ``_place_on_device`` / ``_set_inference_mode``.

        Raises:
            RuntimeError:
                If ``torch.hub.load`` fails for any reason (no network
                access, corrupt local hub cache, invalid entry point,
                incompatible torch/torchvision version, etc.). The
                original exception is chained via ``from exc`` so the
                root cause is never hidden.
        """
        try:
            return torch.hub.load(cls._TORCH_HUB_REPO, variant)
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise RuntimeError(
                f"Failed to load DINOv2 backbone {variant!r} from "
                f"torch.hub repository {cls._TORCH_HUB_REPO!r}. This "
                f"usually means no network access is available to "
                f"download the weights, or the local torch.hub cache "
                f"is corrupt. Original error: {exc}"
            ) from exc

    @staticmethod
    def _place_on_device(
        model: "torch.nn.Module", device: "torch.device",
    ) -> "torch.nn.Module":
        """Move a loaded model to the resolved device.

        Args:
            model: The freshly loaded DINOv2 backbone.
            device: The validated target device.

        Returns:
            The same model, moved to ``device`` (``Module.to()``
            mutates and returns ``self``, but the return value is used
            explicitly here rather than relying on that mutation, to
            keep this helper correct even if a future subclass swaps
            in a wrapper whose ``.to()`` returns a new object).

        Raises:
            RuntimeError:
                If moving the model fails (e.g. out of GPU memory).
        """
        try:
            return model.to(device)
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise RuntimeError(
                f"Failed to move DINOv2 backbone to device {device}: {exc}"
            ) from exc

    @staticmethod
    def _set_inference_mode(model: "torch.nn.Module") -> None:
        """Put a model into a safe, non-trainable inference state.

        Two independent things are done here, both required for a
        benchmark extractor that never trains:

        1. ``model.eval()`` — switches layers with train/eval-dependent
           behaviour (e.g. dropout, batch normalisation) into inference
           mode.
        2. ``requires_grad_(False)`` on every parameter — disables
           gradient tracking so that ``extract()`` calls in a future
           step can run without wrapping every call in
           ``torch.no_grad()`` and without accumulating autograd graph
           memory, since no parameter of this frozen backbone is ever
           meant to be updated.

        Args:
            model: The device-placed DINOv2 backbone to mutate in
                place.
        """
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"variant={self._variant!r}, "
            f"device={self._device}, "
            f"initialized={self._initialized})"
        )
