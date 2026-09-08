"""Self-contained movie I/O for the independently installable audit core.

The full research project retains ``src.inference.movie_io``.  This module
implements the same public loading and scale-bookkeeping interface without
requiring the rest of that source tree, so the formal contract/reference-free
core can be distributed as a bounded package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from .contracts import validate_numeric_array


_TIFF_EXT = {".tif", ".tiff"}
_NPY_EXT = {".npy"}
_NPZ_EXT = {".npz"}
_H5_EXT = {".h5", ".hdf5"}
_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def supported_extensions() -> list[str]:
    return sorted(_TIFF_EXT | _NPY_EXT | _NPZ_EXT | _H5_EXT | _VIDEO_EXT)


def _read_tiff(path: Path) -> np.ndarray:
    try:
        return tifffile.memmap(path)
    except Exception:
        return tifffile.imread(path)


def _load_npz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        for key in ("movie", "denoised", "stack", "arr_0"):
            if key in archive.files:
                return np.asarray(archive[key])
        if archive.files:
            return np.asarray(archive[archive.files[0]])
    raise ValueError(f"{path}: .npz contains no arrays")


def _load_h5(path: Path) -> np.ndarray:
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            f"Reading {path!s} requires the optional h5py dependency; "
            "install calcium-denoiser-audit[hdf5] or convert to TIFF/NumPy"
        ) from exc
    with h5py.File(path, "r") as handle:
        candidates: list[tuple[str, object]] = []
        handle.visititems(
            lambda name, obj: candidates.append((name, obj)) if hasattr(obj, "shape") else None
        )
        named = {name: obj for name, obj in candidates}
        for key in ("mov", "movie", "data", "images", "Y"):
            if key in named:
                return np.asarray(named[key][()])
        for _, obj in candidates:
            if getattr(obj, "ndim", 0) == 3:
                return np.asarray(obj[()])
        if candidates:
            return np.asarray(candidates[0][1][()])
    raise ValueError(f"{path}: no datasets found in HDF5 file")


def _load_video(path: Path) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            f"Reading video {path!s} requires OpenCV; install the project bridge or convert the movie"
        ) from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame.astype(np.float32))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"video has no readable frames: {path}")
    return np.stack(frames, axis=0)


def canonicalize_stack(array: np.ndarray, stack_order: str = "auto") -> np.ndarray:
    """Return a 2-D/3-D array in canonical ``[T,H,W]`` order."""
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D stack, found shape={array.shape}")
    order = stack_order.upper()
    if order == "THW":
        return array
    if order == "HWT":
        return np.moveaxis(array, -1, 0)
    if order != "AUTO":
        raise ValueError("stack_order must be one of: auto, THW, HWT")
    if array.shape[-1] > max(array.shape[0], array.shape[1]) and array.shape[-1] > 512:
        return np.moveaxis(array, -1, 0)
    return array


def load_movie(
    path: str | Path,
    stack_order: str = "auto",
    max_frames: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load a supported input as contiguous float32 ``[T,H,W]`` plus provenance metadata."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"input movie not found: {source}")
    extension = source.suffix.lower()
    if extension in _TIFF_EXT:
        raw, source_format = np.asarray(_read_tiff(source)), "tiff"
    elif extension in _NPY_EXT:
        raw, source_format = np.asarray(np.load(source, allow_pickle=False)), "npy"
    elif extension in _NPZ_EXT:
        raw, source_format = _load_npz(source), "npz"
    elif extension in _H5_EXT:
        raw, source_format = _load_h5(source), "hdf5"
    elif extension in _VIDEO_EXT:
        raw, source_format = _load_video(source), "video"
    else:
        raise ValueError(
            f"unsupported input extension {extension!r}; supported: {', '.join(supported_extensions())}"
        )

    validate_numeric_array(raw)
    original_shape = tuple(int(size) for size in raw.shape)
    original_dtype = str(raw.dtype)
    canonical = np.asarray(canonicalize_stack(raw, stack_order=stack_order))
    canonical_shape = list(canonical.shape)
    if max_frames is not None and int(max_frames) <= 0:
        raise ValueError("max_frames must be a positive integer")
    truncated = max_frames is not None and canonical.shape[0] > int(max_frames)
    if truncated:
        canonical = canonical[: int(max_frames)]
    stack = np.ascontiguousarray(canonical, dtype=np.float32)
    metadata: dict[str, object] = {
        "source_format": source_format,
        "source_path": str(source),
        "original_shape": list(original_shape),
        "original_dtype": original_dtype,
        "canonical_shape": canonical_shape,
        "n_frames": int(stack.shape[0]),
        "height": int(stack.shape[1]),
        "width": int(stack.shape[2]),
        "stack_order": stack_order,
        "truncated_to_max_frames": bool(truncated),
    }
    return stack, metadata


def normalize_for_model(
    stack: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float | str]]:
    array = np.asarray(stack, dtype=np.float32)
    data_min, data_max = float(np.min(array)), float(np.max(array))
    normalized = (array - data_min) / (data_max - data_min + eps)
    scale: dict[str, float | str] = {
        "method": "minmax",
        "data_min": data_min,
        "data_max": data_max,
        "eps": float(eps),
        "span": data_max - data_min,
    }
    return normalized.astype(np.float32), scale


def restore_scale(denoised: np.ndarray, scale: dict[str, float | str] | None) -> np.ndarray:
    if not scale or scale.get("method") != "minmax":
        return np.asarray(denoised, dtype=np.float32)
    data_min = float(scale["data_min"])
    data_max = float(scale["data_max"])
    span = data_max - data_min + float(scale.get("eps", 0.0))
    return (np.asarray(denoised, dtype=np.float32) * span + data_min).astype(np.float32)


def save_array(
    path: str | Path,
    array: np.ndarray,
    compressed: bool = False,
    key: str = "denoised",
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = np.asarray(array, dtype=np.float32)
    if destination.suffix.lower() == ".npz" or compressed:
        destination = destination if destination.suffix.lower() == ".npz" else destination.with_suffix(".npz")
        np.savez_compressed(destination, **{key: value})
    else:
        destination = destination if destination.suffix.lower() == ".npy" else destination.with_suffix(".npy")
        np.save(destination, value, allow_pickle=False)
    return str(destination)
