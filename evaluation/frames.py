"""Video frames for the judge's ``video`` block (D30, D31): selection, decoding and encoding.

- ``plan_frames`` is pure: from the event step, the episode's last step, the video's length and the
  config it picks the steps to show. No decoder is involved, so it is tested without one.
- ``FrameSource`` is the Protocol the renderer depends on. ``PyAVFrameSource`` is the real decoder
  (PyAV decodes, Pillow resizes and encodes JPEG); both are imported lazily, so nothing here needs
  the optional ``video`` extra until a frame is actually read. ``CachingFrameSource`` remembers
  frames across events (windows of neighbouring events overlap).
- JPEG bytes depend only on the pixels, the size and the quality, never on a clock, so the same
  config always yields the same request hash (for one PyAV / Pillow build).
"""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol

from .configuration import VideoEpisode, VideoResize, VideoSampling

VIDEO_EXTRA_HINT = "install the optional 'video' extra: uv sync --extra video (PyAV and Pillow)"
JPEG_MEDIA_TYPE = "image/jpeg"
FPS_RELATIVE_TOLERANCE = 1e-3  # the video's own frame rate must match the configured one
TIME_DECIMALS = 6  # step * step_seconds is rounded to this many decimals before it is shown


class FrameSourceError(ValueError):
    """A video is missing, unreadable, or does not match its configuration. Messages are safe to log."""


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float


@dataclass(frozen=True)
class EncodedFrame:
    frame_index: int
    jpeg: bytes
    width: int
    height: int

    @cached_property
    def data_url(self) -> str:
        """``data:image/jpeg;base64,...`` (computed once, shared by every request using the frame)."""
        return f"data:{JPEG_MEDIA_TYPE};base64,{base64.b64encode(self.jpeg).decode('ascii')}"


class FrameSource(Protocol):
    def info(self, path: Path) -> VideoInfo:
        """Frame count and frame rate. Raises ``FrameSourceError`` for a missing or unreadable video."""
        ...

    def frames(self, path: Path, frame_indices: Sequence[int], resize: VideoResize) -> tuple[EncodedFrame, ...]:
        """JPEG frames for the 0-based ``frame_indices`` (ascending), resized to fit and encoded."""
        ...


# --- selection (pure) -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FramePlan:
    steps: tuple[int, ...]  # ascending; always contains the event step when it has a frame
    frame_indices: tuple[int, ...]  # same length and order as ``steps``


def frame_index_of_step(step: int, episode: VideoEpisode) -> int:
    return episode.frame_offset + step * episode.frames_per_step


def check_video_info(info: VideoInfo, episode: VideoEpisode) -> None:
    if info.frame_count < 1:
        raise FrameSourceError(f"video {episode.path} has no frames")
    if not math.isclose(info.fps, episode.fps, rel_tol=FPS_RELATIVE_TOLERANCE):
        raise FrameSourceError(f"video {episode.path} runs at {info.fps} fps, the config says {episode.fps}")


def _candidate_steps(event_step: int, sampling: VideoSampling) -> list[int]:
    """Event step +- k * stride, so the event step itself is always a candidate."""
    before = sampling.before_steps // sampling.stride_steps
    after = sampling.after_steps // sampling.stride_steps
    return [event_step + k * sampling.stride_steps for k in range(-before, after + 1)]


def plan_frames(
    *, event_step: int, last_step: int, info: VideoInfo, episode: VideoEpisode, sampling: VideoSampling
) -> FramePlan:
    """The steps to show for one event.

    Candidates are ``event_step + k * stride`` inside the configured window. A candidate is clipped
    when its step lies outside ``[0, last_step]`` or its frame outside ``[0, frame_count - 1]``. When
    more than ``max_frames`` remain, the ones closest to the event are kept (the earlier one on a tie).
    """
    check_video_info(info, episode)
    kept = [
        step
        for step in _candidate_steps(event_step, sampling)
        if 0 <= step <= last_step and 0 <= frame_index_of_step(step, episode) < info.frame_count
    ]
    closest = sorted(kept, key=lambda step: (abs(step - event_step), step))[: sampling.max_frames]
    steps = tuple(sorted(closest))
    return FramePlan(steps=steps, frame_indices=tuple(frame_index_of_step(step, episode) for step in steps))


def fitted_size(width: int, height: int, resize: VideoResize) -> tuple[int, int]:
    """Largest size within ``max_width x max_height`` that keeps the aspect ratio; never upscales."""
    scale = min(resize.max_width / width, resize.max_height / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


# --- decoding -------------------------------------------------------------------------------------


class PyAVFrameSource:
    """Decodes with PyAV and encodes JPEG with Pillow; both are imported on first use."""

    def info(self, path: Path) -> VideoInfo:
        _require_file(path)
        return _open_stream(_import_av(), path, lambda container, stream: _stream_info(stream, path))

    def frames(self, path: Path, frame_indices: Sequence[int], resize: VideoResize) -> tuple[EncodedFrame, ...]:
        _require_file(path)
        av = _import_av()
        image_module = _import_pillow()
        wanted = sorted(set(frame_indices))
        if not wanted:
            return ()

        def read(container: Any, stream: Any) -> tuple[EncodedFrame, ...]:
            return _read_frames(container, stream, wanted, resize, image_module, path)

        return _open_stream(av, path, read)


def _import_av() -> Any:
    try:
        import av
    except ImportError as exc:
        raise FrameSourceError(f"PyAV is not installed; {VIDEO_EXTRA_HINT}") from exc
    return av


def _import_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise FrameSourceError(f"Pillow is not installed; {VIDEO_EXTRA_HINT}") from exc
    return Image


def _require_file(path: Path) -> None:
    if not Path(path).is_file():
        raise FrameSourceError(f"video file not found: {path}")


def _open_stream(av: Any, path: Path, action: Any) -> Any:
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise FrameSourceError(f"video {path} has no video stream")
            return action(container, container.streams.video[0])
    except FrameSourceError:
        raise
    except Exception as exc:  # PyAV raises its own error types; the cause is chained, not echoed
        raise FrameSourceError(f"cannot decode video {path}: {type(exc).__name__}") from exc


def _stream_info(stream: Any, path: Path) -> VideoInfo:
    if not stream.frames or stream.average_rate is None:
        raise FrameSourceError(f"video {path} does not report its frame count or frame rate")
    return VideoInfo(frame_count=int(stream.frames), fps=float(stream.average_rate))


def _read_frames(
    container: Any, stream: Any, wanted: list[int], resize: VideoResize, image_module: Any, path: Path
) -> tuple[EncodedFrame, ...]:
    """Seek to the key frame before the first wanted frame, then decode forward."""
    info = _stream_info(stream, path)
    if wanted[0] < 0 or wanted[-1] >= info.frame_count:
        raise FrameSourceError(f"frame index outside video {path} (0..{info.frame_count - 1})")
    ticks_per_frame = 1 / (info.fps * float(stream.time_base))
    start_ticks = stream.start_time or 0
    container.seek(start_ticks + int(wanted[0] * ticks_per_frame), stream=stream, backward=True)
    found: dict[int, EncodedFrame] = {}
    for frame in container.decode(stream):
        index = round((frame.pts - start_ticks) / ticks_per_frame)
        if index in wanted:
            found[index] = _encode(frame, index, resize, image_module)
        if index >= wanted[-1]:
            break
    missing = [index for index in wanted if index not in found]
    if missing:
        raise FrameSourceError(f"could not decode frames {missing} of video {path}")
    return tuple(found[index] for index in wanted)


def _encode(frame: Any, index: int, resize: VideoResize, image_module: Any) -> EncodedFrame:
    image = frame.to_image()  # RGB
    size = fitted_size(image.width, image.height, resize)
    if size != image.size:
        image = image.resize(size, image_module.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=resize.jpeg_quality)
    return EncodedFrame(frame_index=index, jpeg=buffer.getvalue(), width=image.width, height=image.height)


# --- cache ----------------------------------------------------------------------------------------


class CachingFrameSource:
    """Remembers ``info`` and decoded frames, so overlapping event windows decode each frame once."""

    def __init__(self, inner: FrameSource) -> None:
        self._inner = inner
        self._info: dict[Path, VideoInfo] = {}
        self._frames: dict[tuple[Path, int, VideoResize], EncodedFrame] = {}

    def info(self, path: Path) -> VideoInfo:
        if path not in self._info:
            self._info[path] = self._inner.info(path)
        return self._info[path]

    def frames(self, path: Path, frame_indices: Sequence[int], resize: VideoResize) -> tuple[EncodedFrame, ...]:
        missing = [index for index in frame_indices if (path, index, resize) not in self._frames]
        for frame in self._inner.frames(path, missing, resize) if missing else ():
            self._frames[(path, frame.frame_index, resize)] = frame
        return tuple(self._frames[(path, index, resize)] for index in frame_indices)
