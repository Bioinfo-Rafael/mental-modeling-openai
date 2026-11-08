"""Judge video block (L3, D30), part 1: config, frame selection, frame cache and the PyAV decoder.

Everything runs without a decoder except the one smoke test, which reads two frames of a committed
video and is skipped when PyAV or Pillow is missing. Request layout and CLI: test_eval_video_request.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from evaluation.configuration import (
    JudgeConfigError,
    VideoEpisode,
    VideoResize,
    VideoSampling,
    parse_config,
)
from evaluation.frames import (
    CachingFrameSource,
    FrameSourceError,
    PyAVFrameSource,
    VideoInfo,
    fitted_size,
    plan_frames,
)
from evaluation.rendering import (
    VideoContext,
    build_judge_request,
)
from evaluation.schemas import EventKeyPoints, KeyPoint, load_translator_run
from tests.translator.tr_fakes import WIN01
from translate import paths

from .eval_fakes import (
    A1_REQUEST,
    FakeFrameSource,
    judge_config_raw,
    make_video_config,
    video_config_raw,
)

KEY_POINTS = (KeyPoint("kp-001", "[PLACEHOLDER] key point one"),)
LAST_STEP = 340
EPISODES = ("mhi_v1.3_ic100601_win", "ours_seed100100_ic100601_loss", "ours_seed100100_ic100606_win")
RUN_NAME = "video1"
COMMITTED_VIDEO = paths.ROOT / "data/mhi/action_videos/mhi_v1.3_ic100601_win_with_actions.mp4"
EPISODE = VideoEpisode(path=Path("v.mp4"), fps=20.0, frame_offset=0, frames_per_step=1, step_seconds=0.2)
SAMPLING = VideoSampling(strategy="event_window", before_steps=20, after_steps=20, stride_steps=5, max_frames=12)
RESIZE = VideoResize(max_width=960, max_height=540, jpeg_quality=85)
BIG_VIDEO = VideoInfo(frame_count=400, fps=20.0)


def _sampling(**changes: int) -> VideoSampling:
    return VideoSampling(**{**SAMPLING.__dict__, **changes})


def _plan(event_step: int, *, last_step: int = LAST_STEP, info: VideoInfo = BIG_VIDEO, **sampling: int) -> Any:
    return plan_frames(
        event_step=event_step, last_step=last_step, info=info, episode=EPISODE, sampling=_sampling(**sampling)
    )


@pytest.fixture(scope="module")
def events(translator_run: Path) -> tuple[dict[str, Any], ...]:
    return load_translator_run(translator_run).translation_events


def _event(events: tuple[dict[str, Any], ...], event_id: str) -> dict[str, Any]:
    return next(event for event in events if event["event_id"] == event_id)


def _request(
    events: tuple[dict[str, Any], ...],
    event_id: str = A1_REQUEST,
    *,
    source: FakeFrameSource | None = None,
    last_step: int = LAST_STEP,
    config: Any = None,
) -> Any:
    source = source or FakeFrameSource.for_committed_videos()
    return build_judge_request(
        event=_event(events, event_id),
        event_log=events,
        key_points=EventKeyPoints(event_id, KEY_POINTS),
        config=config or make_video_config(),
        video=VideoContext(frame_source=source, episode_id=WIN01, last_step=last_step),
    )


# --- config -----------------------------------------------------------------------------------------


def test_committed_video_config_enables_the_three_episodes_and_text_config_keeps_it_off() -> None:
    video_config = parse_config(video_config_raw())
    text_config = parse_config(judge_config_raw())

    assert video_config.inputs.video.include and "video" in video_config.inputs.included()
    assert sorted(video_config.inputs.video.videos) == sorted(EPISODES)
    assert all(episode.path.is_absolute() for episode in video_config.inputs.video.videos.values())
    assert not text_config.inputs.video.include and "video" not in text_config.inputs.included()
    assert text_config.inputs.video.image_tokens.status == "unverified"


def test_committed_videos_exist_and_match_their_episode_ids() -> None:
    for episode_id, episode in make_video_config().inputs.video.videos.items():
        assert episode.path.is_file(), episode_id
        assert episode.path.name == f"{episode_id}_with_actions.mp4"


def test_video_config_paths_resolve_against_the_repository_root() -> None:
    episode = make_video_config().inputs.video.videos[WIN01]

    assert episode.path == paths.ROOT / "data/mhi/action_videos" / f"{WIN01}_with_actions.mp4"
    assert (episode.fps, episode.frame_offset, episode.frames_per_step, episode.step_seconds) == (20, 0, 1, 0.2)


def test_video_options_are_read_from_the_config() -> None:
    video = make_video_config().inputs.video

    assert (video.sampling.before_steps, video.sampling.after_steps) == (20, 20)
    assert (video.sampling.stride_steps, video.sampling.max_frames) == (5, 12)
    assert (video.resize.max_width, video.resize.max_height, video.resize.jpeg_quality) == (960, 540, 85)
    assert (video.detail, video.image_tokens.per_image, video.image_tokens.status) == ("high", 1105, "unverified")


@pytest.mark.parametrize("key", ["videos", "sampling", "resize", "detail", "label_format", "image_tokens", "source"])
def test_each_video_key_is_required(key: str) -> None:
    raw = video_config_raw()
    video = {name: value for name, value in raw["inputs"]["video"].items() if name != key}

    with pytest.raises(JudgeConfigError, match=rf"missing \['{key}'\]"):
        parse_config({**raw, "inputs": {**raw["inputs"], "video": video}})


def test_unknown_video_keys_are_rejected() -> None:
    with pytest.raises(JudgeConfigError, match=r"unknown \['fps'\]"):
        make_video_config(fps=20)
    with pytest.raises(JudgeConfigError, match=r"unknown \['stride'\]"):
        make_video_config(sampling={**video_config_raw()["inputs"]["video"]["sampling"], "stride": 2})
    with pytest.raises(JudgeConfigError, match=r"unknown \['formula'\]"):
        make_video_config(image_tokens={"per_image": 1, "status": "unverified", "formula": "x"})


def _episode_entry(**changes: Any) -> dict[str, Any]:
    return {**video_config_raw()["inputs"]["video"]["videos"][WIN01], **changes}


@pytest.mark.parametrize(
    "changes",
    [
        {"path": ""},
        {"path": 7},
        {"fps": 0},
        {"fps": True},
        {"fps": float("inf")},
        {"frame_offset": -1},
        {"frame_offset": 1.5},
        {"frames_per_step": 0},
        {"frames_per_step": True},
        {"step_seconds": 0},
        {"step_seconds": -0.2},
        {"step_seconds": "0.2"},
    ],
)
def test_episode_entry_values_are_validated(changes: dict[str, Any]) -> None:
    with pytest.raises(JudgeConfigError):
        make_video_config(videos={WIN01: _episode_entry(**changes)})


@pytest.mark.parametrize("missing", ["path", "fps", "frame_offset", "frames_per_step", "step_seconds"])
def test_episode_entry_without_a_key_is_rejected(missing: str) -> None:
    entry = {k: v for k, v in _episode_entry().items() if k != missing}

    with pytest.raises(JudgeConfigError, match=rf"missing \['{missing}'\]"):
        make_video_config(videos={WIN01: entry})


def test_episode_entry_with_an_unknown_key_is_rejected() -> None:
    with pytest.raises(JudgeConfigError, match=r"unknown \['codec'\]"):
        make_video_config(videos={WIN01: _episode_entry(codec="h264")})


@pytest.mark.parametrize("episode_id", ["", "a/b", "a b", "../x", "a:b"])
def test_episode_ids_must_be_safe_ids(episode_id: str) -> None:
    with pytest.raises(JudgeConfigError, match="keys must match"):
        make_video_config(videos={episode_id: _episode_entry()})


def test_included_video_needs_at_least_one_episode_but_an_excluded_one_may_have_none() -> None:
    with pytest.raises(JudgeConfigError, match="at least one episode"):
        make_video_config(videos={})

    config = make_video_config(videos={}, include=False)

    assert config.inputs.video.videos == {} and "video" not in config.inputs.included()


@pytest.mark.parametrize(
    "label",
    [
        "",
        "no fields",
        "only {step}",
        "only {time_s}",
        "{step} {time_s} {frame}",
        "{step:03d} {time_s}",
        "{step!r} {time_s}",
        "{} {time_s} {step}",
        "{step} {time_s} {",
        "{step}\n{time_s}",
        7,
    ],
)
def test_label_format_must_use_step_and_time_with_plain_known_fields(label: object) -> None:
    with pytest.raises(JudgeConfigError, match="label_format"):
        make_video_config(label_format=label)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"detail": "ultra"}, "detail"),
        ({"source": "other"}, "source"),
        ({"renderer": "png"}, "renderer"),
        ({"image_tokens": {"per_image": -1, "status": "unverified"}}, "per_image"),
        ({"image_tokens": {"per_image": 1.5, "status": "unverified"}}, "per_image"),
        ({"image_tokens": {"per_image": 100, "status": "guess"}}, "status"),
        ({"resize": {"max_width": 0, "max_height": 540, "jpeg_quality": 85}}, "max_width"),
        ({"resize": {"max_width": 960, "max_height": 540, "jpeg_quality": 0}}, "jpeg_quality"),
        ({"resize": {"max_width": 960, "max_height": 540, "jpeg_quality": 101}}, "jpeg_quality"),
    ],
)
def test_other_video_options_are_validated(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(JudgeConfigError, match=match):
        make_video_config(**changes)


@pytest.mark.parametrize(
    ("key", "value"), [("before_steps", -1), ("after_steps", -1), ("stride_steps", 0), ("max_frames", 0)]
)
def test_window_and_sampling_values_are_validated(key: str, value: int) -> None:
    sampling = {**video_config_raw()["inputs"]["video"]["sampling"], key: value}

    with pytest.raises(JudgeConfigError, match=key):
        make_video_config(sampling=sampling)


def test_a_zero_width_window_is_allowed() -> None:
    sampling = {**video_config_raw()["inputs"]["video"]["sampling"], "before_steps": 0, "after_steps": 0}

    assert make_video_config(sampling=sampling).inputs.video.sampling.before_steps == 0


# --- frame selection ----------------------------------------------------------------------------------


def test_window_is_event_step_plus_minus_strides() -> None:
    plan = _plan(100)

    assert plan.steps == (80, 85, 90, 95, 100, 105, 110, 115, 120)
    assert plan.frame_indices == plan.steps  # frame = step


def test_window_start_is_clipped_at_step_zero() -> None:
    assert _plan(0).steps == (0, 5, 10, 15, 20)
    assert _plan(7).steps == (2, 7, 12, 17, 22, 27)


def test_window_end_is_clipped_at_the_episode_last_step() -> None:
    assert _plan(335, last_step=340).steps == (315, 320, 325, 330, 335, 340)
    assert _plan(340, last_step=340).steps == (320, 325, 330, 335, 340)


def test_window_end_is_clipped_at_the_last_available_video_frame() -> None:
    plan = _plan(100, info=VideoInfo(frame_count=108, fps=20.0))  # frames 0..107

    assert plan.steps == (80, 85, 90, 95, 100, 105)


def test_frame_index_uses_offset_and_frames_per_step() -> None:
    episode = VideoEpisode(Path("v.mp4"), 20.0, frame_offset=3, frames_per_step=2, step_seconds=0.2)
    plan = plan_frames(
        event_step=10, last_step=LAST_STEP, info=VideoInfo(60, 20.0), episode=episode, sampling=_sampling(max_frames=3)
    )

    assert plan.steps == (5, 10, 15)
    assert plan.frame_indices == (13, 23, 33)


def test_frames_past_the_video_are_clipped_when_frames_per_step_is_two() -> None:
    episode = VideoEpisode(Path("v.mp4"), 20.0, frame_offset=0, frames_per_step=2, step_seconds=0.2)
    plan = plan_frames(
        event_step=20, last_step=LAST_STEP, info=VideoInfo(50, 20.0), episode=episode, sampling=_sampling()
    )

    assert plan.steps == (0, 5, 10, 15, 20)  # step 25 and later map to frames >= 50
    assert plan.frame_indices == (0, 10, 20, 30, 40)


def test_max_frames_keeps_those_closest_to_the_event_and_the_earlier_one_on_a_tie() -> None:
    assert _plan(100, max_frames=4).steps == (90, 95, 100, 105)
    assert _plan(100, max_frames=1).steps == (100,)
    assert _plan(100, max_frames=2).steps == (95, 100)


def test_max_frames_equal_to_candidates_keeps_all_and_stays_in_time_order() -> None:
    plan = _plan(100, max_frames=9)

    assert plan.steps == tuple(sorted(plan.steps)) and len(plan.steps) == 9


def test_window_not_a_multiple_of_the_stride_stops_at_the_last_whole_stride() -> None:
    assert _plan(100, before_steps=7, after_steps=12).steps == (95, 100, 105, 110)


def test_zero_width_window_shows_only_the_event_frame() -> None:
    assert _plan(100, before_steps=0, after_steps=0).steps == (100,)


def test_stride_larger_than_the_window_shows_only_the_event_frame() -> None:
    assert _plan(100, stride_steps=50).steps == (100,)


def test_event_beyond_the_video_leaves_no_frame_when_the_window_is_empty() -> None:
    plan = _plan(450, last_step=500, before_steps=0, after_steps=0)

    assert plan.steps == () and plan.frame_indices == ()


def test_video_with_another_frame_rate_than_configured_is_an_error() -> None:
    with pytest.raises(FrameSourceError, match="10.0 fps, the config says 20.0"):
        _plan(100, info=VideoInfo(400, 10.0))


def test_video_without_frames_is_an_error() -> None:
    with pytest.raises(FrameSourceError, match="no frames"):
        _plan(100, info=VideoInfo(0, 20.0))


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((1920, 1080), (960, 540)),
        ((960, 540), (960, 540)),
        ((100, 50), (100, 50)),  # never upscales
        ((1000, 1000), (540, 540)),
        ((540, 1080), (270, 540)),
        ((1920, 100), (960, 50)),
        ((10000, 1), (960, 1)),
    ],
)
def test_fitted_size_keeps_the_aspect_ratio_inside_the_box(size: tuple[int, int], expected: tuple[int, int]) -> None:
    assert fitted_size(*size, RESIZE) == expected


# --- frame source cache -------------------------------------------------------------------------------------


def test_caching_source_decodes_each_frame_once_and_shares_the_object() -> None:
    inner = FakeFrameSource.for_committed_videos()
    cache = CachingFrameSource(inner)
    path = Path("mhi_v1.3_ic100601_win_with_actions.mp4")

    first = cache.frames(path, [1, 2, 3], RESIZE)
    second = cache.frames(path, [2, 3, 4], RESIZE)

    assert [call[1] for call in inner.frame_calls] == [(1, 2, 3), (4,)]
    assert second[0] is first[1] and second[1] is first[2]
    assert [f.frame_index for f in second] == [2, 3, 4]
    assert cache.frames(path, [], RESIZE) == () and len(inner.frame_calls) == 2


def test_caching_source_asks_the_video_for_its_info_once() -> None:
    inner = FakeFrameSource.for_committed_videos()
    cache = CachingFrameSource(inner)
    path = Path("mhi_v1.3_ic100601_win_with_actions.mp4")

    assert cache.info(path) == cache.info(path) and len(inner.info_calls) == 1


def test_caching_source_keys_frames_by_resize() -> None:
    inner = FakeFrameSource.for_committed_videos()
    cache = CachingFrameSource(inner)
    path = Path("mhi_v1.3_ic100601_win_with_actions.mp4")

    cache.frames(path, [1], RESIZE)
    cache.frames(path, [1], VideoResize(480, 270, 85))

    assert len(inner.frame_calls) == 2


def test_caching_source_does_not_cache_a_missing_video() -> None:
    cache = CachingFrameSource(FakeFrameSource({}))

    for _ in range(2):
        with pytest.raises(FrameSourceError):
            cache.info(Path("nope.mp4"))


# --- decoder (lazy import, real smoke) -------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["av", "PIL"])
def test_missing_decoder_package_names_the_video_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: str
) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"not a video")
    monkeypatch.setitem(sys.modules, module, None)  # `import <module>` now raises ImportError

    with pytest.raises(FrameSourceError, match="'video' extra"):
        PyAVFrameSource().frames(video, [0], RESIZE)


def test_pyav_source_reports_a_missing_file_without_a_decoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "av", None)

    with pytest.raises(FrameSourceError, match="video file not found"):
        PyAVFrameSource().info(tmp_path / "missing.mp4")
    with pytest.raises(FrameSourceError, match="video file not found"):
        PyAVFrameSource().frames(tmp_path / "missing.mp4", [0], RESIZE)


def test_pyav_source_wraps_decode_errors(tmp_path: Path) -> None:
    pytest.importorskip("av")
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not an mp4 file")

    with pytest.raises(FrameSourceError, match="cannot decode video"):
        PyAVFrameSource().info(broken)


def test_pyav_source_decodes_two_frames_of_a_committed_video() -> None:
    pytest.importorskip("av")
    pytest.importorskip("PIL")
    source = PyAVFrameSource()

    info = source.info(COMMITTED_VIDEO)
    frames = source.frames(COMMITTED_VIDEO, [3, 12], RESIZE)

    assert (info.frame_count, info.fps) == (342, 20.0)
    assert [f.frame_index for f in frames] == [3, 12]
    assert all(f.jpeg[:3] == b"\xff\xd8\xff" and (f.width, f.height) == (960, 540) for f in frames)
    assert frames[0].jpeg != frames[1].jpeg
    assert source.frames(COMMITTED_VIDEO, [3], RESIZE)[0].jpeg == frames[0].jpeg  # deterministic encoding
