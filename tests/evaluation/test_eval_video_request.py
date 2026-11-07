"""Judge video block (L3, D30), part 2: request layout, redaction, token estimate, preview and journals.

Everything runs on a fake ``FrameSource``; no decoder, network or paid call. Config, frame selection
and the decoder: test_eval_video.py.
"""

from __future__ import annotations

import base64
import json
import shutil
import socket
from pathlib import Path
from typing import Any

import pytest

from evaluation import cli
from evaluation.configuration import (
    VideoEpisode,
    VideoResize,
    VideoSampling,
)
from evaluation.frames import (
    CachingFrameSource,
    EncodedFrame,
    FrameSourceError,
    VideoInfo,
    plan_frames,
)
from evaluation.rendering import (
    HEADING_VIDEO,
    IMAGE_OMITTED,
    JudgeRenderError,
    VideoContext,
    approx_input_tokens,
    approx_text_tokens,
    build_judge_request,
    redact_images,
)
from evaluation.schemas import EventKeyPoints, EventLogError, KeyPoint, load_translator_run, run_episode
from mtd_rl_xavier import paths
from mtd_rl_xavier.llm.openai_responses import build_responses_body
from mtd_rl_xavier.translator.schema import canonical_json, sha256_text
from tests.llm.llm_fakes import ScriptedResponsesClient
from tests.translator.tr_fakes import WIN01

from .eval_fakes import (
    A1_REQUEST,
    A1_START,
    EXAMPLE_JUDGED,
    EXAMPLE_KEY_POINTS,
    FAKE_FRAME_HEIGHT,
    FAKE_FRAME_WIDTH,
    VIDEO_CONFIG,
    FakeFrameSource,
    RecordingFactory,
    make_video_config,
    read_jsonl,
    verdict_response,
    video_config_raw,
    with_video,
    write_yaml,
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


# --- request layout --------------------------------------------------------------------------------------


def _content(request: Any) -> list[dict[str, Any]]:
    return list(request.openai_request.input_items)


def test_each_image_is_immediately_preceded_by_its_label_in_time_order(events: tuple[dict[str, Any], ...]) -> None:
    request = _request(events, A1_REQUEST)  # step 10: window 0..30

    items = _content(request)
    images = [i for i, item in enumerate(items) if item["type"] == "input_image"]
    assert [items[i - 1]["type"] for i in images] == ["input_text"] * len(images)
    labels = [items[i - 1]["text"] for i in images]
    assert labels == [
        "Video frame at step 0 (0.0 s, -10 steps from the judged step):",
        "Video frame at step 5 (1.0 s, -5 steps from the judged step):",
        "Video frame at step 10 (2.0 s, 0 steps from the judged step):",
        "Video frame at step 15 (3.0 s, +5 steps from the judged step):",
        "Video frame at step 20 (4.0 s, +10 steps from the judged step):",
        "Video frame at step 25 (5.0 s, +15 steps from the judged step):",
        "Video frame at step 30 (6.0 s, +20 steps from the judged step):",
    ]
    assert [s.step for s in request.frames] == [0, 5, 10, 15, 20, 25, 30]
    assert images == sorted(images) and all(b - a == 2 for a, b in zip(images, images[1:], strict=False))


def test_request_layout_is_header_instructions_video_then_the_other_blocks(events: tuple[dict[str, Any], ...]) -> None:
    request = _request(events)

    texts = [item["text"] for item in _content(request) if item["type"] == "input_text"]
    heading_order = [t.split("\n", 1)[0] for t in texts if t.startswith("## ")]
    assert request.blocks == ("instructions", "video", "timestamps", "trajectory", "translated_intent", "key_points")
    assert heading_order[:2] == ["## INSTRUCTIONS IN EFFECT", HEADING_VIDEO]
    assert heading_order[2:] == ["## TIMESTAMPS", "## TRAJECTORY", "## TRANSLATED INTENT", "## KEY POINTS"]
    assert texts[0].startswith("Judged translation target")


def test_images_carry_the_configured_detail_and_data_url(events: tuple[dict[str, Any], ...]) -> None:
    config = make_video_config(detail="low")
    request = _request(events, config=config)

    images = [item for item in _content(request) if item["type"] == "input_image"]
    assert {image["detail"] for image in images} == {"low"}
    first = images[0]["image_url"]
    assert first.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(first.split(",", 1)[1]) == b"FAKEJPEG-0"


def test_video_intro_says_the_frames_include_what_the_translator_never_saw(events: tuple[dict[str, Any], ...]) -> None:
    intros = [
        i["text"]
        for i in _content(_request(events))
        if i["type"] == "input_text" and i["text"].startswith(HEADING_VIDEO)
    ]

    assert len(intros) == 1 and "Translator never saw" in intros[0]


def test_label_format_is_configurable(events: tuple[dict[str, Any], ...]) -> None:
    config = make_video_config(label_format="[t={time_s}|s={step}]")

    items = _content(_request(events, config=config))

    assert "[t=2.0|s=10]" in [item["text"] for item in items if item["type"] == "input_text"]


def test_frames_use_the_config_window_stride_and_cap(events: tuple[dict[str, Any], ...]) -> None:
    sampling = {"strategy": "event_window", "before_steps": 10, "after_steps": 10, "stride_steps": 10, "max_frames": 2}
    request = _request(events, A1_REQUEST, config=make_video_config(sampling=sampling))

    assert [frame.step for frame in request.frames] == [0, 10]  # candidates 0, 10, 20; the tie 0/20 keeps the earlier
    assert request.image_tokens == 2 * 1105


def test_excluded_video_leaves_no_image_no_label_and_never_touches_the_source(
    events: tuple[dict[str, Any], ...],
) -> None:
    source = FakeFrameSource.for_committed_videos()
    config = make_video_config(include=False)

    request = _request(events, source=source, config=config)

    serialised = json.dumps(build_responses_body(request.openai_request))
    assert "video" not in request.blocks and request.frames == () and request.image_tokens == 0
    assert "input_image" not in serialised and "Video frame" not in serialised and HEADING_VIDEO not in serialised
    assert source.untouched


def test_included_video_without_a_frame_source_context_is_an_error(events: tuple[dict[str, Any], ...]) -> None:
    with pytest.raises(JudgeRenderError, match="no frame source"):
        build_judge_request(
            event=_event(events, A1_REQUEST),
            event_log=events,
            key_points=EventKeyPoints(A1_REQUEST, KEY_POINTS),
            config=make_video_config(),
        )


def test_episode_without_a_configured_video_is_an_error(events: tuple[dict[str, Any], ...]) -> None:
    other = {k: v for k, v in video_config_raw()["inputs"]["video"]["videos"].items() if k != WIN01}

    with pytest.raises(JudgeRenderError, match=f"no video for episode '{WIN01}'"):
        _request(events, config=make_video_config(videos=other))


def test_video_file_missing_for_a_judged_event_is_an_error(events: tuple[dict[str, Any], ...]) -> None:
    with pytest.raises(FrameSourceError, match="video file not found"):
        _request(events, source=FakeFrameSource({}))


def test_video_with_too_few_frames_for_the_event_is_an_error(events: tuple[dict[str, Any], ...]) -> None:
    sampling = {"strategy": "event_window", "before_steps": 0, "after_steps": 0, "stride_steps": 1, "max_frames": 1}
    source = FakeFrameSource.for_committed_videos(frame_count=5)  # the event is at step 10

    with pytest.raises(JudgeRenderError, match="no video frame"):
        _request(events, source=source, config=make_video_config(sampling=sampling))


def test_frame_source_returning_other_frames_is_rejected(events: tuple[dict[str, Any], ...]) -> None:
    class WrongFrames(FakeFrameSource):
        def frames(self, path: Path, frame_indices: Any, resize: VideoResize) -> tuple[EncodedFrame, ...]:
            return (EncodedFrame(999, b"x", 1, 1),)

    with pytest.raises(JudgeRenderError, match="other frames"):
        _request(events, source=WrongFrames.for_committed_videos())


def test_frame_summary_has_sizes_and_no_bytes(events: tuple[dict[str, Any], ...]) -> None:
    request = _request(events)

    summary = request.frames[0]
    assert (summary.width, summary.height) == (FAKE_FRAME_WIDTH, FAKE_FRAME_HEIGHT)
    assert summary.jpeg_bytes == len(b"FAKEJPEG-0") and summary.time_s == 0.0
    assert not any(isinstance(value, (bytes, str)) for value in summary.__dict__.values())


# --- determinism, redaction, tokens --------------------------------------------------------------------------


def test_request_hash_is_deterministic_and_changes_with_the_pixels(events: tuple[dict[str, Any], ...]) -> None:
    first = canonical_json(build_responses_body(_request(events).openai_request))
    second = canonical_json(build_responses_body(_request(events).openai_request))

    class Tampered(FakeFrameSource):
        def frames(self, path: Path, frame_indices: Any, resize: VideoResize) -> tuple[EncodedFrame, ...]:
            return tuple(
                EncodedFrame(f.frame_index, f.jpeg + b"!", f.width, f.height)
                for f in super().frames(path, frame_indices, resize)
            )

    other = canonical_json(
        build_responses_body(_request(events, source=Tampered.for_committed_videos()).openai_request)
    )
    assert sha256_text(first) == sha256_text(second)
    assert sha256_text(first) != sha256_text(other)


def test_request_hash_is_the_same_with_and_without_the_frame_cache(events: tuple[dict[str, Any], ...]) -> None:
    source = CachingFrameSource(FakeFrameSource.for_committed_videos())
    cached = [_request(events, e, source=source) for e in (A1_START, A1_REQUEST)]
    fresh = [_request(events, e, source=FakeFrameSource.for_committed_videos()) for e in (A1_START, A1_REQUEST)]

    hashes = [sha256_text(canonical_json(build_responses_body(r.openai_request))) for r in cached + fresh]
    assert hashes[:2] == hashes[2:] and hashes[0] != hashes[1]


def test_redact_images_removes_every_base64_and_keeps_the_text(events: tuple[dict[str, Any], ...]) -> None:
    body = build_responses_body(_request(events).openai_request)
    original = json.dumps(body)

    redacted = redact_images(body)

    serialised = json.dumps(redacted)
    assert "base64" not in serialised and "data:image" not in serialised and IMAGE_OMITTED in serialised
    assert json.dumps(body) == original  # the input is not modified
    url = next(p["image_url"] for p in body["input"][0]["content"] if p["type"] == "input_image")
    assert f"{len(url)} chars, sha256 {sha256_text(url)}" in serialised
    texts = [p["text"] for p in redacted["input"][0]["content"] if p["type"] == "input_text"]
    assert texts == [p["text"] for p in body["input"][0]["content"] if p["type"] == "input_text"]


def test_redact_images_leaves_a_text_only_body_equal() -> None:
    body = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}], "model": "m"}

    assert redact_images(body) == body


def test_input_tokens_add_the_configured_image_estimate_to_the_text_estimate(
    events: tuple[dict[str, Any], ...],
) -> None:
    request = _request(events)

    assert request.image_tokens == len(request.frames) * 1105 == 7 * 1105
    assert approx_input_tokens(request) == approx_text_tokens(request) + 7 * 1105


def test_image_token_estimate_comes_from_the_config(events: tuple[dict[str, Any], ...]) -> None:
    config = make_video_config(image_tokens={"per_image": 300, "status": "verified"})

    assert _request(events, config=config).image_tokens == 7 * 300


# --- episode context -------------------------------------------------------------------------------------------


def test_run_episode_reads_the_episode_id_and_last_step(translator_run: Path) -> None:
    episode = run_episode(load_translator_run(translator_run))

    assert (episode.episode_id, episode.last_complete_step) == (WIN01, LAST_STEP)


@pytest.mark.parametrize(
    "inputs",
    [
        None,
        {},
        {"episode": 3},
        {"episode": {"episode_id": "a/b", "last_complete_step": 1}},
        {"episode": {"episode_id": "ok", "last_complete_step": -1}},
        {"episode": {"episode_id": "ok", "last_complete_step": True}},
        {"episode": {"episode_id": "ok"}},
    ],
)
def test_run_episode_rejects_a_run_json_without_a_usable_episode(translator_run: Path, inputs: Any) -> None:
    run = load_translator_run(translator_run)
    broken = type(run)(**{**run.__dict__, "run_json": {"status": "completed", "inputs": inputs}})

    with pytest.raises(EventLogError):
        run_episode(broken)


# --- CLI ------------------------------------------------------------------------------------------------------


def _no_network(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("no test may touch the network")


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket, "getaddrinfo", _no_network)


def _args(tmp_path: Path, translator_run: Path, *extra: str, key_points: Path = EXAMPLE_KEY_POINTS) -> list[str]:
    return [
        "--config",
        str(VIDEO_CONFIG),
        "--translator-run",
        str(translator_run),
        "--key-points",
        str(key_points),
        "--output-root",
        str(tmp_path / "out"),
        "--run-name",
        RUN_NAME,
        *extra,
    ]


def _all_text(run_dir: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(run_dir.iterdir()))


def test_preview_reports_frame_summaries_and_estimates_and_writes_no_image_data(
    tmp_path: Path, translator_run: Path, offline: None
) -> None:
    source = FakeFrameSource.for_committed_videos()
    factory = RecordingFactory(error=AssertionError("no client may be built"))

    code = cli.main(_args(tmp_path, translator_run), openai_client_factory=factory, frame_source=source)

    run_dir = tmp_path / "out" / RUN_NAME
    preview = json.loads((run_dir / "preview.json").read_text())
    judged = [event for event in preview["plan"]["events"] if event["planned"] == "judged"]
    assert code == cli.EXIT_OK and factory.calls == [] and preview["api_requests_made"] == 0
    assert [event["frames"]["steps"] for event in judged] == [[0, 5, 10, 15, 20], [0, 5, 10, 15, 20, 25, 30]]
    assert [event["frames"]["count"] for event in judged] == [5, 7]
    assert judged[1]["frames"]["sizes"] == ["960x540"] and judged[1]["frames"]["image_tokens"] == 7 * 1105
    assert all("frames" not in event for event in preview["plan"]["events"] if event["planned"] == "not_judged")
    estimate = preview["estimate"]
    assert (estimate["images_total"], estimate["image_tokens_total"]) == (12, 12 * 1105)
    assert estimate["input_tokens_total"] == estimate["text_tokens_total"] + estimate["image_tokens_total"]
    assert (estimate["image_tokens_per_image"], estimate["image_tokens_status"]) == (1105, "unverified")
    assert any("unverified" in w for w in preview["warnings"]) and any("never saw" in w for w in preview["warnings"])
    assert preview["plan"]["included_blocks"][1] == "video"


def test_preview_files_contain_no_base64_and_no_image_bytes(
    tmp_path: Path, translator_run: Path, offline: None
) -> None:
    cli.main(_args(tmp_path, translator_run), frame_source=FakeFrameSource.for_committed_videos())

    run_dir = tmp_path / "out" / RUN_NAME
    text = _all_text(run_dir)
    example = json.loads((run_dir / "example_judge_request.json").read_text())
    parts = example["input"][0]["content"]
    assert "base64" not in text and "data:image" not in text and "FAKEJPEG" not in text
    assert any(p["type"] == "input_image" and IMAGE_OMITTED in p["image_url"] for p in parts)
    assert any(p["type"] == "input_text" and p["text"].startswith("Video frame at step") for p in parts)
    base64_of_fake = base64.b64encode(b"FAKEJPEG-0").decode()
    assert base64_of_fake not in text


def test_run_record_names_the_videos_and_decoder_versions(tmp_path: Path, translator_run: Path) -> None:
    cli.main(_args(tmp_path, translator_run), frame_source=FakeFrameSource.for_committed_videos())

    record = json.loads((tmp_path / "out" / RUN_NAME / "run.json").read_text())
    video = record["provenance"]["video"]
    assert sorted(video["videos"]) == sorted(EPISODES)
    assert video["videos"][WIN01]["sha256"] and set(video) == {"videos", "av", "pillow"}
    assert record["config"]["inputs"]["video"]["include"] is True


def test_text_only_run_has_no_video_provenance_and_never_touches_a_frame_source(
    tmp_path: Path, translator_run: Path
) -> None:
    source = FakeFrameSource.for_committed_videos()

    code = cli.main(
        [
            "--translator-run",
            str(translator_run),
            "--key-points",
            str(EXAMPLE_KEY_POINTS),
            "--output-root",
            str(tmp_path / "out"),
            "--run-name",
            RUN_NAME,
        ],
        frame_source=source,
    )

    record = json.loads((tmp_path / "out" / RUN_NAME / "run.json").read_text())
    assert code == cli.EXIT_OK and source.untouched and record["provenance"]["video"] is None
    preview = json.loads((tmp_path / "out" / RUN_NAME / "preview.json").read_text())
    assert preview["estimate"]["images_total"] == 0 and preview["estimate"]["image_tokens_status"] is None


def test_missing_video_is_a_preflight_error_before_any_client_or_run_directory(
    tmp_path: Path, translator_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    factory = RecordingFactory(error=AssertionError("no client may be built"))

    code = cli.main(
        _args(tmp_path, translator_run, "--execute", "--confirm-paid-api"),
        openai_client_factory=factory,
        frame_source=FakeFrameSource({}),
    )

    assert code == cli.EXIT_FAILED and factory.calls == [] and not (tmp_path / "out").exists()
    assert "video file not found" in capsys.readouterr().err


def test_episode_without_a_configured_video_is_a_preflight_error(
    tmp_path: Path, translator_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    others = {k: v for k, v in video_config_raw()["inputs"]["video"]["videos"].items() if k != WIN01}
    config = write_yaml(tmp_path / "judge.yaml", with_video(video_config_raw(), videos=others))
    argv = _args(tmp_path, translator_run)
    argv[1] = str(config)

    code = cli.main(argv, frame_source=FakeFrameSource.for_committed_videos())

    assert code == cli.EXIT_FAILED and not (tmp_path / "out").exists()
    assert f"no video for episode '{WIN01}'" in capsys.readouterr().err


def test_run_json_without_an_episode_is_a_preflight_error(
    tmp_path: Path, translator_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "run"
    shutil.copytree(translator_run, broken)
    (broken / "run.json").write_text(json.dumps({"schema_version": 1, "mode": "execute", "status": "completed"}))

    code = cli.main(_args(tmp_path, broken), frame_source=FakeFrameSource.for_committed_videos())

    assert code == cli.EXIT_FAILED and "inputs.episode" in capsys.readouterr().err


def test_no_key_points_means_no_video_is_read(tmp_path: Path, translator_run: Path) -> None:
    key_points = write_yaml(tmp_path / "kp.yaml", {"schema_version": 1, "events": []})
    source = FakeFrameSource({})  # every video would be "not found"

    code = cli.main(_args(tmp_path, translator_run, key_points=key_points), frame_source=source)

    assert code == cli.EXIT_OK and source.untouched


def test_execute_sends_the_images_but_journals_only_their_hashes(tmp_path: Path, translator_run: Path) -> None:
    source = FakeFrameSource.for_committed_videos()
    client = ScriptedResponsesClient([verdict_response(("kp-001", "kp-002")), verdict_response(("kp-001",))])

    code = cli.main(
        _args(tmp_path, translator_run, "--execute", "--confirm-paid-api"),
        openai_client_factory=RecordingFactory(result=client),
        frame_source=source,
    )

    run_dir = tmp_path / "out" / RUN_NAME
    sent = [[p for p in r["input"][0]["content"] if p["type"] == "input_image"] for r in client.requests]
    requests_text = (run_dir / "requests.jsonl").read_text()
    journal = read_jsonl(run_dir / "requests.jsonl")
    results = read_jsonl(run_dir / "results.jsonl")
    assert code == cli.EXIT_OK and [len(images) for images in sent] == [5, 7]
    assert all(p["image_url"].startswith("data:image/jpeg;base64,") for images in sent for p in images)
    assert "base64" not in requests_text and "base64" not in _all_text(run_dir)
    assert [r["call_id"] for r in journal] == list(EXAMPLE_JUDGED)
    # the hash is over the real payload (images included); the journal holds the redacted one
    judged = [r for r in results if r["status"] == "judged"]
    assert [r["request_sha256"] for r in judged] == [sha256_text(canonical_json(body)) for body in client.requests]
    assert [r["payload"] for r in journal] == [redact_images(body) for body in client.requests]


def test_request_hashes_are_the_same_in_two_runs(tmp_path: Path, translator_run: Path) -> None:
    def hashes(name: str) -> list[str]:
        client = ScriptedResponsesClient([verdict_response(("kp-001", "kp-002")), verdict_response(("kp-001",))])
        argv = _args(tmp_path, translator_run, "--execute", "--confirm-paid-api")
        argv[argv.index(RUN_NAME)] = name
        source = FakeFrameSource.for_committed_videos()
        cli.main(argv, openai_client_factory=RecordingFactory(result=client), frame_source=source)
        results = read_jsonl(tmp_path / "out" / name / "results.jsonl")
        return [r["request_sha256"] for r in results if r["request_sha256"]]

    first, second = hashes("one"), hashes("two")

    assert first == second and len(first) == 2
