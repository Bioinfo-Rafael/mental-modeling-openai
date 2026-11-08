"""Judge request rendering: the configured input blocks of one translation event -> one request.

Every word of the request comes from the config's prompt file (``config.prompt``,
``configs/evaluation/prompts/``); this module renders the material into its templates. The user
message is one ``input_text`` item per part: the header naming the judged target, then each
included block in ``INPUT_BLOCKS`` order. An excluded block is absent from the request
entirely (heading, text and, for key points, the output-schema ids). The trajectory is the event's
``input.trajectory.text`` copied verbatim, so the judge sees exactly what the translation layer saw
(D34). Nothing names the episode id, the scenario id or an event id.

Video block (L3, D30): selected frames of the episode's action video, in time order, each image item
immediately preceded by one text item holding its configured label (step and time). The video is
evaluation input only: it shows A-team actions and outcomes after the current step that the online
Translator never saw. ``redact_images`` is what every file written to disk uses instead of the
base64 payload.

Prompt-injection envelope (D38): every free text the judge did not write (the translated intent,
its evidence, instruction texts, key-point texts) is rendered as one JSON string literal
(``quoted``), so it can never contain a raw line break and can never start a heading or end the
window data. Ids are checked against safe-id patterns when the event log is loaded, and the
trajectory must be JSON lines. The system prompt tells the judge that all of it is data.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from translate.events import canonical_json, sha256_text, step_time_s
from translate.llm.openai_responses import OpenAIRequest, input_image, input_text

from .configuration import (
    BLOCK_INSTRUCTIONS,
    BLOCK_KEY_POINTS,
    BLOCK_TIMESTAMPS,
    BLOCK_TRAJECTORY,
    BLOCK_TRANSLATED_INTENT,
    BLOCK_VIDEO,
    JudgeConfig,
    JudgeInputs,
    VideoEpisode,
    VideoInput,
)
from .frames import TIME_DECIMALS, FrameSource, plan_frames
from .prompt_files import JudgePrompt
from .schemas import OUTPUT_SCHEMA_NAME, EventKeyPoints, KeyPoint, build_judge_output_schema

CHARS_PER_TOKEN = 4  # approximate text tokens = ceil(chars / 4); labelled approximate everywhere
EPISODE_START_STEP = 0
IMAGE_OMITTED = "image data omitted"
# Characters str.splitlines() and many readers treat as line breaks that json.dumps leaves raw.
UNICODE_LINE_BREAK_ESCAPES = {"\x85": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"}


class JudgeRenderError(ValueError):
    """A judge request cannot be rendered from the event (unknown renderer, no video for the episode)."""


@dataclass(frozen=True)
class FrameSummary:
    """One image of the request, without its bytes (what the preview and the run record may show)."""

    step: int
    time_s: float
    frame_index: int
    width: int
    height: int
    jpeg_bytes: int


@dataclass(frozen=True)
class VideoContext:
    """What the video block needs besides the config: a frame source and the run's episode."""

    frame_source: FrameSource
    episode_id: str
    last_step: int  # last complete step of the episode; frames after it are never shown


@dataclass(frozen=True)
class JudgeRequest:
    event_id: str
    openai_request: OpenAIRequest
    key_points: tuple[KeyPoint, ...]  # what the verdict must score; () when the key-point block is excluded
    blocks: tuple[str, ...]  # included block names, in request order
    frames: tuple[FrameSummary, ...] = ()  # the images, in request order; () without the video block
    image_tokens: int = 0  # estimate from the config (status "unverified" until checked)


# --- blocks ---------------------------------------------------------------------------------------


def quoted(text: str) -> str:
    """Untrusted free text as one JSON string literal on one line: it cannot leave its envelope."""
    literal = json.dumps(text, ensure_ascii=False)
    for char, escape in UNICODE_LINE_BREAK_ESCAPES.items():
        literal = literal.replace(char, escape)
    return literal


def _target_label(target: dict[str, Any]) -> str:
    return f"{target['kind']} {target['id']}"


def _time(prompt: JudgePrompt, time_s: float, show: bool) -> str:
    return prompt.optional("time", {"time_s": time_s} if show else None)


def render_header(event: dict[str, Any], prompt: JudgePrompt) -> str:
    aircraft = ", ".join(event["input"]["target_aircraft"])
    return prompt.text("header", target=_target_label(event["target"]), target_aircraft=aircraft)


def _instruction(
    instruction: dict[str, Any], primary_id: str, inputs: JudgeInputs, prompt: JudgePrompt
) -> dict[str, str]:
    options = inputs.instructions
    parts = []
    if options.show_ids:
        parts.append(prompt.part("instruction", "id", instruction_id=instruction["instruction_id"]))
    if options.show_targets:
        parts.append(prompt.part("instruction", "target", target=_target_label(instruction["target"])))
    if options.show_issue_steps:
        parts.append(prompt.part("instruction", "issue_step", issue_step=instruction["issue_step"]))
    primary = instruction["instruction_id"] == primary_id
    details = prompt.part("instruction", "details", parts=prompt.word("instruction", "part_joiner").join(parts))
    return {
        "primary": prompt.word("instruction", "primary_marker") if primary else "",
        "details": details if parts else "",
        "text": quoted(instruction["text"]),
    }


def render_instructions(
    event: dict[str, Any], event_log: tuple[dict[str, Any], ...], inputs: JudgeInputs, prompt: JudgePrompt
) -> str:
    payload = event["input"]
    ordered = sorted(payload["instructions_in_effect"], key=lambda item: (item["issue_step"], item["instruction_id"]))
    items = [_instruction(item, payload["primary_instruction_id"], inputs, prompt) for item in ordered]
    return prompt.text("instructions", instructions=prompt.items("instruction", items))


def _reason_text(reason: dict[str, Any], show_sources: bool, prompt: JudgePrompt) -> str:
    if not show_sources:
        return prompt.part("reason", "plain", kind=reason["kind"])
    details = [prompt.part("reason", "source", source=reason["source"])]
    if reason["probability"] is not None:
        details.append(prompt.part("reason", "probability", probability=reason["probability"]))
    joined = prompt.word("reason", "detail_joiner").join(details)
    return prompt.part("reason", "detailed", kind=reason["kind"], details=joined)


def render_timestamps(
    event: dict[str, Any], event_log: tuple[dict[str, Any], ...], inputs: JudgeInputs, prompt: JudgePrompt
) -> str:
    """Scope ``through_event``: episode start, this target's earlier translations, this event."""
    show = inputs.timestamps.show_sim_seconds
    earlier = sorted(
        (item for item in event_log if item["target"] == event["target"] and item["step"] < event["step"]),
        key=lambda item: item["step"],
    )
    earlier_items = [
        {"step": item["step"], "time": _time(prompt, item["time_s"], show), "primary_trigger": item["primary_trigger"]}
        for item in earlier
    ]
    triggers = [
        prompt.part("trigger", "item", reason=_reason_text(reason, inputs.timestamps.show_sources, prompt))
        for reason in event["trigger_reasons"]
    ]
    return prompt.text(
        "timestamps",
        episode_start_step=EPISODE_START_STEP,
        episode_start_time=_time(prompt, step_time_s(EPISODE_START_STEP), show),
        earlier=prompt.items("earlier", earlier_items),
        step=event["step"],
        time=_time(prompt, event["time_s"], show),
        triggers=prompt.word("trigger", "joiner").join(triggers),
    )


def render_trajectory(
    event: dict[str, Any], event_log: tuple[dict[str, Any], ...], inputs: JudgeInputs, prompt: JudgePrompt
) -> str:
    trajectory = event["input"]["trajectory"]
    legend = prompt.blocks["legends"].get(trajectory["renderer"])
    if legend is None:
        raise JudgeRenderError(f"no judge legend for trajectory renderer {trajectory['renderer']!r}")
    window = None
    if inputs.trajectory.show_window_metadata:
        window = {
            "start_step": trajectory["start_step"],
            "end_step": trajectory["end_step"],
            "anchor_step": trajectory["anchor_step"],
            "object_ids": ", ".join(trajectory["object_ids"]),
            "renderer": trajectory["renderer"],
            "renderer_version": trajectory["renderer_version"],
        }
    return prompt.text(
        "trajectory", window=prompt.optional("window", window), legend=legend, trajectory=trajectory["text"]
    )


def render_translated_intent(
    event: dict[str, Any], event_log: tuple[dict[str, Any], ...], inputs: JudgeInputs, prompt: JudgePrompt
) -> str:
    output = event["output"]
    evidence = prompt.items("evidence", [{"item": quoted(item)} for item in output["evidence"]])
    return prompt.text("translated_intent", intent=quoted(output["intent"]), evidence=evidence)


def render_key_points(key_points: tuple[KeyPoint, ...], prompt: JudgePrompt) -> str:
    items = [
        {"index": index, "key_point_id": item.key_point_id, "text": quoted(item.text)}
        for index, item in enumerate(key_points, start=1)
    ]
    return prompt.text("key_points", key_points=prompt.items("key_point", items))


BlockRenderer = Callable[[dict[str, Any], tuple[dict[str, Any], ...], JudgeInputs, JudgePrompt], str]
EVENT_BLOCK_RENDERERS: dict[str, BlockRenderer] = {
    BLOCK_INSTRUCTIONS: render_instructions,
    BLOCK_TIMESTAMPS: render_timestamps,
    BLOCK_TRAJECTORY: render_trajectory,
    BLOCK_TRANSLATED_INTENT: render_translated_intent,
}


# --- video ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoBlock:
    items: tuple[dict[str, Any], ...]  # intro text, then (label text, image) pairs
    frames: tuple[FrameSummary, ...]


def frame_time_s(step: int, episode: VideoEpisode) -> float:
    return round(step * episode.step_seconds, TIME_DECIMALS)


def frame_label(options: VideoInput, episode: VideoEpisode, step: int, event_step: int) -> str:
    relative = step - event_step
    return options.label_format.format(
        step=step,
        time_s=frame_time_s(step, episode),
        relative_step=f"{relative:+d}" if relative else "0",
    )


def render_video(event: dict[str, Any], options: VideoInput, context: VideoContext, prompt: JudgePrompt) -> VideoBlock:
    """Frames around the event: ``before_steps`` .. ``after_steps`` clipped to the episode and the video."""
    episode = options.videos.get(context.episode_id)
    if episode is None:
        raise JudgeRenderError(f"inputs.video.videos has no video for episode {context.episode_id!r}")
    source = context.frame_source
    plan = plan_frames(
        event_step=event["step"],
        last_step=context.last_step,
        info=source.info(episode.path),
        episode=episode,
        sampling=options.sampling,
    )
    if not plan.steps:
        raise JudgeRenderError(
            f"no video frame of episode {context.episode_id!r} lies in the window of step {event['step']}"
        )
    encoded = source.frames(episode.path, plan.frame_indices, options.resize)
    if tuple(frame.frame_index for frame in encoded) != plan.frame_indices:
        raise JudgeRenderError("the frame source returned other frames than requested")
    items: list[dict[str, Any]] = [input_text(prompt.text("video"))]
    for step, frame in zip(plan.steps, encoded, strict=True):
        items.append(input_text(frame_label(options, episode, step, event["step"])))
        items.append(input_image(frame.data_url, detail=options.detail))
    frames = tuple(
        FrameSummary(step, frame_time_s(step, episode), frame.frame_index, frame.width, frame.height, len(frame.jpeg))
        for step, frame in zip(plan.steps, encoded, strict=True)
    )
    return VideoBlock(items=tuple(items), frames=frames)


def redact_images(body: dict[str, Any]) -> dict[str, Any]:
    """A copy of a Responses body whose image data URLs are replaced by their length and sha256.

    The only form in which a request body is written to disk or shown: no base64 and no image bytes.
    """
    redacted = copy.deepcopy(body)
    for message in redacted.get("input", []):
        for part in message.get("content", []):
            if part.get("type") == "input_image":
                url = part["image_url"]
                part["image_url"] = f"[{IMAGE_OMITTED}: {len(url)} chars, sha256 {sha256_text(url)}]"
    return redacted


# --- request --------------------------------------------------------------------------------------


def _block_text(
    block: str,
    event: dict[str, Any],
    event_log: tuple[dict[str, Any], ...],
    key_points: EventKeyPoints,
    config: JudgeConfig,
) -> str:
    if block == BLOCK_KEY_POINTS:
        return render_key_points(key_points.key_points, config.prompt)
    return EVENT_BLOCK_RENDERERS[block](event, event_log, config.inputs, config.prompt)


def build_judge_request(
    *,
    event: dict[str, Any],
    event_log: tuple[dict[str, Any], ...],
    key_points: EventKeyPoints,
    config: JudgeConfig,
    video: VideoContext | None = None,
) -> JudgeRequest:
    """One request for one translation event. ``event_log`` holds the run's translation events.

    ``video`` is required exactly when the video block is included; otherwise it is never touched.
    """
    if key_points.event_id != event["event_id"]:
        raise JudgeRenderError("key points belong to a different event")
    inputs = config.inputs
    blocks = inputs.included()
    items: list[dict[str, Any]] = [input_text(render_header(event, config.prompt))]
    frames: tuple[FrameSummary, ...] = ()
    for block in blocks:
        if block != BLOCK_VIDEO:
            items.append(input_text(_block_text(block, event, event_log, key_points, config)))
            continue
        if video is None:
            raise JudgeRenderError("the video block is included but no frame source was given")
        rendered = render_video(event, inputs.video, video, config.prompt)
        items.extend(rendered.items)
        frames = rendered.frames
    scored = key_points.key_points if inputs.key_points.include else ()
    request = OpenAIRequest(
        model=config.model,
        instructions=config.prompt.text("system"),
        input_items=tuple(items),
        schema_name=OUTPUT_SCHEMA_NAME,
        schema=build_judge_output_schema(scored),
        reasoning_effort=config.reasoning_effort,
        max_output_tokens=config.max_output_tokens,
    )
    return JudgeRequest(
        event_id=event["event_id"],
        openai_request=request,
        key_points=scored,
        blocks=blocks,
        frames=frames,
        image_tokens=len(frames) * inputs.video.image_tokens.per_image,
    )


def approx_input_tokens(request: JudgeRequest) -> int:
    """Text tokens ``ceil(chars / 4)`` over the instructions, text items and schema, plus the image estimate."""
    return approx_text_tokens(request) + request.image_tokens


def approx_text_tokens(request: JudgeRequest) -> int:
    openai_request = request.openai_request
    chars = len(openai_request.instructions) + len(canonical_json(openai_request.schema))
    chars += sum(len(item["text"]) for item in openai_request.input_items if item["type"] == "input_text")
    return math.ceil(chars / CHARS_PER_TOKEN)
