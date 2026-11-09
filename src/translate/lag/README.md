# `translate.lag`

LAG (Light Aircraft Game) episode loading and trajectory rendering for the Translator and the
judge, promoted from experiments 13 and 17 of the workspace project. Nothing in this package imports
or reads the workspace project. The committed episodes are in `data/mhi/` (see its README).

## Loader (`dataset.py`)

- `load_episode(path, *, verify_hash=True)` reads one LAG episode CSV (64 columns, one row per step per
  object, 0.2 s per step). It verifies the CSV SHA-256 against the `manifest.json` beside it, applies
  the manifest dtypes (nullable `Int64` / `Float64` / `string`) and sets
  `frame.attrs = {csv_sha256, schema_version, dt_s}`.
- `last_complete_step(frame)` is the last step at which all four aircraft have a CSV record
  (`has_csv_record == 1`). For the three committed episodes it is 340, 293 and 495. The step after
  it exists only in the ACMI log and has no CSV state or action, so it is never rendered.
- `history_steps`, `select_records`, `index_records` are experiment 13's helpers, unchanged.
- The definitions (`AIRCRAFT`, `COLUMNS`, `STATE_FEATURES`, `ACTION_LABELS`, ...) live in
  `dataset.py` with the same values as experiment 13's `lag_data/definitions.py`.
- `EpisodeLoader` is the callable interface (`(path, *, verify_hash=True) -> DataFrame`) that a
  replacement loader for instruction-conditioned recordings has to satisfy (D12).
- Errors: `LagDataError` (a `ValueError`; the workspace code raised plain `ValueError`).

The OpenAI Batch runners of experiments 14-17 (`execution.py`, `batch.py`) moved to the workspace
project together with a frozen copy of `dataset.py`.

## Renderers (`rendering.py`)

Both return a frozen `RenderedTrajectory` (renderer name and version, the steps actually rendered,
anchor, objects, fields, text, SHA-256 of the UTF-8 text). Text has no trailing newline.

| Renderer | Use | Content |
|----------|-----|---------|
| `render_lag17_jsonl` (`lag17-jsonl`, v1) | Translation input (D22) | One JSON line per step with all aircraft and missiles: `step`, `time_s`, `relative_step`, `entities` (metadata, `state` array, `action` array, `missiles_remaining`). |
| `render_detector_compact` (`lag-detector-compact`, v1) | Detector input (D27) | One JSON line per step: `step`, `relative_step`, `objects` with only the requested objects and fields. Floats are rounded to `float_precision`. |

Rules shared by both:

- `relative_step = step - anchor_step`. Requires `start_step <= anchor_step <= end_step`.
- The window is clipped to `[0, last_complete_step]`; the result reports the clipped
  `start_step` / `end_step`. An anchor beyond `last_complete_step` raises `RenderError`.
- Downed aircraft keep their rows with `alive == 0`; windows after a death render normally.
- State features are `null` while `state_valid != 1`.

Compact renderer details: objects follow the `object_ids` order and fields follow the `fields`
order; an object with no row at a step is omitted from that line; unknown objects or fields,
reserved field names (`step`, `relative_step`, `object_id`) and bad precision raise `RenderError`.
`DEFAULT_DETECTOR_FIELDS`, `DEFAULT_DETECTOR_WINDOW_STEPS` (10) and `DEFAULT_FLOAT_PRECISION` (3)
hold the D27 defaults; the Translator config decides what is actually used. Note that 3 decimals
on `lon_deg` / `lat_deg` is about 100 m, coarser than one step of movement.

## Parity with experiments 13 and 17 (D22)

`render_lag17_jsonl` output equals the window body that experiment 17's `commentary_experiment`
builds (`build_window` followed by the
`json.dumps(..., ensure_ascii=False, allow_nan=False, separators=(",", ":"))` join in
`prompts.build_messages`) byte for byte. Those reference outputs were recorded once, before the
core/workspace split, in `tests/lag/fixtures/lag17-renderer-parity.json` (SHA-256 of every
centre +/-30 window on all three episodes, the prompt embedding, and experiment 13's constants,
dtypes, attrs and helper results). `tests/lag/test_lag_contract.py` compares against that fixture
and never imports the workspace project.

Differences, all deliberate:

- Windows are not fixed to +/-30 around a centre of at least 30; any `[start, end]` works and is clipped.
- `episode_alias` is not part of the lines (the workspace format does not place it there); it is
  stored on `RenderedTrajectory.episode_alias` for provenance.
- Errors are `LagDataError` / `RenderError`, both `ValueError` subclasses.
- The unused `HERE` / `PROCESSED` path constants of the workspace loader are dropped.
