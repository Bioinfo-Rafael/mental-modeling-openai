# `mtd_rl_xavier.lag`

LAG (language-annotated game) episode loading and trajectory rendering, promoted from
`workspace/13_LAG` and `workspace/17_commentary_LAG` so that the Translator and the judge do not
import from `workspace/`. Nothing in this package imports `workspace/`.

## Loader (`dataset.py`)

- `load_episode(path, *, verify_hash=True)` reads one 13_LAG CSV (64 columns, one row per step per
  object, 0.2 s per step). It verifies the CSV SHA-256 against the `manifest.json` beside it, applies
  the manifest dtypes (nullable `Int64` / `Float64` / `string`) and sets
  `frame.attrs = {csv_sha256, schema_version, dt_s}`.
- `last_complete_step(frame)` is the last step at which all four aircraft have a CSV record
  (`has_csv_record == 1`). For the three committed episodes it is 340, 293 and 495. The step after
  it exists only in the ACMI log and has no CSV state or action, so it is never rendered.
- `history_steps`, `select_records`, `index_records` are the workspace helpers, unchanged.
- The definitions (`AIRCRAFT`, `COLUMNS`, `STATE_FEATURES`, `ACTION_LABELS`, ...) live in
  `dataset.py` with the same values as `workspace/13_LAG/lag_data/definitions.py`.
- `EpisodeLoader` is the callable interface (`(path, *, verify_hash=True) -> DataFrame`) that a
  replacement loader for instruction-conditioned recordings has to satisfy (D12).
- Errors: `LagDataError` (a `ValueError`; the workspace code raised plain `ValueError`).

## Frozen-plan execution (`execution.py`, `batch.py`)

Promoted from the byte-identical copies in `workspace/{15,16,17}_*/*_experiment/` (14 held the
same `execution.py` without the batch-transport guard); 14-17 now import these modules.

- `execution.py`: `canonical` / `digest` (they define every saved `plan_id`, `request_sha256` and
  `query_id`, so they must stay byte-for-byte stable), `verify_plan`, `save_plan`, journals,
  `run_lock`, and the synchronous `execute`.
- `batch.py`: OpenAI Batch submit / recover / collect. The experiment's offline export is passed in
  as `evaluate=` (`evaluate(plan, directory)`) to `execute` and `summarize`; everything else,
  including `job_key` and the `lag_job_key` metadata, is unchanged.
- Experiment 14's `batch_run.py` (Batch after a synchronous prefix) is a different adapter and
  stays in `workspace/14_mental-modeling_LAG/`.

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

## Parity with the workspace

`render_lag17_jsonl` output equals the window body that
`workspace/17_commentary_LAG/commentary_experiment` builds (`build_window` followed by the
`json.dumps(..., ensure_ascii=False, allow_nan=False, separators=(",", ":"))` join in
`prompts.build_messages`) byte for byte. `tests/lag/test_lag_contract.py` checks this for centre
+/-30 windows (61 steps) on all three episodes, and checks that constants, dtypes, attrs and
helper results equal workspace 13. The tests import the workspace modules only to compute
references, and they check that the workspace files stay byte-identical.

Differences, all deliberate:

- Windows are not fixed to +/-30 around a centre of at least 30; any `[start, end]` works and is clipped.
- `episode_alias` is not part of the lines (the workspace format does not place it there); it is
  stored on `RenderedTrajectory.episode_alias` for provenance.
- Errors are `LagDataError` / `RenderError`, both `ValueError` subclasses.
- The unused `HERE` / `PROCESSED` path constants of the workspace loader are dropped.
