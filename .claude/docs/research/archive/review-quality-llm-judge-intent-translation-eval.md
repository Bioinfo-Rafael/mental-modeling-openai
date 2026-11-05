# Quality Review: llm-judge-intent-translation-eval

Reviewer: quality-reviewer. Scope: the 53 files in `.claude/logs/review-scope-llm-judge-intent-translation-eval.txt`. Review only; no source, test or config file was edited.

## Verdict

High: 0. Medium: 1. Low: 9. No finding blocks merge on the CLI path. The one Medium (M1) makes the judge unable to load a Translator run whose text contains certain Unicode line separators. Fix it before the first real judged run.

Checks run:
- `uv run pytest tests/llm tests/lag tests/translator tests/evaluation -q`: 749 passed.
- `uv run ruff check` and `ruff format --check` on the 43 scope `.py` files: clean.
- Python 3.10 syntax: all scope `.py` files parse with `ast.parse(feature_version=(3, 10))`. No 3.11-only API found (grep for `tomllib`, `datetime.UTC`, `StrEnum`, `Self`, `except*`, `TaskGroup`).
- Probes (scratchpad `/tmp/claude-15729/-home-user151005-workspace/9d3aaa3e-c4c0-471b-bfec-2bd69359af7f/scratchpad/quality/`): `probe_plan.py`, `probe_ls.py`, `probe_codex.py`, `probe_render.py`.

## Semantics verified (no defect found)

| Item | How verified |
|------|--------------|
| Detector tick every 5 steps on the fixed grid, also when another trigger coincides (D9/D28) | Read `pipeline.py:460-462`, `is_tick` at `:176`; fuzz probe |
| At most one translation per target per step; precedence start > instruction > external_change > detector_change > request (D10) | `schema.py:36-42,192-201`; `_process_target` has a single `_translate` call |
| Mid-episode first instruction: `instruction` trigger, no previous translation, no detector call before it (D24/D35) | `pipeline.py:460` guards on `state.previous`; plan uses `range(first + 1, end)` |
| Dead targets: closed once, never reactivated, `trigger_suppressed`, no model call (D37) | `_step`/`_close`/`_process_target`; `closed` flag carried across re-instructions |
| Window clipping at 0 and `last_step` | `_clip_window` (`lag/rendering.py:99-113`), callers pass `max(0, step-W+1)` |
| Caps never send call cap+1 | `check_cap` uses `attempted >= cap` before building each call (`pipeline.py:371-376`), which is at least as strict as the plan's "completed" |
| `plan_schedule` equals the actual run | Fuzz probe: 180 random scenarios (aircraft and team targets, mid-episode and re-instructions, requests, external changes) across the 3 episodes; detector call count and mandatory translation count matched in every one (0 mismatches) |
| Judge D36 partial judging | `match_key_points` rejects unknown ids; `parse_key_points` rejects duplicate ids; `run_judge` writes `not_judged` with no call; the cap counts attempted calls only |
| Trajectory text identical between the Translator event and the judge request | `evaluation/rendering.py:179-191` copies `input.trajectory.text` verbatim; `schemas.py:166` checks its SHA-256 on load |
| Parser strictness | Exact key sets, bool and float rejected as ints, blank rationale rejected, `parse_constant` rejects NaN in the judge parser |

## Findings

### M1 — Medium — `evaluation/schemas.py:186` — `splitlines()` splits JSONL on Unicode separators (CONFIRMED)

Current code:
```python
for index, line in enumerate(text.splitlines()):
```
`JsonlRunSink` writes records with `ensure_ascii=False` (`translator/cli.py:434`). `json.dumps` escapes only control characters below U+0020, so U+2028, U+2029 and U+0085 stay raw in the file. `str.splitlines()` treats all three as line breaks.

Failing scenario: a Translator intent or evidence string (model output), or an instruction text in the scenario YAML (for example text pasted from a document), contains U+2028, U+2029 or U+0085. The Translator run completes with status `completed`. `load_translator_run` then raises `EventLogError: events.jsonl line 1 is not valid JSON`, and the judge cannot process any event of that run. Reproduced for all three characters with `probe_ls.py`.

Suggested fix: split on `"\n"` only, for example `for index, line in enumerate(text.split("\n"))` and skip the single trailing empty element, or iterate the open text file (universal newlines handle only `\n`, `\r`, `\r\n`). Add a regression test with U+2028 in `output.intent`.

### L1 — Low — `translator/pipeline.py:132-145,216-236,428-433` — pipeline does not enforce "reference at or after the first instruction" (CONFIRMED, API path only)

`config._check_scenario_rules` (`config.py:579-587`) rejects a request or change that precedes its target's first instruction, but `pipeline.validate_scenario` does not. A `Scenario` built directly (not through `parse_scenario`) with a request at step 2 for a target first instructed at step 5 is accepted. `_step` iterates `run.states` only, so the request is silently ignored, while `plan_schedule` counts it. In `probe_codex.py`: the run executed 2 translations while the plan predicted 3.

Codex rated this High. I rate it Low because the CLI and `load_scenario` cannot reach it, and `Translator.run` is documented to take a validated scenario. Suggested fix: move the reference check into a function shared by `parse_scenario` and `validate_scenario`, or raise `PipelineError` in `_process_target` when `scenario_triggers` holds a key whose target has no state.

### L2 — Low — `translator/pipeline.py:357-369` — `events_written` is overstated when `write_event` fails (CONFIRMED)

`emit` does `self.sequence += 1` before `self.sink.write_event(...)`. If the write raises, the sequence has advanced but no record exists, and `summary()` reports `events_written=self.sequence`. `probe_codex.py`: the run reports `events_written=2` while the sink holds 1 record. Suggested fix: build `sequence = self.sequence + 1`, write, then assign `self.sequence`.

### L3 — Low — `translator/translation.py:185` and `evaluation/schemas.py:436` — duplicate JSON keys accepted, `RecursionError` not handled (CONFIRMED / PLAUSIBLE)

`json.loads` keeps the last of duplicate keys, so `{"intent":"a","intent":"b","evidence":["e"]}` parses as intent `b` (confirmed). Deep nesting raises `RecursionError`, which is not a `ValueError` and so bypasses `except (TypeError, ValueError)`; the run still ends as `failed` through the generic handler, but without the typed error (PLAUSIBLE, not run). The strict output schema normally prevents both. Suggested fix: `object_pairs_hook` that raises on duplicates, and add `RecursionError` to the except tuple in both parsers.

### L4 — Low — `translator/cli.py:432-436`, `pipeline.py:478-481` — a lone surrogate in a raw response is reported as `uncertain` (CONFIRMED)

`_answered` writes the raw response to `responses.jsonl` before `parse` runs. A raw body that contains a valid JSON escape such as `"\ud800"` makes `handle.write` raise `UnicodeEncodeError`. `in_flight` is still set, so `_crash_status` records `uncertain` (exit 3), although a complete response was received and the failure is deterministic. `probe_ls.py` shows `status=uncertain`, `uncertain=1`. The same applies to `evaluation/judge.py:318`. It also makes the `_is_utf8` checks in `translation.py:200` and `schemas.py:392` unreachable for text that is part of `raw`. Suggested fix: make the sink encode defensively (`errors="backslashreplace"` for the journal only) or catch `UnicodeEncodeError` around the response write and journal a redacted placeholder, then continue to the parse error (`failed`, exit 1).

### L5 — Low — `evaluation/judge.py:365-367` — `results.jsonl` is not a complete ledger after `cap_reached` or a failure (PLAUSIBLE)

Events after the stopping event, including those without key points, get no `not_judged` row. D36 says events without key points are written as `not_judged`. Suggested fix: write all `not_judged` rows first (they need no call), or state in `evaluation/README.md` that a stopped run leaves later rows unwritten.

### L6 — Low — `translator/config.py:543` — wrong name in the `targets` list error (CONFIRMED by reading)

`_list(section["targets"], name)` passes `name` (`requests[0]`), so the message reads `requests[0] must be a list (use [] for none)` for a bad `targets` value. Suggested fix: `_list(section["targets"], f"{name}.targets")`.

### L7 — Low — duplicated infrastructure across the two CLIs (PLAUSIBLE maintainability risk)

Near-identical code exists in `translator/cli.py` and `evaluation/{cli,judge}.py`:
- `finite_json` (`translator/cli.py:414`, `judge.py:137`);
- `JsonlRunSink` and `JsonlJudgeSink`;
- `_git_state`, `_code_hashes`, `_new_run_dir`, `_write_json`, `_sha256_file`, `_redact`.

The strict YAML loader is duplicated in `translator/config.py:222-251` and `evaluation/configuration.py:200-231`, and `_is_utf8` appears four times (the `llm` copies are intentional independence). A fix applied to one copy (for example L4) will be missed in the other. Suggested fix: one shared helper module for run artifacts and strict YAML, outside `llm/`.

### L8 — Low — domain constants hard-coded in prompts (PLAUSIBLE)

`translation.py:33-70` and `evaluation/rendering.py:39-80` hard-code "0.2 s", the two-versus-two setup and the aircraft ids, although `STEP_SECONDS` exists (`lag/rendering.py:38`). The detector state uses the constant, the prompts do not, so changing the constant leaves the text stale. This breaks the "no magic numbers" rule. Suggested fix: format the step length into the templates.

### L9 — Low (design risk) — `translator/translation.py:194` — one over-long intent aborts the whole paid run (PLAUSIBLE)

The prompt asks for at most `max_intent_chars` (1000) characters; the strict schema cannot enforce `maxLength`. One answer of 1001 characters returns `failed` (exit 1), the already-paid events stay in the log, and there is no resume. With up to 250 paid calls this is an expensive failure mode. The "never repair" decision is respected, so this is a policy question for the lead: either accept it and say so in the README, or enforce the limit with a schema `maxLength` if the API supports it.

### L10 — Low (performance, informational) — `lag/rendering.py:186-194` (PLAUSIBLE, measured)

`render_lag17_jsonl` takes about 0.5 s per 60-step window (per-step filtering and `iterrows`); `render_detector_compact` takes about 0.011 s; `last_complete_step` takes about 4 ms and runs on every call. A full preview with many scheduled ticks and a 250-call run cost minutes of CPU, negligible next to API latency. No change needed now.

## Parser and contract notes (no defect)

- Plan versus code: `alive == 0` is used where the plan text says "alive != 1". The real episodes have no NA `alive` at steps up to `last_step`, and no aircraft returns to `alive == 1`, so the behaviour is identical on the data. Treating NA as alive is the safer reading.
- Known deviations accepted by the lead (file sizes, five trigger kinds, `max_intent_chars`, exit code 4) were not re-reported.

## Codex Consultations

- Prompt: `.claude/logs/codex/prompt-quality-review.md`. Run with `--sandbox read-only --caller quality-reviewer`, label `quality-review`, duration 338 s, exit 0, no timeout.
- Response: `.claude/logs/codex/20261104T015438Z-quality-review.md`.
- Result:
  - Finding 1 (Codex: High; request before first instruction desyncs `plan_schedule` and the run). I reproduced it, but it is reachable only by building a `Scenario` directly. `parse_scenario` already rejects it. Recorded as L1 (Low).
  - Finding 2 (Medium; `emit` increments `sequence` before the write). Reproduced; recorded as L2 (Low).
  - Verified OK by Codex and consistent with my own reading and fuzz run: coincident first instruction and tick, death on a tick step, staggered team deaths, re-instruction after close, first instruction after the down step, cap checks (no cap+1), `attempted == completed + failed + uncertain` on all exit paths, NA and first-death handling, window and `range(first + 1, end)` boundaries.
- `edits.created_files` in the wrapper payload lists `review-tests-llm-judge-intent-translation-eval.md`. That file was created by the concurrent tests reviewer, not by Codex (the sandbox was read-only).
