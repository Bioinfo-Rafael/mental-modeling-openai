## Implementation Plan: Prompt composition refactor

### Purpose

Make the evaluation prompt directly editable as text files and make its composition visible from the CLI. Preserve the current rendered requests and preview/paid execution behavior while separating prompt composition, material selection, API serialization, and sending.

Planning baseline: `rafael/evaluation`, commit `dd56c6a9ee643d592fef449f6eef33292e30fc3d` (2026-11-09 JST). Implementation completed on this branch on 2026-11-09 JST. The user authorized implementation and push to both remotes.

### Scope

Confirmed current implementation:

- `src/evaluation/rendering.py` is absent at the baseline. Its previous hardcoded system prompt is visible at `570c294^:src/evaluation/rendering.py`; commit `570c294` externalized the old prompts, and `51e1a3f` deleted that D29 judge. The plan targets today's judge, not the deleted module.
- Full judge templates already live outside Python, in `configs/evaluation/judge.yaml:23-132`: `system`, `user`, `rating_rule`, and `future`. `src/evaluation/config.py:97-109` validates their placeholders before use.
- `src/evaluation/material.py:223-256` builds an explicit values dictionary and fills the user template. It does not dispatch rendering functions by a keyword registry. Fixed text fragments remain in its helpers, including empty markers and conditional/FUTURE criterion wording (`:108-140`).
- The current evaluation core uses `translate.backends.openai.OpenAIBackend`; its dependency path does not pass through the vendored LLM-Xavier backend. Request text and API serialization are distinct responsibilities (`src/evaluation/judge.py:82-88`, `src/translate/backends/openai.py:104-131`).
- Preview uses the planned request body and returns before client creation (`src/evaluation/cli.py:138-166`). The paid path sends `call.body` unchanged through `backend.send` (`:127-135`). A second dry-run API wrapper is not present in this path.

Current evaluation path:

```text
cli.main -> run_judge -> _calls -> judge.plan_call
    -> material.user_text / system_text -> Prompt
    -> OpenAIBackend.request_body -> Call(body, request_sha256)
    -> preview.json OR existing guarded run_calls -> backend.send -> responses.create
```

Proposed path, with composition calls visible together in `cli._calls`:

```text
load_config -> read template files once -> validated resolved prompt texts
cli._calls: criteria_for -> prompts.build_prompt -> OpenAIBackend.request_body
    -> Call(event, criteria, body, request_sha256(body))
run_judge: preview.json OR existing guarded run_calls -> backend.send
```

`build_prompt` owns all four template expansions and the verdict schema. `material.py` owns run loading, timeline/window selection, and data formatting, returning base material values and optional FUTURE values without expanding templates. The backend owns only Responses field mapping, SDK invocation, and response parsing. The CLI exposes these boundaries without containing prompt prose or telemetry rendering.

New files:

- `configs/evaluation/prompts/judge/system.txt`
- `configs/evaluation/prompts/judge/user.txt`
- `configs/evaluation/prompts/judge/rating_rule.txt`
- `configs/evaluation/prompts/judge/future.txt`
- `src/evaluation/prompts.py`
- `tests/evaluation/test_prompt_files.py`

Modified files:

- `configs/evaluation/judge.yaml`: replace the four inline texts with explicit file references; retain criteria, anchors, scale, and execution settings.
- `src/evaluation/config.py`: resolve file references, validate resolved texts, retain provenance.
- `src/evaluation/material.py`: expose material values; move top-level template filling into `prompts.py`.
- `src/evaluation/judge.py`: remove prompt assembly from `plan_call`; retain `Call`, execution planning, guards, journaling, and verdict parsing.
- `src/evaluation/cli.py`: explicitly build prompt/body/Call and record template provenance.
- `tests/evaluation/{fakes.py,test_judge_material.py,test_judge_guards.py}`: update config helpers and prompt entry-point imports; add behavioral checks.
- `src/evaluation/README.md` and the relevant root `README.md` instructions: document editing, placeholders, paths, preview, and changed-prompt run naming.

No new dependency is needed. Existing `str.format_map`, `check_fields`, strict YAML loading, `Prompt`, and request hashing suffice.

Translation follow-up is optional: the registry described by the user does exist at `src/translate/pipeline.py:89-94` (`INPUTS[name].render`), while complete translation template layout lives in `configs/translate/*.yaml`. Moving `translation_prompt` into `translate/prompts.py` would improve discoverability, but its per-step execution must remain inside the causal pipeline. Do this separately if translation is also in scope; do not unroll configurable input selection into hardcoded CLI branches.

### Implementation Steps

Implement the following steps in dependency order, keeping prompt-content experiments separate from the structural migration.

#### Step 1: Capture the current observable contract

- [x] Add focused offline regression cases for start, change, request, coincident kinds, first/previous translation, and free text containing braces, quotes, and newlines.
- [x] Capture the rendered system/user text and complete request body from the current baseline for representative synthetic events. Use small fixtures or an implementation-time before/after comparison; do not recreate the retired repository-wide freeze machinery.
- [x] Assert that the exact preview body equals the fake SDK's submitted body and the journaled body for the same event/config, including its request hash.

**Verification:** Existing evaluation tests pass at baseline; new checks detect text or API-body changes, and the preview test uses a client factory that raises if constructed. No real API call.

#### Step 2: Extract templates and load them once

- [x] Write the four `.txt` files from the YAML-decoded strings, preserving their exact UTF-8 content. In particular, `future: |2-` has intentional leading blank lines, and the `|-` scalars omit their terminal newline.
- [x] Use explicit config keys such as `system_file`, `user_file`, `rating_rule_file`, and `future_file`; resolve relative paths against the config file's directory, never the working directory.
- [x] For migration, accept exactly one complete form: the existing four inline strings or the four file references. Reject partial/mixed forms and unknown keys. Normalize both to the current resolved `Prompt` model and run its existing validators. This keeps private inline configs usable.
- [x] Read each file only during config loading. Fail with the config key and path on missing files, invalid UTF-8, empty content, malformed/unknown placeholders, or missing required material placeholders. Preserve whitespace; do not use `.strip()` or `.rstrip()` on loaded templates.
- [x] Record resolved texts and source paths/content hashes in `run.json`. Keep path/provenance metadata outside the API body so relocation alone does not change `request_sha256`.
- [x] Update test helpers that copy `judge.yaml` to temporary directories: copy the prompt directory too, or use absolute source references, so config-relative resolution remains valid.

**Verification:** Inline and file-backed configs yield identical resolved texts and request hashes; path resolution works from another working directory; editing a template changes the request hash. Missing/invalid templates fail before any API client is constructed. Provenance captures the text actually loaded even if the source file changes later.

#### Step 3: Expose the composition boundary at the CLI

- [x] Add `evaluation.prompts.build_prompt(event, run, frame, config, criteria) -> Prompt`. Give it explicit local variables for material values, rating rule, system text, user text, and schema.
- [x] Move all four template expansions and `verdict_schema` to this module. Refactor `material._future` to return raw FUTURE values or `None`, and expose base material values separately. Keep temporal selection, quoting, and dynamic material formatting in `material.py`.
- [x] In `build_prompt`, expand FUTURE (empty when absent), rating rule, system, then user in dependency order. Keep criterion FUTURE availability based on the same event/window eligibility. No material helper imports the prompt builder or expands a configured template.
- [x] In `cli._calls`, compute criteria once, call `build_prompt`, call `OpenAIBackend.request_body`, and construct `Call` with the hash. Replace the former `judge.plan_call` use and update its internal test callers.
- [x] Keep the full user-message order visible in `user.txt`. Keep four principal templates rather than splitting every heading into a separately dispatched renderer.
- [x] Leave the existing small data-formatting fragments unchanged in this migration. If experimental editing of criterion annotations or empty-state wording is needed, extract them in a separate semantic change with its own request diff.

**Verification:** Representative bodies and hashes are identical to Step 1. Existing tests for quoting, FUTURE cutoffs, human-visible windows, union of criteria, and independence from translator metadata still pass. Imports remain acyclic: `judge.py` no longer imports prompt builders; `prompts.py` does not import the CLI.

#### Step 4: Retain one sending path and verify the full workflow

- [x] Preserve the current preview branch and delayed paid-client construction. Keep `OpenAIBackend.request_body` as the single serialization function; `send(body)` must not rebuild prompt text.
- [x] Preserve paid flags, root-wide request scanning, changed-bytes refusal, marker locking, call caps, pre-send fsync, uncertain outcomes, and explicit `--only-event` recovery.
- [x] Document the editing path: template text -> preview.json exact request -> existing explicit paid execution. Document that prompt text changes are experiments and require another judgement name when the current directory already contains different bytes.
- [x] Update diagrams/import references and verify the complete diff contains only the listed scope.

**Verification:** Run `uv run pytest -q tests/evaluation tests/translate/test_cli_config.py`, then targeted `uv run ruff check` and `uv run ruff format --check` on changed Python files. Existing no-resend and preview tests plus the new preview/journal/fake-SDK equality check must pass. No paid validation is required for this refactor.

### Risks & Considerations

- Whitespace affects request hashes and therefore duplicate-request detection. Preserve exact YAML-decoded text during extraction; request hash equality is the migration acceptance criterion, not file-byte equality between YAML and TXT.
- Config paths become references to additional files. Resolve consistently, read once, and record resolved text/provenance; a YAML file hash alone no longer identifies all prompt inputs.
- Existing inline private configs and temporary test copies need explicit migration support as described above. Keep `load_config`'s resolved config behavior stable; store provenance separately from prompt interpolation fields.
- Material semantics must remain unchanged: free text stays JSON-quoted; trajectory ends at t; FUTURE remains evaluation-only, correctly bounded, and restricted by criterion. Reordering CRITERIA/SCALE would change the prompt's stated trust boundary and belongs in a separate content change.
- Commenting out API calls makes dry-run behavior depend on a source edit. The existing explicit preview branch already permits inspecting exact bodies without a client and should be retained.
- The initial planning environment lacked `uv`. Implementation validation used a temporary uv/Python 3.12 environment synchronized from the unchanged lockfile. No paid calls were made. The full suite has 15 existing trajectory-hash parity failures in `tests/translate/test_inputs.py`; the same 15 failures were reproduced from a clean archive of baseline `dd56c6a` under the same environment. Their dependency/platform cause remains unverified.

### Open Questions

- No blocker for the proposed evaluation-only refactor. The checked-out `rafael/evaluation` branch is the planning target; the older `rendering.py` design is historical.
- Optional later scope: apply the same file-backed template pattern and discoverable builder module to translation/detector prompts. That extension would also require incorporating external template contents into translation run identity, whose default name currently uses the config-file hash.
- Optional later content change: externalize the remaining small material-formatting phrases. This initial migration deliberately preserves their emitted text and the existing rubric.

### Implementation Validation

- `uv run --no-sync pytest -q tests/evaluation tests/translate/test_cli_config.py`: 99 passed, independently rerun by the parent.
- `uv run --no-sync pytest -q --tb=no`: 133 passed, 15 failed; all failures are the baseline trajectory-hash parity cases described above.
- `uv run --no-sync ruff check src tests scripts` and `ruff format --check src tests scripts`: passed (44 Python files).
- Six synthetic events rendered with the baseline and current source: complete API bodies and request hashes identical.
- All four extracted files equal the original YAML-decoded UTF-8 bytes; all other judge config values unchanged.
- Preview, fake SDK submission and request journal equality covered by the new integration test.
- `git diff --check`, `bash -n slurm/validate.sbatch`, both CLI help commands outside the checkout, and source/test line ceilings: passed.
- Independent source review: PASS. The Slurm job itself was not submitted; local offline tests cover this change.
