"""Messages shown to the LLM: the first message and the per-turn follow-ups.

Only whitelisted spec fields are rendered (alias, description, inputs, clip, architecture,
outputs); provenance and notes never reach the model. ``task_mode="blind"`` withholds the task,
domain and all input/output names and keeps only the I/O layout.
"""

from __future__ import annotations

from dataclasses import dataclass

from weightreading.architecture import equations
from weightreading.protocol import FINAL_SECTIONS, HYPOTHESIS_SECTION
from weightreading.spec import OutputSpec, Spec
from weightreading.tensors import Tensor


@dataclass(frozen=True)
class ExecutorLimits:
    max_runs: int
    timeout_s: int
    stdout_cap_chars: int
    stderr_tail_chars: int


RULES = """\
# Rules (strict)
You must NOT run the network or any of its layers. Concretely, you must not:
- apply any of its nonlinearities (ReLU, LayerNorm normalisation, sigmoid, tanh, softmax, argmax over
  logits) to a vector derived from an observation, an activation or any hand-built input vector;
- iterate the recurrence, or evaluate sigmoid/tanh/exp of biases (e.g. "resting" gate values);
- import torch, use any other deep-learning library, or search for anything.
You MAY do any linear algebra on the weights and biases themselves: slicing (including per-gate
slices of the GRU matrices), norms, statistics, sorting, correlations, products of weight matrices
across layers, folding LayerNorm gains/biases into the following Linear layer, SVD and
eigendecomposition (e.g. of the GRU hidden-to-hidden blocks). Reasoning in words about what a
nonlinearity would do is allowed; computing it is not."""


def input_label(spec: Spec, index: int, task_mode: str) -> str:
    return spec.inputs[index].name if task_mode == "shown" else f"x{index}"


def head_label(spec: Spec, head: int, task_mode: str) -> str:
    return spec.outputs[head].name if task_mode == "shown" else f"head_{head}"


def _task_section(spec: Spec, task_mode: str) -> str:
    if task_mode == "shown":
        return spec.description
    return (
        f"A recurrent policy network trained with reinforcement learning. Every step it reads "
        f"{len(spec.inputs)} numbers and chooses one bin for each of {len(spec.outputs)} output heads. "
        "What the system, the task, the inputs and the outputs mean is withheld from you."
    )


def _protocol_section(limits: ExecutorLimits) -> str:
    return f"""\
# How you work
- Every reply must be EITHER a ```python code block, which is executed and whose output you receive
  in the next message, OR your final answer (with no ```python block at all). Prefer exactly one
  block per reply: if you send several, they are merged in order into ONE run (separated by
  `# --- block N ---` comments) and count as one of your runs.
- A reply with neither a code block nor a complete final answer (see "Final answer" below) is a
  protocol error, not a final answer.
- Each code block runs in a fresh Python 3 process with numpy. Preloaded: `np` (numpy), `json`, and
  `W`, a dict mapping every parameter name below to a float32 numpy array. Files you write under
  /work persist between runs; /work/weights is read-only. There is no network and no other data.
- stdout is truncated to {limits.stdout_cap_chars} characters; the last {limits.stderr_tail_chars}
  characters of stderr are shown; each run is killed after {limits.timeout_s} s.
- You may execute at most {limits.max_runs} code blocks. Print compact summaries, not whole matrices."""


def _inputs_section(spec: Spec, task_mode: str) -> str:
    lines = [f"# Inputs (observation vector x, {len(spec.inputs)} entries)"]
    if task_mode == "shown":
        lines += [
            f"- x[{item.index}] {item.name}: {item.definition}" + (f" [{item.unit}]" if item.unit else "")
            for item in spec.inputs
        ]
    else:
        lines.append(f"x[0] .. x[{len(spec.inputs) - 1}]; refer to them as x0, x1, ... Their meaning is withheld.")
    if spec.input_clip is not None:
        low, high = spec.input_clip
        lines.append(f"Every entry of x is clipped to [{low:g}, {high:g}] before the network sees it.")
    return "\n".join(lines)


def _linspace_line(item: OutputSpec, label: str) -> str:
    return (
        f"- head {item.head} `{label}`: {item.bins} bins; bin k (0..{item.bins - 1}) means value "
        f"{item.low:g} + k * {item.high - item.low:g} / {item.bins - 1}  (from {item.low:g} to {item.high:g})"
    )


def _values_line(item: OutputSpec, label: str, task_mode: str) -> str:
    """Explicit per-bin values; bin labels and the unit are semantics, so ``blind`` drops them."""
    shown = task_mode == "shown"
    bins = [
        f"bin {k} = {value:g}" + (f" ({item.labels[k]})" if shown and item.labels else "")
        for k, value in enumerate(item.values)
    ]
    unit = f" [{item.unit}]" if shown and item.unit else ""
    return f"- head {item.head} `{label}`: {item.bins} bins; " + ", ".join(bins) + unit


def _outputs_section(spec: Spec, task_mode: str) -> str:
    lines = ["# Outputs (one categorical head per output; bins are ordinal)"]
    for item in spec.outputs:
        label = head_label(spec, item.head, task_mode)
        lines.append(_values_line(item, label, task_mode) if item.explicit else _linspace_line(item, label))
    if any(item.explicit for item in spec.outputs):
        lines.append(
            "A head's value is the number its chosen bin stands for (listed above), not the bin index; "
            "for some heads the value decreases as the bin index increases."
        )
    return "\n".join(lines)


def _parameters_section(tensors: tuple[Tensor, ...]) -> str:
    total = sum(t.numel for t in tensors)
    rows = [f"- `{t.name}` shape {list(t.shape)}" for t in tensors]
    return "\n".join([f"# Parameters (keys of W; {len(tensors)} arrays, {total} numbers)", *rows])


def _architecture_section(spec: Spec) -> str:
    flow = " -> ".join(["observation", *[block.block_id for block in spec.blocks]])
    return "\n\n".join([f"# Architecture (data flow: {flow})", *[equations(block) for block in spec.blocks]])


def _section_line(name: str, description: str) -> str:
    return f"## {name} ({description})" if description else f"## {name}"


def _final_answer_section(spec: Spec, task_mode: str) -> str:
    inputs = ", ".join(input_label(spec, i, task_mode) for i in range(len(spec.inputs)))
    heads = ", ".join(head_label(spec, h, task_mode) for h in range(len(spec.outputs)))
    sections = list(FINAL_SECTIONS)
    if task_mode == "blind":
        sections.append(HYPOTHESIS_SECTION)
    section_lines = "\n".join(_section_line(name, description) for name, description in sections)
    return f"""\
# Final answer
When you are done, reply WITHOUT any ```python block, using these sections:
{section_lines}
Then exactly one ```json block: {{"claims": [{{"input": <input>, "head": <head>, "direction": "+" | "-" | "none",
"confidence": <0..1>}}, ...]}}. "direction" says whether increasing that input, all else fixed, moves
the head's chosen value up (+) or down (-). Inputs: {inputs}. Heads: {heads}."""


def first_message(spec: Spec, tensors: tuple[Tensor, ...], task_mode: str, limits: ExecutorLimits) -> str:
    intro = (
        f'You are given the trained weights of a neural network called "{spec.alias}". Explain what it computes '
        "and how, by reading its weights. You will never see it run."
    )
    sections = [
        intro,
        "# Task\n" + _task_section(spec, task_mode),
        _protocol_section(limits),
        RULES,
        _architecture_section(spec),
        _inputs_section(spec, task_mode),
        _outputs_section(spec, task_mode),
        _parameters_section(tensors),
        _final_answer_section(spec, task_mode),
    ]
    return "\n\n".join(sections) + "\n"


def run_result_message(
    run_index: int,
    max_runs: int,
    stdout: str,
    stderr_tail: str,
    exit_code: int | None,
    timed_out: bool,
    stdout_truncated: bool,
    merged_blocks: int = 1,
) -> str:
    status = "TIMED OUT" if timed_out else f"exit code {exit_code}"
    parts = [f"OUTPUT of run {run_index} of at most {max_runs} ({status}):", "```text", stdout, "```"]
    if merged_blocks > 1:
        parts.append(f"Note: your reply had {merged_blocks} ```python blocks; they were merged into this one run.")
    if stdout_truncated:
        parts.append("(stdout was truncated)")
    if stderr_tail:
        parts += ["stderr (tail):", "```text", stderr_tail, "```"]
    remaining = max_runs - run_index
    parts.append(f"Runs remaining: {remaining}." if remaining else "No runs remain: reply with your final answer now.")
    return "\n".join(parts)


def protocol_error_message(problems: tuple[str, ...], runs_remaining: int) -> str:
    missing = "\n".join(f"- {problem}" for problem in problems)
    remaining = (
        f"Runs remaining: {runs_remaining}." if runs_remaining > 0 else "No runs remain: reply with your final answer."
    )
    return (
        "Your reply had neither a ```python code block nor a well-formed final answer. Missing:\n"
        f"{missing}\n"
        "Reply with a ```python block to run code, or with the complete final answer and no ```python block.\n"
        f"{remaining}"
    )


def budget_exhausted_message() -> str:
    return "The code-execution budget is used up. Reply with your final answer now, with no ```python block."


def spec_sections(spec: Spec, tensors: tuple[Tensor, ...], task_mode: str) -> dict[str, str]:
    """The whitelisted spec renderings (architecture, inputs, outputs, parameters) for other prompts."""
    return {
        "architecture": _architecture_section(spec),
        "inputs": _inputs_section(spec, task_mode),
        "outputs": _outputs_section(spec, task_mode),
        "parameters": _parameters_section(tensors),
    }
