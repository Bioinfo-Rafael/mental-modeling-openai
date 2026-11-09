"""Text renderings of a policy for a prompt: architecture equations, inputs, outputs, parameters, activations.

Only whitelisted spec fields are rendered (alias, description, inputs, clip, architecture,
outputs); the spec's ``provenance``, ``notes`` and ``evaluation`` sections never reach a model.
These renderings are data, like the trajectory window: the wording around them belongs to the
prompt file.
"""

from __future__ import annotations

from translate.policy.architecture import equations
from translate.policy.spec import OutputSpec, Spec
from translate.policy.tensors import Tensor
from translate.policy.trace import ActivationDoc


def _inputs_section(spec: Spec) -> str:
    lines = [f"# Inputs (observation vector x, {len(spec.inputs)} entries)"]
    lines += [
        f"- x[{item.index}] {item.name}: {item.definition}" + (f" [{item.unit}]" if item.unit else "")
        for item in spec.inputs
    ]
    if spec.input_clip is not None:
        low, high = spec.input_clip
        lines.append(f"Every entry of x is clipped to [{low:g}, {high:g}] before the network sees it.")
    return "\n".join(lines)


def _linspace_line(item: OutputSpec) -> str:
    return (
        f"- head {item.head} `{item.name}`: {item.bins} bins; bin k (0..{item.bins - 1}) means value "
        f"{item.low:g} + k * {item.high - item.low:g} / {item.bins - 1}  (from {item.low:g} to {item.high:g})"
    )


def _values_line(item: OutputSpec) -> str:
    bins = [
        f"bin {k} = {value:g}" + (f" ({item.labels[k]})" if item.labels else "") for k, value in enumerate(item.values)
    ]
    unit = f" [{item.unit}]" if item.unit else ""
    return f"- head {item.head} `{item.name}`: {item.bins} bins; " + ", ".join(bins) + unit


def _outputs_section(spec: Spec) -> str:
    lines = ["# Outputs (one categorical head per output; bins are ordinal)"]
    lines += [_values_line(item) if item.explicit else _linspace_line(item) for item in spec.outputs]
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


def spec_sections(spec: Spec, tensors: tuple[Tensor, ...]) -> dict[str, str]:
    """The whitelisted spec renderings, keyed by the prompt value they fill."""
    return {
        "policy_task": spec.description,
        "policy_architecture": _architecture_section(spec),
        "policy_inputs": _inputs_section(spec),
        "policy_outputs": _outputs_section(spec),
        "policy_parameters": _parameters_section(tensors),
    }


def activation_rows(docs: tuple[ActivationDoc, ...], window_length: int) -> str:
    """One line per activation key: its shape in the window and what it is."""
    return "\n".join(f"- `A['{doc.key}']` shape {[window_length, *doc.shape_tail]}: {doc.description}" for doc in docs)
