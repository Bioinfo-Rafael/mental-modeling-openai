"""Architecture blocks: which state_dict keys each block owns, their shapes, and equation text.

Supported block types (all PyTorch conventions):

* ``mlp``: ``nn.Sequential(Linear, ReLU, LayerNorm, Linear, ReLU, LayerNorm, ...)``; Linear j at
  index ``3*j``, LayerNorm j at ``3*j + 2``.
* ``gru``: single-layer ``nn.GRU``; gate blocks ``[r; z; n]``.
* ``layernorm``: ``nn.LayerNorm(size)``.
* ``categorical_heads``: one ``Linear(input_size, bins)`` per output head.
"""

from __future__ import annotations

from dataclasses import dataclass

MLP_STRIDE = 3  # Linear, ReLU, LayerNorm per layer
MLP_NORM_OFFSET = 2
GRU_GATES = 3

BLOCK_TYPES = ("mlp", "gru", "layernorm", "categorical_heads")


@dataclass(frozen=True)
class Block:
    """One architecture block. ``input`` names the producing block id or ``observation``."""

    block_id: str
    block_type: str
    prefix: str
    input: str
    input_size: int
    sizes: tuple[int, ...]  # mlp: hidden sizes; gru/layernorm: (hidden,); heads: bins per head
    eps: float = 1e-5
    param: str = "logits_net"

    @property
    def output_size(self) -> int:
        if self.block_type == "mlp":
            return self.sizes[-1]
        if self.block_type == "categorical_heads":
            return sum(self.sizes)
        return self.sizes[0]


def expected_params(block: Block) -> dict[str, tuple[int, ...]]:
    """Map of full state_dict key -> shape that this block owns."""
    if block.block_type == "mlp":
        return _mlp_params(block)
    if block.block_type == "gru":
        hidden = block.sizes[0]
        rows = GRU_GATES * hidden
        return {
            f"{block.prefix}weight_ih_l0": (rows, block.input_size),
            f"{block.prefix}weight_hh_l0": (rows, hidden),
            f"{block.prefix}bias_ih_l0": (rows,),
            f"{block.prefix}bias_hh_l0": (rows,),
        }
    if block.block_type == "layernorm":
        return {f"{block.prefix}weight": block.sizes, f"{block.prefix}bias": block.sizes}
    if block.block_type == "categorical_heads":
        params: dict[str, tuple[int, ...]] = {}
        for head, bins in enumerate(block.sizes):
            params[f"{block.prefix}{head}.{block.param}.weight"] = (bins, block.input_size)
            params[f"{block.prefix}{head}.{block.param}.bias"] = (bins,)
        return params
    raise ValueError(f"unknown block type: {block.block_type}")


def _mlp_params(block: Block) -> dict[str, tuple[int, ...]]:
    params: dict[str, tuple[int, ...]] = {}
    previous = block.input_size
    for layer, hidden in enumerate(block.sizes):
        linear = f"{block.prefix}{MLP_STRIDE * layer}"
        norm = f"{block.prefix}{MLP_STRIDE * layer + MLP_NORM_OFFSET}"
        params[f"{linear}.weight"] = (hidden, previous)
        params[f"{linear}.bias"] = (hidden,)
        params[f"{norm}.weight"] = (hidden,)
        params[f"{norm}.bias"] = (hidden,)
        previous = hidden
    return params


def _layer_norm_text(eps: float) -> str:
    return f"LN(v) = g * (v - mean(v)) / sqrt(var(v) + {eps:g}) + beta   (mean/var over the vector's own entries)"


def equations(block: Block) -> str:
    """Exact math for one block (text, not runnable code)."""
    source = "the observation x" if block.input == "observation" else f"the output of block `{block.input}`"
    if block.block_type == "mlp":
        return _mlp_equations(block, source)
    if block.block_type == "gru":
        return _gru_equations(block, source)
    if block.block_type == "layernorm":
        return (
            f"Block `{block.block_id}` (LayerNorm, size {block.sizes[0]}), input: {source}.\n"
            f"  y = LN(v) with g = W['{block.prefix}weight'], beta = W['{block.prefix}bias'];\n"
            f"  {_layer_norm_text(block.eps)}"
        )
    return _heads_equations(block, source)


def _mlp_equations(block: Block, source: str) -> str:
    lines = [f"Block `{block.block_id}` (MLP, {len(block.sizes)} layers), input: {source}."]
    previous = "x" if block.input == "observation" else "u"
    for layer, hidden in enumerate(block.sizes):
        linear = f"{block.prefix}{MLP_STRIDE * layer}"
        norm = f"{block.prefix}{MLP_STRIDE * layer + MLP_NORM_OFFSET}"
        lines.append(
            f"  a{layer + 1} = LN{layer + 1}(relu(W['{linear}.weight'] @ {previous} + W['{linear}.bias']))"
            f"   ({hidden} units; LN{layer + 1} gain W['{norm}.weight'], bias W['{norm}.bias'])"
        )
        previous = f"a{layer + 1}"
    lines.append(f"  output = {previous}. Order is Linear -> ReLU -> LayerNorm (LayerNorm comes AFTER the ReLU).")
    lines.append(f"  {_layer_norm_text(block.eps)}")
    return "\n".join(lines)


def _gru_equations(block: Block, source: str) -> str:
    hidden = block.sizes[0]
    p = block.prefix
    return "\n".join(
        [
            f"Block `{block.block_id}` (GRU, PyTorch convention, hidden size {hidden}), input u: {source}.",
            f"  Rows of W['{p}weight_ih_l0'], W['{p}weight_hh_l0'], W['{p}bias_ih_l0'], W['{p}bias_hh_l0'] are"
            f" stacked gate blocks [r; z; n] of {hidden} rows each (rows 0-{hidden - 1} = r,"
            f" {hidden}-{2 * hidden - 1} = z, {2 * hidden}-{3 * hidden - 1} = n).",
            "  r  = sigmoid(W_ir u + b_ir + W_hr h + b_hr)",
            "  z  = sigmoid(W_iz u + b_iz + W_hz h + b_hz)",
            "  n  = tanh(W_in u + b_in + r * (W_hn h + b_hn))",
            "  h' = (1 - z) * n + z * h",
            "  h = 0 at the start of every episode; exactly one GRU step per decision.",
            "  The carried recurrent state is h' itself (before any LayerNorm); the block output is h'.",
        ]
    )


def _heads_equations(block: Block, source: str) -> str:
    lines = [f"Block `{block.block_id}` ({len(block.sizes)} categorical output heads), input v: {source}."]
    for head, bins in enumerate(block.sizes):
        key = f"{block.prefix}{head}.{block.param}"
        lines.append(f"  logits_{head} = W['{key}.weight'] @ v + W['{key}.bias']   ({bins} bins)")
    lines.append("  Each head's action is argmax of its logits (deterministic at deployment); heads are independent.")
    return "\n".join(lines)
