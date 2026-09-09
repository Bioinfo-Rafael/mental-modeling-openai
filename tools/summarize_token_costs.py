#!/usr/bin/env python3
"""Combine offline input counts and pilot output means with user-supplied prices."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from tools.llmx_adapter import WORKSPACE_ROOT


def estimate_costs(
    input_summary: dict[str, Any],
    pilot_summary: dict[str, Any] | None,
    *,
    input_price_per_million: float,
    output_price_per_million: float,
) -> dict[str, Any]:
    if input_summary.get('schema_version') == 2:
        rows = input_summary['by_history']
        if len(rows) != 1:
            raise ValueError('Choose one prompt_mode/history_size scope before estimating costs; mixed histories must not be silently summed.')
        if any(s['count_scope'] != 'full_selected_scope' for s in input_summary['scopes']):
            raise ValueError('Sample token totals are not full-dataset cost estimates.')
        query_count = int(rows[0]['number_of_queries'])
        input_tokens = int(rows[0]['total_input_tokens'])
    else:
        query_count = int(input_summary["overall"]["number_of_queries"])
        input_tokens = int(input_summary["overall"]["total_input_tokens"])
    mean_output = (
        float(pilot_summary["overall"]["output_tokens"]["mean"])
        if pilot_summary and pilot_summary["overall"]["output_tokens"]["mean"] is not None
        else None
    )
    projected_output = mean_output * query_count if mean_output is not None else None
    input_cost = input_tokens / 1_000_000 * input_price_per_million
    output_cost = (
        projected_output / 1_000_000 * output_price_per_million
        if projected_output is not None
        else None
    )
    return {
        "schema_version": 1,
        "prices_are_user_supplied": True,
        "query_count": query_count,
        "input_tokens": input_tokens,
        "pilot_mean_output_tokens": mean_output,
        "projected_output_tokens": projected_output,
        "input_price_per_million_usd": input_price_per_million,
        "output_price_per_million_usd": output_price_per_million,
        "estimated_input_cost_usd": input_cost,
        "estimated_output_cost_usd": output_cost,
        "estimated_total_cost_usd": input_cost + output_cost if output_cost is not None else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-summary",
        type=Path,
        default=WORKSPACE_ROOT / "outputs" / "token_counts" / "summary.json",
    )
    parser.add_argument("--pilot-summary", type=Path)
    parser.add_argument("--input-price-per-million", type=float, required=True)
    parser.add_argument("--output-price-per-million", type=float, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "outputs" / "estimated_costs.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from tools.output_safety import derived_path
    args.output = derived_path(args.output)
    input_summary = json.loads(args.input_summary.read_text())
    pilot_summary = json.loads(args.pilot_summary.read_text()) if args.pilot_summary else None
    result = estimate_costs(
        input_summary,
        pilot_summary,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, delete=False) as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
