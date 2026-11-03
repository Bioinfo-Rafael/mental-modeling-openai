"""Short human-readable ``evaluation.md`` rendered from the ``evaluation.json`` payload."""

from __future__ import annotations

PERCENT = 100


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{PERCENT * value:.0f}%"


def _acc(block: dict) -> str:
    return f"{_pct(block['accuracy'])} ({block['correct']}/{block['n']})"


def _ci(interval: list[float] | None) -> str:
    return "n/a" if interval is None else f"{_pct(interval[0])}-{_pct(interval[1])}"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:+.{digits}f}"


def _plain(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _header(payload: dict) -> list[str]:
    run, settings, parsed = payload["run"], payload["settings"], payload["claims_parsed"]
    lines = [
        f"# Evaluation: {payload['run_name']}",
        "",
        f"- network `{payload['network_alias']}`, model `{run['model']}`, task mode `{run['task_mode']}`",
        f"- claims scored: {parsed['n']}; rejected: {len(parsed['rejected'])}; duplicate pairs: {len(parsed['duplicate_pairs'])}",
        f"- base states: {settings['samples']} sampled from the spec's `evaluation` ranges (seed {settings['seed']}); "
        f"warm-up {settings['warmup_steps']} steps; step {settings['step_fraction']} of the input range; "
        f"angle mode `{settings['angle_mode']}`",
        f"- truth: >= {_pct(settings['agree'])} of states agree in sign AND |mean effect| >= {settings['min_effect']} "
        "of the head's value span over the input's whole range; else 'none'. Effect = d E[v] (expected bin value).",
        f"- primary condition: `{payload['primary_condition']}` (episode start, the only non-synthetic hidden state)",
        "- caveat: inputs are sampled independently (no recorded trajectories), so this is behaviour on a "
        "synthetic state distribution, not on deployment data",
    ]
    if run["spec_changed_since_run"]:
        lines.append("- note: the spec file changed since the run (expected: the harness-only `evaluation` section)")
    audit = payload.get("audit")
    if audit:
        lines.append(f"- audit: flagged runs {audit['flagged_runs']}, human verdict {audit['human_verdict']}")
    return lines


def _headline(payload: dict) -> list[str]:
    lines = [
        "",
        "## Headline",
        "",
        "| condition | overall (95% CI) | signed claims (95% CI) | 'none' claims | recall of signed truths | Brier"
        " | argmax-truth agreement | top-k coverage | truths +/-/none |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in payload["conditions_order"]:
        block = payload["conditions"][name]
        summary, boot, counts, rec = (
            block["summary"],
            block["bootstrap"],
            block["truth_counts"],
            block["summary"]["recall"],
        )
        marker = " (primary)" if name == payload["primary_condition"] else ""
        lines.append(
            f"| {name}{marker} | {_acc(summary['overall'])} ({_ci(boot['overall_accuracy_ci95'])})"
            f" | {_acc(summary['signed_claims'])} ({_ci(boot['signed_accuracy_ci95'])}) | {_acc(summary['none_claims'])}"
            f" | {_pct(rec['recall'])} ({rec['signed_truths_claimed_correctly']}/{rec['signed_truths']})"
            f" | {_plain(summary['brier'])} | {_acc(summary['argmax_secondary'])} | {_pct(block['coverage_hit_rate'])}"
            f" | {counts['+']}/{counts['-']}/{counts['none']} |"
        )
    first = payload["conditions"][payload["conditions_order"][0]]["summary"]["recall"]
    agreement = ", ".join(
        f"{key} {_pct(value)}" for key, value in payload["truth_agreement_between_conditions"].items()
    )
    return lines + [
        "",
        f"Pairs claimed: {first['pairs_claimed']} of {first['pairs_total']}. "
        f"Truth agreement between conditions (all pairs): {agreement}.",
    ]


def _confidence(payload: dict) -> list[str]:
    names = payload["conditions_order"]
    buckets = list(payload["conditions"][names[0]]["summary"]["by_confidence"])
    lines = ["", "## By confidence", "", "| confidence | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for bucket in buckets:
        cells = [_acc(payload["conditions"][name]["summary"]["by_confidence"][bucket]) for name in names]
        lines.append(f"| {bucket} | " + " | ".join(cells) + " |")
    return lines


def _claims(payload: dict) -> list[str]:
    names = payload["conditions_order"]
    per_condition = [payload["conditions"][name]["claims"] for name in names]
    header = (
        "| input | head | claim | conf | " + " | ".join(f"{n}: truth (mean effect, stability)" for n in names) + " |"
    )
    lines = ["", "## Claims", "", header, "|---" * (4 + len(names)) + "|"]
    for rows in zip(*per_condition, strict=True):
        first = rows[0]
        cells = [
            f"{'ok' if r['correct'] else 'X'} {r['truth']} ({_num(r['mean_effect'])}, {_pct(r['truth_stability'])})"
            for r in rows
        ]
        lines.append(
            f"| {first['input']} | {first['head']} | {first['claimed']} | {first['confidence']:.2f} | "
            + " | ".join(cells)
            + " |"
        )
    return lines


def _coverage(payload: dict) -> list[str]:
    lines = ["", "## Strongest true effects per head (coverage)"]
    for name in payload["conditions_order"]:
        rows = payload["conditions"][name]["coverage"]
        items = [
            f"{r['head']}#{r['rank']} {r['input']} {r['truth']} ({_num(r['mean_effect'])}) "
            f"{'claimed ' + r['claimed'] if r['claimed'] else 'NOT claimed'}"
            for r in rows
        ]
        lines += ["", f"- **{name}**: " + "; ".join(items)]
    return lines


def _sensitivity(payload: dict) -> list[str]:
    lines = ["", "## Threshold sensitivity (accuracy range over the agree x min_effect grid)", ""]
    for name in payload["conditions_order"]:
        grid = payload["conditions"][name]["sensitivity"]
        overall = [row["accuracy"] for row in grid if row["accuracy"] is not None]
        signed = [row["signed_accuracy"] for row in grid if row["signed_accuracy"] is not None]
        lines.append(
            f"- {name}: overall {_pct(min(overall, default=None))}-{_pct(max(overall, default=None))}, "
            f"signed {_pct(min(signed, default=None))}-{_pct(max(signed, default=None))}"
        )
    return lines


def render_markdown(payload: dict) -> str:
    sections = _header(payload) + _headline(payload) + _confidence(payload) + _claims(payload)
    return "\n".join(sections + _coverage(payload) + _sensitivity(payload)) + "\n"
