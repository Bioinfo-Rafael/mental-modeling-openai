# develop / v001 / baseline_h40

Model: `gpt-6-luna` · H=40 · stride=5

Prompt SHA256: `d0d08bc930bc1264a620658fdefe1672186189c8e002a4e2daab0d8db0c51c78`

| condition | valid/planned | TP | TN | FP | FN | recall | F2 | p50 s | p95 s | known USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| responses_diagnostic | 0/10 | 0 | 0 | 0 | 0 | — | — | — | — | — |

- Classification metrics above use valid outputs only; errors, pending counts and error-as-miss metrics are in summary.json.
- Cost is a usage-based estimate, not an invoice. Unknown-cost attempts are reported separately; null is not zero.
- Step matching requires the same aircraft and is one-to-one. Exact and ±tolerance metrics are both saved.
- Binary event coverage is only an overlapping-window proxy. Localization and duplicate counts require diagnostic output.
- Reasons are for manual review; no automatic score claims that a tactical interpretation is correct.
- Single episode reused for development and evaluation; not a held-out test.
- Unannotated windows are weak negatives, not verified absence of important events.
- Window-wide retrospective detection, not future-event prediction.

[Review responses](review.html) · [Predictions](predictions.csv) · [Detailed metrics](summary.json)
