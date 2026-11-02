# develop / v001 / concise_h40

Model: `gpt-6-luna` · H=40 · stride=5

Prompt SHA256: `5ec5489de24e414ee8dc86d8331d09d7e086c15637419fd9ed61b5331e123341`

| condition | valid/planned | TP | TN | FP | FN | recall | F2 | p50 s | p95 s | known USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| responses_diagnostic | 10/10 | 5 | 1 | 4 | 0 | 1.0000 | 0.8621 | 4.5088 | 9.1279 | 0.0387 |

- Classification metrics above use valid outputs only; errors, pending counts and error-as-miss metrics are in summary.json.
- Cost is a usage-based estimate, not an invoice. Unknown-cost attempts are reported separately; null is not zero.
- Step matching requires the same aircraft and is one-to-one. Exact and ±tolerance metrics are both saved.
- Binary event coverage is only an overlapping-window proxy. Localization and duplicate counts require diagnostic output.
- Reasons are for manual review; no automatic score claims that a tactical interpretation is correct.
- Single episode reused for development and evaluation; not a held-out test.
- Unannotated windows are weak negatives, not verified absence of important events.
- Window-wide retrospective detection, not future-event prediction.

[Review responses](review.html) · [Predictions](predictions.csv) · [Detailed metrics](summary.json)
