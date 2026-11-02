# develop / v002 / onset_localization_h40

Model: `gpt-6-luna` · H=40 · stride=5

Prompt SHA256: `f188c2b83bbaf195ad30cde1d07fb757ee6697ab5dfd9dc74fba7eb8d0a8dfd4`

| condition | valid/planned | TP | TN | FP | FN | recall | F2 | p50 s | p95 s | known USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| responses_diagnostic | 9/10 | 4 | 2 | 3 | 0 | 1.0000 | 0.8696 | 5.7656 | 10.1323 | 0.0391 |

- Classification metrics above use valid outputs only; errors, pending counts and error-as-miss metrics are in summary.json.
- Cost is a usage-based estimate, not an invoice. Unknown-cost attempts are reported separately; null is not zero.
- Step matching requires the same aircraft and is one-to-one. Exact and ±tolerance metrics are both saved.
- Binary event coverage is only an overlapping-window proxy. Localization and duplicate counts require diagnostic output.
- Reasons are for manual review; no automatic score claims that a tactical interpretation is correct.
- Single episode reused for development and evaluation; not a held-out test.
- Unannotated windows are weak negatives, not verified absence of important events.
- Window-wide retrospective detection, not future-event prediction.

[Review responses](review.html) · [Predictions](predictions.csv) · [Detailed metrics](summary.json)
