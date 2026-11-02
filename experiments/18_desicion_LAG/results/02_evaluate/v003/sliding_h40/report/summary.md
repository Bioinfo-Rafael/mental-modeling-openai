# evaluate / v003 / sliding_h40

Model: `gpt-6-luna` · H=40 · stride=5

Prompt SHA256: `1e3acc10e8c9fe704ce61dcbaa017ed51b92fdf356ba757278c8bc6b1e55ee1e`

| condition | valid/planned | TP | TN | FP | FN | recall | F2 | p50 s | p95 s | known USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| responses_diagnostic | 169/186 | 88 | 25 | 38 | 18 | 0.8302 | 0.8000 | 5.5634 | 10.7074 | 0.3190 |
| decisions | 186/186 | 96 | 30 | 33 | 27 | 0.7805 | 0.7729 | 1.7835 | 4.5428 | 0.2890 |

| condition | aircraft set accuracy | A1 recall | A2 recall |
|---|---:|---:|---:|
| responses_diagnostic | 0.5858 | 0.9589 | 0.5606 |
| decisions | 0.5323 | 0.8000 | 0.5833 |

- Classification metrics above use valid outputs only; errors, pending counts and error-as-miss metrics are in summary.json.
- Cost is a usage-based estimate, not an invoice. Unknown-cost attempts are reported separately; null is not zero.
- Step matching requires the same aircraft and is one-to-one. Exact and ±tolerance metrics are both saved.
- Binary event coverage is only an overlapping-window proxy. Localization and duplicate counts require diagnostic output.
- Aircraft-set accuracy requires the exact NONE/A1/A2/BOTH set. It does not establish correct step localization.
- Responses diagnostic and Decisions choice produce different outputs; latency/cost compare these workloads, not an isolated API effect.
- Reasons are for manual review; no automatic score claims that a tactical interpretation is correct.
- Single episode reused for development and evaluation; not a held-out test.
- Unannotated windows are weak negatives, not verified absence of important events.
- Window-wide retrospective detection, not future-event prediction.

[Review responses](review.html) · [Predictions](predictions.csv) · [Detailed metrics](summary.json)
