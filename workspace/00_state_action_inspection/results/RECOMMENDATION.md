# State 10-bin recommendation

Primary design: use observation-space equal-width bins. This matches the existing Pendulum action-bin design: fixed task-level bounds, bins 0–8 are `[lower, upper)`, and bin 9 is `[lower, upper]`. Quantile bins are included only as a distribution diagnostic.

## MountainCar-v0

### state[0] — position

- Observation-space range: `[-1.2, 0.6]`
- Observed data range: `[-0.9283018708, 0.4993931949]`
- Distribution: Observed span covers 79.3% of the theoretical span; 9/10 observation-space bins are occupied; largest-bin share is 22.3%; top-two-bin share is 41.2%.
- Recommended method: Use observation-space equal-width bins.
- Recommended boundaries: `-1.2, -1.02, -0.84, -0.66, -0.48, -0.3, -0.12, 0.06, 0.24, 0.42, 0.6`
- Reason: The documented finite task range contains the data without severe concentration and preserves a fixed, dataset-independent class definition.

### state[1] — velocity

- Observation-space range: `[-0.07, 0.07]`
- Observed data range: `[-0.02155416459, 0.04503487796]`
- Distribution: Observed span covers 47.6% of the theoretical span; 6/10 observation-space bins are occupied; largest-bin share is 21.1%; top-two-bin share is 41.5%.
- Recommended method: Use observation-space equal-width bins.
- Recommended boundaries: `-0.07, -0.056, -0.042, -0.028, -0.014, 0, 0.014, 0.028, 0.042, 0.056, 0.07`
- Reason: The documented finite task range contains the data without severe concentration and preserves a fixed, dataset-independent class definition.

## Pendulum-v1

### state[0] — x = cos(theta)

- Observation-space range: `[-1, 1]`
- Observed data range: `[-0.9999824762, 0.999999702]`
- Distribution: Observed span covers 100.0% of the theoretical span; 10/10 observation-space bins are occupied; largest-bin share is 89.0%; top-two-bin share is 91.5%.
- Recommended method: Observation-space binning has a problem and needs review. Keep it as the explicit baseline; do not silently substitute observed or quantile bounds.
- Recommended boundaries: `-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1`
- Reason: observation-space bins are concentrated (occupied=10/10, largest=89.0%, top-two=91.5%)

### state[1] — y = sin(theta)

- Observation-space range: `[-1, 1]`
- Observed data range: `[-0.9999985695, 0.9999751449]`
- Distribution: Observed span covers 100.0% of the theoretical span; 10/10 observation-space bins are occupied; largest-bin share is 78.8%; top-two-bin share is 84.2%.
- Recommended method: Observation-space binning has a problem and needs review. Keep it as the explicit baseline; do not silently substitute observed or quantile bounds.
- Recommended boundaries: `-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1`
- Reason: observation-space bins are concentrated (occupied=10/10, largest=78.8%, top-two=84.2%)

### state[2] — angular velocity

- Observation-space range: `[-8, 8]`
- Observed data range: `[-8, 8]`
- Distribution: Observed span covers 100.0% of the theoretical span; 10/10 observation-space bins are occupied; largest-bin share is 47.8%; top-two-bin share is 90.3%.
- Recommended method: Observation-space binning has a problem and needs review. Keep it as the explicit baseline; do not silently substitute observed or quantile bounds.
- Recommended boundaries: `-8, -6.4, -4.8, -3.2, -1.6, 0, 1.6, 3.2, 4.8, 6.4, 8`
- Reason: observation-space bins are concentrated (occupied=10/10, largest=47.8%, top-two=90.3%)

## Joint prediction interpretation

Use `delta = next_state - current_state`: `INC` when delta > 0.0001, `DEC` when delta < -0.0001, and `UNCH` otherwise. The bin is the absolute next-state bin, while the real value remains the unrounded next-state value. See `state_delta_summary.csv` and the delta histograms before fixing this threshold in a prompt.

Source for names and theoretical ranges: `upstream/LLM-Xavier/llm_x/task.py` (`MountainCarTask`, `PendulumTask`).
