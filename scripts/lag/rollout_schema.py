"""Shared layout of recorded policy rollouts (numpy-only; imported by the recorder and the replay).

One ``episode_seed<seed>.npz`` per episode (and one ``head_weights.npz``: ``weight`` (13, 128), ``bias``
(13,), ``head_sizes``), T = decisions in the episode (one per env step, 0.2 s).
Rows ``R`` of the recurrent state depend on the protocol: 2 for ``per_agent_v1`` (one per learner),
1 for ``mhi_render_v1_3`` (one row shared by both learners, stepped in packed order).
"""

from __future__ import annotations

SCHEMA = "weightreading_rollout_v1"
DEFAULT_SCENARIO = "2v2/ShootMissile/HierarchyVsbaseline_mhi_v1_3"
# mtd-rl-lag scripts/eval/evaluate_mhi_team_batch.py SEEDS start at 100600.
DEFAULT_SEEDS = (100600, 100601, 100602, 100603)
PROTOCOLS = ("per_agent_v1", "mhi_render_v1_3")
PROTOCOL_ROWS = {"per_agent_v1": 2, "mhi_render_v1_3": 1}

DECISION_FIELDS = {
    "obs": "(T, 2, 33) float32: actor input rows exactly as passed to PPOActor.forward, learner packed order",
    "rnn_in": "(T, R, 1, 128) float32: recurrent state passed to the actor (pre-LayerNorm h)",
    "masks": "(T, 2, 1) float32: GRU masks passed to the actor (all ones)",
    "gru_out": "(T, 2, 128) float32: nn.GRU output per learner row = pre-LayerNorm h' after that row",
    "act_in": "(T, 2, 128) float32: input of every action head (the act MLP output), learner packed order",
    "logits": "(T, 2, 13) float32: raw head logits concatenated in head order (action_head_sizes)",
    "actions": "(T, 2, 4) int64: chosen bins per head (deterministic = argmax)",
    "log_probs": "(T, 2, 1) float32: summed log-probability of the chosen bins",
    "rnn_out": "(T, R, 1, 128) float32: recurrent state returned by the actor (carried to step t+1)",
}

CONTEXT_FIELDS = {
    "step": "(T,) int64: env.current_step before the decision is applied",
    "alive": "(T, 4) bool: aircraft alive before the step, aircraft_uids order",
    "shotdown": "(T, 4) bool: aircraft shot down before the step",
    "crash": "(T, 4) bool: aircraft crashed before the step",
    "geodetic": "(T, 4, 3) float64: (longitude deg, latitude deg, altitude m)",
    "position_neu": "(T, 4, 3) float64: (north, east, up) m relative to the battle-field centre",
    "rpy": "(T, 4, 3) float64: (roll, pitch, yaw) rad",
    "velocity_neu": "(T, 4, 3) float64: (v_north, v_east, v_up) m/s",
    "missiles_left": "(T, 4) int64: missiles not yet launched",
    "reward": "(T, 4, k) float64: env rewards returned by this step",
    "done": "(T, 4, k) bool: env dones returned by this step",
}


def episode_file(seed: int) -> str:
    return f"episode_seed{seed}.npz"
