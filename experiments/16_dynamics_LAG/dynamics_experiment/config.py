"""Experiment settings and their validation, kept together for review."""
from dataclasses import dataclass

from lag_data.definitions import STATE_FEATURES

# The same aliases/model IDs and request options as Experiments 05/06/08-12.
MODELS = {"sol": "gpt-5.6-sol", "terra": "gpt-5.6-terra", "luna": "gpt-5.6-luna"}
TASKS = ("trajectory",)
OBSERVATION_SCOPES = ("self", "all_aircraft_missiles")
EPISODES = (
    "mhi_v1.3_ic100601_win",
)
TARGET_ACTION_FIELDS = ("altitude_action", "heading_action", "speed_action")
REQUEST_OPTIONS = {"reasoning_effort": "medium"}


@dataclass(frozen=True)
class Experiment:
    name: str = "dynamics_batch_v1"
    models: tuple[str, ...] = ("sol", "terra", "luna")
    histories: tuple[int, ...] = (20,)
    horizon: int = 10
    episodes: tuple[str, ...] = EPISODES
    target_ids: tuple[str, ...] = ("A0100", "A0200")
    n: int = 5
    sampling: str = "uniform_time_bins"
    tasks: tuple[str, ...] = TASKS
    observation_scopes: tuple[str, ...] = OBSERVATION_SCOPES
    state_fields: tuple[str, ...] = STATE_FEATURES

    def validate(self):
        if type(self.horizon) is not int or self.horizon != 10:
            raise ValueError("This experiment jointly predicts exactly ten future steps")
        if self.sampling != "uniform_time_bins":
            raise ValueError("This experiment selects one start from each of n equal temporal bins")
        if not self.tasks or len(set(self.tasks)) != len(self.tasks) or not set(self.tasks) <= set(TASKS):
            raise ValueError("Choose prediction tasks: " + ", ".join(TASKS))
        for values, allowed, name in ((self.models, MODELS, "models"), (self.episodes, EPISODES, "episodes"),
                                      (self.target_ids, ("A0100", "A0200"), "target_ids"),
                                      (self.observation_scopes, OBSERVATION_SCOPES, "observation_scopes")):
            if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
                raise ValueError(f"Invalid {name}")
        if not self.histories or len(set(self.histories)) != len(self.histories) or any(type(h) is not int or h < 3 for h in self.histories):
            raise ValueError("Histories must be distinct integers of at least three steps for the attitude prior")
        if type(self.n) is not int or self.n < 1:
            raise ValueError("n must be a positive integer")
        if tuple(self.state_fields) != STATE_FEATURES:
            raise ValueError("This experiment uses the specified ten state fields in their fixed order")
        if not self.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.name):
            raise ValueError("Invalid experiment name")
