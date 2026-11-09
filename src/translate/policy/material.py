"""The ``policy`` input: a recorded policy's weights and its validated replay activations.

``load_policy_material`` runs once per translation run: it reads the checkpoint named in the rollout
manifest (allow-list unpickler, SHA-256 checked), checks it against the network spec, replays the
episode free-running from ``h = 0`` in float64 and refuses unless the replay reproduces the
recording (logits, ``h'``, ``h_prev`` vs ``rnn_in`` within 1e-4, every action). Per slot,
``policy_values`` gives the prompt values and ``write_session_inputs`` writes the read-only
bundle a code session reads as ``W`` and ``A``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from translate.policy.reference import ReferencePolicy, build_policy
from translate.policy.render import activation_rows, spec_sections
from translate.policy.rollout import Rollout, RolloutError, load_rollout, sha256_file
from translate.policy.session_inputs import SessionInputs, write_inputs
from translate.policy.spec import Spec, check_against_tensors, load_spec
from translate.policy.torch_reader import Checkpoint, read_checkpoint
from translate.policy.trace import (
    ActivationDoc,
    TraceError,
    activation_docs,
    trace_episode,
    validate_trace,
    window_arrays,
    window_bounds,
)


@dataclass(frozen=True)
class PolicyMaterial:
    seed: int
    spec: Spec
    spec_path: Path
    checkpoint: Checkpoint
    rollout: Rollout
    policy: ReferencePolicy
    trace: dict[str, np.ndarray]
    validation: dict
    docs: tuple[ActivationDoc, ...]
    sections: dict[str, str]

    def provenance(self) -> dict:
        return {
            "spec_path": str(self.spec_path),
            "spec_sha256": sha256_file(self.spec_path),
            "network_alias": self.spec.alias,
            "checkpoint_path": str(self.checkpoint.path),
            "checkpoint_sha256": self.checkpoint.sha256,
            "rollout_path": str(self.rollout.path),
            "rollout_sha256": self.rollout.sha256,
            "rollout_manifest_sha256": self.rollout.manifest_sha256,
            "trace_validation": self.validation,
        }


def load_policy_material(
    rollout_dir: Path, seed: int, spec_path: Path, checkpoint_path: Path | None = None
) -> PolicyMaterial:
    rollout = load_rollout(rollout_dir, seed)
    checkpoint = read_checkpoint(checkpoint_path or rollout.checkpoint_path)
    if checkpoint.sha256 != rollout.checkpoint_sha256:
        raise RolloutError("checkpoint sha256 differs from the actor recorded in the rollout manifest")
    spec = load_spec(spec_path)
    check_against_tensors(spec, checkpoint.tensors, checkpoint.sha256)
    policy = build_policy(spec, checkpoint.tensors)
    arrays = rollout.arrays
    trace = trace_episode(policy, arrays["obs"])
    validation = validate_trace(trace, policy, arrays["logits"], arrays["gru_out"], arrays["actions"], arrays["rnn_in"])
    if not validation["ok"]:
        raise TraceError(f"replayed activations do not reproduce the recording: {validation}")
    sections = spec_sections(spec, checkpoint.tensors)
    return PolicyMaterial(
        seed, spec, spec_path, checkpoint, rollout, policy, trace, validation, activation_docs(policy), sections
    )


def policy_values(material: PolicyMaterial, step: int, past_steps: int) -> dict[str, object]:
    """Prompt values of the ``policy`` input for the window ending at ``step``."""
    start, end = window_bounds(step, past_steps)
    length = end - start + 1
    return {
        **material.sections,
        "policy_alias": material.spec.alias,
        "policy_activations": activation_rows(material.docs, length),
        "window_start": start,
        "window_end": end,
        "window_length": length,
    }


def write_session_inputs(
    material: PolicyMaterial, aircraft: str, step: int, past_steps: int, directory: Path
) -> SessionInputs:
    """``W`` (the checkpoint tensors) and ``A`` (this aircraft's window) for one code session."""
    start, end = window_bounds(step, past_steps)
    row = material.rollout.learner_row(aircraft)
    if step >= material.rollout.decisions:
        raise RolloutError(f"step {step} is after the end of the recorded rollout")
    metadata = {
        "network_alias": material.spec.alias,
        "checkpoint_sha256": material.checkpoint.sha256,
        "rollout_sha256": material.rollout.sha256,
        "seed": material.seed,
        "aircraft": aircraft,
        "learner_row": row,
        "step": step,
        "window": [start, end],
    }
    arrays = window_arrays(material.trace, row, start, end)
    return write_inputs(directory, material.checkpoint.tensors, arrays, metadata)
