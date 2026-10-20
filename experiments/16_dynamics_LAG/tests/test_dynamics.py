"""Verify the experimental intervention and its coefficient provenance."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'13_LAG'), str(ROOT/'upstream/LLM-Xavier')]
from dynamics_experiment import dynamics, runner
from dynamics_experiment.config import Experiment


def test_only_prior_changes_from_saved_baseline():
    baseline = json.loads((HERE.parent/'15_trajectory_LAG/results/trajectory_batch_v1/manifest.json').read_text())
    plan = runner.make_plan(Experiment())
    assert plan['counts'] == baseline['counts']
    assert plan['coverage'] == baseline['coverage']
    assert plan['request_options'] == baseline['request_options']
    assert {k:v for k,v in plan['experiment'].items() if k!='name'} == {k:v for k,v in baseline['experiment'].items() if k!='name'}
    key = lambda s:(s['problem_id'],s['observation_scope'])
    old = {key(s):s for s in baseline['samples'].values()}
    for sample in plan['samples'].values():
        original = old[key(sample)]
        assert {k:v for k,v in sample.items() if k!='messages'} == {k:v for k,v in original.items() if k!='messages'}
        assert sample['messages'][1] == original['messages'][1]
        new_system = sample['messages'][0]['content']
        assert new_system.count(dynamics.prior_text()) == 1
        assert new_system.replace(dynamics.prior_text(),'') == original['messages'][0]['content']
    for path in [dynamics.MODEL_PATH,dynamics.TEMPLATE_PATH]:
        assert plan['source_sha256'][str(path.relative_to(ROOT))] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_coefficients_and_rendered_equation_match_source():
    model = dynamics.load_model()
    source_path = ROOT/model['source_results']
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == model['source_results_sha256']
    source = json.loads(source_path.read_text())['final_models']['history_and_velocity']
    for key in ['feature_names','coefficient_rows_features_columns_angles','intercept_deg']:
        assert model[key] == source[key]
    text = dynamics.prior_text()
    matrix = json.loads(text.split('Coefficient matrix B:\n')[1].split('\n\nFor the first forecast')[0])
    intercept = json.loads(text.split('b, ordered [roll, pitch, yaw]:\n')[1].split('\n\nB has')[0])
    names = json.loads(text.split('Row feature names, in order:\n')[1].split('\nCoefficient matrix B:')[0])
    assert matrix == source['coefficient_rows_features_columns_angles']
    assert intercept == source['intercept_deg'] and names == source['feature_names']
    # Numerical check of the row-feature/column-output convention in the prompt.
    x = np.arange(24,dtype=float)/7
    expected = np.array(source['intercept_deg']) + x@np.array(source['coefficient_rows_features_columns_angles'])
    assert np.allclose(np.array(intercept)+np.array(matrix).T@x,expected)


def test_short_history_cannot_satisfy_prior():
    for h in (1,2):
        with pytest.raises(ValueError,match='three'):
            replace(Experiment(),histories=(h,)).validate()
