"""Render a frozen, auditable empirical prior without accessing future telemetry."""
import hashlib
import json
import math
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parents[1] / 'dynamics'
MODEL_PATH = DIRECTORY / 'attitude_model.json'
TEMPLATE_PATH = DIRECTORY / 'attitude_prior.txt'
VERSION = 'lag-attitude-history-velocity-full3-v1'


def load_model():
    model = json.loads(MODEL_PATH.read_text())
    if model['angle_order'] != ['Roll', 'Pitch', 'Yaw'] or model['dt_seconds'] != 0.2:
        raise ValueError('Unexpected attitude convention or sample interval')
    names = model['feature_names']
    matrix = model['coefficient_rows_features_columns_angles']
    intercept = model['intercept_deg']
    if len(names) != 24 or len(set(names)) != 24 or len(matrix) != 24 or len(intercept) != 3:
        raise ValueError('Expected 24 features and three output angles')
    if any(len(row) != 3 for row in matrix):
        raise ValueError('Expected coefficient matrix shape (24, 3)')
    if not all(type(x) in (float, int) and math.isfinite(x) for x in intercept + [v for row in matrix for v in row]):
        raise ValueError('Nonfinite or nonnumeric coefficient')
    return model


def prior_text():
    model = load_model()
    text = TEMPLATE_PATH.read_text()
    values = {'INTERCEPT_JSON':model['intercept_deg'], 'FEATURE_NAMES_JSON':model['feature_names'],
              'COEFFICIENTS_JSON':model['coefficient_rows_features_columns_angles']}
    for token, value in values.items():
        placeholder = '{{' + token + '}}'
        if text.count(placeholder) != 1:
            raise ValueError(f'Expected one {placeholder} in prior template')
        text = text.replace(placeholder, json.dumps(value, indent=2, allow_nan=False))
    if '{{' in text or '}}' in text:
        raise ValueError('Unresolved prior template placeholder')
    return text.rstrip()


def provenance():
    model = load_model()
    return dict(version=VERSION, model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                template_sha256=hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest(),
                rendered_sha256=hashlib.sha256(prior_text().encode()).hexdigest(),
                source_results=model['source_results'], source_results_sha256=model['source_results_sha256'],
                source_report=model['source_report'], source_report_sha256=model['source_report_sha256'],
                evaluation_overlap=model['evaluation_overlap'])
