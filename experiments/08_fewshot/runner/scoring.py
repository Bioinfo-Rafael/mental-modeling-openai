"""Exp.08 Joint action scoring: final IDs for accuracy, numeric torque for error."""
from experiments.common_analysis.joint_data import parse_response
from llm_x.metrics import bin_actions

VERSION = '08_joint_final_action_v1'
EVALUATION_FIELDS = ('index', 'status', 'prediction', 'ground_truth', 'element_accuracy',
                     'parse_error', 'action_value_prediction', 'action_value_ground_truth',
                     'absolute_error', 'action_value_parse_error', 'scoring_version')
CSV_FIELDS = ('query_id', 'condition_id', 'model_alias', 'task', 'metric', 'H', 'shots',
              'score_pattern', 'ordinal', 'status', 'prediction', 'ground_truth',
              'assistant_text', 'input_tokens', 'output_tokens', 'total_tokens',
              'request_elapsed_seconds', 'action_value_prediction',
              'action_value_ground_truth', 'absolute_error', 'scoring_version')


def score(row, text, episode):
    if row['metric'] != 'next-action' or row['task'] not in ('MountainCar-v0', 'Pendulum-v1'):
        raise ValueError('Exp.08 scoring supports only MountainCar/Pendulum next-action')
    index = row['query_index']
    mountain = row['task'] == 'MountainCar-v0'
    truth = episode.action_vector(index).astype(float).tolist()
    expected = episode.discrete_action(index) if mountain else bin_actions(truth, start=-2, stop=2, bins=10)
    result = dict(index=index, ground_truth=expected, prediction=None, scoring_version=VERSION,
                  action_value_ground_truth=truth, action_value_prediction=None, absolute_error=None)
    if not isinstance(text, str) or not text.strip():
        return {**result, 'status': 'empty_response'}
    parsed = parse_response(text, row['task'], row['metric'])
    discrete = parsed['action' if mountain else 'action_bin']
    numeric = parsed['action' if mountain else 'action_value']
    if discrete['ok']:
        prediction = discrete['value'][0] if mountain else discrete['value']
        result.update(prediction=prediction, status='match' if prediction == expected else 'mismatch')
        if not mountain:
            result['element_accuracy'] = float(prediction == expected)
    else:
        result.update(status='ignored', parse_error=discrete['error'])
    if numeric['ok']:
        result.update(action_value_prediction=numeric['value'],
                      absolute_error=abs(numeric['value'][0] - truth[0]))
    else:
        result['action_value_parse_error'] = numeric['error']
    return result
