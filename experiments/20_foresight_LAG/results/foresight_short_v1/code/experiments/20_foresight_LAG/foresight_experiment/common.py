from importlib import import_module
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
ex = import_module('experiments.17_commentary_LAG.commentary_experiment.execution')
batch = import_module('experiments.17_commentary_LAG.commentary_experiment.batch')
answers = import_module('experiments.17_commentary_LAG.commentary_experiment.evaluation')
prompt_tools = import_module('experiments.17_commentary_LAG.commentary_experiment.prompts')
records = import_module('experiments.18_desicion_LAG.decision_experiment.dataset')
lag = import_module('experiments.13_LAG.lag_data.dataset')


def run_directory(name):
    path = (HERE / 'results' / name).resolve()
    if path == (HERE / 'results').resolve() or not path.is_relative_to((HERE / 'results').resolve()):
        raise ValueError('Run must be a subdirectory of results/')
    return path
