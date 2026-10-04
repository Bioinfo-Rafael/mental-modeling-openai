"""Offline context comparison, using Exp.11's metrics, bars and red baseline segments."""
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
plots = import_module('experiments.11_ICRL_n10.plot_results')
HERE = Path(__file__).resolve().parent
ORDER = [(context, model) for context in ('fewshot4', 'episodes_E9') for model in ('terra', 'luna')]

if __name__ == '__main__':
    plots.main(root=HERE / 'results', group_field='context_method', order=ORDER,
               group_labels=('fewshot4', 'episodes_E9'), title='Few-shot / Episodes',
               overview_name='context_comparison')
