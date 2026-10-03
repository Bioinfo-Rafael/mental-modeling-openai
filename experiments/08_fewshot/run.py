"""Terra/Luna next-action few-shot sweep. No flags: offline prompt preview only."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from importlib import import_module
# @Rafaメモ:experiments/08_fewshot/runner/fewshot_runner.pyをrunnerとしてimport
runner = import_module('experiments.08_fewshot.runner.fewshot_runner') 

# Conditions live here, following Exp.06's small run.py entry point.
# @Rafaメモ: runner.pyのFewshotExperimentクラスを使って、実験の条件を定義。これは単なる型定義
EXPERIMENT = runner.FewshotExperiment(
    name='08_fewshot',
    models=('terra', 'luna'),
    tasks=('MountainCar-v0', 'Pendulum-v1'),
    metric='next-action',
    histories=(5, 20),
    shot_counts=(4, 8, 12),  # Total examples: half correct, half incorrect.
    score_patterns=(
        runner.ScorePattern('pattern1', correct_scores=(1,), incorrect_scores=(3, 4)),
        runner.ScorePattern('pattern2', correct_scores=(1, 2), incorrect_scores=(1, 2, 3)),
    ),
    n=3,
    seed=42,
    selection_policy="balanced_next",  # Six per role; model balance + next-action priority.
)

# Empty: automatically use the only candidate_pool JSON in data_prep/.
# If multiple exports exist, list the intended file(s), relative to this directory:
# INPUT_FILES = ('data_prep/selected_examples_2026-....json',)
INPUT_FILES = ()
#@Rafaメモ: runner.pyのrun関数を呼び出して、実験を実行
if __name__ == '__main__':
    raise SystemExit(runner.run(EXPERIMENT, input_files=INPUT_FILES)) 
