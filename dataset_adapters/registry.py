"""Raw dataset registry. Does not define model features or prompts."""
from .base import inventories, sources, RawAdapter
from .llmx import LLMXAdapter

EXTERNAL = ('f16capstone', 'trajair', 'aircombat_wez', 'calculated_moves', 'baidu_fighter_jet')

def dataset_ids():
    return sorted(t['task'] for t in inventories()[0]['tasks']) + list(EXTERNAL)

class NotAcquired(RawAdapter):
    dataset_id = 'baidu_fighter_jet'
    def sequences(self): return []
    def describe(self):
        return dict(dataset_id=self.dataset_id, display_name=self.dataset_id, category='candidate', source=sources()[self.dataset_id], status='NOT_ACQUIRED', format=[], number_of_files=0, number_of_sequences=None, number_of_records=None, sequential=None, multi_agent=None, state_available=None, action_available=None, reward_available=None, state_dimension=None, action_dimension=None, reward_dimension=None, mental_modeling_compatibility='unknown', schema_doc='data/candidate_datasets/baidu_fighter_jet/NOT_ACQUIRED.md')

def get_adapter(dataset_id):
    if dataset_id not in dataset_ids(): raise ValueError(f'unknown dataset: {dataset_id}')
    if dataset_id not in EXTERNAL: return LLMXAdapter(dataset_id)
    if dataset_id == 'baidu_fighter_jet': return NotAcquired()
    from importlib import import_module
    names = dict(f16capstone='F16CapstoneAdapter', trajair='TrajAirAdapter', aircombat_wez='AirCombatWEZAdapter', calculated_moves='CalculatedMovesAdapter')
    return getattr(import_module('dataset_adapters.' + dataset_id), names[dataset_id])()
