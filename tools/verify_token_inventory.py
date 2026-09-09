#!/usr/bin/env python3
"""Read-only verification of inventory totals, identifiers, and config digests."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset_adapters.base import ROOT

def main():
    out=ROOT/'outputs/token_counts'
    groups=defaultdict(lambda:[0,0]);seen=set();configs={}
    with (out/'queries.csv').open() as f:
        for r in csv.DictReader(f):
            key=(r['dataset_id'],r['prompt_mode'],r['history_size'],r['preprocessing_config_sha256'])
            identity=key+(r['source_file'],r['sequence_id'],r['query_index'])
            assert identity not in seen,identity
            seen.add(identity)
            text=r['preprocessing_config'];digest=r['preprocessing_config_sha256']
            if digest not in configs:
                assert hashlib.sha256(text.encode()).hexdigest()==digest
                assert (out/'preprocessing_configs'/f'{digest}.json').read_text().strip()==text
                configs[digest]=text
            assert configs[digest]==text
            assert int(r['total_content_tokens'])==int(r['system_tokens'])+int(r['user_tokens'])
            assert int(r['estimated_api_input_tokens'])==int(r['total_content_tokens'])+9
            assert (ROOT/r['source_file']).is_file()
            groups[key][0]+=1;groups[key][1]+=int(r['estimated_api_input_tokens'])
    with (out/'token_inventory.csv').open() as f:
        inventory=list(csv.DictReader(f))
    assert len(inventory)==len(groups)
    for r in inventory:
        key=(r['dataset_id'],r['prompt_mode'],r['history_size'],r['preprocessing_config_sha256'])
        assert groups[key]==[int(r['n_units']),int(r['total_input_tokens'])]
    summary=json.loads((out/'summary.json').read_text())
    assert len(seen)==summary['number_of_queries']
    print(f'PASS: {len(seen)} unique query/H/config rows; {len(groups)} scope totals; {len(configs)} config hashes; API calls=0')
if __name__=='__main__': main()
