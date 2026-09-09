import json
import socket
import zipfile

import numpy as np
import pytest
from scipy.io import savemat

from tools.inventory_candidates import (fingerprint, inspect_csv, inspect_mat,
    inspect_one, inspect_processed, inspect_text, inspect_zip, merge_dtype, scan, write_results)
from tools.inventory_llmx_data import array_spec, create_inventory, write_detailed_inventory


def test_csv_all_chunks_and_missing(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("ID,x,label\na,1,ok\nb,,bad\na,2.5,ok\n")
    r = inspect_csv(path, chunksize=1)
    assert r["shape"] == [3, 3]
    assert r["missing"]["x"] == 1
    assert r["unique"]["ID"] == 2
    assert r["schema"]["fields"][1]["dtype"] == "float64"


def test_csv_schema_group_ignores_length_but_not_dtype(tmp_path):
    a, b, c = [tmp_path / n for n in ("a.csv", "b.csv", "c.csv")]
    a.write_text("x,y\n1,2\n")
    b.write_text("x,y\n3,4\n5,6\n")
    c.write_text("x,y\nfoo,2\n")
    assert fingerprint(inspect_csv(a)["schema"]) == fingerprint(inspect_csv(b)["schema"])
    assert fingerprint(inspect_csv(a)["schema"]) != fingerprint(inspect_csv(c)["schema"])


def test_mat_numeric_and_struct(tmp_path):
    path = tmp_path / "test.mat"
    savemat(path, {"vals": np.arange(12, dtype=np.float64).reshape(4, 3), "config": {"rate": 10}})
    r = inspect_mat(path)
    assert r["variables"]["vals"]["shape"] == [4, 3]
    assert r["variables"]["vals"]["dtype"] == "float64"
    assert r["variables"]["vals"]["matlab_class"] == "double"
    assert "rate" in r["variables"]["config"]["struct_fields"]
    json.dumps(r)


def test_array_dtype_shape_serialization():
    a = array_spec(np.ones((2, 1, 3), dtype=np.float32))
    assert json.loads(json.dumps(a))["num_elements"] == 6
    assert a["ndim"] == 3 and not a["hasobject"]
    assert array_spec(np.array([object()]))["hasobject"]
    assert merge_dtype({"int64", "float64"}) == "float64"


def test_npz_never_loads_pickle(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, values=np.array([object()]))
    with pytest.raises(ValueError):
        inspect_one(path, "fixture")


def test_tracks_not_scenes_and_gaps(tmp_path):
    path = tmp_path / "scene.txt"
    path.write_text("0 10 1 2 3 4 5\n1 10 1 2 3 4 5\n3 10 1 2 3 4 5\n1 20 1 2 3 4 5\n")
    r = inspect_processed(path)
    assert r["track_rows"] == {"10": 3, "20": 1}
    assert r["max_simultaneous"] == 2
    assert r["contiguous_1frame_segments"] == 3
    assert r["frame_deltas"] == {"1.0": 1, "2.0": 1}


def test_zip_tree_and_script_blocks(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("nested/data.csv", "x\n1\n")
    assert inspect_zip(archive)["archive"]["uncompressed_bytes"] == 4
    script = tmp_path / "scripts-blue1.txt"
    script.write_text("Encounter 1\n[50.0] [2] default-fire\n\nEncounter 2\n[1.0] [3] evadeRWR180\n")
    r = inspect_text(script)
    assert r["encounter_count"] == 2 and r["rule_rows"] == 2
    assert r["unmatched_lines"] == []


def test_full_scan_and_source_preservation(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network prohibited")
    monkeypatch.setattr(socket, "create_connection", blocked)
    source = tmp_path / "source" / "example" / "repository"
    source.mkdir(parents=True)
    (source / "data.csv").write_text("x\n1\n")
    original = (source / "data.csv").read_bytes()
    result = scan(tmp_path / "source")
    assert len(result["files"]) == 1 and result["source_size_mtime_unchanged"]
    write_results(result, tmp_path / "output")
    assert (source / "data.csv").read_bytes() == original
    assert (tmp_path / "output/candidate_columns.csv").exists()


def test_llmx_detailed_outputs_and_T_normalization(tmp_path):
    path = tmp_path / "offline_data/physics_data/raw_transitions/CartPole-v1/episodes"
    path.mkdir(parents=True)
    for length in [2, 3]:
        np.savez(path / f"episode{length}.npz", states=np.zeros((length, 1, 4)), actions=np.zeros((length, 1), dtype=np.int64), rewards=np.ones((length, 1)), episodic_return=np.ones((1,)))
    inv = create_inventory(tmp_path)
    assert inv["tasks"][0]["schema_consistent_except_T"]
    write_detailed_inventory(inv, tmp_path / "output")
    assert "num_elements" in (tmp_path / "output/llmx_files.csv").read_text()


def test_empty_and_semicolon_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    assert inspect_csv(path)["rows"] == 0
    assert inspect_csv(path)["schema"]["kind"] == "empty_csv"
    path.write_text("date;flight;value\nday;1;4\n")
    r = inspect_one(path, "fixture")
    assert r["columns"] == 3 and r["rows"] == 1
    assert r["schema"]["delimiter"] == ";"


def test_cross_file_alignment_by_key():
    import pandas as pd
    from tools.audit_dataset_relationships import compare_by_key
    left = pd.DataFrame({"Case": [1, 2], "x": [5, 6]})
    right = pd.DataFrame({"Case": [2, 1], "x": [6, 5]})
    assert compare_by_key(left, right, "Case", ["x"])["matching_rows"] == 2
    right["Case"] = [1, 1]
    assert not compare_by_key(left, right, "Case", ["x"])["unique_keys"]


def test_raw_timestamp_chunk_boundaries(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("ID,Date,Time\n1,01/01/2020,00:00:00.000\n1,01/01/2020,00:00:01.000\n1,01/01/2020,00:00:01.000\n")
    r = inspect_csv(path, chunksize=1)
    assert r["raw_time_delta_ms_counts"] == {"0": 1, "1000": 1}
    assert r["raw_track_rows"] == {"1": 3}
