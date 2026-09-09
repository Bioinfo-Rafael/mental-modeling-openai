#!/usr/bin/env python3
"""Offline cross-file checks; no model, simulator, API, or tokenization."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.inventory_llmx_data import _atomic_text


def compare_by_key(left, right, key, columns):
    a, b = left.set_index(key), right.set_index(key)
    if not a.index.is_unique or not b.index.is_unique:
        return {"unique_keys": False, "matching_rows": None}
    common = a.index.intersection(b.index)
    equal = np.isclose(a.loc[common, columns].to_numpy(float), b.loc[common, columns].to_numpy(float), rtol=1e-9, atol=1e-10, equal_nan=True).all(axis=1)
    return {"unique_keys": True, "left_rows": len(a), "right_rows": len(b), "common_keys": len(common), "matching_rows": int(equal.sum()), "columns": columns}


def main():
    candidate_root = ROOT / "data/candidate_datasets"
    inv = json.loads((ROOT / "outputs/dataset_inventory/candidate_schema.json").read_text())
    runs = defaultdict(dict)
    for r in inv["files"]:
        if r["dataset"] == "calculated_moves" and Path(r["relative_path"]).parent.name.isdecimal():
            runs[str(Path(r["relative_path"]).parent)][Path(r["relative_path"]).name] = r
    cm = {"run_count": len(runs), "agent_row_minus_maxEncounters": Counter(), "winner_count_minus_maxEncounters": Counter(), "script_count_minus_maxEncounters": Counter(), "exceptions": []}
    for path, files in runs.items():
        maximum = int(files["SimulationConfiguration.txt"]["configuration"]["maxEncounters"])
        for name, r in files.items():
            if name in {"blue1.csv", "blue2.csv", "red.csv", "red1.csv", "red2.csv"}:
                diff = r["rows"] - maximum
                cm["agent_row_minus_maxEncounters"][str(diff)] += 1
                if diff != 1:
                    cm["exceptions"].append(path + "/" + name)
            elif name == "winHistory.csv":
                cm["winner_count_minus_maxEncounters"][str(r["logical_records"] - maximum)] += 1
            elif name.startswith("scripts-"):
                cm["script_count_minus_maxEncounters"][str(r["encounter_count"] - maximum)] += 1
    wezroot = candidate_root / "aircombat_wez/repository/Data"
    frames = {p.stem: pd.read_csv(p) for p in sorted(wezroot.glob("*.csv"))}
    inputs = ["BL_Speed", "RD_Speed", "rad", "BL_Hdg", "RD_Hdg", "BL_Alt", "RD_Alt"]
    factorial = frames["FactorialExperiment"]
    wez = {"factorial_rows": len(factorial), "factorial_input_levels": {k: sorted(factorial[k].unique().tolist()) for k in inputs},
           "factorial_unique_input_combinations": len(factorial[inputs].drop_duplicates()), "random_case_correspondence": {}}
    base = frames["RandomExperiment_1000_WEZ"]
    for key in ["RandomExperiment_1000_NEZ", "RandomExperiment_1000_RMAX"]:
        other = frames[key]
        wez["random_case_correspondence"][key] = compare_by_key(base, other, "Case", inputs)
        wez["random_case_correspondence"][key]["columns_in_source"] = list(other.columns)
        output = "RNez" if key.endswith("NEZ") else "RMax"
        target = "maxRange" if "maxRange" in other else output
        wez["random_case_correspondence"][key]["output_correspondence"] = compare_by_key(base, other.rename(columns={target: output}), "Case", [output])
    froot = candidate_root / "f16capstone/repository/Mathematical Model"
    prep = (froot / "System Identification/prep_flight_data.m").read_text()
    labels = re.findall(r'"([^"]+)"', prep.split("header = [", 1)[1].split("];", 1)[0])
    csvfiles = sorted((froot / "System Identification").glob("*.csv"))
    f16 = {"header55": labels, "csv_header_correspondence": {}}
    for p in csvfiles:
        fields = list(pd.read_csv(p, nrows=0).columns)
        f16["csv_header_correspondence"][p.name] = [v.replace("-", "_") for v in fields] == labels
    clip = loadmat(froot / "System Identification/Elev_Clipped_1.mat", variable_names=["vals"])["vals"]
    time = loadmat(froot / "Matlab Script/time.mat", variable_names=["time"])["time"]
    control = loadmat(froot / "Matlab Script/Control_Inputs.mat", variable_names=["control_IN"])["control_IN"]
    f16["time_mat_equals_clipped1_time"] = bool(np.array_equal(time[:, 0], clip[:, 37]))
    f16["control_IN_first_row"] = control[0].tolist()
    f16["clipped1_rc_0_to_3_first_row"] = clip[0, 43:47].tolist()
    f16["control_IN_columns_equal_raw_rc"] = {str(i): [j for j in range(12) if np.allclose(control[:, i], clip[:, 43+j], atol=1e-10, rtol=1e-9)] for i in range(4)}
    f16["clipped_csv_timestamp_matches"] = {}
    full = pd.read_csv(froot / "System Identification/25_05_2021__23_20_29.csv")
    for name in ["Elev_Clipped_1.mat", "Elev_Clipped_2.mat"]:
        values = loadmat(froot / "System Identification" / name, variable_names=["vals"])["vals"]
        times = full["m-currentPhysicsTime-SEC"].to_numpy()
        indexes = np.searchsorted(times, values[:, 37])
        indexes = np.clip(indexes, 0, len(times)-1)
        prev = np.clip(indexes-1, 0, len(times)-1)
        indexes = np.where(abs(times[indexes]-values[:, 37]) <= abs(times[prev]-values[:, 37]), indexes, prev)
        f16["clipped_csv_timestamp_matches"][name] = {"rows": len(values), "matching_timestamps_atol_1e-8": int(np.isclose(times[indexes], values[:, 37], atol=1e-8, rtol=0).sum()), "max_absolute_time_error": float(abs(times[indexes]-values[:, 37]).max())}
    result = {"calculated_moves": cm, "aircombat_wez": wez, "f16capstone": f16}
    _atomic_text(ROOT / "outputs/dataset_inventory/relationships.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
