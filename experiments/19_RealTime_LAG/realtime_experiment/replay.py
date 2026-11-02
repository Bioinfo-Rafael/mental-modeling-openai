"""No API calls: convert measured latencies into causally ordered replay events."""
import math


def build_replay(manifest, records, catalog):
    config = manifest["config"]
    dt, deadline = config["dt_s"], config["deadline"]["limit_seconds"]
    labels = {int(k): {n["value"]: n for n in nodes} for k, nodes in catalog["levels"].items()}
    events = []
    for q in manifest["queries"]:
        r = records.get(q["id"])
        if not r or r.get("latency_s") is None:
            continue
        source_s = q["source_end_step"] * dt
        e = {"id": q["id"], "stage": q["stage"], "repeat": q["repeat"],
             "choice_count": q["choice_count"], "aircraft_id": q["aircraft_id"],
             "source_end_step": q["source_end_step"], "source_time_s": source_s,
             "latency_s": r["latency_s"], "api_status": r["api_status"], "choice": r.get("choice"),
             "label": None, "gate_latency_s": None}
        send_s = source_s
        if q["stage"] == "goal":
            gate = records.get(q["gate_query_id"], {})
            if gate.get("api_status") != "ok" or gate.get("choice") not in ("A1", "A2", "BOTH") or gate.get("latency_s", math.inf) > deadline:
                continue
            e["gate_latency_s"] = gate["latency_s"]
            send_s += gate["latency_s"] + r.get("dispatch_offset_s", 0)
        late = r["latency_s"] > deadline
        if late:
            e["choice"] = None  # the answer was not available when the deadline expired
        e.update(send_sim_s=send_s, ready_sim_s=send_s + r["latency_s"],
                 event_sim_s=send_s + min(r["latency_s"], deadline),
                 status="late" if late else r["api_status"], deadline_sim_s=send_s + deadline)
        if not late and r["api_status"] == "ok":
            if q["stage"] == "gate":
                e["status"] = "gate_none" if r["choice"] == "NONE" else "gate_trigger"
            else:
                e["label"] = labels[q["choice_count"]][r["choice"]]["label"]
                e["status"] = "candidate"
                e["broad_goal"] = r["choice"][0]
        e["display_step"] = math.ceil(e["event_sim_s"] / dt - 1e-10)
        events.append(e)
    events.sort(key=lambda e: (e["event_sim_s"], e["id"]))
    last = {}
    for e in events:
        if e["status"] != "candidate":
            continue
        key = (e["repeat"], e["choice_count"], e["aircraft_id"])
        previous = last.get(key)
        if previous and e["source_end_step"] <= previous["source_end_step"]:
            e["status"] = "stale"
        else:
            e["status"] = "same_goal" if previous and previous["choice"] == e["choice"] else "displayed"
            last[key] = e
    return {"schema_version": 1, "run_id": manifest["run_id"], "episode_id": config["episode_id"],
            "created_at": manifest["created_at"], "dt": dt, "deadline_s": deadline,
            "repeats": manifest["repeats"], "choice_counts": [4, 8, 32], "events": events,
            "complete": len(records) == len(manifest["queries"]),
            "recording_last_step": manifest["recording_last_step"],
            "last_step": max([manifest["recording_last_step"]] + [e["display_step"] for e in events]),
            "timing_note": "各APIの実測待ち時間を1倍速のシミュレーション時刻へ加算。継続並行送信時の混雑は再現していません。"}
