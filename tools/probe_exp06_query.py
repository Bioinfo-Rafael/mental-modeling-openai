"""Probe the saved failing second Exp.06 request exactly once; never resume a batch."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import common
from openai import OpenAI, APIStatusError

SOURCE = ROOT / "experiments/06_new_models_n10/results/batches/612bf8080963456b9286c5481885a700"
QUERY_ID = "ffc4c5f124cadce47e6d7d151dd1059899efdcd39a5e317924e53b7b4f7429d2"
OUTPUT = ROOT / "outputs/exp06_single_query_probe"


def load_request():
    requests = common.read_jsonl(SOURCE / "requests.jsonl")
    selected = [r for r in requests if r["query_id"] == QUERY_ID and r["api_request_made"]]
    if len(selected) != 1:
        raise ValueError("Expected exactly one saved attempt for the specified second query")
    queries = common.read_json(SOURCE / "manifest.json")["queries"]
    q = next(q for q in queries if q["query_id"] == QUERY_ID)
    kwargs = selected[0]["kwargs"]
    common.verify_prompt(q, kwargs["messages"][0]["content"], kwargs["messages"][1]["content"])
    if (q["ordinal"] != 1 or q["query_index"] != 6
            or kwargs != common.expected_api_request("gpt-5.6-sol", q["system_prompt"], q["user_prompt"])):
        raise ValueError("Saved request is not the intended second query/settings")
    return kwargs


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(common.redact(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("sol", "terra", "luna"), default="sol",
                        help="Change only the model; reuse the exact saved sol messages/options")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid-api", action="store_true")
    args = parser.parse_args(argv)
    if args.execute != args.confirm_paid_api:
        parser.error("Both --execute and --confirm-paid-api are required")
    original = load_request()
    kwargs = copy.deepcopy(original)
    kwargs["model"] = common.MODELS[args.model]
    metadata = {"source_query_id": QUERY_ID, "model": kwargs["model"], "model_alias": args.model,
                "source_model": original["model"], "model_only_override": kwargs["model"] != original["model"]}
    print(f"Source query: {QUERY_ID}\n{args.model} / MountainCar / next-action / H5 / ordinal=1 / index=6")
    if not args.execute:
        print("DRY RUN: saved request validated; only model selected. Maximum API attempts: 1; retries: 0. API calls: 0.")
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set in your terminal")
    destination = OUTPUT / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + args.model + "_" + uuid.uuid4().hex[:8])
    if destination.resolve() != destination:
        raise ValueError("Symlinked output directory is not allowed")
    destination.mkdir(parents=True, exist_ok=False)
    save(destination / "request.json", {"query_id": QUERY_ID, "source": str(SOURCE), "kwargs": kwargs,
         **metadata,
         "source_requests_sha256": hashlib.sha256((SOURCE / "requests.jsonl").read_bytes()).hexdigest(),
         "sdk_max_retries": 0, "included_in_experiment": False})
    print(f"Probe output: {destination}", flush=True)
    # No upstream retry loop and SDK retry disabled. The generation kwargs are untouched.
    with OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1",
                max_retries=0, timeout=60.0) as client:
        result = {"query_id": QUERY_ID, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                  **metadata,
                  "included_in_experiment": False, "sdk_create_calls": 1}
        started = time.perf_counter()
        try:
            with common.quiet_sdk_logging():
                response = client.chat.completions.create(**kwargs)
            result.update(status="success", request_id=getattr(response, "_request_id", None),
                          raw_response=response.model_dump(mode="json"))
        except APIStatusError as exc:
            result.update(status="api_error", status_code=exc.status_code, request_id=exc.request_id,
                          error_type=type(exc).__name__, error_body=exc.body, error_message=str(exc))
        except (Exception, KeyboardInterrupt) as exc:
            result.update(status="unknown_outcome", error_type=type(exc).__name__, error_message=str(exc))
        finally:
            result.update(elapsed_seconds=time.perf_counter() - started,
                          finished_at_utc=datetime.now(timezone.utc).isoformat())
            save(destination / "result.json", result)
    print(f"Status: {result['status']}; request_id: {result.get('request_id')}; seconds: {result['elapsed_seconds']:.3f}")
    print(f"Saved: {destination / 'result.json'}")
    return 0 if result["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
