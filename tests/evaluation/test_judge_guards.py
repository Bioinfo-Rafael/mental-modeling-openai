"""The paid gate and the no-resend guards of the judge: preview, root scan, marker, changed bytes, outcomes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from evaluation import cli
from evaluation.config import load_config
from evaluation.judge import MARKER, JudgeError, known_requests
from evaluation.material import load_run
from translate.backends import BackendError

from .fakes import ROOT, Factory, FakeSDK, asked_criteria, event, judge_config, write_run

PAID_SHA256 = "2486c8aba84bf343d5b8d1c2d3c12ee6c5c1e6331c08e304ebb458b52a79cfab"  # the 2026-11-05 assess request
LEGACY_JOURNAL = ROOT / "tests" / "fixtures" / "legacy-assess-journal.jsonl"


def _setup(tmp_path: Path, steps: tuple[int, ...] = (45, 145)) -> tuple[Path, Path]:
    run_dir = write_run(
        tmp_path / "runs" / "trial",
        [event(step, ("start",) if i == 0 else ("change",)) for i, step in enumerate(steps)],
    )
    return run_dir, judge_config(tmp_path)


def _judge(run_dir: Path, config: Path, factory: Factory, *flags: str) -> int:
    return cli.main(["judge", "--run", str(run_dir), "--config", str(config), *flags], client_factory=factory)


PAID = ("--execute", "--confirm-paid-api")


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []


def test_preview_makes_no_call_and_writes_no_journal(tmp_path: Path) -> None:
    run_dir, config = _setup(tmp_path)
    factory = Factory()

    code = _judge(run_dir, config, factory)

    judged = tmp_path / "evaluation" / "trial" / "judge"
    assert (code, factory.calls) == (0, 0)
    assert json.loads((judged / "run.json").read_text())["plan"]["to_send"] == 2
    assert not (judged / "requests.jsonl").exists() and not (judged / "responses.jsonl").exists()


@pytest.mark.parametrize("flags", [("--execute",), ("--confirm-paid-api",)])
def test_one_paid_flag_alone_is_a_usage_error(tmp_path: Path, flags: tuple[str, ...]) -> None:
    run_dir, config = _setup(tmp_path)

    with pytest.raises(SystemExit) as exc:
        _judge(run_dir, config, Factory(), *flags)

    assert exc.value.code == 2


def test_the_scan_recognises_the_paid_assess_journal_in_its_current_layout(tmp_path: Path) -> None:
    target = tmp_path / "evaluation" / "assess" / "intentcmp-seed100601-ja-A-s045-A0100" / "judge_requests.jsonl"
    target.parent.mkdir(parents=True)
    shutil.copy(LEGACY_JOURNAL, target)

    assert PAID_SHA256 in known_requests(tmp_path / "evaluation")


def test_the_scan_follows_a_symlinked_directory_once(tmp_path: Path) -> None:
    archive = tmp_path / "archive" / "assess"
    (archive / "old-run").mkdir(parents=True)
    shutil.copy(LEGACY_JOURNAL, archive / "old-run" / "judge_requests.jsonl")
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "assess").symlink_to(archive)
    (archive / "loop").symlink_to(tmp_path / "archive")  # a cycle is walked once

    assert PAID_SHA256 in known_requests(tmp_path / "evaluation")


def test_a_request_in_a_legacy_journal_is_never_resent(tmp_path: Path) -> None:
    run_dir, config_path = _setup(tmp_path, (45,))
    run = load_run(run_dir)
    planned = cli._calls(run, load_config(config_path), run.path)[0][0]
    legacy = json.loads(LEGACY_JOURNAL.read_text(encoding="utf-8"))
    journal = tmp_path / "evaluation" / "assess" / "old-run" / "judge_requests.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps({**legacy, "request_sha256": planned.sha256, "payload": planned.body}) + "\n")
    factory = Factory()

    code = _judge(run_dir, config_path, factory, *PAID)

    judged = tmp_path / "evaluation" / "trial" / "judge"
    status = json.loads((judged / "run.json").read_text())
    assert (code, factory.calls, factory.sdk.bodies) == (1, 0, [])  # never resent; results stay with the old journal
    assert status["plan"]["already_sent"] == 1 and status["status"] == "failed"


def test_a_paid_run_sends_each_request_once_and_a_rerun_sends_nothing(tmp_path: Path) -> None:
    run_dir, config = _setup(tmp_path)
    first, second = Factory(), Factory()

    codes = (_judge(run_dir, config, first, *PAID), _judge(run_dir, config, second, *PAID))

    judged = tmp_path / "evaluation" / "trial" / "judge"
    assert codes == (0, 0) and len(first.sdk.bodies) == 2 and second.calls == 0
    assert len(_lines(judged / "requests.jsonl")) == 2
    assert {r["status"] for r in _lines(judged / "responses.jsonl")} == {"ok"}
    assert len(_lines(judged / "results.jsonl")) == sum(len(asked_criteria(body)) for body in first.sdk.bodies)
    assert not (tmp_path / "evaluation" / MARKER).exists()


def test_the_marker_admits_one_paid_run_and_names_its_file(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    run_dir, config = _setup(tmp_path)
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / MARKER).write_text("{}")
    factory = Factory()

    code = _judge(run_dir, config, factory, *PAID)

    assert (code, factory.calls) == (1, 0)
    assert MARKER in capsys.readouterr().err
    assert not (tmp_path / "evaluation" / "trial" / "judge" / "requests.jsonl").exists()


def test_an_event_judged_here_with_other_bytes_is_refused(tmp_path: Path) -> None:
    run_dir, config = _setup(tmp_path, (45,))
    judged = tmp_path / "evaluation" / "trial" / "judge"
    judged.mkdir(parents=True)
    old = {"call_id": event(45)["event_id"], "request_sha256": "a" * 64, "body": {"model": "earlier prompt"}}
    (judged / "requests.jsonl").write_text(json.dumps(old) + "\n")
    factory = Factory()

    preview, paid = _judge(run_dir, config, factory), _judge(run_dir, config, factory, *PAID)

    assert json.loads((judged / "preview.json").read_text())["plan"]["refused"] == [old["call_id"]]
    assert (preview, paid, factory.calls) == (0, 1, 0)


def test_a_journal_line_without_a_request_hash_fails_closed(tmp_path: Path) -> None:
    run_dir, config = _setup(tmp_path)
    broken = tmp_path / "evaluation" / "other" / "requests.jsonl"
    broken.parent.mkdir(parents=True)
    broken.write_text('{"call_id": "x"}\n')

    with pytest.raises(JudgeError, match="requests.jsonl"):
        known_requests(tmp_path / "evaluation")
    assert _judge(run_dir, config, Factory(), *PAID) == 1


@pytest.mark.parametrize(
    ("outcome", "code", "status"),
    [
        (BackendError("timed out", uncertain=True), 3, "uncertain"),
        (BackendError("HTTP 400", uncertain=False), 1, "error"),
        (KeyboardInterrupt(), 3, "uncertain"),
        ('{"criteria": []}', 1, "ok"),
    ],
)
def test_a_failed_call_stops_the_run_journaled_and_never_retried(
    tmp_path: Path, outcome: object, code: int, status: str
) -> None:
    run_dir, config = _setup(tmp_path)
    factory = Factory(FakeSDK(outcomes=[outcome]))

    result = _judge(run_dir, config, factory, *PAID)

    judged = tmp_path / "evaluation" / "trial" / "judge"
    assert (result, len(factory.sdk.bodies)) == (code, 1)
    assert len(_lines(judged / "requests.jsonl")) == 1 and _lines(judged / "responses.jsonl")[0]["status"] == status
    assert not (judged / "results.jsonl").exists()


def test_the_call_cap_stops_before_call_cap_plus_one(tmp_path: Path) -> None:
    run_dir, _ = _setup(tmp_path, (45, 145, 245))
    config = judge_config(tmp_path, limits={"max_calls": 2})
    factory = Factory()

    code = _judge(run_dir, config, factory, *PAID)

    assert (code, len(factory.sdk.bodies)) == (4, 2)


def test_a_request_journaled_under_the_guard_root_is_not_resent_when_output_points_elsewhere(tmp_path: Path) -> None:
    run_dir, _ = _setup(tmp_path, (45,))
    first, second = Factory(), Factory()
    _judge(run_dir, judge_config(tmp_path), first, *PAID)

    code = _judge(run_dir, judge_config(tmp_path / "elsewhere", output=str(tmp_path / "other-root")), second, *PAID)

    assert (code, len(first.sdk.bodies), second.calls) == (1, 1, 0)  # never resent; its results live elsewhere


def test_a_run_that_repeats_an_event_is_refused(tmp_path: Path) -> None:
    record = event(45)
    run_dir = write_run(tmp_path / "runs" / "trial", [record, record])
    factory = Factory()

    assert _judge(run_dir, judge_config(tmp_path), factory, *PAID) == 1 and factory.calls == 0


def test_a_sent_event_without_results_is_never_resent_and_keeps_the_run_failed(tmp_path: Path) -> None:
    run_dir, config = _setup(tmp_path)
    first, second = Factory(FakeSDK(outcomes=['{"criteria": []}'])), Factory()

    codes = (_judge(run_dir, config, first, *PAID), _judge(run_dir, config, second, *PAID))

    assert codes == (1, 1) and len(first.sdk.bodies) == 1 and len(second.sdk.bodies) == 1
    status = json.loads((tmp_path / "evaluation" / "trial" / "judge" / "run.json").read_text())
    assert status["status"] == "failed" and status["error"]["first"] == event(45)["event_id"]


def test_a_paid_run_from_a_git_worktree_is_refused_naming_the_main_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    run_dir, config = _setup(tmp_path)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: /srv/main/.git/worktrees/worktree\n")
    monkeypatch.setattr(cli, "CHECKOUT", worktree)
    factory = Factory()

    paid, preview = _judge(run_dir, config, factory, *PAID), _judge(run_dir, config, factory)

    assert (paid, preview, factory.calls) == (1, 0, 0)  # a preview sends nothing and stays allowed
    assert "Run it from /srv/main" in capsys.readouterr().err
    assert not (tmp_path / "evaluation" / MARKER).exists()


def test_a_judgement_directory_refuses_a_run_that_changed(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    run_dir, config = _setup(tmp_path)
    first = _judge(run_dir, config, Factory())
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**event(245, ("change",)), "sequence": 3}) + "\n")

    assert (first, _judge(run_dir, config, Factory())) == (0, 1)
    assert "judge it under another --name" in capsys.readouterr().err


# --- --only-event: the one explicit re-send of a stuck event ---------------------------------------


def _stuck(tmp_path: Path) -> tuple[Path, Path, str, dict]:
    """A paid run whose first event (step 45) got a malformed verdict: it is stuck; step 145 was never sent."""
    run_dir, config = _setup(tmp_path)
    first = Factory(FakeSDK(outcomes=['{"criteria": []}']))
    assert _judge(run_dir, config, first, *PAID) == 1
    return run_dir, config, event(45)["event_id"], first.sdk.bodies[0]


@pytest.mark.parametrize(
    ("flags", "event_id"),
    [((), "e"), (("--execute",), "e"), (("--confirm-paid-api",), "e"), (PAID, ""), (PAID, " ")],
)
def test_only_event_without_both_paid_flags_or_an_id_is_a_usage_error(
    tmp_path: Path, flags: tuple[str, ...], event_id: str
) -> None:
    run_dir, config = _setup(tmp_path)
    factory = Factory()

    with pytest.raises(SystemExit) as exc:
        _judge(run_dir, config, factory, *flags, "--only-event", event_id)

    assert (exc.value.code, factory.calls) == (2, 0)


def test_only_event_refuses_an_event_that_is_not_stuck(tmp_path: Path) -> None:
    run_dir, config, stuck, body = _stuck(tmp_path)
    factory = Factory()
    never_sent = event(145, ("change",))["event_id"]
    codes = [_judge(run_dir, config, factory, *PAID, "--only-event", e) for e in (never_sent, "no-such-event")]
    journal = tmp_path / "evaluation" / "trial" / "judge" / "requests.jsonl"
    original = journal.read_text(encoding="utf-8")
    journal.write_text(original + json.dumps({**_lines(journal)[0], "call_id": "another-event"}) + "\n")
    codes.append(_judge(run_dir, config, factory, *PAID, "--only-event", stuck))  # same bytes, another id here
    journal.write_text(original, encoding="utf-8")
    elsewhere = tmp_path / "evaluation" / "assess" / "old-run" / "judge_requests.jsonl"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text(json.dumps({"call_id": "old", "payload": body}) + "\n")  # the same bytes, sent elsewhere

    codes.append(_judge(run_dir, config, factory, *PAID, "--only-event", stuck))

    assert (codes, factory.calls) == ([1, 1, 1, 1], 0)


def test_only_event_resends_exactly_one_body_journaled_as_user_requested(tmp_path: Path) -> None:
    run_dir, config, stuck, body = _stuck(tmp_path)
    factory = Factory()

    code = _judge(run_dir, config, factory, *PAID, "--only-event", stuck)

    judged = tmp_path / "evaluation" / "trial" / "judge"
    lines = _lines(judged / "requests.jsonl")
    assert (code, factory.sdk.bodies) == (0, [body])
    assert [line["call_id"] for line in lines] == [stuck, stuck] and "resend" not in lines[0]
    assert lines[1]["resend"] == {"requested_by": "user", "option": "--only-event", "previous_sends": 1}
    assert {row["event_id"] for row in _lines(judged / "results.jsonl")} == {stuck}
    assert json.loads((judged / "run.json").read_text())["only_event"] == stuck


def test_after_a_resend_the_guard_still_blocks_every_known_request(tmp_path: Path) -> None:
    run_dir, config, stuck, body = _stuck(tmp_path)
    _judge(run_dir, config, Factory(), *PAID, "--only-event", stuck)
    rerun, again = Factory(), Factory()

    codes = (_judge(run_dir, config, rerun, *PAID), _judge(run_dir, config, again, *PAID, "--only-event", stuck))

    assert codes == (0, 1) and again.calls == 0  # judged now, so no longer stuck
    assert len(rerun.sdk.bodies) == 1 and rerun.sdk.bodies[0] != body  # only the never-sent event 145
