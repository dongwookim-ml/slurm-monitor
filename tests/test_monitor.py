import ast
import io
import json
import os
import threading
import urllib.error
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from rich.console import Console

import slurm_monitor as sm

WEBHOOK = "https://hooks.slack.com/services/TEST/TEST/REDACTED"


def render(value, width=100, height=30):
    stream = io.StringIO()
    Console(file=stream, width=width, height=height, color_system=None).print(value)
    return stream.getvalue()


@pytest.mark.parametrize(
    "text,total,types",
    [
        ("gpu:4", 4, {}),
        ("gpu:A100:4", 4, {"a100": 4}),
        ("gpu:a100:4,gpu:v100:2", 6, {"a100": 4, "v100": 2}),
        ("gres/gpu=8,gres/gpu:a100=4,gres/gpu:v100=4", 8, {"a100": 4, "v100": 4}),
        ("gpu:a100:2(IDX:0,1)", 2, {"a100": 2}),
        ("cpu=8,mem=32G", 0, {}),
        ("N/A", None, {}),
        ("gpu:a100:bad", None, {}),
    ],
)
def test_gpu_parsing(text, total, types):
    assert asdict(sm.parse_gpus(text)) == {"total": total, "types": types}


def test_json_versions_and_multinode_actual_allocations(snapshot):
    running, pending, _ = snapshot.jobs
    assert running.gpus == running.requested_gpus == 8
    assert running.gpu_per_node == {"a100": 4, "": 4}
    assert pending.gpus is None and pending.requested_gpus == 2
    assert pending.nodes == 1 and pending.cpus == 4 and pending.memory_per_node == 8192
    assert running.name == "train|phase2[/red]"


@pytest.mark.parametrize(
    "body", ["", "{}", '{"jobs":null}', '{"jobs":[{}]}', '{"jobs":[],"errors":[{}]}']
)
def test_malformed_json_is_failure_not_empty(body):
    with pytest.raises(sm.QueryError):
        sm.parse_queue_json(body)


def test_successful_empty_queue():
    assert sm.parse_queue_json('{"jobs":[],"errors":[]}') == []


def test_text_unit_separator_keeps_pipe_names():
    values = [
        "101",
        "train|phase[/red]",
        "alice",
        "gpu",
        "RUNNING",
        "0:05",
        "1:00:00",
        "2",
        "",
        "cpu=8,node=2,gres/gpu=8",
        "gres/gpu:4",
        "8",
        "1G",
        "",
        "normal",
        "research",
        "",
        "1000",
        "",
    ]
    job = sm.parse_queue_text(sm.SEP.join(values))[0]
    assert job.name == "train|phase[/red]" and job.gpus == 8
    assert job.memory_per_node is None  # Missing CPU/node scope stays unknown.
    with pytest.raises(sm.QueryError):
        sm.parse_queue_text(sm.SEP.join(values).replace("train|phase", "train" + sm.SEP + "phase"))


def test_shared_nodes_unique_header_and_partition_usage(snapshot):
    assert sm.known_sum([node.gpus for node in snapshot.nodes]) == 34
    info = {row["name"]: row for row in sm.gpu_summary(snapshot, "alice")}
    assert info["gpu"] == dict(name="gpu", total=34, used=10, free=8, mine=8)
    assert (
        info["shared"]["total"] == 10
        and info["shared"]["used"] == 6
        and info["shared"]["free"] == 4
    )
    assert info["gpu"]["mine"] == snapshot.jobs[0].gpus
    assert (
        sm.parse_nodes("NodeName=n State=MIXED Gres=gpu:8 CPUAlloc=60 CPUTot=64")[0].free_gpus
        is None
    )


@pytest.mark.parametrize(
    "state", ["DOWN", "IDLE+DRAIN", "MIXED+DRAINING", "IDLE*", "IDLE~", "MAINT", "POWERED_DOWN"]
)
def test_unusable_nodes_offer_no_gpus(state):
    node = sm.parse_nodes(f"NodeName=n State={state} Gres=gpu:8 GresUsed=gpu:0")[0]
    assert node.free_gpus == 0


def test_cpu_allocation_does_not_estimate_gpus():
    node = sm.parse_nodes("NodeName=n State=MIXED Gres=gpu:8 GresUsed=gpu:1 CPUAlloc=60 CPUTot=64")[
        0
    ]
    assert node.free_gpus == 7
    node = sm.parse_nodes(
        "NodeName=n State=IDLE Gres=gpu:8 AllocTRES=cpu=2,gres/gpu=2 GresDrain=gpu:1"
    )[0]
    assert node.free_gpus == 5


def test_generic_usage_on_mixed_gpu_types_is_unknown():
    node = sm.parse_nodes("NodeName=n State=MIXED Gres=gpu:a100:4,gpu:v100:4 AllocTRES=gres/gpu=2")[
        0
    ]
    assert node.free_gpus == 6 and node.free_type("a100") is None


def test_duplicate_and_malformed_nodes(fixture_text):
    line = fixture_text("nodes.txt").splitlines()[0]
    assert len(sm.parse_nodes(line + "\n" + line)) == 1
    with pytest.raises(sm.QueryError):
        sm.parse_nodes(line + "\n" + line.replace("CPUAlloc=8", "CPUAlloc=9"))
    with pytest.raises(sm.QueryError):
        sm.parse_nodes("Could not load nodes")


def scripted_runner(fixture_text, queue=None):
    commands = []

    def run(argv):
        commands.append(argv)
        if argv[0] == "squeue":
            return queue() if queue else fixture_text("queue.json")
        if argv[0] == "sacct":
            return fixture_text("accounting.txt")
        return fixture_text("nodes.txt" if "nodes" in argv else "partitions.txt")

    return run, commands


def test_collector_two_queries_after_policy_cache(fixture_text):
    run, commands = scripted_runner(fixture_text)
    collector = sm.Collector(run, clock=lambda: 3000)
    collector.collect()
    assert len(commands) == 3
    collector.collect()
    assert len(commands) == 5
    assert sum(command[0] == "squeue" for command in commands) == 2
    assert all(isinstance(command, list) for command in commands)


def test_collector_failure_preserves_last_success(fixture_text):
    fail = [False]

    def queue():
        if fail[0]:
            raise sm.QueryError("squeue: query failed")
        return fixture_text("queue.json")

    run, _ = scripted_runner(fixture_text, queue)
    collector = sm.Collector(run, clock=lambda: 3000)
    first = collector.collect()
    fail[0] = True
    failed = collector.collect({"101"})
    assert not failed.jobs_ok and failed.jobs == first.jobs and failed.jobs_at == 3000
    assert failed.accounting == []


def test_json_fallback_is_capability_only_and_cached(fixture_text):
    commands = []

    def run(argv):
        commands.append(argv)
        if "--json" in argv:
            raise sm.QueryError("squeue: unsupported format (exit 1)")
        return ""

    collector = sm.Collector(run)
    assert collector.queue() == collector.queue() == []
    assert len(commands) == 3 and "--json" not in commands[-1]
    assert sm.SEP in commands[-1][-1]
    collector = sm.Collector(
        lambda argv: (_ for _ in ()).throw(sm.QueryError("squeue: query failed"))
    )
    with pytest.raises(sm.QueryError):
        collector.queue()
    assert collector.json_supported


def test_accounting_is_batched_and_only_for_missing(fixture_text):
    run, commands = scripted_runner(fixture_text)
    collector = sm.Collector(run)
    collector.collect({"101", "104", "105"})
    accounting = [cmd for cmd in commands if cmd[0] == "sacct"]
    assert len(accounting) == 1 and accounting[0][accounting[0].index("--jobs") + 1] == "104,105"
    assert "--allocations" in accounting[0]


@pytest.mark.parametrize(
    "exception", [sm.subprocess.TimeoutExpired("squeue", 15), FileNotFoundError(), OSError()]
)
def test_runner_failures_are_explicit(monkeypatch, exception):
    monkeypatch.setattr(sm.subprocess, "run", Mock(side_effect=exception))
    with pytest.raises(sm.QueryError):
        sm.run_command(["squeue", "--json"])


def test_runner_rejects_shell_and_hides_sensitive_stderr(monkeypatch):
    with pytest.raises(TypeError):
        sm.run_command("squeue; echo bad")
    result = sm.subprocess.CompletedProcess([], 1, stdout="", stderr=WEBHOOK)
    run = Mock(return_value=result)
    monkeypatch.setattr(sm.subprocess, "run", run)
    with pytest.raises(sm.QueryError) as error:
        sm.run_command(["squeue", "--user", "alice;true"])
    assert "shell" not in run.call_args.kwargs
    assert WEBHOOK not in str(error.value)
    assert run.call_args.args[0][-1] == "alice;true"


def test_failure_and_recovery_cannot_emit_false_endings(snapshot):
    tracker = sm.JobTracker(notify=True)
    tracker.observe(snapshot, snapshot.jobs)
    tracker.observe(replace(snapshot, jobs_ok=False, jobs=[]), [])
    tracker.observe(snapshot, snapshot.jobs)
    assert tracker.outbox == [] and tracker.awaiting == {}


def test_completing_accounting_delay_and_timeout(snapshot, fixture_text):
    tracker = sm.JobTracker(notify=True)
    running = snapshot.jobs[0]
    tracker.observe(snapshot, [running])
    tracker.observe(snapshot, [replace(running, state="COMPLETING")])
    tracker.observe(snapshot, [])
    assert set(tracker.awaiting) == {"101"} and tracker.outbox == []
    ended = replace(snapshot, accounting=sm.parse_accounting(fixture_text("accounting.txt")))
    tracker.observe(ended, [])
    assert tracker.history[0].state == "TIMEOUT" and tracker.history[0].exit_code == "0:15"
    assert tracker.history[0].name == running.name
    assert tracker.outbox[0]["kind"] == "timeout"
    tracker.observe(ended, [])
    assert len(tracker.outbox) == 1


def test_pending_configuring_started_once_and_requeue(snapshot):
    tracker = sm.JobTracker(notify=True)
    pending = snapshot.jobs[1]
    tracker.observe(snapshot, [pending])
    tracker.observe(snapshot, [replace(pending, state="CONFIGURING")])
    tracker.observe(snapshot, [replace(pending, state="RUNNING", gpus=2)])
    tracker.observe(snapshot, [replace(pending, state="RUNNING", gpus=2)])
    assert len(tracker.outbox) == 1
    tracker.observe(snapshot, [replace(pending, state="PENDING", restart_count=1)])
    tracker.observe(snapshot, [replace(pending, state="RUNNING", restart_count=1, gpus=2)])
    assert len(tracker.outbox) == 2


def test_accounting_exit_and_derived_step_failure(fixture_text, snapshot):
    records = sm.parse_accounting(fixture_text("accounting.txt"))
    assert records[1].state == "CANCELLED"
    assert records[2].state == "COMPLETED" and records[2].derived_exit_code == "1:0"
    output = render(sm.diagnostic_panel(records[2], snapshot, None), width=160)
    assert "batch exit: 0:0" in output and "derived step exit: 1:0" in output
    with pytest.raises(sm.QueryError):
        sm.parse_accounting("104.batch|COMPLETED|0:0|0:0|1:00|cpu=1|gpu|alice")


def test_atomic_private_journal_restart_and_ack(tmp_path, snapshot):
    path = tmp_path / "state.json"
    tracker = sm.JobTracker(path, notify=True)
    tracker.observe(snapshot, [snapshot.jobs[1]])
    tracker.observe(snapshot, [replace(snapshot.jobs[1], state="RUNNING", gpus=2)])
    assert len(tracker.outbox) == 1 and os.stat(path).st_mode & 0o777 == 0o600
    assert WEBHOOK not in path.read_text()
    tracker.close()
    restarted = sm.JobTracker(path, notify=True)
    restarted.observe(snapshot, [replace(snapshot.jobs[1], state="RUNNING", gpus=2)])
    sender = Mock(return_value=sm.SendResult(True))
    assert restarted.deliver(sender) and sender.call_count == 1 and restarted.outbox == []
    restarted.close()
    assert json.loads(path.read_text())["outbox"] == []


def test_journal_lock_corruption_and_symlink(tmp_path):
    path = tmp_path / "state.json"
    tracker = sm.JobTracker(path, True)
    with pytest.raises(sm.StateError, match="locked"):
        sm.JobTracker(path, True)
    tracker.close()
    path.write_text("broken")
    with pytest.raises(sm.StateError):
        sm.JobTracker(path, True)
    assert path.read_text() == "broken"
    path.unlink()
    target = tmp_path / "target"
    target.write_text("preserve")
    path.symlink_to(target)
    with pytest.raises(sm.StateError, match="symlink"):
        sm.JobTracker(path, True)
    assert target.read_text() == "preserve"


def test_write_failure_pauses_delivery_then_recovers(tmp_path, monkeypatch, snapshot):
    tracker = sm.JobTracker(tmp_path / "state.json", True)
    tracker.observe(snapshot, [snapshot.jobs[1]])
    replace_file = sm.os.replace
    monkeypatch.setattr(sm.os, "replace", Mock(side_effect=OSError("storage unavailable")))
    tracker.observe(snapshot, [replace(snapshot.jobs[1], state="RUNNING", gpus=2)])
    sender = Mock(return_value=sm.SendResult(True))
    assert not tracker.deliver(sender) and not sender.called and len(tracker.outbox) == 1
    monkeypatch.setattr(sm.os, "replace", replace_file)
    assert tracker.deliver(sender) and tracker.outbox == []
    tracker.close()


def test_retry_after_permanent_failure_and_restart(tmp_path, snapshot):
    now = [3000.0]
    tracker = sm.JobTracker(tmp_path / "state.json", True, clock=lambda: now[0])
    tracker.observe(snapshot, [snapshot.jobs[1]])
    tracker.observe(snapshot, [replace(snapshot.jobs[1], state="RUNNING", gpus=2)])
    assert tracker.deliver(lambda _: sm.SendResult(False, True, 30, "HTTP 429"))
    assert tracker.outbox[0]["next_attempt"] == 3030
    assert not tracker.deliver(lambda _: pytest.fail("retry happened too soon"))
    now[0] = 3030
    assert tracker.deliver(lambda _: sm.SendResult(False, False, error="HTTP 403"))
    assert tracker.outbox[0]["next_attempt"] is None
    tracker.close()
    tracker = sm.JobTracker(tmp_path / "state.json", True, clock=lambda: now[0])
    assert tracker.deliver(lambda _: sm.SendResult(True))
    tracker.close()


def test_slack_batch_limit_and_safe_mentions(snapshot):
    tracker = sm.JobTracker(notify=True)
    tracker.initialized = True
    jobs = [replace(snapshot.jobs[0], id=str(i), name="<!channel> " + "x" * 240) for i in range(30)]
    tracker.observe(snapshot, jobs)
    messages = []
    while tracker.outbox:
        tracker.deliver(lambda text: messages.append(text) or sm.SendResult(True))
    assert len(messages) > 1 and all(len(message) <= 3500 for message in messages)
    assert all("<!channel>" not in message for message in messages)
    assert "&lt;!channel&gt;" in messages[0]


@pytest.mark.parametrize(
    "exception,retry",
    [
        (TimeoutError(), True),
        (urllib.error.URLError("transport"), True),
        (urllib.error.HTTPError(WEBHOOK, 429, "limit", {"Retry-After": "30"}, None), True),
        (urllib.error.HTTPError(WEBHOOK, 500, "server", {}, None), True),
        (urllib.error.HTTPError(WEBHOOK, 403, "forbidden", {}, None), False),
    ],
    ids=["timeout", "transport", "rate-limit", "server", "forbidden"],
)
def test_slack_transport_failures_are_retained_and_redacted(monkeypatch, exception, retry):
    monkeypatch.setattr(sm.urllib.request, "urlopen", Mock(side_effect=exception))
    result = sm.send_slack_notification(WEBHOOK, "test")
    assert not result.success and result.retry == retry and WEBHOOK not in result.error
    if isinstance(exception, urllib.error.HTTPError) and exception.code == 429:
        assert result.delay == 30


def test_slack_success_disables_markup(monkeypatch):
    response = Mock(status=200)
    response.read.return_value = b"ok"
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    post = Mock(return_value=response)
    monkeypatch.setattr(sm.urllib.request, "urlopen", post)
    assert sm.send_slack_notification(WEBHOOK, "text").success
    payload = json.loads(post.call_args.args[0].data)
    assert payload == {"text": "text", "mrkdwn": False}


@pytest.mark.parametrize(
    "url",
    [
        "http://hooks.slack.com/services/TEST",
        "https://evil.invalid/services/TEST",
        "https://hooks.slack.com@evil.invalid/services/TEST",
        '"' + WEBHOOK + '"',
        WEBHOOK + "?secret=x",
    ],
)
def test_webhook_validation_never_echoes_secrets(url):
    with pytest.raises(ValueError) as error:
        sm.validate_webhook(url)
    assert url not in str(error.value)


def test_dotenv_quotes_precedence_and_unrelated_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sm.Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(sm.Path, "home", lambda: tmp_path / "home")
    (tmp_path / "home").mkdir()
    (tmp_path / ".env").write_text("UNRELATED=1\n")
    (tmp_path / "home" / ".slurm-monitor.env").write_text(
        f'export SLACK_WEBHOOK_URL="{WEBHOOK}" # comment\n'
    )
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert sm.load_webhook() == WEBHOOK
    monkeypatch.setenv("SLACK_WEBHOOK_URL", WEBHOOK + "/environment")
    assert sm.load_webhook() == WEBHOOK + "/environment"
    assert sm.load_webhook(WEBHOOK + "/cli") == WEBHOOK + "/cli"


def test_worker_does_not_block_observation_and_interrupted_delivery_restarts(tmp_path, snapshot):
    tracker = sm.JobTracker(tmp_path / "state.json", True)
    tracker.observe(snapshot, [snapshot.jobs[1]])
    tracker.observe(snapshot, [replace(snapshot.jobs[1], state="RUNNING", gpus=2)])
    entered, release = threading.Event(), threading.Event()

    def send(_):
        entered.set()
        assert release.wait(2)
        return sm.SendResult(False, error="interrupted transport")

    worker = sm.DeliveryWorker(tracker, send)
    worker.start()
    assert entered.wait(2)
    tracker.observe(snapshot, [replace(snapshot.jobs[1], state="COMPLETING", gpus=2)])
    assert tracker.previous["102"].state == "COMPLETING"
    release.set()
    assert worker.close()
    tracker.close()
    restarted = sm.JobTracker(tmp_path / "state.json", True)
    assert len(restarted.outbox) == 1 and restarted.outbox[0]["attempts"] == 1
    restarted.close()


def test_suitability_is_not_a_scheduler_guarantee(snapshot):
    job = snapshot.jobs[1]
    status, reasons = sm.resource_suitability(job, snapshot, "gpu")
    assert status == "Potential resource fit; scheduling unknown"
    assert any("reservations" in reason for reason in reasons)
    status, reasons = sm.resource_suitability(replace(job, requested_gpus=100), snapshot, "gpu")
    assert status.startswith("Insufficient") and "insufficient visible free GPUs" in reasons
    status, reasons = sm.resource_suitability(
        replace(job, gpu_per_node={"a100": 8}), snapshot, "gpu"
    )
    assert status.startswith("Insufficient") and any("per-node GPU" in reason for reason in reasons)
    assert sm.resource_suitability(job, replace(snapshot, nodes_ok=False), "gpu")[0].startswith(
        "Unknown"
    )
    assert sm.resource_suitability(replace(job, memory_per_node=None), snapshot, "gpu")[
        0
    ].startswith("Unknown")


def test_policy_and_unknown_constraints(snapshot):
    job = replace(snapshot.jobs[1], nodes=10, time_limit="10:00:00", account="other", qos="other")
    status, reasons = sm.resource_suitability(job, snapshot, "gpu")
    assert status.startswith("Insufficient")
    assert any("MaxNodes" in reason for reason in reasons) and any(
        "MaxTime" in reason for reason in reasons
    )
    assert any("account" in reason for reason in reasons) and any(
        "QoS" in reason for reason in reasons
    )
    assert "priority" in sm.pending_explanation("Priority")
    assert "QoS" in sm.pending_explanation("QOSMaxGRESPerUser")


def test_terminal_literal_text_hidden_counts_and_actual_interval(snapshot):
    jobs = [replace(snapshot.jobs[0], id=str(i)) for i in range(20)]
    output = render(sm.job_table(jobs, 15), width=200)
    assert "5 hidden" in output and "[/red]" in output and "train|phase2" in output
    assert sm.clean("\x1b[31mhello\x1b[0m\x00\n" + WEBHOOK).startswith("hello  [webhook redacted]")
    args = sm.parse_args(["--interval", "12", "--compact"])
    output = render(sm.render_snapshot(snapshot, sm.JobTracker(), args, "alice", 100, 30))
    assert "Refresh: 12s" in output
    output = render(
        sm.render_snapshot(snapshot, sm.JobTracker(), args, "alice", 45, 8, True),
        width=45,
        height=8,
    )
    assert "SLURM Monitor" in output and "Refresh 12s" in output


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "0.5"])
def test_interval_validation(value):
    with pytest.raises(SystemExit) as error:
        sm.parse_args(["--interval=" + value])
    assert error.value.code == 2


def test_cli_compatibility_and_python38_syntax():
    assert sm.parse_args(["-1", "-a", "-c", "-p", "gpu"]).once
    assert sm.parse_args(["--max-jobs", "0"]).max_jobs == 0
    source = Path(sm.__file__).read_text()
    tree = ast.parse(source, feature_version=8)
    assert any(
        isinstance(node, ast.ImportFrom) and node.module == "__future__" for node in tree.body
    )


def test_main_once_json_historical_and_partition_tracking(
    tmp_path, monkeypatch, fixture_text, capsys
):
    run, commands = scripted_runner(fixture_text)
    collector = sm.Collector(run, clock=lambda: 3000)
    monkeypatch.setattr(sm, "Collector", lambda: collector)
    monkeypatch.setattr(sm.getpass, "getuser", lambda: "alice")
    assert sm.main(["--once", "--compact"]) == 0
    assert "train|phase2" in capsys.readouterr().out
    assert sm.main(["--json", "--job", "104"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert any(
        job["id"] == "104" and job["derived_exit_code"] == "1:0" for job in output["accounting"]
    )
    worker = Mock()
    worker.close.return_value = True
    monkeypatch.setattr(sm, "DeliveryWorker", lambda *_: worker)
    monkeypatch.setattr(sm, "load_webhook", lambda _: WEBHOOK)
    assert (
        sm.main(
            [
                "--once",
                "--partition",
                "gpu",
                "--slack",
                "--state-file",
                str(tmp_path / "state.json"),
            ]
        )
        == 0
    )
    assert json.loads((tmp_path / "state.json").read_text())["initialized"]
    assert worker.start.called and worker.close.called
    assert all(command[0] in {"squeue", "scontrol", "sacct"} for command in commands)


def test_array_and_heterogeneous_identifiers():
    body = {
        "jobs": [dict(job_id=502, array_job_id=500, array_task_id=2, job_state="RUNNING")],
        "errors": [],
    }
    assert sm.parse_queue_json(json.dumps(body))[0].id == "500_2"
    body["jobs"][0] = dict(job_id=602, het_job_id=600, het_job_offset=2, job_state="RUNNING")
    assert sm.parse_queue_json(json.dumps(body))[0].id == "600+2"
    body["jobs"][0] = dict(
        job_id=500,
        array_job_id=500,
        array_task_id=4294967294,
        array_task_string="1-4%2",
        job_state="PENDING",
    )
    with pytest.raises(sm.QueryError, match="unsupported format for grouped arrays"):
        sm.parse_queue_json(json.dumps(body))
    calls = []

    def run(argv):
        calls.append(argv)
        return json.dumps(body) if "--json" in argv else ""

    assert sm.Collector(run).queue() == []
    assert "--array" in calls[-1] and "JobArrayID:0" in calls[-1][-1]


def test_accounting_cache_and_delayed_backoff(fixture_text):
    now = [3000.0]
    run, commands = scripted_runner(fixture_text)
    collector = sm.Collector(run, clock=lambda: now[0])
    collector.collect({"104", "105"})
    collector.collect({"104", "105"})
    assert sum(command[0] == "sacct" for command in commands) == 1
    now[0] += 15
    collector.collect({"104", "105"})
    assert sum(command[0] == "sacct" for command in commands) == 2
    assert commands[-1][commands[-1].index("--jobs") + 1] == "105"


def test_reused_job_id_preserves_old_generation_and_new_start(snapshot):
    tracker = sm.JobTracker(notify=True)
    old = snapshot.jobs[0]
    new = replace(old, submitted="2000", name="new generation")
    tracker.observe(snapshot, [old])
    tracker.observe(snapshot, [new])
    assert tracker.awaiting[old.id].submitted == old.submitted
    assert tracker.previous[old.id].submitted == "2000"
    assert tracker.outbox[0]["kind"] == "started"
    wrong = replace(new, state="COMPLETED", exit_code="0:0")
    tracker.observe(replace(snapshot, accounting=[wrong]), [new])
    assert old.id in tracker.awaiting
    correct = replace(old, state="COMPLETED", exit_code="0:0")
    tracker.observe(replace(snapshot, accounting=[correct]), [new])
    assert tracker.awaiting == {} and tracker.history[0].name == old.name


def test_collector_queries_accounting_for_reused_generation(fixture_text, snapshot):
    old = replace(snapshot.jobs[0], submitted="900")
    run, commands = scripted_runner(fixture_text)
    collector = sm.Collector(run)
    collector.collect({"101": old})
    assert any(command[0] == "sacct" and "101" in command for command in commands)


def test_stale_policy_error_stays_visible(fixture_text):
    def run(argv):
        if "partitions" in argv:
            raise sm.QueryError("scontrol: partition query failed")
        return fixture_text("queue.json" if argv[0] == "squeue" else "nodes.txt")

    collector = sm.Collector(run, clock=lambda: 3000)
    assert (
        collector.collect().errors
        == collector.collect().errors
        == ["scontrol: partition query failed"]
    )


def test_partition_live_tracking_and_interrupt_are_persisted(tmp_path, monkeypatch, fixture_text):
    queues = [fixture_text("queue.json")]
    body = json.loads(queues[0])
    body["jobs"][1]["job_state"] = "RUNNING"
    body["jobs"][1]["tres_alloc_str"] = body["jobs"][1]["tres_req_str"]
    queues.append(json.dumps(body))

    def queue():
        return queues.pop(0)

    run, _ = scripted_runner(fixture_text, queue)
    collector = sm.Collector(run, clock=lambda: 3000)
    monkeypatch.setattr(sm, "Collector", lambda: collector)
    monkeypatch.setattr(sm.getpass, "getuser", lambda: "alice")
    monkeypatch.setattr(
        sm,
        "Console",
        lambda: Console(file=io.StringIO(), force_terminal=True, width=120, height=40),
    )
    live = Mock()
    live.__enter__ = Mock(return_value=live)
    live.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(sm, "Live", lambda *a, **k: live)
    sleeper = Mock(side_effect=[None, KeyboardInterrupt()])
    monkeypatch.setattr(sm.time, "sleep", sleeper)
    worker = Mock()
    worker.close.return_value = True
    monkeypatch.setattr(sm, "DeliveryWorker", lambda *_: worker)
    monkeypatch.setattr(sm, "load_webhook", lambda _: WEBHOOK)
    path = tmp_path / "state.json"
    assert sm.main(["--partition", "gpu", "--slack", "--state-file", str(path)]) == 0
    assert live.update.call_count == 1
    payload = json.loads(path.read_text())
    assert [event["kind"] for event in payload["outbox"]] == ["started"]
    assert payload["previous"]["102"]["state"] == "RUNNING"


def test_json_failure_exit_status_and_no_secret_output(monkeypatch, capsys):
    collector = sm.Collector(
        lambda _: (_ for _ in ()).throw(sm.QueryError("query unavailable")), clock=lambda: 3000
    )
    monkeypatch.setattr(sm, "Collector", lambda: collector)
    assert sm.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert not payload["jobs_ok"] and not payload["nodes_ok"] and len(payload["errors"]) == 3


def test_malformed_journal_outbox_is_preserved(tmp_path):
    path = tmp_path / "state.json"
    contents = json.dumps(
        dict(
            version=1,
            previous={},
            awaiting={},
            history=[],
            initialized=True,
            outbox=[dict(id="x", kind="started", job={"id": "1"}, attempts=0, next_attempt="bad")],
        )
    )
    path.write_text(contents)
    with pytest.raises(sm.StateError, match="Malformed"):
        sm.JobTracker(path, True)
    assert path.read_text() == contents


def test_unknown_states_and_reason_cannot_invent_free_gpus():
    node = sm.parse_nodes(
        "NodeName=n State=DOWN Gres=gpu:8 GresUsed=gpu:0 Reason=maintenance State=IDLE"
    )[0]
    assert node.free_gpus == 0
    node = sm.parse_nodes("NodeName=n State= Gres=gpu:8 GresUsed=gpu:0")[0]
    assert node.free_gpus == 0
    with pytest.raises(sm.QueryError):
        sm.parse_nodes("NodeName=n State=DOWN State=IDLE Gres=gpu:8")


def test_total_memory_and_alternate_partition_constraints(snapshot):
    job = snapshot.jobs[1]
    assert job.memory_total == 8192
    status, reasons = sm.resource_suitability(replace(job, memory_total=1024**3), snapshot, "gpu")
    assert status.startswith("Insufficient") and "insufficient visible free memory" in reasons
    status, reasons = sm.resource_suitability(replace(job, partition="gpu"), snapshot, "shared")
    assert "partition is outside the current job request" in reasons


def test_generic_and_typed_per_node_requirements_are_both_retained():
    body = {
        "jobs": [
            dict(
                job_id=1,
                job_state="PENDING",
                node_count=2,
                tres_per_node="gres/gpu=8,gres/gpu:a100=4",
            )
        ],
        "errors": [],
    }
    job = sm.parse_queue_json(json.dumps(body))[0]
    assert job.gpu_per_node == {"": 8, "a100": 4}
    assert job.requested_gpus == 16 and job.requested_types == {"a100": 8}


def test_no_completion_when_accounting_submit_identity_is_unknown(snapshot):
    tracker = sm.JobTracker(notify=True)
    job = snapshot.jobs[0]
    tracker.observe(snapshot, [job])
    tracker.observe(
        replace(snapshot, accounting=[replace(job, state="COMPLETED", submitted="")]), []
    )
    assert tracker.outbox == [] and set(tracker.awaiting) == {job.id}


def test_terminal_queue_record_uses_accounting_exit_details(snapshot):
    queued = replace(snapshot.jobs[0], state="COMPLETED")
    accounted = replace(queued, exit_code="0:0", derived_exit_code="1:0")
    sample = replace(snapshot, jobs=[queued], accounting=[accounted])
    args = sm.parse_args(["--job", "101", "--once"])
    output = render(sm.render_snapshot(sample, sm.JobTracker(), args, "alice", 160, 40), 160)
    assert "batch exit: 0:0" in output and "derived step exit: 1:0" in output


def test_queue_scope_uses_argv_user_filter_and_json_hides_other_users(
    monkeypatch, fixture_text, capsys
):
    run, commands = scripted_runner(fixture_text)
    collector = sm.Collector(run, clock=lambda: 3000)
    monkeypatch.setattr(sm, "Collector", lambda: collector)
    monkeypatch.setattr(sm.getpass, "getuser", lambda: "alice")
    assert sm.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert all(job["user"] == "alice" for job in payload["jobs"])
    assert commands[0][commands[0].index("--user") + 1] == "alice"


def test_blocked_journal_event_schema_is_checked(tmp_path):
    path = tmp_path / "state.json"
    contents = json.dumps(
        dict(
            version=1,
            previous={},
            awaiting={},
            history=[],
            initialized=True,
            outbox=[dict(id="x", kind=None, job={"id": "1"}, attempts=0, next_attempt=3000)],
        )
    )
    path.write_text(contents)
    with pytest.raises(sm.StateError, match="Malformed"):
        sm.JobTracker(path, True)
    assert path.read_text() == contents


def test_suitability_unknown_gpu_type_does_not_report_a_false_fit(snapshot):
    mixed = sm.parse_nodes(
        "NodeName=m Partitions=gpu State=MIXED Gres=gpu:a100:4,gpu:v100:4 AllocTRES=gres/gpu=2 CPUAlloc=0 CPUTot=64 RealMemory=65536 AllocMem=0"
    )[0]
    status, reasons = sm.resource_suitability(
        snapshot.jobs[1], replace(snapshot, nodes=[mixed]), "gpu"
    )
    assert status.startswith("Unknown") and any(
        "type allocation is unknown" in reason for reason in reasons
    )


def test_effective_cpus_exclude_reserved_cores(snapshot):
    node = sm.parse_nodes(
        "NodeName=n Partitions=gpu State=IDLE CPUTot=64 CPUEfctv=56 CPUAlloc=0 Gres=gpu:0 AllocTRES= RealMemory=65536 AllocMem=0"
    )[0]
    assert node.cpus == 56
    job = replace(snapshot.jobs[1], cpus=60, requested_gpus=0, requested_types={}, gpu_per_node={})
    status, reasons = sm.resource_suitability(job, replace(snapshot, nodes=[node]), "gpu")
    assert status.startswith("Insufficient") and "insufficient visible free CPUs" in reasons


def test_delivery_worker_paces_batches_to_one_per_second():
    tracker = Mock()
    tracker.deliver.return_value = True
    worker = sm.DeliveryWorker(tracker, Mock())
    stop = Mock()
    stop.is_set.side_effect = [False, True]
    worker.stop = stop
    worker._run()
    stop.wait.assert_called_once_with(1)


def test_malformed_accounting_exit_cannot_enter_a_notification():
    with pytest.raises(sm.QueryError, match="malformed exit code"):
        sm.parse_accounting("101|FAILED|<!channel>|0:0|00:01|cpu=2|gpu|alice|1000")
