import io
import re
from dataclasses import replace

import pytest
from rich.console import Console, Group
from rich.table import Table

import slurm_monitor as sm


def sample(snapshot, jobs=80, partitions=18):
    nodes = [
        replace(snapshot.nodes[i % 4], name=f"node{i:02}", partitions=[f"p{i:02}"])
        for i in range(partitions)
    ]
    policies = [sm.Partition(f"p{i:02}", state="UP") for i in range(partitions)]
    records = [
        replace(
            snapshot.jobs[0],
            id=str(1000 + i),
            name=f"training_{i}",
            partition=f"p{i % max(1, partitions):02}",
        )
        for i in range(jobs)
    ]
    return replace(snapshot, jobs=records, nodes=nodes, partitions=policies)


def render(view, width):
    console = Console(file=io.StringIO(), width=width, color_system=None)
    lines = console.render_lines(view, console.options.update(height=None), pad=False)
    return "\n".join("".join(segment.text for segment in line) for line in lines), len(lines)


@pytest.mark.parametrize("width,height", [(100, 56), (160, 56), (42, 90)])
def test_eighteen_partitions_all_fit_without_arbitrary_more(snapshot, width, height):
    data = sample(snapshot, jobs=1)
    view = sm.render_snapshot(
        data, sm.JobTracker(), sm.parse_args([]), "alice", width, height, True
    )
    output, lines = render(view, width)
    assert lines <= height
    assert all(f"p{i:02}" in output for i in range(18))
    assert "more" not in output and "hidden" not in output


def test_default_job_count_is_not_capped_at_fifteen(snapshot):
    data = sample(snapshot, jobs=80)
    args = sm.parse_args(["--compact"])
    output, lines = render(
        sm.render_snapshot(data, sm.JobTracker(), args, "alice", 160, 120, True), 160
    )
    assert args.max_jobs is None and "Jobs: 80/80 shown" in output
    assert "hidden" not in output and lines < 120


@pytest.mark.parametrize("width", [42, 80, 160])
@pytest.mark.parametrize("height", [8, 18, 24, 28, 40, 56, 90])
def test_mixed_sections_fit_real_viewport_without_clipping_controls(snapshot, width, height):
    data = sample(snapshot)
    data.errors = ["sacct unavailable; endings remain unconfirmed"]
    tracker = sm.JobTracker()
    tracker.history = [
        replace(data.jobs[0], id=str(9000 + i), state="FAILED", exit_code="1:0") for i in range(3)
    ]
    output, lines = render(
        sm.render_snapshot(data, tracker, sm.parse_args([]), "alice", width, height, True), width
    )
    assert lines <= height
    assert all(len(line) <= width for line in output.splitlines())
    assert "Refresh: 5s" in output and "Ctrl+C" in output
    assert "errors" in output.lower()
    assert all(symbol in output for symbol in "█░×?")


@pytest.mark.parametrize("width,height", [(100, 40), (160, 56), (42, 56)])
def test_hidden_rows_only_when_another_row_would_exceed_viewport(snapshot, width, height):
    data = sample(snapshot)
    ordered_jobs = sorted(data.jobs, key=lambda job: (job.state != "PENDING", job.id))
    view = sm.render_snapshot(
        data, sm.JobTracker(), sm.parse_args([]), "alice", width, height, True
    )
    output, lines = render(view, width)
    assert lines <= height and ("hidden" in output or "more" in output)
    parts = list(view.renderables)
    for index, part in enumerate(parts):
        if not isinstance(part, Table):
            continue
        title = str(part.title)
        if title.startswith("Jobs:"):
            shown = int(re.search(r"Jobs: (\d+)/", title).group(1))
            if shown == len(data.jobs):
                continue
            larger = sm.job_table(ordered_jobs, shown + 1, width < 100)
        elif title.startswith("Partition GPUs"):
            text, _ = render(part, width)
            shown = sum(f"p{i:02}" in text for i in range(18))
            if shown == 18:
                continue
            larger = sm.gpu_table(data, "alice", shown + 1, data.jobs, width)
        else:
            continue
        candidate = list(parts)
        candidate[index] = larger
        assert render(Group(*candidate), width)[1] > height


@pytest.mark.parametrize("width,height", [(42, 28), (100, 56), (160, 90)])
def test_partition_node_details_share_height_with_many_jobs(snapshot, width, height):
    data = sample(snapshot)
    data.nodes = [replace(node, partitions=["p00"]) for node in data.nodes]
    data.jobs = [replace(job, partition="p00") for job in data.jobs]
    args = sm.parse_args(["--partition", "p00"])
    output, lines = render(
        sm.render_snapshot(data, sm.JobTracker(), args, "alice", width, height, True), width
    )
    assert lines <= height and "Ctrl+C" in output
    assert "node" in output and "U" in output


def test_resize_increases_visible_rows_and_once_ignores_viewport(snapshot):
    data = sample(snapshot)
    short = sm.render_snapshot(data, sm.JobTracker(), sm.parse_args([]), "alice", 100, 28, True)
    tall = sm.render_snapshot(data, sm.JobTracker(), sm.parse_args([]), "alice", 100, 70, True)
    short_count = int(re.search(r"Jobs: (\d+)/", render(short, 100)[0]).group(1))
    tall_count = int(re.search(r"Jobs: (\d+)/", render(tall, 100)[0]).group(1))
    assert tall_count > short_count
    full, lines = render(
        sm.render_snapshot(data, sm.JobTracker(), sm.parse_args(["--once"]), "alice", 100, 10), 100
    )
    assert "Jobs: 80/80 shown" in full and "more" not in full
    assert lines > 10


def test_explicit_job_cap_and_zero_auto_preserve_overflow_guard(snapshot):
    data = sample(snapshot, jobs=40)
    capped, _ = render(
        sm.render_snapshot(
            data,
            sm.JobTracker(),
            sm.parse_args(["--max-jobs", "5", "--compact"]),
            "alice",
            100,
            90,
            True,
        ),
        100,
    )
    assert "Jobs: 5/40 shown" in capped
    auto, lines = render(
        sm.render_snapshot(
            data, sm.JobTracker(), sm.parse_args(["--max-jobs", "0"]), "alice", 100, 28, True
        ),
        100,
    )
    assert lines <= 28 and "hidden" in auto


def test_wrapped_errors_and_diagnostics_use_measured_space(snapshot):
    data = sample(snapshot, jobs=1)
    data.jobs[0].state = "PENDING"
    data.jobs[0].reason = "QOSMaxGRESPerUser"
    data.errors = ["query failure " * 12]
    args = sm.parse_args(["--job", data.jobs[0].id])
    output, lines = render(
        sm.render_snapshot(data, sm.JobTracker(), args, "alice", 80, 70, True), 80
    )
    assert lines <= 70 and "Job diagnostics" in output
    assert "QOSMaxGRESPerUser" in output and "Ctrl+C" in output


@pytest.mark.parametrize("width,height", [(42, 28), (100, 28), (160, 56)])
def test_actual_terminal_color_render_uses_the_same_viewport(snapshot, width, height):
    data = sample(snapshot)
    view = sm.render_snapshot(
        data, sm.JobTracker(), sm.parse_args([]), "alice", width, height, True
    )
    console = Console(
        file=io.StringIO(),
        width=width,
        height=height,
        force_terminal=True,
        color_system="truecolor",
    )
    assert console.width == width and console.height == height
    lines = console.render_lines(view, console.options.update(height=None), pad=False)
    assert len(lines) <= height
    assert "Ctrl+C" in "".join(segment.text for line in lines for segment in line)
