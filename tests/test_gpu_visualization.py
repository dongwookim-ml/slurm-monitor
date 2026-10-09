import io
from dataclasses import replace

import pytest
from rich.console import Console

import slurm_monitor as sm


def render(value, width):
    stream = io.StringIO()
    Console(file=stream, width=width, color_system=None).print(value)
    output = stream.getvalue()
    assert max(map(len, output.splitlines()), default=0) <= width
    return output


@pytest.mark.parametrize(
    "total,used,idle,bar,labels",
    [
        (8, 4, 2, "████░░××", "U4 I2 X2 /8"),
        (8, 0, 0, "××××××××", "U0 I0 X8 /8"),
        (8, 8, 0, "████████", "U8 I0 X0 /8"),
        (8, 0, 8, "░░░░░░░░", "U0 I8 X0 /8"),
        (8, 2, None, "██??????", "U2 I? X? /8"),
        (8, None, 4, "░░░░????", "U? I4 X? /8"),
        (8, None, None, "????????", "U? I? X? /8"),
        (None, 4, None, "????????", "U4 I? X? /?"),
        (0, 0, 0, "No GPUs", "U0 I0 X0 /0"),
        (8, 9, 0, "????????", "inconsistent"),
        (8, 2, 7, "????????", "inconsistent"),
    ],
)
def test_gpu_meter_capacity_and_unknown_honesty(total, used, idle, bar, labels):
    meter = sm.gpu_meter(total, used, idle, 8, stacked=True)
    assert meter.plain.splitlines()[0] == bar
    assert labels in meter.plain


def test_bar_rounding_and_small_segments_keep_exact_counts():
    meter = sm.gpu_meter(1000, 1, 998, 10)
    bar = meter.plain.split("  ")[0]
    assert len(bar) == 10 and all(symbol in bar for symbol in "█░×")
    assert "U1 I998 X1 /1000" in meter.plain
    colors = {span.style for span in meter.spans}
    assert {"bold bright_red", "bright_green", "bright_black", "dim"} <= colors


@pytest.mark.parametrize("width", [42, 60, 80, 100, 140])
def test_partition_bars_mixed_types_drains_and_zero_total(snapshot, width):
    before = sm.gpu_summary(snapshot, "alice")
    output = render(sm.gpu_table(snapshot, "alice", width=width), width)
    assert "U10 I8 X16 /34" in output
    assert "U6 I4 X0 /10" in output
    assert "No GPUs" in output and "U0 I0 X0 /0" in output
    assert all(symbol in output for symbol in "█░×")
    assert all(word in output for word in ["Used", "Idle", "Unavailable", "Unknown"])
    assert sm.gpu_summary(snapshot, "alice") == before


@pytest.mark.parametrize("width", [42, 100])
def test_unknown_allocations_never_draw_idle_and_drains_show_unavailable(snapshot, width):
    unknown = sm.Node("unknown", ["unknown"], ["IDLE"], gpus=8)
    unknown_snapshot = replace(snapshot, nodes=[unknown], partitions=[])
    output = render(sm.gpu_table(unknown_snapshot, "alice", width=width), width)
    assert "U? I? X? /8" in output and "??????????" in output
    meter = sm.gpu_meter(8, None, None, 10)
    assert "░" not in meter.plain
    node_output = render(sm.node_table(snapshot, "gpu", width=width), width)
    assert "U0 I0 X8 /8" in node_output
    assert "DRAIN" in node_output and "DOWN" in node_output


def test_narrow_live_dashboard_keeps_inventory_and_compact_mode(snapshot):
    tracker = sm.JobTracker()
    output = render(
        sm.render_snapshot(snapshot, tracker, sm.parse_args([]), "alice", 42, 40, True), 42
    )
    assert "GPU allocation" in output and "U10 I8 X16 /34" in output
    assert "more" not in output  # The three partitions fit; no arbitrary cap.
    compact = render(
        sm.render_snapshot(snapshot, tracker, sm.parse_args(["--compact"]), "alice", 42, 40, True),
        42,
    )
    assert "GPU allocation" not in compact


def test_literal_partition_name_cannot_be_rich_markup(snapshot):
    node = replace(snapshot.nodes[0], partitions=["[/red]"])
    output = render(sm.gpu_table(replace(snapshot, nodes=[node], partitions=[]), "alice"), 100)
    assert "[/red]" in output and "U6 I4 X0 /10" in output


def test_limited_inventory_prioritizes_gpu_capacity(snapshot):
    output = render(sm.gpu_table(snapshot, "alice", limit=1, width=42), 42)
    assert "U10 I8 X16 /34" in output and "2 more" in output
    assert "No GPUs" not in output


def test_narrow_short_live_view_keeps_gpu_bar_and_legend(snapshot):
    output = render(
        sm.render_snapshot(snapshot, sm.JobTracker(), sm.parse_args([]), "alice", 42, 28, True), 42
    )
    assert "U10 I8 X16 /34" in output
    assert "X=Unavailable" in output and "?=Unknown" in output
    assert len(output.splitlines()) <= 28


@pytest.mark.parametrize("state,labels", [("DOWN", "U6 I0 X4 /10"), ("UNKNOWN", "U6 I? X? /10")])
def test_partition_policy_does_not_turn_blocked_or_unknown_capacity_into_idle(
    snapshot, state, labels
):
    selected = replace(snapshot.nodes[0], partitions=["shared"])
    policy = sm.Partition("shared", state=state)
    snapshot = replace(snapshot, nodes=[selected], partitions=[policy])
    record = sm.gpu_summary(snapshot, "alice")[0]
    meter = sm.gpu_meter(record["total"], record["used"], record["free"], 10)
    assert labels in meter.plain and "░" not in meter.plain
    assert labels in render(sm.gpu_table(snapshot, "alice", width=42), 42)
