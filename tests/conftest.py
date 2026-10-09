"""Tests are offline by construction, including accidental command/HTTP attempts."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import slurm_monitor as sm  # noqa: E402


@pytest.fixture(autouse=True)
def prohibit_external_effects(monkeypatch):
    monkeypatch.setattr(
        sm.subprocess,
        "run",
        Mock(side_effect=AssertionError("Real subprocess prohibited in tests")),
    )
    monkeypatch.setattr(
        sm.urllib.request,
        "urlopen",
        Mock(side_effect=AssertionError("Real HTTP prohibited in tests")),
    )


@pytest.fixture
def fixture_text():
    return lambda name: (Path(__file__).parent / "fixtures" / name).read_text()


@pytest.fixture
def snapshot(fixture_text):
    return sm.Snapshot(
        sm.parse_queue_json(fixture_text("queue.json")),
        sm.parse_nodes(fixture_text("nodes.txt")),
        sm.parse_partitions(fixture_text("partitions.txt")),
        collected_at=3000,
        jobs_at=3000,
        nodes_at=3000,
        partitions_at=3000,
    )
