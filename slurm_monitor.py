#!/usr/bin/env python3
"""Read-only SLURM dashboard. Collection, tracking, delivery and rendering are separate."""

from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import json
import math
import os
import re
import shlex
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

VERSION = "0.2.2"
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "PREEMPTED",
    "DEADLINE",
    "REVOKED",
}
UNUSABLE = {
    "DOWN",
    "DRAIN",
    "DRAINED",
    "DRAINING",
    "FAIL",
    "FAILING",
    "MAINT",
    "FUTURE",
    "UNKNOWN",
    "NOT_RESPONDING",
    "POWERED_DOWN",
    "POWERING_DOWN",
    "POWER_DOWN",
    "REBOOT_ISSUED",
    "REBOOT_REQUESTED",
    "PLANNED",
    "RESERVED",
}
SEP = "\x1f"
QUEUE_FIELDS = [
    "JobArrayID",
    "Name",
    "UserName",
    "Partition",
    "State",
    "TimeUsed",
    "TimeLimit",
    "NumNodes",
    "Reason",
    "tres-alloc",
    "tres-per-node",
    "NumCPUs",
    "MinMemory",
    "Feature",
    "QOS",
    "Account",
    "Reservation",
    "SubmitTime",
    "tres-per-task",
]


class QueryError(RuntimeError):
    """A failed query or malformed response, never an empty successful snapshot."""


class StateError(RuntimeError):
    """Local delivery journal cannot be safely used."""


def clean(value: object, limit: int = 500) -> str:
    """Strip terminal/control sequences and redact webhook URLs in diagnostics."""
    text = str(value if value is not None else "")
    text = re.sub(r"\x1b\][^\x07]*(?:\x07|\x1b\\)", "", text)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"[\u202a-\u202e\u2066-\u2069]", "", text)
    text = re.sub(r"https?://hooks\.slack(?:-gov)?\.com/\S+", "[webhook redacted]", text)
    return text[:limit]


def literal(value: object, style: str = "") -> Text:
    return Text(clean(value), style=style)


def number(value: object) -> Optional[int]:
    if isinstance(value, dict):
        if value.get("set") is False or value.get("infinite") is True:
            return None
        value = value.get("number")
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (ValueError, TypeError):
        return None
    return result if 0 <= result < 0xFFFFFFFE else None


def state_name(value: object) -> str:
    if isinstance(value, list):
        # COMPLETING is a flag in several JSON schema versions.
        values = [str(v).upper() for v in value]
        if "COMPLETING" in values:
            return "COMPLETING"
        value = values[0] if values else "UNKNOWN"
    return str(value or "UNKNOWN").upper().split()[0].rstrip("+")


def submitted_token(value: object) -> str:
    numeric = number(value)
    if numeric is not None:
        return str(numeric)
    try:
        return str(int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()))
    except (ValueError, TypeError, OverflowError):
        return ""


def duration(value: object) -> Optional[float]:
    text = str(value or "").strip()
    if text.upper() in {"UNLIMITED", "INFINITE"}:
        return math.inf
    try:
        days, _, clock = text.rpartition("-")
        pieces = [int(v) for v in (clock if days else text).split(":")]
        if len(pieces) == 3:
            seconds = pieces[0] * 3600 + pieces[1] * 60 + pieces[2]
        elif len(pieces) == 2:
            seconds = pieces[0] * 60 + pieces[1]
        elif len(pieces) == 1:
            seconds = pieces[0] * 60  # SLURM bare limits are minutes.
        else:
            return None
        return seconds + (int(days) * 86400 if days else 0)
    except ValueError:
        return None


def memory_mib(value: object) -> Optional[int]:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)(?:[cn])?", str(value).strip(), re.I)
    if not match:
        return None
    return int(
        float(match[1]) * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2}[match[2].upper()]
    )


@dataclass
class GPUCount:
    total: Optional[int] = None
    types: dict[str, int] = field(default_factory=dict)


def parse_gpus(value: object, empty_is_zero: bool = False) -> GPUCount:
    """Read GRES and TRES, preserving types and avoiding generic+typed TRES double counts."""
    text = re.sub(r"\([^)]*\)", "", str(value or ""))
    if text.strip() in {"", "(null)", "None", "N/A", "Unknown"}:
        return GPUCount(0 if empty_is_zero else None)
    counts: dict[str, int] = {}
    for token in text.split(","):
        match = re.fullmatch(r"(?:gres/)?gpu(?::([^:=]+))?[:=](\d+)", token.strip(), re.I)
        if match:
            key = (match[1] or "").lower()
            counts[key] = counts.get(key, 0) + int(match[2])
        elif re.match(r"(?:gres/)?gpu(?:[:=]|$)", token.strip(), re.I):
            return GPUCount()  # Do not silently ignore a malformed GPU resource.
    if not counts:
        return GPUCount(0)
    total = counts[""] if "=" in text and "" in counts else sum(counts.values())
    return GPUCount(total, {k: v for k, v in counts.items() if k})


def tres_value(value: object, key: str) -> str:
    for token in str(value or "").split(","):
        if token.startswith(key + "="):
            return token.split("=", 1)[1]
    return ""


@dataclass
class Job:
    id: str
    name: str = ""
    user: str = ""
    partition: str = ""
    state: str = "UNKNOWN"
    runtime: str = ""
    time_limit: str = ""
    nodes: Optional[int] = None
    gpus: Optional[int] = None
    gpu_types: dict[str, int] = field(default_factory=dict)
    requested_gpus: Optional[int] = None
    requested_types: dict[str, int] = field(default_factory=dict)
    gpu_per_node: dict[str, int] = field(default_factory=dict)
    cpus: Optional[int] = None
    memory_per_node: Optional[int] = None
    features: str = ""
    qos: str = ""
    account: str = ""
    reservation: str = ""
    reason: str = ""
    submitted: str = ""
    restart_count: int = 0
    exit_code: str = ""
    derived_exit_code: str = ""
    task_resources: str = ""
    memory_total: Optional[int] = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL


@dataclass
class Node:
    name: str
    partitions: list[str] = field(default_factory=list)
    states: list[str] = field(default_factory=list)
    gpus: Optional[int] = None
    gpu_types: dict[str, int] = field(default_factory=dict)
    used_gpus: Optional[int] = None
    used_types: dict[str, int] = field(default_factory=dict)
    cpus: Optional[int] = None
    used_cpus: Optional[int] = None
    memory: Optional[int] = None
    used_memory: Optional[int] = None
    features: list[str] = field(default_factory=list)
    drained_gpus: Optional[int] = 0
    drained_types: dict[str, int] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return bool(
            {"IDLE", "MIXED", "ALLOCATED", "COMPLETING"}.intersection(self.states)
        ) and not UNUSABLE.intersection(self.states)

    @property
    def free_gpus(self) -> Optional[int]:
        if not self.usable:
            return 0
        if (
            self.gpus is None
            or self.used_gpus is None
            or self.drained_gpus is None
            or self.used_gpus + self.drained_gpus > self.gpus
        ):
            return None
        return self.gpus - self.used_gpus - self.drained_gpus

    def free_type(self, kind: str) -> Optional[int]:
        if not self.usable:
            return 0
        if self.free_gpus is None:
            return None
        if len(self.gpu_types) == 1:
            return self.free_gpus if kind in self.gpu_types else 0
        if (
            sum(self.used_types.values()) != self.used_gpus
            or sum(self.drained_types.values()) != self.drained_gpus
        ):
            return None
        return max(
            0,
            self.gpu_types.get(kind, 0)
            - self.used_types.get(kind, 0)
            - self.drained_types.get(kind, 0),
        )


@dataclass
class Partition:
    name: str
    state: str = "UNKNOWN"
    max_time: str = ""
    max_nodes: Optional[int] = None
    accounts: str = ""
    qos: str = ""
    groups: str = ""


@dataclass
class Snapshot:
    jobs: list[Job] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    partitions: list[Partition] = field(default_factory=list)
    accounting: list[Job] = field(default_factory=list)
    jobs_ok: bool = True
    nodes_ok: bool = True
    errors: list[str] = field(default_factory=list)
    collected_at: float = 0
    jobs_at: float = 0
    nodes_at: float = 0
    partitions_at: float = 0


def run_command(argv: list[str], timeout: float = 15) -> str:
    """Run only an argv list. Hide stderr (which can contain URLs/job secrets)."""
    if isinstance(argv, str):
        raise TypeError("Commands must be argv lists")
    environment = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("SQUEUE_", "SINFO_", "SACCT_"))
        and k not in {"SLURM_CLUSTERS", "SLURM_JSON", "SLURM_YAML"}
    }
    environment["LC_ALL"] = "C"
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False, env=environment
        )
    except FileNotFoundError:
        raise QueryError(f"{Path(argv[0]).name}: command not found") from None
    except (subprocess.TimeoutExpired, OSError, UnicodeError):
        raise QueryError(f"{Path(argv[0]).name}: query timed out or could not execute") from None
    if result.returncode:
        # Only retain the capability signal internally, never raw output in diagnostics.
        unsupported = any(
            v in result.stderr.lower()
            for v in ("unrecognized option", "unknown option", "data_parser", "invalid option")
        )
        raise QueryError(
            f"{Path(argv[0]).name}: {'unsupported format' if unsupported else 'query failed'} (exit {result.returncode})"
        )
    return result.stdout


def parse_queue_json(output: str) -> list[Job]:
    try:
        payload = json.loads(output)
        if (
            not isinstance(payload, dict)
            or payload.get("errors")
            or not isinstance(payload.get("jobs"), list)
        ):
            raise ValueError("Missing jobs or server errors")
        jobs = []
        for row in payload["jobs"]:
            jid = number(row.get("job_id"))
            if jid is None or not row.get("job_state"):
                raise ValueError("Missing job identifier/state")
            identifier = str(jid)
            array = number(row.get("array_job_id"))
            task = number(row.get("array_task_id"))
            if array:
                if task is None:
                    raise QueryError(
                        "squeue: unsupported format for grouped arrays; using expanded text"
                    )
                identifier = f"{array}_{task}"
            heterogeneous = number(row.get("het_job_id"))
            if heterogeneous:
                identifier = f"{heterogeneous}+{number(row.get('het_job_offset')) or 0}"
            state = state_name(row["job_state"])
            allocated = parse_gpus(row.get("tres_alloc_str"))
            requested = parse_gpus(
                row.get("tres_req_str") or (row.get("tres_alloc_str") if state == "PENDING" else "")
            )
            per_node = parse_gpus(row.get("tres_per_node"))
            node_requirements = dict(per_node.types)
            if per_node.total:
                node_requirements[""] = per_node.total
            requested_nodes = number(row.get("node_count") or row.get("minimum_nodes"))
            if (
                requested.total is None
                and per_node.total is not None
                and requested_nodes is not None
            ):
                requested = GPUCount(
                    per_node.total * requested_nodes,
                    {key: value * requested_nodes for key, value in per_node.types.items()},
                )
            limit = number(row.get("time_limit"))
            start, end = number(row.get("start_time")), number(row.get("end_time"))
            elapsed = max(0, (end or int(time.time())) - start) if start else None
            jobs.append(
                Job(
                    identifier,
                    str(row.get("name") or ""),
                    str(row.get("user_name") or ""),
                    str(row.get("partition") or ""),
                    state,
                    str(elapsed) + "s" if elapsed is not None else "",
                    str(limit) if limit is not None else "",
                    requested_nodes,
                    allocated.total if state != "PENDING" else None,
                    allocated.types,
                    requested.total,
                    requested.types,
                    node_requirements,
                    number(row.get("cpus") or row.get("minimum_cpus")),
                    number(row.get("memory_per_node") or row.get("minimum_memory_per_node")),
                    str(row.get("features") or ""),
                    str(row.get("qos") or ""),
                    str(row.get("account") or ""),
                    str(row.get("reservation") or ""),
                    str(row.get("state_reason") or ""),
                    submitted_token(row.get("submit_time")),
                    number(row.get("restart_cnt")) or 0,
                    task_resources=str(row.get("tres_per_task") or ""),
                    memory_total=memory_mib(tres_value(row.get("tres_req_str"), "mem")),
                )
            )
        return jobs
    except (ValueError, TypeError, KeyError, AttributeError):
        raise QueryError("squeue: malformed JSON response; previous jobs retained") from None


def parse_queue_text(output: str) -> list[Job]:
    jobs = []
    for line in output.split("\n"):
        if not line.strip():
            continue
        columns = [v.strip() for v in line.split(SEP)]
        if len(columns) != len(QUEUE_FIELDS) or not re.fullmatch(
            r"\d+(?:[_+][\d\[\],%?-]+)?", columns[0]
        ):
            raise QueryError("squeue: malformed text response; previous jobs retained")
        (
            identifier,
            name,
            user,
            partition,
            state,
            runtime,
            limit,
            nodes,
            reason,
            alloc,
            per_node,
            cpus,
            mem,
            features,
            qos,
            account,
            reservation,
            submit,
            task,
        ) = columns
        allocation, per = parse_gpus(alloc), parse_gpus(per_node)
        types = dict(per.types)
        if per.total:
            types[""] = per.total
        # MinMemory's CPU/node scope varies with submission mode; do not invent a scope.
        jobs.append(
            Job(
                identifier,
                name,
                user,
                partition,
                state_name(state),
                runtime,
                limit,
                number(nodes),
                allocation.total if state != "PENDING" else None,
                allocation.types,
                allocation.total if state == "PENDING" else None,
                allocation.types,
                types,
                number(cpus),
                features=features,
                qos=qos,
                account=account,
                reservation=reservation,
                reason=reason,
                submitted=submitted_token(submit),
                task_resources=task,
            )
        )
    return jobs


def key_values(line: str) -> dict[str, str]:
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=", line))
    if len({match[1] for match in matches}) != len(matches):
        raise QueryError("scontrol: ambiguous duplicate fields")
    return {
        m[1]: line[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(line)].strip()
        for i, m in enumerate(matches)
    }


def parse_nodes(output: str) -> list[Node]:
    nodes: dict[str, Node] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        row = key_values(line.split(" Reason=", 1)[0])
        if not row.get("NodeName") or "State" not in row:
            raise QueryError("scontrol: malformed node response; previous nodes retained")
        configured = parse_gpus(row.get("Gres"), empty_is_zero="Gres" in row)
        used = parse_gpus(
            row.get("GresUsed", row.get("AllocTRES")),
            empty_is_zero="GresUsed" in row or "AllocTRES" in row,
        )
        drained = parse_gpus(row.get("GresDrain"), empty_is_zero=True)
        if configured.total == 0:
            used = GPUCount(0)
        states = [v.upper().rstrip("*~#%!$@^-") for v in row["State"].split("+")]
        if "*" in row["State"]:
            states.append("NOT_RESPONDING")
        if re.search(r"[~#%!$@^]", row["State"]):
            states.append("UNKNOWN")
        node = Node(
            row["NodeName"],
            [v for v in row.get("Partitions", "").split(",") if v and v != "(null)"],
            states,
            configured.total,
            configured.types,
            used.total,
            used.types,
            number(row.get("CPUEfctv"))
            if number(row.get("CPUEfctv")) is not None
            else number(row.get("CPUTot")),
            number(row.get("CPUAlloc")),
            number(row.get("RealMemory")),
            number(row.get("AllocMem")),
            [v for v in row.get("ActiveFeatures", "").split(",") if v and v != "(null)"],
            drained.total,
            drained.types,
        )
        if node.name in nodes and nodes[node.name] != node:
            raise QueryError("scontrol: conflicting duplicate node records")
        nodes[node.name] = node
    return list(nodes.values())


def parse_partitions(output: str) -> list[Partition]:
    parts = []
    for line in output.splitlines():
        if not line.strip():
            continue
        row = key_values(line)
        if not row.get("PartitionName"):
            raise QueryError("scontrol: malformed partition response")
        parts.append(
            Partition(
                row["PartitionName"],
                row.get("State", "UNKNOWN").upper(),
                row.get("MaxTime", ""),
                number(row.get("MaxNodes")),
                row.get("AllowAccounts", ""),
                row.get("AllowQos", ""),
                row.get("AllowGroups", ""),
            )
        )
    return parts


def parse_accounting(output: str) -> list[Job]:
    jobs = []
    # Names are deliberately absent: only non-free-text fields use the pipe delimiter.
    for line in output.splitlines():
        if not line.strip():
            continue
        columns = line.split("|")
        if len(columns) != 9 or not re.fullmatch(r"\d+(?:[_+]\d+)?", columns[0]):
            raise QueryError("sacct: malformed accounting response; endings remain unconfirmed")
        jid, state, exit_code, derived, elapsed, allocated, partition, user, submitted = columns
        if any(
            value and value != "Unknown" and not re.fullmatch(r"\d+:\d+", value)
            for value in (exit_code, derived)
        ):
            raise QueryError("sacct: malformed exit code; ending remains unconfirmed")
        count = parse_gpus(allocated)
        jobs.append(
            Job(
                jid,
                user=user,
                partition=partition,
                state=state_name(state),
                runtime=elapsed,
                gpus=count.total,
                gpu_types=count.types,
                exit_code=exit_code,
                derived_exit_code=derived,
                submitted=submitted_token(submitted),
            )
        )
    return jobs


class Collector:
    def __init__(
        self,
        runner: Callable[[list[str]], str] = run_command,
        clock: Callable[[], float] = time.time,
        partition_ttl: float = 60,
    ):
        self.run = runner
        self.clock = clock
        self.partition_ttl = partition_ttl
        self.previous = Snapshot(jobs_ok=False, nodes_ok=False)
        self.json_supported = True
        self.partitions_attempted_at: Optional[float] = None
        self.partition_error = ""
        self.accounting_error = ""
        self.accounting_cache: dict[str, list[Job]] = {}
        self.accounting_attempts: dict[str, float] = {}
        self.user: Optional[str] = None

    def queue(self) -> list[Job]:
        base = ["squeue", "--local", "--states=all", "--array"]
        if self.user:
            base += ["--user", self.user]
        if self.json_supported:
            try:
                return parse_queue_json(self.run(base + ["--json"]))
            except QueryError as error:
                if "unsupported format" not in str(error):
                    raise
                self.json_supported = False
        fmt = ",".join(
            name + ":0" + (SEP if i + 1 < len(QUEUE_FIELDS) else "")
            for i, name in enumerate(QUEUE_FIELDS)
        )
        return parse_queue_text(self.run(base + ["--noheader", "--Format", fmt]))

    def collect(
        self, tracked: set[str] | dict[str, Job] = frozenset(), inspect_job: Optional[str] = None
    ) -> Snapshot:
        now = self.clock()
        old = self.previous
        result = Snapshot(
            old.jobs,
            old.nodes,
            old.partitions,
            collected_at=now,
            jobs_at=old.jobs_at,
            nodes_at=old.nodes_at,
            partitions_at=old.partitions_at,
        )
        try:
            result.jobs = self.queue()
            result.jobs_at = now
        except QueryError as error:
            result.jobs_ok = False
            result.errors.append(str(error))
        try:
            result.nodes = parse_nodes(self.run(["scontrol", "show", "nodes", "--oneliner"]))
            result.nodes_at = now
        except QueryError as error:
            result.nodes_ok = False
            result.errors.append(str(error))
        if (
            self.partitions_attempted_at is None
            or now - self.partitions_attempted_at >= self.partition_ttl
        ):
            self.partitions_attempted_at = now
            try:
                result.partitions = parse_partitions(
                    self.run(["scontrol", "show", "partitions", "--oneliner"])
                )
                result.partitions_at = now
                self.partition_error = ""
            except QueryError as error:
                self.partition_error = str(error)
        if self.partition_error:
            result.errors.append(self.partition_error)
        if result.jobs_ok:
            current = {job.id for job in result.jobs if not job.terminal}
            missing = set(tracked) - current
            if isinstance(tracked, dict):
                missing.update(
                    job.id
                    for job in result.jobs
                    if job.id in tracked
                    and job.submitted
                    and tracked[job.id].submitted
                    and job.submitted != tracked[job.id].submitted
                )
            missing.update(job.id for job in result.jobs if job.terminal)
            if inspect_job and inspect_job not in current:
                missing.add(inspect_job)
            # Grouped pending-array IDs cannot be safely turned into a sacct job selector.
            ids = sorted(jid for jid in missing if re.fullmatch(r"\d+(?:[_+]\d+)?", jid))
            cached = [record for jid in ids for record in self.accounting_cache.get(jid, [])]

            def cached_matches(jid: str) -> bool:
                previous = tracked.get(jid) if isinstance(tracked, dict) else None
                return any(
                    record.terminal
                    and record.submitted
                    and (not previous or record.submitted == previous.submitted)
                    for record in self.accounting_cache.get(jid, [])
                )

            query_ids = [
                jid
                for jid in ids
                if not cached_matches(jid)
                and now - self.accounting_attempts.get(jid, -math.inf) >= 15
            ]
            result.accounting = cached
            if not ids:
                self.accounting_error = ""
            if query_ids:
                try:
                    records = []
                    for offset in range(0, len(query_ids), 500):
                        chunk = query_ids[offset : offset + 500]
                        self.accounting_attempts.update({jid: now for jid in chunk})
                        records += [
                            record
                            for record in parse_accounting(
                                self.run(
                                    [
                                        "sacct",
                                        "--local",
                                        "--allocations",
                                        "--noheader",
                                        "--parsable2",
                                        "--jobs",
                                        ",".join(chunk),
                                        "--format",
                                        "JobID,State,ExitCode,DerivedExitCode,Elapsed,AllocTRES,Partition,User,Submit",
                                    ]
                                )
                            )
                            if record.id in chunk
                        ]
                    result.accounting += records
                    for jid in query_ids:
                        self.accounting_cache[jid] = [
                            record for record in records if record.id == jid and record.terminal
                        ]
                    self.accounting_error = ""
                    while len(self.accounting_cache) > 2000:
                        self.accounting_cache.pop(next(iter(self.accounting_cache)))
                except QueryError as error:
                    self.accounting_error = str(error)
        if self.accounting_error:
            result.errors.append(self.accounting_error)
        self.previous = result
        return result


def known_sum(values: list[Optional[int]]) -> Optional[int]:
    return None if any(value is None for value in values) else sum(value or 0 for value in values)


def gpu_summary(snapshot: Snapshot, user: str) -> list[dict]:
    metadata = {part.name: part for part in snapshot.partitions}
    names = sorted(set(metadata).union(*(set(node.partitions) for node in snapshot.nodes)))
    records = []
    for name in names:
        nodes = [node for node in snapshot.nodes if name in node.partitions]
        policy = metadata.get(name)
        free = (
            known_sum([node.free_gpus for node in nodes])
            if not policy or policy.state == "UP"
            else None
            if policy.state == "UNKNOWN"
            else 0
        )
        mine = known_sum(
            [
                job.gpus
                for job in snapshot.jobs
                if job.user == user
                and job.partition == name
                and not job.terminal
                and job.state != "PENDING"
            ]
        )
        records.append(
            dict(
                name=name,
                total=known_sum([node.gpus for node in nodes]),
                used=known_sum([node.used_gpus for node in nodes]),
                free=free,
                mine=mine,
            )
        )
    return records


def pending_explanation(reason: str) -> str:
    if reason.startswith(("QOS", "Assoc")):
        return "Account/QoS limit; free hardware alone is insufficient."
    return {
        "Resources": "Waiting for requested resources or node placement.",
        "Priority": "Other eligible jobs have higher scheduling priority.",
        "Dependency": "Waiting for a job dependency.",
        "DependencyNeverSatisfied": "A dependency cannot be satisfied.",
        "JobHeldUser": "Held by the user.",
        "JobHeldAdmin": "Held by an administrator.",
        "InvalidAccount": "Account or partition authorization needs correction.",
        "PartitionTimeLimit": "Requested time exceeds the partition limit.",
        "PartitionNodeLimit": "Requested node count exceeds the partition limit.",
        "Reservation": "Waiting for a reservation.",
        "ReqNodeNotAvail": "Requested nodes are unavailable or reserved.",
    }.get(reason, "SLURM supplied this reason; additional policy or constraints may apply.")


def resource_suitability(job: Job, snapshot: Snapshot, partition: str) -> tuple[str, list[str]]:
    """Necessary visible capacity checks only; never predict a scheduling decision."""
    nodes = [node for node in snapshot.nodes if partition in node.partitions]
    reasons = []
    unknown = [
        "priority, reservations, topology, licenses and scheduler placement are not fully evaluated"
    ]
    if job.partition and partition not in job.partition.split(","):
        reasons.append("partition is outside the current job request")
    policy = next((p for p in snapshot.partitions if p.name == partition), None)
    if not snapshot.jobs_ok or not snapshot.nodes_ok:
        return "Unknown (stale data)", [
            "A resource query failed; retained data is not suitable for a current assessment."
        ]
    if not nodes:
        return "Unknown", ["No visible nodes in this partition; visibility may be restricted."]
    if policy:
        if policy.state != "UP":
            reasons.append("partition is not UP")
        if policy.max_nodes is not None and job.nodes is not None and job.nodes > policy.max_nodes:
            reasons.append("requested node count exceeds MaxNodes")
        maximum, requested = duration(policy.max_time), duration(job.time_limit)
        if maximum is not None and requested is not None and requested > maximum:
            reasons.append("requested time exceeds MaxTime")
        for value, allowed, label in (
            (job.account, policy.accounts, "account"),
            (job.qos, policy.qos, "QoS"),
        ):
            if allowed and allowed.upper() not in {"ALL", "(NULL)"}:
                if value and value not in allowed.split(","):
                    reasons.append(f"{label} is outside the visible allow list")
                elif not value:
                    unknown.append(f"{label} is unknown")
        if policy.groups.upper() not in {"ALL", "(NULL)"}:
            unknown.append("group membership is not evaluated")
    else:
        unknown.append("partition policy is unavailable")
    if not snapshot.partitions_at or snapshot.collected_at - snapshot.partitions_at > 60:
        unknown.append("partition policy may be stale")
    active = [node for node in nodes if node.usable]
    if job.nodes is None:
        unknown.append("requested node count is unknown")
    elif len(active) < job.nodes:
        reasons.append("too few active visible nodes")
    if job.cpus is None:
        unknown.append("requested CPU count is unknown")
    else:
        capacity = known_sum(
            [
                max(0, node.cpus - node.used_cpus)
                if node.cpus is not None and node.used_cpus is not None
                else None
                for node in active
            ]
        )
        if capacity is None:
            unknown.append("free CPU counts are unknown")
        elif job.cpus > capacity:
            reasons.append("insufficient visible free CPUs")
    free = known_sum([node.free_gpus for node in active])
    if job.requested_gpus is None or free is None:
        unknown.append("GPU request or free counts are unknown")
    elif job.requested_gpus > free:
        reasons.append("insufficient visible free GPUs")
    for kind, count in job.requested_types.items():
        free_type = known_sum([node.free_type(kind) for node in active])
        if free_type is None:
            unknown.append(f"free {kind} counts are unknown")
        elif count > free_type:
            reasons.append(f"insufficient visible free {kind} GPUs")
    if job.gpu_per_node:
        candidates = []
        for node in active:
            if node.free_gpus is None:
                unknown.append("per-node GPU allocation is unknown")
                continue
            if any(node.free_type(kind) is None for kind in job.gpu_per_node if kind):
                unknown.append("per-node GPU type allocation is unknown")
                continue
            if all(
                (node.free_gpus if not kind else node.free_type(kind)) >= count
                for kind, count in job.gpu_per_node.items()
            ):
                candidates.append(node)
        if (
            job.nodes is not None
            and len(candidates) < job.nodes
            and not any("per-node GPU" in reason for reason in unknown)
        ):
            reasons.append("too few nodes satisfy the per-node GPU request")
    if job.memory_per_node is not None:
        eligible = [
            node
            for node in active
            if node.memory is not None
            and node.used_memory is not None
            and node.memory - node.used_memory >= job.memory_per_node
        ]
        if any(node.memory is None or node.used_memory is None for node in active):
            unknown.append("free node memory is unknown")
        elif job.nodes is not None and len(eligible) < job.nodes:
            reasons.append("too few nodes satisfy the per-node memory request")
    else:
        unknown.append("memory request/distribution is not fully known")
    if job.memory_total is not None:
        capacity = known_sum(
            [
                max(0, node.memory - node.used_memory)
                if node.memory is not None and node.used_memory is not None
                else None
                for node in active
            ]
        )
        if capacity is not None and job.memory_total > capacity:
            reasons.append("insufficient visible free memory")
    if job.features or job.task_resources:
        unknown.append("feature expressions and per-task placement require scheduler evaluation")
    if reasons:
        return "Insufficient visible resources/policy", reasons + unknown
    incomplete = any(
        "unknown" in reason
        or "unavailable" in reason
        or "stale" in reason
        or "not fully known" in reason
        for reason in unknown
    )
    return (
        "Unknown (partial capacity checks)"
        if incomplete
        else "Potential resource fit; scheduling unknown"
    ), unknown


@dataclass
class SendResult:
    success: bool
    retry: bool = True
    delay: float = 0
    error: str = ""


def validate_webhook(value: str) -> None:
    try:
        url = urllib.parse.urlsplit(value)
        valid = (
            url.scheme == "https"
            and url.hostname in {"hooks.slack.com", "hooks.slack-gov.com"}
            and url.path.startswith("/services/")
            and not url.username
            and not url.password
            and url.port in {None, 443}
            and not url.query
            and not url.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Webhook must be an HTTPS Slack incoming-webhook URL (value redacted).")


def load_webhook(cli_value: Optional[str] = None) -> Optional[str]:
    if cli_value:
        return cli_value
    if os.environ.get("SLACK_WEBHOOK_URL"):
        return os.environ["SLACK_WEBHOOK_URL"]
    for path in [
        Path.cwd() / ".env",
        Path.home() / ".slurm-monitor.env",
        Path(__file__).parent / ".env",
    ]:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            raise ValueError(
                "Cannot read a Slack configuration file (contents redacted)."
            ) from None
        for line in lines:
            match = re.match(r"^\s*(?:export\s+)?SLACK_WEBHOOK_URL\s*=\s*(.*)$", line)
            if match:
                try:
                    values = shlex.split(match[1], comments=True)
                except ValueError:
                    raise ValueError(
                        "Malformed Slack configuration quoting (contents redacted)."
                    ) from None
                if len(values) != 1:
                    raise ValueError("Slack configuration requires one URL (contents redacted).")
                return values[0]
    return None


def slack_text(value: object) -> str:
    return clean(value, 240).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def event_line(event: dict) -> str:
    job = event["job"]
    gpu = "?" if job.get("gpus") is None else str(job["gpus"])
    exit_info = (
        f"; exit {slack_text(job.get('exit_code') or '?')}, derived {slack_text(job.get('derived_exit_code') or '?')}"
        if event["kind"] != "started"
        else ""
    )
    return (
        f"{event['kind'].upper()}: {slack_text(job['id'])} {slack_text(job.get('name', ''))} "
        f"({slack_text(job.get('partition', ''))}; {gpu} GPUs{exit_info})"
    )


def send_slack_notification(webhook_url: str, message: str) -> SendResult:
    """No secrets in errors. Slack has no webhook idempotency; delivery is at least once."""
    try:
        validate_webhook(webhook_url)
        payload = json.dumps({"text": message, "mrkdwn": False}).encode("utf-8")
        request = urllib.request.Request(
            webhook_url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status == 200 and response.read(32).strip() == b"ok":
                return SendResult(True)
            return SendResult(False, False, error="Slack returned an unexpected response")
    except urllib.error.HTTPError as error:
        try:
            delay = min(86400, max(1, float(error.headers.get("Retry-After", "5"))))
            if not math.isfinite(delay):
                delay = 5
        except (ValueError, TypeError, AttributeError):
            delay = 5
        return SendResult(
            False,
            error.code == 429 or error.code >= 500,
            delay,
            f"Slack HTTP {error.code} (URL redacted)",
        )
    except (urllib.error.URLError, TimeoutError, OSError):
        return SendResult(False, error="Slack transport failure; event retained")
    except ValueError:
        return SendResult(False, False, error="Invalid Slack URL (value redacted)")


class JobTracker:
    """State transitions and an atomic, private delivery journal, independent of the UI."""

    def __init__(
        self,
        state_file: Optional[Path] = None,
        notify: bool = False,
        clock: Callable[[], float] = time.time,
    ):
        self.path = state_file
        self.notify = notify
        self.clock = clock
        self.lock = threading.RLock()
        self.file_lock = None
        self.previous: dict[str, Job] = {}
        self.awaiting: dict[str, Job] = {}
        self.history: list[Job] = []
        self.outbox: list[dict] = []
        self.initialized = False
        self.error = ""
        self.ready = True
        if state_file:
            self._open()

    def _open(self) -> None:
        assert self.path is not None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.path.with_suffix(self.path.suffix + ".lock")
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            self.file_lock = os.fdopen(descriptor, "a+")
            os.fchmod(self.file_lock.fileno(), 0o600)
            fcntl.flock(self.file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.is_symlink():
                raise StateError("Refusing a symlink delivery journal.")
            if self.path.exists():
                os.chmod(self.path, 0o600)
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if payload.get("version") != 1:
                    raise StateError(
                        "Unsupported delivery journal version; existing file preserved."
                    )
                self.previous = {key: Job(**value) for key, value in payload["previous"].items()}
                self.awaiting = {key: Job(**value) for key, value in payload["awaiting"].items()}
                self.history = [Job(**value) for value in payload["history"]]
                self.outbox = payload["outbox"]
                self.initialized = bool(payload["initialized"])
                if not isinstance(payload["initialized"], bool):
                    raise StateError("Malformed delivery journal; existing file preserved.")
                for event in self.outbox:
                    if (
                        not isinstance(event, dict)
                        or not {"id", "kind", "job", "attempts", "next_attempt"} <= event.keys()
                    ):
                        raise StateError("Malformed delivery journal; existing file preserved.")
                    Job(**event["job"])
                    if event["kind"] not in {"started", "completed"} | {
                        state.lower() for state in TERMINAL
                    }:
                        raise StateError("Malformed delivery journal; existing file preserved.")
                    if (
                        not isinstance(event["attempts"], int)
                        or event["attempts"] < 0
                        or not isinstance(event["id"], str)
                    ):
                        raise StateError("Malformed delivery journal; existing file preserved.")
                    deadline = event["next_attempt"]
                    if deadline is not None and (
                        not isinstance(deadline, (int, float)) or not math.isfinite(deadline)
                    ):
                        raise StateError("Malformed delivery journal; existing file preserved.")
                    # A new invocation can correct URL/policy errors; one retry is then allowed.
                    if event["next_attempt"] is None:
                        event["next_attempt"] = self.clock()
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            self.close()
            label = (
                "Journal is locked by another monitor"
                if isinstance(error, BlockingIOError)
                else "Cannot safely read or lock delivery journal"
            )
            raise StateError(label + "; existing file preserved.") from None
        except StateError:
            self.close()
            raise

    def close(self) -> None:
        if self.file_lock:
            self.file_lock.close()
            self.file_lock = None

    def _save(self) -> bool:
        if not self.path:
            return True
        temporary = None
        try:
            payload = dict(
                version=1,
                initialized=self.initialized,
                previous={k: asdict(v) for k, v in self.previous.items()},
                awaiting={k: asdict(v) for k, v in self.awaiting.items()},
                history=[asdict(v) for v in self.history],
                outbox=self.outbox,
            )
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=".slurm-monitor-",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                os.fchmod(output.fileno(), 0o600)
                json.dump(payload, output, ensure_ascii=False)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
            self.ready = True
            if self.error.startswith("Journal write"):
                self.error = ""
            return True
        except (OSError, TypeError, ValueError):
            self.ready = False
            self.error = (
                "Journal write failed: delivery paused; quit only after storage is repaired."
            )
            return False
        finally:
            try:
                if temporary and temporary.exists():
                    temporary.unlink()
            except OSError:
                pass  # Leave the private temporary file for recovery if storage fails.

    @property
    def tracked_ids(self) -> set[str]:
        with self.lock:
            return set(self.previous) | set(self.awaiting)

    @property
    def tracked_jobs(self) -> dict[str, Job]:
        with self.lock:
            return {**self.previous, **self.awaiting}

    def _enqueue(self, job: Job, kind: str) -> None:
        if not self.notify:
            return
        token = f"{job.id}|{job.submitted}|{job.restart_count}|{kind}|{job.state}|{job.exit_code}"
        identifier = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if any(event["id"] == identifier for event in self.outbox):
            return
        self.outbox.append(
            dict(id=identifier, kind=kind, job=asdict(job), attempts=0, next_attempt=self.clock())
        )

    def observe(self, snapshot: Snapshot, jobs: list[Job]) -> None:
        with self.lock:
            if not snapshot.jobs_ok:
                return  # Failed queries cannot alter the baseline or emit endings.
            current = {job.id: job for job in jobs if not job.terminal}
            if not self.initialized:
                self.previous = current
                self.initialized = True
                self._save()
                return  # Snooze initial running jobs, including after a fresh installation.
            for jid, job in current.items():
                previous = self.previous.get(jid) or self.awaiting.get(jid)
                if (
                    previous
                    and previous.submitted
                    and job.submitted
                    and previous.submitted != job.submitted
                ):
                    self.awaiting.setdefault(jid, previous)
                    previous = None  # Numeric job ID reuse is a new generation.
                if job.state == "RUNNING" and (
                    previous is None
                    or previous.state in {"PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_HOLD"}
                ):
                    self._enqueue(job, "started")
                pending = self.awaiting.get(jid)
                if (
                    pending is None
                    or not pending.submitted
                    or not job.submitted
                    or pending.submitted == job.submitted
                ):
                    self.awaiting.pop(jid, None)
            for jid, job in self.previous.items():
                if jid not in current and "[" not in jid:
                    self.awaiting.setdefault(jid, job)
            for record in snapshot.accounting:
                if not record.terminal or record.id not in self.awaiting:
                    continue
                expected = self.awaiting[record.id]
                if expected.submitted and expected.submitted != record.submitted:
                    continue
                previous = self.awaiting.pop(record.id)
                ended = Job(
                    **{
                        **asdict(previous),
                        **{
                            k: v
                            for k, v in asdict(record).items()
                            if k
                            in {
                                "state",
                                "runtime",
                                "gpus",
                                "gpu_types",
                                "exit_code",
                                "derived_exit_code",
                            }
                        },
                    }
                )
                self.history = ([ended] + [job for job in self.history if job.id != ended.id])[:100]
                self._enqueue(
                    ended, "completed" if ended.state == "COMPLETED" else ended.state.lower()
                )
            self.previous = current
            self._save()  # Persist transitions and events before any HTTP request.

    def deliver(self, sender: Callable[[str], SendResult]) -> bool:
        with self.lock:
            if not self.ready and not self._save():
                return False
            eligible = [
                event
                for event in self.outbox
                if event["next_attempt"] is not None and event["next_attempt"] <= self.clock()
            ]
            selected = []
            lines = []
            for event in eligible:
                line = event_line(event)
                if sum(len(v) + 1 for v in lines) + len(line) > 3500:
                    break
                selected.append(event)
                lines.append(line)
            if not selected:
                return False
            # Journal was saved by observe/load; in-memory events need no persistence.
            if not self._save():
                return False
        try:
            result = sender("\n".join(lines))
        except Exception:
            result = SendResult(False, error="Slack delivery failed; event retained")
        with self.lock:
            ids = {event["id"] for event in selected}
            if result.success:
                self.outbox = [event for event in self.outbox if event["id"] not in ids]
                self.error = ""
            else:
                self.error = result.error
                for event in self.outbox:
                    if event["id"] in ids:
                        event["attempts"] += 1
                        event["next_attempt"] = (
                            (
                                self.clock()
                                + max(
                                    result.delay, min(300, 5 * 2 ** min(event["attempts"] - 1, 6))
                                )
                            )
                            if result.retry
                            else None
                        )
            self._save()
        return True

    def status(self) -> tuple[int, int, str]:
        with self.lock:
            return len(self.outbox), len(self.awaiting), self.error


class DeliveryWorker:
    def __init__(self, tracker: JobTracker, sender: Callable[[str], SendResult]):
        self.tracker = tracker
        self.sender = sender
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="slurm-monitor-slack", daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            attempted = self.tracker.deliver(self.sender)
            self.stop.wait(1 if attempted else 0.5)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> bool:
        self.stop.set()
        self.thread.join(timeout=11)
        return not self.thread.is_alive()


def fmt_count(value: Optional[int]) -> str:
    return "?" if value is None else str(value)


def visible_jobs(
    snapshot: Snapshot,
    user: str,
    show_all: bool = False,
    partition: Optional[str] = None,
    job_id: Optional[str] = None,
) -> list[Job]:
    return [
        job
        for job in snapshot.jobs
        if (show_all or job.user == user or job_id == job.id)
        and (not partition or partition in job.partition.split(","))
        and (not job_id or job.id == job_id)
    ]


def job_table(jobs: list[Job], limit: int, narrow: bool = False) -> Table:
    shown = jobs if limit == 0 else jobs[:limit]
    title = f"Jobs: {len(shown)}/{len(jobs)} shown"
    if len(shown) < len(jobs):
        title += f" ({len(jobs) - len(shown)} hidden)"
    table = Table(title=literal(title), expand=True, show_lines=False)
    for name in (
        ["ID", "Name", "State", "GPU", "Reason / exit"]
        if narrow
        else ["ID", "Name", "User", "Partition", "State", "GPUs", "Runtime", "Reason / exit"]
    ):
        table.add_column(name, overflow="ellipsis", no_wrap=True)
    for job in shown:
        info = (
            job.reason
            if job.state == "PENDING"
            else (f"{job.exit_code} / {job.derived_exit_code}" if job.terminal else "")
        )
        gpu = fmt_count(job.requested_gpus if job.state == "PENDING" else job.gpus)
        style = (
            "yellow"
            if job.state == "PENDING"
            else "green"
            if job.state == "RUNNING"
            else "red"
            if job.terminal and job.state != "COMPLETED"
            else "cyan"
        )
        row = (
            [
                literal(job.id),
                literal(job.name),
                literal(job.state, style),
                literal(gpu),
                literal(info),
            ]
            if narrow
            else [
                literal(job.id),
                literal(job.name),
                literal(job.user),
                literal(job.partition),
                literal(job.state, style),
                literal(gpu),
                literal(job.runtime),
                literal(info),
            ]
        )
        table.add_row(*row)
    if not shown:
        table.add_row(
            *[literal("No visible jobs" if i == 1 else "-") for i in range(len(table.columns))]
        )
    return table


def gpu_legend() -> Text:
    legend = Text()
    for text, style in (
        ("█ U=Used  ", "bold bright_red"),
        ("░ I=Idle*  ", "bright_green"),
        ("× X=Unavailable  ", "bright_black"),
        ("?=Unknown", "bright_yellow"),
    ):
        legend.append(text, style)
    return legend


def gpu_meter(
    total: Optional[int],
    used: Optional[int],
    idle: Optional[int],
    width: int = 20,
    stacked: bool = False,
) -> Text:
    """Proportional inventory, with unknown capacity never painted as idle."""
    width = max(1, width)
    invalid = any(value is not None and value < 0 for value in (total, used, idle))
    known = (used or 0) + (idle or 0)
    invalid |= total is not None and known > total
    unavailable = (
        total - known
        if total is not None and used is not None and idle is not None and not invalid
        else None
    )
    bar = Text()
    if invalid or total is None:
        bar.append("?" * width, "bright_yellow")
    elif total == 0:
        bar.append("No GPUs", "dim")
    else:
        counts = [
            used or 0,
            idle or 0,
            unavailable or 0,
            total - known if unavailable is None else 0,
        ]
        quotas = [count * width / total for count in counts]
        cells = [int(quota) for quota in quotas]
        for i in sorted(range(4), key=lambda i: quotas[i] - cells[i], reverse=True)[
            : width - sum(cells)
        ]:
            cells[i] += 1
        # Keep small nonzero segments visible when there are enough cells.
        if width >= sum(count > 0 for count in counts):
            for i, count in enumerate(counts):
                if count and not cells[i]:
                    donor = max(range(4), key=lambda j: cells[j])
                    cells[donor] -= 1
                    cells[i] = 1
        for cells_count, symbol, style in zip(
            cells,
            ("█", "░", "×", "?"),
            ("bold bright_red", "bright_green", "bright_black", "bright_yellow"),
        ):
            bar.append(symbol * cells_count, style)
    bar.append("\n" if stacked else "  ")
    bar.append(
        f"U{fmt_count(used)} I{fmt_count(idle)} X{fmt_count(unavailable)} /{fmt_count(total)}",
        "dim",
    )
    if invalid:
        bar.append(" (inconsistent)", "yellow")
    return bar


def gpu_table(
    snapshot: Snapshot,
    user: str,
    limit: int = 0,
    jobs: Optional[list[Job]] = None,
    width: int = 100,
) -> Table:
    narrow = width < 75
    table = Table(
        title="Partition GPUs (shared nodes may overlap)",
        caption=gpu_legend(),
        expand=True,
    )
    table.add_column(
        "Partition",
        max_width=20 if not narrow else max(8, width // 3),
        overflow="ellipsis",
        no_wrap=True,
    )
    if not narrow:
        for name in ["Run", "Pend", "Mine"] if width >= 100 else ["R/P", "Mine"]:
            table.add_column(name, justify="right", no_wrap=True)
    table.add_column("GPU allocation", ratio=2, overflow="fold")
    records = gpu_summary(snapshot, user)
    if limit:
        records.sort(key=lambda record: (record["total"] == 0, record["name"]))
    for record in records[: limit or None]:
        partition_jobs = [
            job
            for job in (jobs if jobs is not None else snapshot.jobs)
            if record["name"] in job.partition.split(",")
        ]
        row = [literal(record["name"])]
        if not narrow:
            running = sum(job.state == "RUNNING" for job in partition_jobs)
            pending = sum(job.state == "PENDING" for job in partition_jobs)
            row += (
                [literal(running), literal(pending)]
                if width >= 100
                else [literal(f"{running}/{pending}")]
            )
            row.append(literal(fmt_count(record["mine"]), "cyan"))
        row.append(
            gpu_meter(
                record["total"],
                record["used"],
                record["free"],
                10 if narrow else 20 if width < 100 else 24,
                narrow,
            )
        )
        table.add_row(*row)
    if limit and len(records) > limit:
        table.add_row(
            literal(f"{len(records) - limit} more"),
            *[literal("…") for _ in table.columns[1:]],
        )
    return table


def node_table(snapshot: Snapshot, partition: str, limit: int = 0, width: int = 100) -> Table:
    nodes = [node for node in snapshot.nodes if partition in node.partitions]
    narrow = width < 75
    table = Table(
        title=literal(f"{partition}: {len(nodes)} nodes"), caption=gpu_legend(), expand=True
    )
    table.add_column(
        "Node / state" if narrow else "Node",
        max_width=max(10, width // 3),
        overflow="ellipsis",
        no_wrap=True,
    )
    if not narrow:
        table.add_column("State", max_width=18, overflow="ellipsis", no_wrap=True)
        table.add_column("CPU used/total", no_wrap=True)
    table.add_column("GPU allocation", ratio=2, overflow="fold")
    for node in nodes[: limit or None]:
        row = [
            Text.assemble(literal(node.name), "\n", literal("+".join(node.states)))
            if narrow
            else literal(node.name)
        ]
        if not narrow:
            row += [
                literal("+".join(node.states)),
                literal(f"{fmt_count(node.used_cpus)}/{fmt_count(node.cpus)}"),
            ]
        row.append(
            gpu_meter(node.gpus, node.used_gpus, node.free_gpus, 10 if narrow else 16, narrow)
        )
        table.add_row(*row)
    if limit and len(nodes) > limit:
        table.add_row(
            literal(f"{len(nodes) - limit} hidden"),
            *[literal("…") for _ in table.columns[1:]],
        )
    return table


def diagnostic_panel(
    job: Job, snapshot: Snapshot, partition: Optional[str], concise: bool = False
) -> Panel:
    lines = [f"Job {job.id}: {job.name} ({job.state})"]
    if job.terminal:
        lines += [
            f"Runtime: {job.runtime}; batch exit: {job.exit_code or 'unknown'}; derived step exit: {job.derived_exit_code or 'unknown'}",
            "Exit format is exit-code:signal. Derived exit can reveal a failed step even when the batch script returned zero.",
        ]
    elif job.state == "PENDING":
        lines += [
            f"Reason: {job.reason or 'unknown'}. {pending_explanation(job.reason)}",
            f"Request: {fmt_count(job.nodes)} nodes, {fmt_count(job.cpus)} CPUs, {fmt_count(job.requested_gpus)} GPUs; "
            f"GPU types: {job.requested_types or 'unspecified/unknown'}; memory {fmt_count(job.memory_total)} MiB total / "
            f"{fmt_count(job.memory_per_node)} MiB per node; time limit: {job.time_limit or 'unknown'}.",
        ]
        candidates = [partition] if partition else [v for v in job.partition.split(",") if v]
        for name in candidates[:6]:
            status, reasons = resource_suitability(job, snapshot, name)
            lines.append(f"{name}: {status}. " + "; ".join(reasons))
        if len(candidates) > 6:
            lines.append(f"{len(candidates) - 6} additional partitions omitted; use --partition.")
        lines.append(
            "Resource suitability is not a scheduler guarantee. No job is submitted or modified."
        )
    else:
        lines += [
            f"Allocated GPUs: {fmt_count(job.gpus)}; runtime: {job.runtime or 'unknown'}",
            "GPU counts represent SLURM allocations, not measured device utilization.",
        ]
    if concise:
        lines = lines[:3] + [
            "Scheduling is not guaranteed; unknown constraints may apply.",
            "Use --once --job ID for full diagnostics.",
        ]
    return Panel(
        Text("\n".join(clean(line, 100 if concise else 1200) for line in lines)),
        title="Job diagnostics",
    )


def render_snapshot(
    snapshot: Snapshot,
    tracker: JobTracker,
    args: argparse.Namespace,
    user: str,
    width: int,
    height: int,
    terminal: bool = False,
) -> Group:
    jobs = visible_jobs(
        snapshot, user, args.all_users, args.partition if not args.job else None, args.job
    )
    jobs = sorted(jobs, key=lambda job: (job.state != "PENDING", job.id))
    pending, waiting, delivery_error = tracker.status()
    errors = list(snapshot.errors) + ([delivery_error] if delivery_error else [])
    if not snapshot.jobs_ok:
        errors.append(
            f"Jobs STALE: last successful snapshot age {max(0, snapshot.collected_at - snapshot.jobs_at):.0f}s"
        )
    if not snapshot.nodes_ok:
        errors.append(
            f"Nodes STALE: last successful snapshot age {max(0, snapshot.collected_at - snapshot.nodes_at):.0f}s"
        )
    total = known_sum([node.gpus for node in snapshot.nodes])
    used = known_sum([node.used_gpus for node in snapshot.nodes])
    header = f"SLURM Monitor | {'all users' if args.all_users else user} | GPUs allocated {fmt_count(used)}/{fmt_count(total)} (unique nodes)"
    footer = f"Refresh: {args.interval:g}s | {datetime.fromtimestamp(snapshot.collected_at).strftime('%H:%M:%S')} | Slack queued: {pending} | endings awaiting accounting: {waiting} | Ctrl+C exits"
    cap = args.max_jobs or 0
    parts = [Panel(literal(header), style="cyan")]
    if errors:
        parts.append(
            Panel(
                Text("\n".join(clean(error) for error in errors)),
                title="Errors / stale data",
                style="red",
            )
        )
    # Each section supplies its actual Rich table; row height can vary with width.
    sections = [
        (min(len(jobs), cap) if cap else len(jobs), lambda rows: job_table(jobs, rows, width < 100))
    ]
    inventory = []
    if args.partition and not args.compact:
        nodes = [node for node in snapshot.nodes if args.partition in node.partitions]
        sections.append(
            (
                min(len(nodes), cap) if cap else len(nodes),
                lambda rows: node_table(snapshot, args.partition, rows, width),
            )
        )
    elif not args.compact and width >= 40:
        inventory = gpu_summary(snapshot, user)
        sections.append((len(inventory), lambda rows: gpu_table(snapshot, user, rows, jobs, width)))
    bottom = []
    history = []
    if args.job:
        selected = next((job for job in jobs if job.id == args.job), None)
        if selected is None:
            selected = next((job for job in snapshot.accounting if job.id == args.job), None)
        if selected:
            accounting = next(
                (
                    record
                    for record in snapshot.accounting
                    if record.id == selected.id and record.terminal
                ),
                None,
            )
            if selected.terminal and accounting:
                selected = Job(
                    **{
                        **asdict(selected),
                        **{
                            key: getattr(accounting, key)
                            for key in ("exit_code", "derived_exit_code", "runtime")
                        },
                    }
                )
            bottom.append(
                diagnostic_panel(selected, snapshot, args.partition, terminal and height < 35)
            )
        else:
            bottom.append(
                Panel(
                    literal("Job not visible; accounting may be delayed, disabled or restricted."),
                    title="Job diagnostics",
                )
            )
    else:
        with tracker.lock:
            history = tracker.history[:3]
        if history:
            sections.append((len(history), lambda rows: job_table(history, rows, width < 100)))
    bottom += [
        literal(
            "* Free = unallocated GPUs on active nodes. CPU/memory, reservations and partition policy can still prevent scheduling. ? = unknown.",
            "dim",
        ),
        literal(footer, "dim"),
    ]
    if not terminal:
        return Group(*parts, *(factory(count) for count, factory in sections), *bottom)

    console = Console(width=max(1, width), height=max(1, height))
    options = console.options.update(height=None)

    def measure(value) -> int:
        return len(console.render_lines(value, options, pad=False))

    fixed_height = sum(measure(value) for value in [*parts, *bottom])
    maximum = [min(count, max(1, height)) for count, _ in sections]
    rows = list(maximum)
    cache = [{} for _ in sections]

    def section_height(index: int, count: int) -> int:
        if count not in cache[index]:
            cache[index][count] = measure(sections[index][1](count))
        return cache[index][count]

    def total_height(counts: list[int]) -> int:
        return fixed_height + sum(section_height(i, count) for i, count in enumerate(counts))

    # Start with all eligible rows and trim the tallest section only when necessary.
    while total_height(rows) > height:
        candidates = [i for i, count in enumerate(rows) if count > 1]
        if not candidates:
            break
        index = max(candidates, key=lambda i: section_height(i, rows[i]))
        rows[index] -= 1
    if total_height(rows) <= height:
        # Reclaim slack caused by wrapped headers or disappearance of a "more" row.
        while True:
            grew = False
            for i, maximum_rows in enumerate(maximum):
                if rows[i] >= maximum_rows:
                    continue
                for candidate in dict.fromkeys([rows[i] + 1, maximum_rows]):
                    trial = list(rows)
                    trial[i] = candidate
                    if total_height(trial) <= height:
                        rows = trial
                        grew = True
                        break
            if not grew:
                break
        return Group(
            *parts, *(factory(count) for (_, factory), count in zip(sections, rows)), *bottom
        )

    # If even one row per table cannot fit, use a bounded overview instead of clipping controls.
    overview = [
        literal(f"SLURM Monitor | GPUs {fmt_count(used)}/{fmt_count(total)}"),
        literal(f"Jobs {len(jobs)} | {len(errors)} errors | Slack queued {pending}"),
        literal("█ used  ░ idle  × unavailable  ? unknown", "dim"),
    ]
    if errors:
        overview.append(literal("Error: " + errors[0], "red"))
    previews = []
    for record in inventory:
        previews.append(
            Text.assemble(
                literal(record["name"] + " "),
                gpu_meter(record["total"], record["used"], record["free"], 6),
            )
        )
    previews += [literal(f"{job.id} {job.state} {job.name}") for job in jobs[: cap or None]]
    previews += [literal(f"Recent: {job.id} {job.state} {job.name}") for job in history]
    if args.partition and not args.compact:
        previews += [
            Text.assemble(
                literal(node.name + " "), gpu_meter(node.gpus, node.used_gpus, node.free_gpus, 6)
            )
            for node in nodes[: cap or None]
        ]
    available = max(0, height - len(overview) - 1)
    shown = available if len(previews) <= available else max(0, available - 1)
    overview += previews[:shown]
    if shown < len(previews) and available:
        overview.append(literal(f"{len(previews) - shown} rows hidden; --once shows full tables"))
    overview.append(literal(f"Refresh: {args.interval:g}s | awaiting sacct: {waiting} | Ctrl+C"))
    overview = overview[: max(0, height)]
    for line in overview:
        line.truncate(max(1, width), overflow="ellipsis")
    return Group(*overview)


def positive_interval(value: str) -> float:
    try:
        result = float(value)
        if not math.isfinite(result) or result < 1:
            raise ValueError
        return result
    except ValueError:
        raise argparse.ArgumentTypeError(
            "interval must be a finite number of at least 1 second"
        ) from None


def nonnegative(value: str) -> int:
    try:
        result = int(value)
        if result < 0:
            raise ValueError
        return result
    except ValueError:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from None


def job_identifier(value: str) -> str:
    if not re.fullmatch(r"\d+(?:[_+]\d+)?", value):
        raise argparse.ArgumentTypeError(
            "job ID must be numeric, optionally with an array/heterogeneous suffix"
        )
    return value


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only SLURM job and GPU monitor")
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument(
        "--interval",
        "-i",
        type=positive_interval,
        default=5,
        help="Refresh seconds (at least 1; default 5)",
    )
    parser.add_argument("--all-users", "-a", action="store_true")
    parser.add_argument("--once", "-1", action="store_true")
    parser.add_argument("--compact", "-c", action="store_true")
    parser.add_argument("--partition", "-p", help="Node details / assessment partition")
    parser.add_argument(
        "--job",
        type=job_identifier,
        help="Inspect pending suitability or completed exit information",
    )
    parser.add_argument(
        "--max-jobs",
        type=nonnegative,
        default=None,
        help="Optional job/node row cap; default or 0 fits the live viewport; --once shows all",
    )
    parser.add_argument(
        "--json", action="store_true", help="One read-only machine-readable snapshot"
    )
    parser.add_argument("--slack", "-s", action="store_true")
    parser.add_argument(
        "--slack-webhook",
        help="Override Slack URL; prefer the environment to avoid shell-history exposure",
    )
    parser.add_argument(
        "--state-file", type=Path, help="Private Slack journal; default is scoped by host/user/view"
    )
    return parser.parse_args(argv)


def default_state_file(args: argparse.Namespace, user: str) -> Path:
    scope = json.dumps([socket.gethostname(), user, args.all_users, args.partition, args.job])
    token = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:16]
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / "slurm-monitor" / f"{token}.json"


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    console = Console()
    user = getpass.getuser()
    webhook = None
    tracker = None
    worker = None
    try:
        if args.slack:
            webhook = load_webhook(args.slack_webhook)
            if not webhook:
                raise ValueError("--slack requires SLACK_WEBHOOK_URL or --slack-webhook")
            validate_webhook(webhook)
        tracker = JobTracker(
            args.state_file or default_state_file(args, user) if webhook else None,
            notify=bool(webhook),
        )
        collector = Collector()
        collector.user = None if args.all_users or args.job else user
        if webhook:
            worker = DeliveryWorker(
                tracker, lambda message: send_slack_notification(webhook, message)
            )
            worker.start()

        def collect() -> Snapshot:
            snapshot = collector.collect(tracker.tracked_jobs, args.job)
            jobs = visible_jobs(
                snapshot, user, args.all_users, args.partition if not args.job else None, args.job
            )
            tracker.observe(snapshot, jobs)
            return snapshot

        snapshot = collect()
        if args.once or args.json or not console.is_terminal:
            if args.json:
                payload = asdict(snapshot)
                payload["jobs"] = [
                    asdict(job)
                    for job in visible_jobs(
                        snapshot,
                        user,
                        args.all_users,
                        args.partition if not args.job else None,
                        args.job,
                    )
                ]
                print(json.dumps(payload, ensure_ascii=False))
            else:
                console.print(
                    render_snapshot(snapshot, tracker, args, user, console.width, console.height)
                )
            return 0 if snapshot.jobs_ok and snapshot.nodes_ok and not snapshot.errors else 1
        with Live(
            render_snapshot(snapshot, tracker, args, user, console.width, console.height, True),
            console=console,
            refresh_per_second=1,
            screen=True,
            vertical_overflow="ellipsis",
        ) as live:
            while True:
                time.sleep(args.interval)
                snapshot = collect()
                live.update(
                    render_snapshot(
                        snapshot, tracker, args, user, console.width, console.height, True
                    )
                )
    except KeyboardInterrupt:
        console.print(
            literal(
                "Monitor stopped; queued Slack events remain in the delivery journal.", "yellow"
            )
        )
        return 0
    except (ValueError, StateError) as error:
        console.print(literal(str(error), "red"))
        return 2
    finally:
        stopped = worker.close() if worker else True
        if tracker and stopped:
            tracker.close()
        if not stopped:
            console.print(
                literal(
                    "Delivery shutdown timed out; pending journal retained. A request may have reached Slack.",
                    "yellow",
                )
            )


if __name__ == "__main__":
    raise SystemExit(main())
