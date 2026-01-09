#!/usr/bin/env python3
"""
SLURM Job Monitor - Real-time terminal visualization for SLURM jobs
Usage: slurm-monitor [--interval SECONDS] [--all-users]
"""

import subprocess
import argparse
import time
import os
import sys
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
except ImportError:
    print("Error: 'rich' library not installed.")
    print("Install it with: pip install rich")
    sys.exit(1)


def run_command(cmd: str) -> str:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def get_jobs(user: str = None) -> list[dict]:
    """Get SLURM jobs with detailed information."""
    user_filter = "" if user is None else f"-u {user}"
    cmd = f"squeue {user_filter} -o '%i|%j|%u|%P|%T|%M|%l|%D|%R|%b' --noheader"
    output = run_command(cmd)

    jobs = []
    if output and not output.startswith("Error"):
        for line in output.split('\n'):
            if line.strip():
                parts = line.split('|')
                if len(parts) >= 10:
                    jobs.append({
                        'id': parts[0].strip(),
                        'name': parts[1].strip()[:30],
                        'user': parts[2].strip(),
                        'partition': parts[3].strip(),
                        'state': parts[4].strip(),
                        'time': parts[5].strip(),
                        'time_limit': parts[6].strip(),
                        'nodes': parts[7].strip(),
                        'reason': parts[8].strip()[:25],
                        'gres': parts[9].strip(),
                    })
    return jobs


def get_cluster_summary() -> dict:
    """Get cluster resource summary."""
    # Get partition info
    cmd = "sinfo -o '%P|%a|%D|%C' --noheader"
    output = run_command(cmd)

    partitions = []
    total_nodes = 0
    total_cpus_alloc = 0
    total_cpus = 0

    if output and not output.startswith("Error"):
        for line in output.split('\n'):
            if line.strip():
                parts = line.split('|')
                if len(parts) >= 4:
                    name = parts[0].strip().rstrip('*')
                    state = parts[1].strip()
                    nodes = parts[2].strip()
                    # CPU format: allocated/idle/other/total
                    cpu_info = parts[3].strip().split('/')
                    if len(cpu_info) == 4:
                        alloc, idle, other, total = map(int, cpu_info)
                        total_cpus_alloc += alloc
                        total_cpus += total
                    total_nodes += int(nodes) if nodes.isdigit() else 0
                    partitions.append({
                        'name': name,
                        'state': state,
                        'nodes': nodes,
                        'cpus': parts[3].strip()
                    })

    # Get GPU usage
    cmd = "squeue -o '%b' --noheader | grep gpu | sed 's/.*gpu://' | sed 's/,.*//' | awk '{s+=$1} END {print s}'"
    gpus_in_use = run_command(cmd)
    gpus_in_use = int(gpus_in_use) if gpus_in_use.isdigit() else 0

    return {
        'partitions': partitions,
        'total_nodes': total_nodes,
        'cpus_alloc': total_cpus_alloc,
        'cpus_total': total_cpus,
        'gpus_in_use': gpus_in_use
    }


def get_gpu_availability() -> list[dict]:
    """Get GPU availability per partition."""
    cmd = "sinfo -o '%P|%G|%D|%t|%C' --noheader"
    output = run_command(cmd)

    partitions = {}
    if output and not output.startswith("Error"):
        for line in output.split('\n'):
            if line.strip():
                parts = line.split('|')
                if len(parts) >= 5:
                    name = parts[0].strip().rstrip('*')
                    gres = parts[1].strip()
                    nodes = int(parts[2].strip()) if parts[2].strip().isdigit() else 0
                    state = parts[3].strip()

                    # Parse GPU info
                    gpu_count = 0
                    gpu_type = ""
                    if 'gpu:' in gres:
                        gpu_part = gres.split('gpu:')[1].split('(')[0]
                        if ':' in gpu_part:
                            gpu_type, gpu_count = gpu_part.rsplit(':', 1)
                            gpu_count = int(gpu_count) if gpu_count.isdigit() else 0
                        else:
                            gpu_count = int(gpu_part) if gpu_part.isdigit() else 0

                    if name not in partitions:
                        partitions[name] = {'total': 0, 'idle': 0, 'gpu_type': gpu_type}

                    partitions[name]['total'] += nodes * gpu_count
                    if state in ['idle', 'mix']:
                        partitions[name]['idle'] += nodes * gpu_count if state == 'idle' else nodes * gpu_count // 2

    return [{'name': k, **v} for k, v in partitions.items()]


def get_partition_summary(jobs: list[dict]) -> list[dict]:
    """Get summary of running jobs and GPU usage per partition."""
    summary = {}
    for job in jobs:
        if job['state'] != 'RUNNING':
            continue
        partition = job['partition']
        if partition not in summary:
            summary[partition] = {'jobs': 0, 'gpus': 0}
        summary[partition]['jobs'] += 1
        # Parse GPU count from gres
        gres = job['gres']
        if 'gpu:' in gres:
            gpu_str = gres.split('gpu:')[1].split(',')[0].split('(')[0]
            # Handle formats like "gpu:4" or "gpu:A100:4"
            if ':' in gpu_str:
                gpu_count = gpu_str.split(':')[-1]
            else:
                gpu_count = gpu_str
            try:
                summary[partition]['gpus'] += int(gpu_count)
            except ValueError:
                pass
    return [{'partition': k, **v} for k, v in sorted(summary.items())]


def create_summary_table(jobs: list[dict]) -> Table:
    """Create a summary table showing jobs and GPUs per partition."""
    summary = get_partition_summary(jobs)

    table = Table(
        title="Running Summary",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
        expand=True
    )

    table.add_column("Partition", style="green", width=12)
    table.add_column("Jobs", style="yellow", justify="right", width=6)
    table.add_column("GPUs", style="magenta", justify="right", width=6)

    total_jobs = 0
    total_gpus = 0
    for s in summary:
        table.add_row(s['partition'], str(s['jobs']), str(s['gpus']))
        total_jobs += s['jobs']
        total_gpus += s['gpus']

    if summary:
        table.add_section()
        table.add_row("[bold]Total[/]", f"[bold]{total_jobs}[/]", f"[bold]{total_gpus}[/]")
    else:
        table.add_row("-", "0", "0")

    return table


def create_job_table(jobs: list[dict], title: str, state_filter: str = None) -> Table:
    """Create a rich table for jobs."""
    filtered = [j for j in jobs if state_filter is None or j['state'] == state_filter]

    table = Table(
        title=title,
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
        expand=True
    )

    table.add_column("ID", style="yellow", width=10)
    table.add_column("Name", style="white", width=25)
    table.add_column("User", style="blue", width=12)
    table.add_column("Partition", style="green", width=12)
    table.add_column("GPUs", style="magenta", width=8)
    table.add_column("Time", style="cyan", width=12)
    table.add_column("Limit", style="dim", width=12)

    if state_filter == "PENDING":
        table.add_column("Reason", style="red", width=20)

    for job in filtered[:15]:  # Limit to 15 jobs per table
        gpu_info = job['gres'].replace('gpu:', '') if 'gpu' in job['gres'] else '-'
        row = [
            job['id'],
            job['name'],
            job['user'],
            job['partition'],
            gpu_info,
            job['time'],
            job['time_limit'],
        ]
        if state_filter == "PENDING":
            row.append(job['reason'])
        table.add_row(*row)

    if not filtered:
        if state_filter == "PENDING":
            table.add_row("-", "No pending jobs", "-", "-", "-", "-", "-", "-")
        else:
            table.add_row("-", "No running jobs", "-", "-", "-", "-", "-")

    return table


def create_gpu_table(gpu_info: list[dict]) -> Table:
    """Create GPU availability table."""
    table = Table(
        title="GPU Availability",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
    )

    table.add_column("Partition", style="green", width=15)
    table.add_column("Available", style="yellow", justify="right", width=10)
    table.add_column("Total", style="dim", justify="right", width=10)
    table.add_column("Usage", width=20)

    for p in sorted(gpu_info, key=lambda x: x['name']):
        if p['total'] > 0:
            usage_pct = ((p['total'] - p['idle']) / p['total']) * 100 if p['total'] > 0 else 0
            bar_width = 15
            filled = int(usage_pct / 100 * bar_width)
            bar = "[green]" + "█" * (bar_width - filled) + "[red]" + "█" * filled + "[/]"

            avail_style = "green" if p['idle'] > 0 else "red"
            table.add_row(
                p['name'],
                f"[{avail_style}]{p['idle']}[/]",
                str(p['total']),
                bar + f" {usage_pct:.0f}%"
            )

    return table


def create_dashboard(user: str, show_all: bool) -> Layout:
    """Create the main dashboard layout."""
    console = Console()

    # Get data
    jobs = get_jobs(None if show_all else user)
    gpu_info = get_gpu_availability()
    cluster = get_cluster_summary()

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )

    layout["main"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1)
    )

    layout["left"].split_column(
        Layout(name="running", ratio=2),
        Layout(name="summary", size=12),
        Layout(name="pending", ratio=2)
    )

    # Header
    user_display = "All Users" if show_all else user
    header_text = Text()
    header_text.append("  SLURM Job Monitor", style="bold white")
    header_text.append(f"  |  User: {user_display}", style="cyan")
    header_text.append(f"  |  GPUs in use: {cluster['gpus_in_use']}", style="yellow")
    header_text.append(f"  |  {datetime.now().strftime('%H:%M:%S')}", style="dim")
    layout["header"].update(Panel(header_text, style="blue"))

    # Job tables
    running_jobs = [j for j in jobs if j['state'] == 'RUNNING']
    pending_jobs = [j for j in jobs if j['state'] == 'PENDING']

    layout["running"].update(create_job_table(jobs, f"Running Jobs ({len(running_jobs)})", "RUNNING"))
    layout["summary"].update(create_summary_table(jobs))
    layout["pending"].update(create_job_table(jobs, f"Pending Jobs ({len(pending_jobs)})", "PENDING"))

    # GPU availability
    layout["right"].update(create_gpu_table(gpu_info))

    # Footer
    footer_text = Text()
    footer_text.append("  Press Ctrl+C to exit", style="dim")
    footer_text.append("  |  Refresh: 5s", style="dim")
    layout["footer"].update(Panel(footer_text, style="dim"))

    return layout


def create_compact_view(user: str, show_all: bool) -> Table:
    """Create a compact single-table view."""
    jobs = get_jobs(None if show_all else user)

    table = Table(
        title=f"SLURM Jobs ({datetime.now().strftime('%H:%M:%S')})",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True
    )

    table.add_column("ID", style="yellow", width=10)
    table.add_column("Name", style="white", width=30)
    table.add_column("Partition", style="green", width=12)
    table.add_column("State", width=10)
    table.add_column("GPUs", style="magenta", width=8)
    table.add_column("Time", style="cyan", width=12)
    table.add_column("Reason", style="dim", width=20)

    for job in jobs:
        state_style = "green" if job['state'] == 'RUNNING' else "yellow"
        gpu_info = job['gres'].replace('gpu:', '') if 'gpu' in job['gres'] else '-'
        reason = job['reason'] if job['state'] == 'PENDING' else ''

        table.add_row(
            job['id'],
            job['name'],
            job['partition'],
            f"[{state_style}]{job['state']}[/]",
            gpu_info,
            job['time'],
            reason
        )

    if not jobs:
        table.add_row("-", "No jobs found", "-", "-", "-", "-", "-")

    return table


def main():
    parser = argparse.ArgumentParser(description="SLURM Job Monitor")
    parser.add_argument('--interval', '-i', type=int, default=5, help="Refresh interval in seconds")
    parser.add_argument('--all-users', '-a', action='store_true', help="Show jobs from all users")
    parser.add_argument('--once', '-1', action='store_true', help="Run once and exit")
    parser.add_argument('--compact', '-c', action='store_true', help="Compact single-table view")
    args = parser.parse_args()

    user = os.environ.get('USER', 'unknown')
    console = Console()

    if args.once:
        if args.compact:
            console.print(create_compact_view(user, args.all_users))
        else:
            console.print(create_dashboard(user, args.all_users))
        return

    try:
        if args.compact:
            with Live(create_compact_view(user, args.all_users), refresh_per_second=1) as live:
                while True:
                    time.sleep(args.interval)
                    live.update(create_compact_view(user, args.all_users))
        else:
            with Live(create_dashboard(user, args.all_users), refresh_per_second=1, screen=True) as live:
                while True:
                    time.sleep(args.interval)
                    live.update(create_dashboard(user, args.all_users))
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped.[/]")


if __name__ == "__main__":
    main()
