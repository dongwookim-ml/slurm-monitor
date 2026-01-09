# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SLURM Monitor is a single-file Python CLI tool that provides real-time terminal dashboards for monitoring SLURM cluster jobs and GPU availability. It uses the Rich library for terminal UI rendering.

## Commands

```bash
# Install in development mode
pip install -e .

# Run directly
python slurm_monitor.py --once        # Single snapshot
python slurm_monitor.py               # Live dashboard
python slurm_monitor.py -c            # Compact view
python slurm_monitor.py -a            # All users
python slurm_monitor.py -s            # With Slack notifications

# Install to ~/bin for personal use
cp slurm_monitor.py ~/bin/slurm-monitor && chmod +x ~/bin/slurm-monitor
```

## Architecture

The entire application is in `slurm_monitor.py` (~650 lines), organized into sections:

1. **Slack Notification Support** (lines 32-213)
   - `load_env_file()` - Loads webhook URL from `.env` or `~/.slurm-monitor.env`
   - `send_slack_notification()` - HTTP POST to Slack webhook
   - `JobTracker` class - Tracks job state changes between polling cycles, batches multiple events into single notifications

2. **SLURM Commands** (lines 216-337)
   - `run_command()` - Subprocess wrapper for shell commands
   - `get_jobs()` - Parses `squeue` output into job dicts
   - `get_cluster_summary()` - Parses `sinfo` for cluster stats
   - `get_gpu_availability()` - Parses GPU info per partition

3. **UI Components** (lines 340-581)
   - `create_summary_table()` - Per-partition running/pending/GPU counts
   - `create_job_table()` - Running or pending jobs table
   - `create_gpu_table()` - GPU availability with usage bars
   - `create_dashboard()` - Full Rich Layout with all components
   - `create_compact_view()` - Single table view

4. **Main Loop** (lines 584-651)
   - Argument parsing, Rich Live display loop, Slack integration

## Key Patterns

- Jobs are represented as dicts with keys: `id`, `name`, `user`, `partition`, `state`, `time`, `time_limit`, `nodes`, `reason`, `gres`
- GPU counts are parsed from GRES strings like `gpu:4` or `gpu:A100:4`
- JobTracker maintains `previous_jobs` dict to detect state transitions (PENDING→RUNNING, RUNNING→gone)
- Slack notifications are batched per polling cycle and grouped by event type

## Slack Webhook Configuration

Webhook URL is loaded from (in order): `./env`, `~/.slurm-monitor.env`, or `--slack-webhook` flag.
