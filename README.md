# SLURM Monitor

A real-time terminal dashboard for monitoring SLURM cluster jobs and GPU availability.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Real-time job monitoring** - Auto-refreshing display of running and pending jobs
- **Running summary** - Per-partition breakdown of running jobs and GPU usage
- **GPU availability tracking** - Visual representation of GPU usage per partition
- **Slack notifications** - Get notified when jobs start or complete
- **Multiple view modes** - Full dashboard or compact table view
- **User filtering** - Monitor your jobs or all cluster users
- **Rich terminal UI** - Beautiful tables and progress bars using the Rich library

## Screenshots

### Full Dashboard
```
╭──────────────────────────────────────────────────────────────────────────────╮
│   SLURM Job Monitor  |  User: username  |  GPUs in use: 128  |  14:32:15    │
╰──────────────────────────────────────────────────────────────────────────────╯
                  Running Jobs (12)                       GPU Availability
╭───────┬─────────────────┬──────┬─────────┬────┬────────╮╭─────────┬────┬─────╮
│ ID    │ Name            │ User │ Partit… │ …  │ Time   ││ Partit… │ …  │ …   │
├───────┼─────────────────┼──────┼─────────┼────┼────────┤├─────────┼────┼─────┤
│ 12345 │ train_model     │ user │ A100    │ 4  │ 2:30:15││ A100    │ 8  │ 16  │
│ 12346 │ inference       │ user │ RTX3090 │ 2  │ 0:45:22││ RTX3090 │ 12 │ 24  │
└───────┴─────────────────┴──────┴─────────┴────┴────────┘└─────────┴────┴─────┘
                   Running Summary
╭───────────────────────┬─────────────┬─────────────╮
│ Partition             │        Jobs │        GPUs │
├───────────────────────┼─────────────┼─────────────┤
│ A100                  │           8 │          32 │
│ RTX3090               │           4 │           8 │
├───────────────────────┼─────────────┼─────────────┤
│ Total                 │          12 │          40 │
╰───────────────────────┴─────────────┴─────────────╯
```

### Compact View
```
                             SLURM Jobs (14:32:15)
╭────────┬──────────────────────┬──────────┬─────────┬──────┬────────┬─────────╮
│ ID     │ Name                 │ Partition│ State   │ GPUs │ Time   │ Reason  │
├────────┼──────────────────────┼──────────┼─────────┼──────┼────────┼─────────┤
│ 12345  │ train_model          │ A100     │ RUNNING │ 4    │ 2:30:15│         │
│ 12347  │ preprocess           │ RTX3090  │ PENDING │ 1    │ 0:00   │Resources│
╰────────┴──────────────────────┴──────────┴─────────┴──────┴────────┴─────────╯
```

## Installation

### Using pip

```bash
pip install slurm-monitor
```

### From source

```bash
git clone https://github.com/dongwookim-ml/slurm-monitor.git
cd slurm-monitor
pip install -e .
```

### Manual installation

```bash
# Clone and copy to your bin directory
git clone https://github.com/dongwookim-ml/slurm-monitor.git
cp slurm-monitor/slurm_monitor.py ~/bin/slurm-monitor
chmod +x ~/bin/slurm-monitor

# Install dependencies
pip install rich
```

## Usage

```bash
# Start the live dashboard (refreshes every 5 seconds)
slurm-monitor

# Compact table view
slurm-monitor --compact
slurm-monitor -c

# Show all users' jobs
slurm-monitor --all-users
slurm-monitor -a

# Custom refresh interval (in seconds)
slurm-monitor --interval 10
slurm-monitor -i 10

# Run once and exit (no live updates)
slurm-monitor --once
slurm-monitor -1

# Combine options
slurm-monitor -c -a -i 3  # Compact view, all users, 3s refresh
```

### Recommended Bash Aliases

Add these to your `~/.bashrc`:

```bash
alias smon='slurm-monitor'              # Full dashboard
alias smon-all='slurm-monitor -a'       # All users
alias smon-c='slurm-monitor -c'         # Compact view
alias sq='squeue -u $USER -o "%.10i %.25j %.10P %.8T %.10M %.10l %.4D %R"'
```

## Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--interval` | `-i` | Refresh interval in seconds (default: 5) |
| `--all-users` | `-a` | Show jobs from all users |
| `--once` | `-1` | Run once and exit |
| `--compact` | `-c` | Use compact single-table view |
| `--slack` | `-s` | Enable Slack notifications |
| `--slack-webhook` | | Slack webhook URL (overrides .env) |

## Slack Notifications

Get notified on Slack when your jobs start or complete.

### Setup

1. Create a Slack webhook:
   - Go to [Slack API](https://api.slack.com/messaging/webhooks)
   - Create an incoming webhook for your workspace
   - Copy the webhook URL

2. Save the webhook URL in a `.env` file:
   ```bash
   # In the slurm-monitor directory or ~/.slurm-monitor.env
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
   ```

3. Run with Slack notifications enabled:
   ```bash
   slurm-monitor --slack
   # Or with inline webhook URL
   slurm-monitor --slack --slack-webhook "https://hooks.slack.com/services/..."
   ```

### Notification Events

- **Job Started** :rocket: - When a pending job starts running
- **Job Completed** :white_check_mark: - When a running job finishes
- **Monitor Started/Stopped** - When the monitor begins or ends

### Example Slack Message

```
🚀 Job Started
• ID: 12345
• Name: train_model
• Partition: A100
• GPUs: 4
```

## Requirements

- Python 3.8+
- SLURM cluster environment (`squeue`, `sinfo` commands available)
- [Rich](https://github.com/Textualize/rich) library for terminal formatting

## How It Works

The monitor uses SLURM commands to gather cluster information:

- `squeue` - Get job information (ID, name, user, partition, state, time, etc.)
- `sinfo` - Get partition and node information

Data is refreshed at the specified interval and displayed using Rich's Live display feature for smooth updates.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Dongwoo Kim ([@dongwookim-ml](https://github.com/dongwookim-ml))
