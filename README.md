# SLURM Monitor

A read-only terminal dashboard for SLURM jobs, GPU allocations and job diagnostics. It never submits, modifies or cancels jobs.

- Live and compact views, per-partition job summaries, and per-node CPU/GPU details.
- Actual GPU allocation counts, including multiple GPU types and nodes shared between partitions.
- Pending reasons, conservative resource suitability checks, and completed/failed/timed-out exit information.
- Optional durable Slack notifications with batched delivery, retries and restart recovery.
- Query errors and stale data stay visible. Missing resource information is shown as `?`.

## Install

Python 3.8+ and Rich 13+ are required. Run on a POSIX host with SLURM clients and access to the cluster.

```bash
pip install slurm-monitor
# From this source checkout:
pip install -e .
# Standalone installation is still supported:
cp slurm_monitor.py ~/bin/slurm-monitor
chmod +x ~/bin/slurm-monitor
pip install -r requirements.txt
```

The new CLI options and restored GPU bars are in source version 0.2.1; an older published package may not contain them.

## Use

```bash
slurm-monitor                          # Current user's dashboard, refresh every 5s
slurm-monitor -a                       # All visible users; with --slack this tracks all visible jobs
slurm-monitor -c -i 10                 # Compact view, refresh every 10s
slurm-monitor -p gpu                   # Jobs and per-node details for a partition
slurm-monitor -1 --max-jobs 0           # One snapshot, all rows
slurm-monitor --job 12345              # Pending reason / resource suitability
slurm-monitor --job 12345 -p gpu        # Assess this specific partition
slurm-monitor -1 --job 12345_2          # Historical array task / exit diagnostics
slurm-monitor --json                   # One machine-readable snapshot
```

Existing `-1`, `-a`, `-c`, `-p`, `-i` and `-s` flags are preserved. Refresh intervals must be finite and at least one second. `--max-jobs` defaults to 15; zero shows all. Hidden row counts are explicit and narrow terminals use fewer columns. Non-interactive output automatically emits one snapshot. Exit codes are 0 for a successful snapshot or Ctrl+C, 1 for query errors, and 2 for invalid configuration.

## What the numbers mean

Jobs use allocated TRES rather than per-node GRES multiplied in selected views. Nodes use `GresUsed`, with `AllocTRES` as a fallback. Generic and typed TRES counts are not added twice. Down/drained/unresponsive nodes contribute no free GPUs. Drained GRES are also removed from the free count. Missing allocation/type detail remains unknown; CPU usage is never used to guess GPU usage.

The header counts physical nodes once. A node may belong to multiple partitions, so **partition totals can overlap and must not be summed**. The same physical allocation is reflected in each partition containing that node. Run/Pend counts describe the visible job scope; Mine describes the current user's visible allocations.

Partition and per-node GPU bars show **red `█` used**, **green `░` idle**, **gray `×` unavailable**, and **yellow `?` unknown**. Symbols and the U/I/X legend remain readable without color. Counts follow the bar, or appear beneath it on narrow terminals. Unavailable means the remaining inventory cannot currently be offered as idle, including down/drained nodes or an inactive partition. Unknown allocation remains `?`, never idle. Zero-total partitions show “No GPUs”. Bars are rounded to character cells; small nonzero segments stay visible and the exact counts are authoritative.

Free means unallocated GPUs on active nodes. CPU/memory availability, reservations, account/QoS policy, exclusivity, feature expressions, topology, licenses and scheduler placement can still prevent their use. The dashboard measures SLURM allocations, not GPU device utilization.

## Pending and exit diagnostics

`--job ID` explains the pending reason and checks necessary visible capacity conditions: aggregate free CPUs/GPUs, requested GPU types, per-node GPU placement, known node memory, node/time limits and visible account/QoS allow lists. Unknown requests, unavailable policy and stale data are reported explicitly. This is **resource suitability, not a scheduler guarantee or a predicted start time**. For example, a Priority or QoS-limited job can still wait with free GPUs. An assessment of an alternate partition does not change the job's requested partitions.

Disappearance from `squeue` is not success. The tracker retains an ending-confirmation record until `sacct` supplies a terminal state. COMPLETING, CONFIGURING, SUSPENDED and requeue transitions are handled without inventing completion. Submit identity distinguishes reused numeric job IDs when available. Accounting delay, disabled accounting and restricted visibility may keep endings unconfirmed. The latest 100 confirmed endings are retained during monitoring; large terminals display the most recent three. `--once --job ID` also queries historical accounting directly.

Exit diagnostics show both batch `ExitCode` and `DerivedExitCode` in `exit-code:signal` format. A derived step failure can be nonzero even when the batch script exits zero. Terminal states include COMPLETED, FAILED, TIMEOUT, CANCELLED, OUT_OF_MEMORY and node/boot failure. Historical accounting may not include a job name.

## Collection and SLURM compatibility

A refresh collects one job snapshot and one physical-node snapshot and shares them with every view and the tracker. Partition policy is refreshed every 60 seconds. Normal refreshes therefore use two client invocations after the policy cache is populated. Each SLURM client invocation may issue multiple internal RPCs; this is not an RPC-count guarantee. Confirmed accounting is cached; unresolved endings are retried at most every 15 seconds in batches of up to 500 IDs.

`squeue --json` is preferred. When JSON is unavailable or groups array tasks, an expanded `--array --Format` fallback uses a control-character separator and actual `tres-alloc` fields, preserving pipe characters in names and formatted array/heterogeneous IDs. Malformed responses and connection failures retain the last successful snapshot rather than invoking a different parser or resetting tracking. Renderers issue no SLURM calls.

Node data comes from `scontrol show nodes --oneliner`, partition policy from `scontrol show partitions --oneliner`, and endings from allocation-only `sacct`. Sites can restrict these through PrivateData. Field availability and JSON schemas differ by SLURM version; unavailable fields are deliberately not approximated.

References: [squeue](https://slurm.schedmd.com/squeue.html), [scontrol](https://slurm.schedmd.com/scontrol.html), [sacct](https://slurm.schedmd.com/sacct.html).

## Slack notifications

Create an [incoming webhook](https://docs.slack.dev/messaging/sending-messages-using-incoming-webhooks/) and keep its URL private. Prefer an environment variable to avoid exposing it in command-line history/process listings:

```bash
export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
slurm-monitor --slack
slurm-monitor --slack -p gpu             # Partition mode also tracks jobs
```

Configuration precedence is `--slack-webhook`, process environment, current-directory `.env`, `~/.slurm-monitor.env`, then a `.env` next to the script. An unrelated `.env` does not mask later configuration. Quoted values and `export` are accepted. Only HTTPS incoming-webhook URLs on `hooks.slack.com` or `hooks.slack-gov.com` are accepted. Names are escaped and messages disable markup, so a job name cannot introduce a channel mention. Webhook values are not stored in the journal or printed in errors.

Events are persisted **before** HTTP delivery in an atomic, mode-0600 JSON journal. The default is `$XDG_STATE_HOME/slurm-monitor/<scope>.json`, or `~/.local/state/slurm-monitor/<scope>.json`. Scope includes host, user and view filters. `--state-file PATH` overrides the destination. A file lock prevents concurrent use of one journal. Corrupt/unsupported journals are preserved and startup fails clearly; repair or select a new file deliberately. A failed journal write pauses delivery and is shown in the UI.

Delivery runs in a separate worker so network latency does not block refresh. Messages are bounded and batched. 429 honors Retry-After; transport failures and 5xx retry with capped exponential backoff. Permanent 4xx failures retain their events and are reported, with another attempt allowed after a new invocation (for example after correcting the webhook). Ctrl+C stops new delivery attempts and retains pending work; it does not send a separate shutdown message.

Delivery is **at least once**. If Slack accepts a request but the response or local acknowledgment is lost, recovery may send a duplicate. Incoming webhooks provide no idempotency key, so exactly-once delivery cannot be promised. Initial running jobs are snoozed in a new journal; a restored journal resumes outstanding work.

## Development and offline checks

```bash
pip install -e '.[dev]'
python -m pytest
ruff check .
ruff format --check .
python -m build
python slurm_monitor.py --help
```

Fixtures are synthetic. Tests prohibit real subprocess and HTTP execution and cover query failures, accounting delay, GPU types/shared partitions, safe rendering, journal restart/locking/write failure, retry behavior, interruption, CLI modes and suitability uncertainty. CI runs supported Python versions, lint/format and package/entry-point checks without SLURM clients or Slack secrets. Live site compatibility and real webhook delivery require a separately authorized smoke test.

MIT license. See [LICENSE](LICENSE).
