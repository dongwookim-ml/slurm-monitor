# Repository guide

SLURM Monitor is a read-only CLI. Keep `slurm_monitor.py` standalone so direct execution and copying it to `~/bin/slurm-monitor` continue to work. Python 3.8 compatibility uses postponed annotations. Package metadata belongs only in `pyproject.toml`.

The module separates normalized job/node models, parsers, a snapshot collector, resource assessment, job tracking, a durable Slack outbox, a delivery worker and pure Rich rendering. Do not query SLURM from renderers. Missing fields stay unknown. Queue failures never become empty successful snapshots. Job disappearance requires accounting confirmation, including Submit identity when known. GPU figures are SLURM allocations, never CPU-based estimates.

Safe local checks:

```bash
python -m pytest
ruff check .
ruff format --check .
python -m build
python slurm_monitor.py --help
```

The regression suite prohibits real subprocesses and HTTP. Fixtures are synthetic and contain no credentials. Test interrupted/restarted and failed/retried delivery flows when changing tracking or persistence. Keep diagnostic text literal; do not interpret job names as Rich or Slack markup. State files contain job data and event queues, never webhook URLs.

Do not use a production cluster, send real Slack messages, submit/cancel jobs, publish packages, push or create PRs without explicit authorization. Offline test success does not establish compatibility with a site's SLURM schema, PrivateData settings or accounting setup.
