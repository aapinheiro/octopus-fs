"""Command line interface.

    octopus_fs run data.parquet --target churn --arms fast --report out.html

Why a CLI at all: it is the cheapest way to run Octopus as a scheduled job
(Databricks task, cron, CI step) without anyone writing a notebook.
"""

from __future__ import annotations


def main() -> int:
    """Entry point registered in pyproject [project.scripts].

    TODO(you):
    - argparse is enough; skip typer/click until there are 3+ subcommands
    - subcommands: `run` (score + optional report), `arms` (list available)
    - read .csv/.parquet by extension; fail clearly on anything else
    - exit codes matter for schedulers: 0 ok, 1 user error, 2 all arms failed
    - print a compact summary to stdout and JSON to --out, never both mixed
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
