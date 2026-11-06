"""Single entry point: python -m src.pipeline --all"""
import argparse
import sys

from src import (
    backtest,
    clean,
    llm_brief,
    model,
    panel,
    redistribute,
    report,
    stock,
)
from src.io import MissingDataError, SchemaMismatchError, ensure_dirs, load_config

STAGES = {
    "profile": report.run_profile,
    "panel": panel.run,
    "clean": clean.run,
    "backtest": backtest.run,
    "model": model.run,
    "stock": stock.run,
    "redistribute": redistribute.run,
    "llm_brief": llm_brief.run,
    "report": report.run,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stockout Sentinel pipeline")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="run every stage in order")
    target.add_argument("--stage", choices=list(STAGES), help="run one stage")
    parser.add_argument("--config", help="path to a config file (default: config.yaml)")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    for name in list(STAGES) if args.all else [args.stage]:
        try:
            status = STAGES[name](cfg)
        except MissingDataError as error:
            print(f"[{name}] stopped: {error}", file=sys.stderr)
            return 1
        except SchemaMismatchError as error:
            print(f"[{name}] stopped: {error}", file=sys.stderr)
            return 2
        print(f"[{name}] {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
