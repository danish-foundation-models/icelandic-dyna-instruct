import argparse
import asyncio
from pathlib import Path

from reverse_instruct.config import load_config
from reverse_instruct.runner import dry_run, run
from reverse_instruct.server import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reverse-instruct")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="Generate instruction data")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")

    serve_parser = commands.add_parser("serve", help="Start the configured vLLM server")
    serve_parser.add_argument("config", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    if args.command == "serve":
        serve(config)
        return

    if args.dry_run:
        dry_run(config, limit=args.limit or 3)
        return

    asyncio.run(run(config, limit=args.limit, resume=args.resume))


if __name__ == "__main__":
    main()
