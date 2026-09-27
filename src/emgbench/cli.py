"""Focused entry points for the three-way comparison and recorded-video demo."""

import argparse
import os
import sys
from pathlib import Path


def background_command(arguments, workspace):
    """Resolve the workspace before the child changes its working directory."""
    forwarded = []
    tokens = iter(arguments)
    for token in tokens:
        if token == "--workspace":
            next(tokens)
        elif token.startswith("--workspace=") or token == "--background":
            continue
        else:
            forwarded.append(token)
    return [sys.executable, "-m", "emgbench", "--workspace", str(workspace), *forwarded]


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="emgbench", description=__doc__)
    parser.add_argument("--workspace", type=Path, help="Data/results directory; defaults to the current directory")
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot_parser = commands.add_parser("snapshot", help="Restore, verify, or build the committed reproduction snapshot")
    snapshot_parser.add_argument("action", choices=["restore", "verify", "build"])
    snapshot_parser.add_argument("--directory", type=Path, default=Path("snapshot"), help="Snapshot directory (default: ./snapshot)")
    commands.add_parser("reference", help="Fetch only the small, pinned authors' checkout for parity tests")
    prepare_parser = commands.add_parser("prepare", help="Download the fixed cohort and build resumable feature caches")
    prepare_parser.add_argument("--workers", type=int, default=6)
    prepare_parser.add_argument("--download-workers", type=int, default=3)
    prepare_parser.add_argument("--background", action="store_true")
    compare_parser = commands.add_parser("compare", help="Run or verify the three fixed pipelines")
    compare_parser.add_argument("--models", nargs="+", choices=["svm-rms", "lda-du", "tabpfn-du"])
    compare_parser.add_argument("--quote", action="store_true", help="Quote pending TabPFN work without fitting")
    compare_parser.add_argument("--max-estimated-tokens", type=int)
    compare_parser.add_argument("--background", action="store_true")
    commands.add_parser("report", help="Verify paired results and write results/comparison.md and JSON")
    training_parser = commands.add_parser("demo-training", help="Cache in-sample replay predictions for four training users")
    training_parser.add_argument("--models", nargs="+", choices=["svm-rms", "lda-du", "tabpfn-du"])
    training_parser.add_argument("--quote", action="store_true")
    training_parser.add_argument("--max-estimated-tokens", type=int)
    training_parser.add_argument("--svm-cache-mb", type=float, default=2048)
    training_parser.add_argument("--background", action="store_true")
    demo_parser = commands.add_parser("demo", help="Export cached predictions and serve the recorded-video demo")
    demo_parser.add_argument("--export-only", action="store_true")
    demo_parser.add_argument("--rebuild", action="store_true", help="Rebuild replay bundles from raw recordings and filtered channels")
    demo_parser.add_argument("--host", default="127.0.0.1")
    demo_parser.add_argument("--port", type=int, default=8765)
    demo_parser.add_argument("--background", action="store_true")
    check_parser = commands.add_parser("check-demo", help="Exercise playback, model switching and mobile layout")
    check_parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args(arguments)
    if args.workspace:
        os.environ["EMGBENCH_WORKSPACE"] = str(args.workspace.resolve())
    if getattr(args, "background", False):
        from .common import ROOT
        from .jobs import launch

        child = background_command(arguments, ROOT)
        print(launch(args.command, child))
        return
    if args.command == "snapshot":
        from .snapshot import build_snapshot, restore_snapshot

        if args.action == "build":
            build_snapshot(args.directory)
        else:
            restore_snapshot(args.directory, verify_only=args.action == "verify")
    elif args.command == "reference":
        from .reference import ensure_reference

        ensure_reference()
    elif args.command == "prepare":
        from .prepare import prepare

        prepare(args.workers, args.download_workers)
    elif args.command == "compare":
        from .comparison import compare

        compare(args.models, args.max_estimated_tokens, args.quote)
    elif args.command == "report":
        from .report import create_report

        create_report()
    elif args.command == "demo-training":
        from .training_replay import prepare_training_replay

        prepare_training_replay(args.models, args.max_estimated_tokens, args.quote, args.svm_cache_mb)
    elif args.command == "demo":
        from .replay import export_replay

        export_replay(rebuild=args.rebuild)
        if not args.export_only:
            import uvicorn

            from .server import create_app

            uvicorn.run(create_app(), host=args.host, port=args.port)
    elif args.command == "check-demo":
        from .browser import check_demo

        check_demo(args.url)
