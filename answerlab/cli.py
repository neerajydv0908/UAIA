"""Small commands so we can learn one part at a time."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .feedback import LABELS, record_feedback, save_run
from .pipeline import render, run_pipeline
from .providers import FixtureProvider, HTTPProvider
from .records import Case, LiveConfig, Run
from .steps import decide_sources, understand

ROOT = Path(__file__).resolve().parent.parent


def parser():
    app = argparse.ArgumentParser(description="Learn our evidence-first pipeline. Default demo costs $0.")
    commands = app.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Run only question/source classification")
    analyze.add_argument("question")
    demo = commands.add_parser("demo", help="Replay clearly labeled fictional teaching answers")
    demo.add_argument("--case", type=Path, default=ROOT / "examples/device_z.json")
    ask = commands.add_parser("ask", help="Explicit live OpenAI/Claude API calls")
    ask.add_argument("question")
    ask.add_argument("--config", type=Path, required=True)
    ask.add_argument("--allow-network", action="store_true")
    answer = commands.add_parser("answer", help="v1: web/saved evidence, cited answers, sentence checks, trained judge")
    answer.add_argument("question")
    answer.add_argument("--config", type=Path, default=ROOT / "examples/live_config_v1.json")
    answer.add_argument("--allow-network", action="store_true")
    answer.add_argument("--no-judge", action="store_true", help="Use the rule-based tie-break instead of the judge")
    for command in (demo, ask, answer):
        command.add_argument("--corpus", type=Path, default=ROOT / "examples/corpus.json")
        command.add_argument("--retrieval", choices=("direct", "mcp"), default="direct")
        command.add_argument("--json", action="store_true", help="Print the full auditable run record")
        command.add_argument("--save", type=Path, help="Explicitly retain the question, sources, and answers")
        command.add_argument("--brief", action="store_true", help="Hide the step trace")
    feedback = commands.add_parser("feedback", help="Store an optional review signal; does not train")
    feedback.add_argument("run", type=Path)
    feedback.add_argument("--label", choices=sorted(LABELS), required=True)
    feedback.add_argument("--note", default="")
    feedback.add_argument("--consent", action="store_true")
    feedback.add_argument("--allow-training", action="store_true")
    feedback.add_argument("--output", type=Path, default=ROOT / "data/private/feedback.jsonl")
    return app


async def execute(args):
    if args.command == "analyze":
        task = understand(args.question)
        print(json.dumps({"step_1": task.model_dump(), "step_2": decide_sources(task).model_dump()}, indent=2))
        return
    if args.command == "feedback":
        run = Run.model_validate_json(args.run.read_text())
        record = record_feedback(run, args.output, args.label, args.note, args.consent, args.allow_training)
        print(f"Feedback saved for run {record['run_id']}; pending review, not training data.")
        return
    if args.save and args.save.exists():
        raise ValueError("That save path already exists; choose a new run filename")
    if args.command == "answer":
        if not args.allow_network:
            raise ValueError("Live mode requires --allow-network; it calls model, search and judge services")
        from .judge_client import judge_pair
        from .pipeline_v1 import build_live, load_env_file, render_v1, run_v1
        load_env_file()
        config = LiveConfig.model_validate_json(args.config.read_text())
        providers, verifier, notes = build_live(config)
        run = await run_v1(args.question, providers, verifier, args.corpus, config.limits,
                           judge_fn=None if args.no_judge else judge_pair, setup_notes=notes,
                           search_usd=config.search_usd_per_call)
        if args.save:
            save_run(run, args.save)
        print(run.model_dump_json(indent=2) if args.json else render_v1(run, not args.brief))
        return
    if args.command == "demo":
        case = Case.model_validate_json(args.case.read_text())
        run = await run_pipeline(case.question, [FixtureProvider(reply) for reply in case.replies],
                                 args.corpus, transport=args.retrieval)
    else:
        if not args.allow_network:
            raise ValueError("Live mode requires --allow-network; use demo for zero-cost testing")
        config = LiveConfig.model_validate_json(args.config.read_text())
        run = await run_pipeline(args.question, [HTTPProvider(p) for p in config.providers],
                                 args.corpus, limits=config.limits, transport=args.retrieval, mode="live")
    if args.save:
        save_run(run, args.save)
    print(run.model_dump_json(indent=2) if args.json else render(run, not args.brief))


def main():
    args = parser().parse_args()
    try:
        asyncio.run(execute(args))
    except ValidationError:
        # Pydantic errors can include the original input; avoid echoing sensitive records.
        print("Invalid record/configuration. Check the documented fields and limits.", file=sys.stderr)
        raise SystemExit(2)
    except (ValueError, OSError) as exc:
        print(f"Cannot complete this run: {exc}", file=sys.stderr)
        raise SystemExit(2)
