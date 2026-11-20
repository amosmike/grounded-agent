from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import AgentDeps, investigate
from .evals import GoldenCase, run_eval, summarise
from .memory import PatternStore
from .model import AnthropicModel
from .router import routes_from_env
from .world import load_world

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORLDS = DATA / "worlds"
PATTERNS = DATA / "patterns.json"


def _cases() -> list[GoldenCase]:
    return [GoldenCase.from_dict(d) for d in json.loads((DATA / "golden.json").read_text())]


def _usd(n: float) -> str:
    return f"${n:.3f}"


def cmd_investigate(case_id: str) -> None:
    case = next((c for c in _cases() if c.id == case_id), None)
    if case is None:
        sys.exit(f"Unknown case {case_id}. Cases: {', '.join(c.id for c in _cases())}")
    memory = PatternStore(PATTERNS)
    r = investigate(case.ticket, load_world(WORLDS / case.world), AgentDeps(AnthropicModel(), routes_from_env(), memory))
    trace_path = r.trace.write(ROOT / "runs" / r.run_id)

    rating = f", evidence {r.rating}" if r.rating else ""
    print(f"\n{case.ticket.id}: {r.outcome}{rating} ({r.turns} turns, {_usd(r.cost_usd)})")
    if r.recalled:
        print(f"Recalled: {', '.join(r.recalled)}")
    if r.answer:
        a = r.answer
        print(f"\nQuestion: {a.question_asked}\nCause: {a.cause}\nFix: {a.fix}\nEscalate: {'yes' if a.escalate else 'no'}")
        print("\nClaims:")
        supported = {c["index"] for c in r.verdict.claims if c["supported"]} if r.verdict else set()
        for i, claim in enumerate(a.claims):
            cited = "cited" if r.citations[i].ok else "UNCITED"
            print(f"  {cited} / {'supported' if i in supported else 'unsupported'}: {claim.text}")
        print(f"\nReply for the customer:\n{a.customer_reply}")
        learned = memory.learn(case.ticket, a, r.rating or "weak")
        memory.save()
        if learned:
            print(f"\nMemory: {learned.id} ({learned.status}). Review it with: grounded-agent review {learned.id} {case.ticket.id} correct|partly|wrong")
    for p in r.proposals:
        print(f"\nAwaiting approval: {p.collection}/{p.record_id}.{p.field} {json.dumps(p.old_value)} -> {json.dumps(p.new_value)} ({p.reason})")
    print(f"\nTrace: {trace_path}")


def cmd_review(pattern_id: str, ticket_id: str, review: str) -> None:
    memory = PatternStore(PATTERNS)
    pattern = memory.apply_review(pattern_id, ticket_id, review)  # type: ignore[arg-type]
    memory.save()
    print(f"{pattern.id} is now {pattern.status}.")


def cmd_eval(k: int) -> None:
    # Evals read memory but never write to it.
    deps = AgentDeps(AnthropicModel(), routes_from_env(), PatternStore(PATTERNS))
    scores = run_eval(_cases(), deps, WORLDS, k)
    print(f"\n{'case':<8}{'run':>4}  {'outcome':<14}{'evidence':<10}{'cause':>6}{'escal':>6}{'fix':>5}  {'pass':<6}{'cost':>8}{'turns':>6}")
    for s in scores:
        print(
            f"{s.case_id:<8}{s.run:>4}  {s.outcome:<14}{s.rating or '-':<10}{s.cause:>6.1f}{s.escalation:>6.0f}{s.change:>5.1f}  "
            f"{'yes' if s.passed else 'no':<6}{_usd(s.cost_usd):>8}{s.turns:>6}"
        )
    sm = summarise(scores, k)
    print(
        f"\n{sm.cases} cases x {k} runs. Right: {sm.mean_right:.0%}. pass^{k}: {sm.pass_hat_k:.0%}. "
        f"Mean {_usd(sm.mean_cost_usd)} and {sm.mean_turns:.1f} turns a run (investigation and verification; judging not included)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="grounded-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("investigate", help="investigate one golden ticket, learn from it and write a trace")
    p.add_argument("case_id")
    p = sub.add_parser("review", help="record an engineer's review of an answer")
    p.add_argument("pattern_id")
    p.add_argument("ticket_id")
    p.add_argument("review", choices=["correct", "partly", "wrong"])
    p = sub.add_parser("eval", help="replay every golden case against its snapshot")
    p.add_argument("--k", type=int, default=1, help="runs per case (reports pass^k)")
    args = parser.parse_args()

    if args.command == "investigate":
        cmd_investigate(args.case_id)
    elif args.command == "review":
        cmd_review(args.pattern_id, args.ticket_id, args.review)
    else:
        if args.k < 1:
            parser.error("--k must be at least 1")
        cmd_eval(args.k)


if __name__ == "__main__":
    main()
