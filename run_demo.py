"""End-to-end demo: policy in, adjudicated claims out.

    python run_demo.py                # auto: LLM if OPENAI_API_KEY is set, else offline
    python run_demo.py --mode demo    # force offline deterministic mode
    python run_demo.py --mode llm     # force real OpenAI agent

Deterministic mode needs no API key and no model download, so anyone who clones
the repo can run it. LLM mode shows the real agentic function-calling loop.
"""

import argparse
import json
import os
from pathlib import Path

from policy_agent.vectorstore import chunk_policy, VectorStore, openai_embed, lexical_embed
from policy_agent.demo_mode import run_demo_agent, load_reference_rules

ROOT = Path(__file__).parent


def _load():
    policy = (ROOT / "data" / "sample_policy.md").read_text()
    claims = json.loads((ROOT / "data" / "sample_claims.json").read_text())
    return policy, claims


def run_offline(policy, claims):
    store = VectorStore(lexical_embed).build(chunk_policy(policy))
    reference = load_reference_rules(ROOT / "data" / "reference_rules.json")
    run_demo_agent(store, claims, reference)


def run_llm(policy, claims):
    from policy_agent.agent import run_agent

    store = VectorStore(openai_embed).build(chunk_policy(policy))
    print("Mode: LLM (real OpenAI function calling).\n")
    answer, _ = run_agent(store, claims)
    print("\n=== FINAL ANSWER ===\n")
    print(answer)


def main():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "demo", "llm"], default="auto")
    args = parser.parse_args()

    policy, claims = _load()
    print(f"Loaded policy ({len(policy)} chars) and {len(claims)} claims.\n")

    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    mode = args.mode
    if mode == "auto":
        mode = "llm" if has_key else "demo"

    if mode == "llm":
        if not has_key:
            print("No OPENAI_API_KEY found; falling back to deterministic mode.\n")
            run_offline(policy, claims)
            return
        try:
            run_llm(policy, claims)
        except Exception as exc:
            print(f"\nLLM mode failed ({type(exc).__name__}: {exc}).")
            print("Falling back to deterministic mode so the demo still runs.\n")
            run_offline(policy, claims)
    else:
        run_offline(policy, claims)


if __name__ == "__main__":
    main()
