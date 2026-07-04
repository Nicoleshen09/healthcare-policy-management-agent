"""Deterministic (offline) demo mode.

This mirrors the same three-step flow the LLM agent follows -- retrieve, extract
rules, adjudicate -- but with no API call:

  * retrieval runs for real, using the offline lexical embedder
  * adjudication runs for real, using the deterministic rule engine
  * only the "extract rules from prose" step is replaced by a reference file
    (data/reference_rules.json)

So a reviewer who clones the repo with no API key still sees the pipeline execute,
not a pre-baked printout.
"""

import json
from pathlib import Path

from .tools import make_tool_dispatch

# Queries a real agent would use to locate each part of the coverage criteria.
PLAN_QUERIES = [
    "covered diagnoses eligible ICD-10 codes",
    "frequency limitation replacement device time period",
    "prior authorization requirement",
    "exclusions not covered members",
]


def load_reference_rules(path) -> dict:
    return json.loads(Path(path).read_text())


def _citation_for(reason: str, citations: dict) -> str:
    r = reason.lower()
    if r.startswith("meets"):
        return ", ".join(sorted(set(citations.values())))
    if "diagnosis" in r:
        return citations.get("covered_diagnosis_prefixes", "")
    if "frequency" in r:
        return citations.get("frequency_limit_days", "")
    if "authorization" in r:
        return citations.get("prior_auth_required", "")
    return ", ".join(sorted(set(citations.values())))


def run_demo_agent(store, claims, reference, verbose=True) -> list:
    """Return a list of (claim_id, decision, reason, citation)."""
    dispatch = make_tool_dispatch(store)
    citations = reference.get("citations", {})

    if verbose:
        print("Mode: DETERMINISTIC (offline).")
        print("Retrieval and adjudication run for real; LLM rule-extraction is")
        print("replaced by data/reference_rules.json.\n")
        print("Step 1  retrieve policy sections")
        for q in PLAN_QUERIES:
            top = dispatch["search_policy"](q)["results"][0]
            print(f"  search_policy('{q}')")
            print(f"      -> {top['citation']}  {top['heading']}")

    rules = {
        k: reference[k]
        for k in [
            "procedure_code",
            "covered_diagnosis_prefixes",
            "frequency_limit_days",
            "prior_auth_required",
        ]
    }
    if verbose:
        print("\nStep 2  extracted rules (from reference file)")
        print("  " + json.dumps(rules))
        print("\nStep 3  adjudicate claims")

    rows = []
    for c in claims:
        out = dispatch["adjudicate_claim"](rules=rules, **c)
        cite = _citation_for(out["reasons"][0], citations)
        rows.append((out["claim_id"], out["decision"], out["reasons"][0], cite))
        if verbose:
            print(f"  {out['claim_id']}: {out['decision']:6s} ({cite})  {out['reasons'][0]}")

    return rows
