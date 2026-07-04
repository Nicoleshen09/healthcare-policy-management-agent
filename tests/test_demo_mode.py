"""Verifies the offline deterministic mode runs with no API key."""

import json
from pathlib import Path

from policy_agent.vectorstore import chunk_policy, VectorStore, lexical_embed
from policy_agent.demo_mode import run_demo_agent, load_reference_rules

ROOT = Path(__file__).resolve().parent.parent


def test_offline_demo_produces_expected_decisions():
    policy = (ROOT / "data" / "sample_policy.md").read_text()
    claims = json.loads((ROOT / "data" / "sample_claims.json").read_text())
    reference = load_reference_rules(ROOT / "data" / "reference_rules.json")

    store = VectorStore(lexical_embed).build(chunk_policy(policy))
    rows = run_demo_agent(store, claims, reference, verbose=False)

    decisions = {claim_id: decision for claim_id, decision, _, _ in rows}
    assert decisions == {
        "CLM-001": "PAY",
        "CLM-002": "DENY",
        "CLM-003": "DENY",
        "CLM-004": "REVIEW",
    }


def test_offline_retrieval_hits_frequency_section():
    policy = (ROOT / "data" / "sample_policy.md").read_text()
    store = VectorStore(lexical_embed).build(chunk_policy(policy))
    top = store.search("frequency limitation replacement device", k=1)[0]
    assert "Frequency" in top["heading"]
