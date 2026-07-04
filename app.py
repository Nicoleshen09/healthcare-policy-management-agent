"""Streamlit UI for the healthcare policy management agent.

Run it with:

    streamlit run app.py

It reuses the same policy_agent package as the CLI. Deterministic mode needs no
API key, so a reviewer can open the app and see the full flow with zero setup;
LLM mode (with a key) shows the real OpenAI function-calling agent.
"""

import json
import os
from pathlib import Path

import streamlit as st

from policy_agent.vectorstore import (
    chunk_policy,
    VectorStore,
    lexical_embed,
    openai_embed,
)
from policy_agent.tools import make_tool_dispatch
from policy_agent.demo_mode import PLAN_QUERIES, load_reference_rules, _citation_for

ROOT = Path(__file__).parent
DATA = ROOT / "data"

BADGE_COLORS = {"PAY": "#16a34a", "DENY": "#dc2626", "REVIEW": "#d97706"}


# --------------------------------------------------------------------------- #
# Data + orchestration (no Streamlit here, so it stays testable)
# --------------------------------------------------------------------------- #
def load_defaults():
    policy = (DATA / "sample_policy.md").read_text()
    claims = json.loads((DATA / "sample_claims.json").read_text())
    reference = load_reference_rules(DATA / "reference_rules.json")
    return policy, claims, reference


def _clean_claims(claims):
    """Normalize edited rows: empty last_service_date becomes None."""
    cleaned = []
    for c in claims:
        c = dict(c)
        if not c.get("last_service_date"):
            c["last_service_date"] = None
        cleaned.append(c)
    return cleaned


def run_deterministic(policy, claims, reference):
    """Offline flow: real retrieval + reference rules + real adjudication."""
    store = VectorStore(lexical_embed).build(chunk_policy(policy))
    dispatch = make_tool_dispatch(store)

    retrieval = [
        {"query": q, "hits": dispatch["search_policy"](q)["results"]}
        for q in PLAN_QUERIES
    ]

    rules = {
        k: reference[k]
        for k in [
            "procedure_code",
            "covered_diagnosis_prefixes",
            "frequency_limit_days",
            "prior_auth_required",
        ]
    }
    citations = reference.get("citations", {})

    decisions = []
    for c in _clean_claims(claims):
        try:
            out = dispatch["adjudicate_claim"](rules=rules, **c)
            out["citation"] = _citation_for(out["reasons"][0], citations)
        except Exception as exc:
            out = {
                "claim_id": c.get("claim_id", "?"),
                "decision": "ERROR",
                "reasons": [f"Could not adjudicate: {exc}"],
                "citation": "",
            }
        decisions.append(out)

    return retrieval, rules, decisions


def parse_llm_transcript(messages):
    """Pull retrieval, extracted rules, and decisions out of the agent messages."""
    calls = {}
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            calls[tc.id] = (tc.function.name, json.loads(tc.function.arguments))

    retrieval, rules, decisions = [], None, []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role != "tool":
            continue
        tcid = m["tool_call_id"] if isinstance(m, dict) else m.tool_call_id
        content = m["content"] if isinstance(m, dict) else m.content
        name, args = calls.get(tcid, (None, {}))
        try:
            data = json.loads(content)
        except Exception:
            continue
        if name == "search_policy":
            retrieval.append({"query": args.get("query", ""), "hits": data.get("results", [])})
        elif name == "adjudicate_claim":
            if rules is None:
                rules = args.get("rules")
            data.setdefault("claim_id", args.get("claim_id"))
            data.setdefault("citation", args.get("citation", ""))
            decisions.append(data)
    return retrieval, rules, decisions


def run_llm(policy, claims):
    from policy_agent.agent import run_agent

    store = VectorStore(openai_embed).build(chunk_policy(policy))
    answer, messages = run_agent(store, _clean_claims(claims), verbose=False)
    retrieval, rules, decisions = parse_llm_transcript(messages)
    return answer, retrieval, rules, decisions


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_retrieval(retrieval):
    st.subheader("1. Evidence retrieval")
    st.caption("The agent searches the policy before extracting any rule.")
    for step in retrieval:
        hits = step["hits"]
        top = hits[0] if hits else None
        label = f"`{step['query']}` -> " + (f"**{top['heading']}**" if top else "no match")
        st.markdown(label)
        if top:
            with st.expander(f"Show source snippet ({top['citation']})"):
                st.write(top["text"])


def render_rules(rules):
    st.subheader("2. Extracted coverage rules")
    st.caption("Written policy converted into a structured, machine-readable object.")
    st.json(rules)


def render_decisions(decisions):
    st.subheader("3. Claim decisions")
    for d in decisions:
        color = BADGE_COLORS.get(d["decision"], "#6b7280")
        badge = (
            f'<span style="background:{color};color:white;padding:2px 10px;'
            f'border-radius:12px;font-weight:600;font-size:0.85em">{d["decision"]}</span>'
        )
        cite = f" &nbsp;<code>{d['citation']}</code>" if d.get("citation") else ""
        st.markdown(
            f'{badge} &nbsp; **{d.get("claim_id","?")}**{cite}<br>'
            f'<span style="color:#555">{d["reasons"][0]}</span>',
            unsafe_allow_html=True,
        )


def render_summary(decisions):
    st.subheader("4. Summary")
    rows = ["| Claim | Decision | Cite | Reason |", "|---|---|---|---|"]
    for d in decisions:
        rows.append(
            f"| {d.get('claim_id','?')} | {d['decision']} | "
            f"{d.get('citation','')} | {d['reasons'][0]} |"
        )
    st.markdown("\n".join(rows))


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Healthcare Policy Management Agent", layout="wide")
    st.title("Healthcare Policy Management Agent")
    st.caption(
        "A lightweight agentic AI workflow for converting healthcare policy text into citation-backed claim review recommendations."
    )

    policy, default_claims, reference = load_defaults()

    # Sidebar controls
    st.sidebar.header("Controls")
    mode_choice = st.sidebar.radio(
        "Mode",
        ["Auto", "Deterministic (offline)", "LLM"],
        help="Deterministic needs no API key. LLM runs the real OpenAI agent.",
    )
    key_input = st.sidebar.text_input("OpenAI API key (optional)", type="password")
    if key_input:
        os.environ["OPENAI_API_KEY"] = key_input
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    run = st.sidebar.button("Run adjudication", type="primary")

    # Inputs
    with st.expander("Coverage policy (input document)", expanded=False):
        st.markdown(policy)

    st.markdown("**Claims** (editable, add or change rows before running)")
    edited = st.data_editor(default_claims, num_rows="dynamic", use_container_width=True)
    claims = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)

    if not run:
        st.info("Set a mode in the sidebar and click Run adjudication.")
        return

    # Resolve mode
    resolved = "demo"
    if mode_choice == "LLM":
        resolved = "llm" if has_key else "demo"
        if not has_key:
            st.warning("No API key provided; running in deterministic mode instead.")
    elif mode_choice == "Auto":
        resolved = "llm" if has_key else "demo"

    st.divider()

    if resolved == "llm":
        st.success("Mode: LLM (real OpenAI function calling)")
        try:
            answer, retrieval, rules, decisions = run_llm(policy, claims)
        except Exception as exc:
            st.error(f"LLM run failed ({type(exc).__name__}: {exc}). Falling back to deterministic mode.")
            retrieval, rules, decisions = run_deterministic(policy, claims, reference)
            answer = None
    else:
        st.info("Mode: Deterministic (offline). Retrieval and adjudication run for real; rule extraction uses the reference file.")
        retrieval, rules, decisions = run_deterministic(policy, claims, reference)
        answer = None

    render_retrieval(retrieval)
    st.divider()
    if rules:
        render_rules(rules)
        st.divider()
    render_decisions(decisions)
    st.divider()
    render_summary(decisions)

    if answer:
        with st.expander("Agent's own written summary"):
            st.markdown(answer)


if __name__ == "__main__":
    main()