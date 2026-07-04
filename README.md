# Healthcare Policy Management Agent

An agentic proof of concept that reads an unstructured healthcare coverage policy,
extracts the coverage rules from it, and adjudicates claims against those rules,
returning a PAY / DENY / REVIEW decision with a citation back to the policy text.

Domain: content management in health care, specifically converting written coverage
policy into executable rules and adjudicating claims against them with an audit trail.

## What it demonstrates

- An agent, not a chatbot: a ReAct loop (OpenAI function calling) that plans a
  multi-step task, calls tools, and decides when it is done.
- Retrieval-grounded reasoning: the agent searches the policy before extracting any
  rule, so it does not invent coverage criteria.
- Policy to rules: the LLM converts prose policy language into a structured rules
  object, the capability payment-integrity work depends on.
- Auditable, deterministic decisions: the pay/deny/review call is made by plain
  Python, not the LLM, so decisions are reproducible and every one carries a citation
  to the policy section it relied on.

## Architecture

```
coverage policy  ->  ingest + embed  ->  Agent (plan + reason)  ->  rules + decisions
                     (vector store)       |            |             (PAY / DENY / REVIEW)
                                     retriever     adjudicator
                                      (RAG)      (deterministic)
```

The agent plans the task and selects tools; retrieval grounds it in the policy text;
the deterministic rule engine makes the actual decision. Only the rule extraction in
the middle is model reasoning.

## Two ways to run it, each with two modes

The project runs as a command-line demo or as a Streamlit web app. Both support:

- Deterministic mode (no API key): retrieval and adjudication run for real, offline.
  Only the "extract rules from prose" step is stood in by a reference file
  (`data/reference_rules.json`). Anyone can run it with no setup and no model download.
- LLM mode (needs `OPENAI_API_KEY`): the real ReAct agent calls OpenAI function
  calling to extract the rules from the policy at runtime.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Deterministic mode needs nothing else. For LLM mode, provide a key (either in a `.env`
file for the CLI, or in the app's sidebar field):

```bash
cp .env.example .env      # then add your OPENAI_API_KEY
```

## Running

Command line:

```bash
python run_demo.py              # auto: LLM if a key is set, else deterministic
python run_demo.py --mode demo  # force deterministic / offline
python run_demo.py --mode llm   # force real OpenAI agent
```

Web app:

```bash
streamlit run app.py
```

The app loads the sample policy and an editable claims table, then shows the retrieval
evidence, the extracted rules, colored decisions, and a summary. Editing a claim value
and re-running shows the decision change live.

## Expected outcome (identical decisions in both modes)

| Claim   | Decision | Cite      | Why                                             |
|---------|----------|-----------|-------------------------------------------------|
| CLM-001 | PAY      | Section 1-4 | Diabetes dx, prior device > 365 days ago, auth on file |
| CLM-002 | DENY     | Section 2 | Diagnosis not in the covered set                |
| CLM-003 | DENY     | Section 3 | Replacement device inside the 12 month window   |
| CLM-004 | REVIEW   | Section 4 | Prior authorization not on file                 |

Citation ids follow the document's own section numbering, so a cite lines up with what
a reader sees in the policy text.

## Deterministic vs LLM: what is the same and what differs

The final decision is made by the same deterministic `adjudicate()` function in both
modes, so given the same rules and claims the PAY / DENY / REVIEW outcome is identical
and reproducible. The LLM is never allowed to decide a claim on its own.

What can differ is upstream of the decision: the search queries the agent chooses, and
which policy section its reasoning cites. For a non-covered diagnosis, for example, the
deterministic path cites the Covered Indications section while the LLM sometimes cites
the more specific Exclusions section. This is expected: the decision converges, the
citation reasoning is where model judgment shows.

Because the reference rules are effectively a ground-truth answer key, they also serve
as a check on the LLM: run LLM mode, compare its extracted rules to the reference, and
a match confirms the extraction was correct.

## Deployment

Deployed on Streamlit Community Cloud (free):

1. Push this repo to GitHub (public).
2. Sign in at share.streamlit.io with GitHub.
3. Create a new app pointing at `app.py`; `requirements.txt` is picked up automatically.

Safety: no API key is committed or stored in Streamlit secrets. The deployed app
defaults to deterministic mode, which needs no key; LLM mode uses only a key a visitor
enters in the sidebar.

## Using a real policy

`data/sample_policy.md` is synthetic. To run against a real public document, drop a CMS
Local or National Coverage Determination (or an MLN article) into `data/` as text and
point `run_demo.py` at it. Those documents are public. Proprietary measure specifications
(for example NCQA HEDIS specifications) are intentionally not used here.

## Design notes and limitations

- Scope is intentionally small (one policy, a handful of rule fields, four claims) to
  prioritize a clear working concept over completeness.
- The REVIEW path is the human-in-the-loop seam: anything the rules cannot clear
  automatically is routed to a person rather than auto-denied.
- A production version would add rule versioning and a policy change comparison mode
  (diff two policy versions, flag which extracted rules changed), broader rule coverage,
  and evaluation of the extraction step against a labeled set.

## Project layout

```
policy_agent/
  rule_engine.py   deterministic adjudicator (PAY / DENY / REVIEW)
  vectorstore.py   chunk, embed (OpenAI or offline lexical), cosine search
  tools.py         tool schemas + implementations
  agent.py         ReAct loop over OpenAI function calling (LLM mode)
  demo_mode.py     offline flow: real retrieval + reference rules + real adjudication
app.py             Streamlit web app (both modes)
run_demo.py        command-line entry point with mode selection
data/              synthetic policy, claims, and reference rules
tests/             deterministic tests (no API key required)
```

## Tests

```bash
pytest -q
```

The rule engine, retrieval mechanics, and offline flow are all tested without an API key.