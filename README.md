# Healthcare Policy Management Agent

An agentic proof of concept for healthcare content management. The system reads an unstructured coverage policy, retrieves relevant policy evidence, extracts structured coverage rules, and generates citation-backed claim review recommendations.


**[▶ Live demo](https://healthcare-policy-management-agent.streamlit.app)**


## Deliverables

- [Written Report (PDF)](Deliverables/Agentic_AI_for_Healthcare_Content_Management.pdf)
- [Written Report (Word)](Deliverables/Agentic_AI_for_Healthcare_Content_Management.docx)
- [PowerPoint Presentation](Deliverables/Agentic_AI_for_Healthcare_Content_Management.pptx)
- [Recorded Demo Video](Deliverables/Yue_Shen_Agentic_AI_POC_Video.mp4)

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
coverage policy  ->  ingest + embed  ->  Agent (plan + reason)  ->  rules + recommended decisions
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
| CLM-001 | PAY      | Section 1-4 | Meets procedure, diagnosis, frequency, and authorization criteria |
| CLM-002 | DENY     | Section 2 | Diagnosis not in the covered set                |
| CLM-003 | DENY     | Section 3 | Frequency limitation is not met                 |
| CLM-004 | REVIEW   | Section 4 | Prior authorization not on file                 |

These labels are used for the prototype workflow. In a production healthcare environment, outputs should be treated as reviewer-facing recommendations and validated by qualified analysts before operational use.

## Tiny RAG layer

The retrieval layer is intentionally small. Because the demo uses one short policy document, it does not require FAISS, Chroma, or a separate vector database.

The policy is split into citable sections. Each section is embedded, and the system uses cosine similarity to retrieve the most relevant policy sections for queries such as diagnosis eligibility, frequency limitations, and prior authorization requirements.

In deterministic mode, the app uses a reproducible hashing-based lexical embedding. In LLM mode, the same vector store can use OpenAI embeddings. This keeps the demo lightweight while preserving the core RAG pattern: grounding outputs in source policy evidence.

## Deterministic vs LLM: what is the same and what differs

The final decision is always made by the same deterministic `adjudicate()` function,
so for a given set of rules and a given claim, the PAY / DENY / REVIEW outcome is fixed
and reproducible. The LLM is never allowed to decide a claim on its own.

Across the four sample claims, both modes produce the same four decisions, because the
LLM extracts the coverage rules correctly. What varies is upstream of the decision: the
search queries the agent chooses, and which policy section its reasoning cites. For the
non-covered-diagnosis claim, for example, the deterministic path cites the Covered
Indications section, while the LLM may instead cite the Exclusions section, which is
arguably a more precise basis for that denial. The decision converges; the citation is
where model judgment shows.

Decisions match only as long as the extraction is correct. If the LLM mis-extracts a
rule, a decision could change, which is exactly why two safeguards exist: the reference
rules act as a ground-truth key to check the extraction (run LLM mode, compare the
extracted rules to the reference, and a match confirms extraction was correct), and the
REVIEW path routes anything uncertain to a human.

## Using a real policy

`data/sample_policy.md` is synthetic. To run against a real public document, drop a CMS
Local or National Coverage Determination (or an MLN article) into `data/` as text and
point `run_demo.py` at it. Those documents are public. Proprietary measure specifications
(for example NCQA HEDIS specifications) are intentionally not used here.

## Design notes

- Scope is intentionally small (one policy, a handful of rule fields, four claims) to
  prioritize a clear working concept over completeness.
- The REVIEW path is the human-in-the-loop seam: anything the rules cannot clear
  automatically is routed to a person rather than auto-denied.

## Limitations

This is a proof of concept, and its scope is deliberately narrow:

- Single synthetic policy. The demo runs on one synthetic coverage policy and four
  sample claims, not a real corpus. It shows the concept working, not production coverage.
- Narrow rule schema. Rules capture procedure code, covered diagnosis prefixes, a
  frequency limit, and prior authorization. Real payer policies involve modifiers, code
  ranges, place of service, date logic, and combinations this schema does not model.
- Simple matching. Diagnosis coverage uses ICD-10 prefix matching, which does not handle
  exclusion codes, code ranges, or finer coding nuances.
- Offline retrieval is lexical. Deterministic mode uses a hashing bag-of-words retriever
  with no semantic understanding, so it can miss paraphrased passages. LLM mode uses
  semantic embeddings.
- No formal evaluation. Extraction is sanity-checked against the reference rules for one
  policy, not benchmarked on a labeled dataset, so extraction accuracy at scale is unmeasured.
- Extraction can fail. The LLM can occasionally emit a malformed or mis-typed rule, which
  the current code does not fully guard against.
- Not production-hardened. There is no PHI handling, access control, persistence, or audit
  logging suitable for real claims data.
- Section-level citations. Citations point to a policy section, not a specific sentence.

A production version would add a richer rule schema, semantic retrieval at scale, policy version comparison, effective-date logic, conflict resolution, labeled evaluation sets, input/output validation, analyst approval workflows, audit logging, and security controls.

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
Deliverables/      report, presentation, and recorded demo video
```

## Tests

```bash
pytest -q
```

The rule engine, retrieval mechanics, and offline flow are all tested without an API key.