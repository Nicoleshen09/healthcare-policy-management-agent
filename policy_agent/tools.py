"""The two tools the agent can call.

  search_policy    -> RAG retrieval over the policy document (grounding + citations)
  adjudicate_claim -> runs the deterministic rule engine on one claim

The agent's reasoning glue in between (reading retrieved text and *extracting* the
CoverageRules) is done by the LLM, which is exactly the "convert written policy
into rules" capability that healthcare payment-integrity work depends on.
"""

from .rule_engine import CoverageRules, Claim, adjudicate

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": (
                "Search the coverage policy for relevant sections. Use it to find "
                "covered diagnoses, frequency limits, prior-authorization rules, and "
                "exclusions. Returns policy excerpts each with a citation id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, e.g. 'frequency limitation'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjudicate_claim",
            "description": (
                "Run the coverage rules you extracted from the policy against a single "
                "claim. Returns PAY, DENY, or REVIEW with reasons. Do not decide claims "
                "yourself; always call this tool so decisions stay deterministic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "procedure_code": {"type": "string"},
                    "diagnosis_code": {"type": "string"},
                    "service_date": {"type": "string", "description": "ISO YYYY-MM-DD"},
                    "last_service_date": {"type": ["string", "null"]},
                    "prior_auth_obtained": {"type": "boolean"},
                    "citation": {
                        "type": "string",
                        "description": "The policy section id you relied on for this decision, e.g. 'Section 2'.",
                    },
                    "rules": {
                        "type": "object",
                        "description": "The coverage rules extracted from the policy.",
                        "properties": {
                            "procedure_code": {"type": "string"},
                            "covered_diagnosis_prefixes": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "frequency_limit_days": {"type": ["integer", "null"]},
                            "prior_auth_required": {"type": "boolean"},
                            "notes": {"type": "string"},
                        },
                        "required": ["procedure_code", "covered_diagnosis_prefixes"],
                    },
                },
                "required": [
                    "claim_id",
                    "procedure_code",
                    "diagnosis_code",
                    "service_date",
                    "rules",
                ],
            },
        },
    },
]


def make_tool_dispatch(store):
    """Bind the tools to a specific vector store and return name -> callable."""

    def search_policy(query: str) -> dict:
        hits = store.search(query, k=3)
        return {
            "results": [
                {"citation": h["id"], "heading": h["heading"],
                 "text": h["text"], "score": round(h["score"], 3)}
                for h in hits
            ]
        }

    def adjudicate_claim(**kw) -> dict:
        r = kw["rules"]
        rules = CoverageRules(
            procedure_code=r["procedure_code"],
            covered_diagnosis_prefixes=r["covered_diagnosis_prefixes"],
            frequency_limit_days=r.get("frequency_limit_days"),
            prior_auth_required=r.get("prior_auth_required", False),
            notes=r.get("notes", ""),
        )
        claim = Claim(
            claim_id=kw["claim_id"],
            procedure_code=kw["procedure_code"],
            diagnosis_code=kw["diagnosis_code"],
            service_date=kw["service_date"],
            last_service_date=kw.get("last_service_date"),
            prior_auth_obtained=kw.get("prior_auth_obtained", False),
        )
        result = adjudicate(claim, rules)
        result["claim_id"] = claim.claim_id
        return result

    return {"search_policy": search_policy, "adjudicate_claim": adjudicate_claim}
