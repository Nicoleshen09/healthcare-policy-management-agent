"""Deterministic adjudication logic.

This module is intentionally free of any LLM or network dependency. The agent's
job is to read an unstructured policy and *extract* a CoverageRules object; the
actual pay/deny/review decision is made here by plain, auditable Python. Keeping
the decision deterministic is a deliberate design choice: it means the LLM never
"decides" a claim on its own, so decisions are reproducible and cannot be
hallucinated.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class CoverageRules:
    procedure_code: str
    covered_diagnosis_prefixes: list  # e.g. ["E10", "E11"]
    frequency_limit_days: Optional[int] = None
    prior_auth_required: bool = False
    notes: str = ""


@dataclass
class Claim:
    claim_id: str
    procedure_code: str
    diagnosis_code: str
    service_date: str  # ISO YYYY-MM-DD
    last_service_date: Optional[str] = None
    prior_auth_obtained: bool = False


def _norm(code: str) -> str:
    return code.replace(".", "").upper().strip()


def _days_between(d1: str, d2: str) -> int:
    return abs((date.fromisoformat(d1) - date.fromisoformat(d2)).days)


def adjudicate(claim: Claim, rules: CoverageRules) -> dict:
    """Return {"decision": PAY|DENY|REVIEW, "reasons": [...]}."""

    # 1. Does this policy govern the billed procedure?
    if claim.procedure_code != rules.procedure_code:
        return {
            "decision": "REVIEW",
            "reasons": [
                f"Procedure {claim.procedure_code} is not governed by this policy "
                f"(policy covers {rules.procedure_code})."
            ],
        }

    # 2. Diagnosis coverage (prefix match on normalized ICD-10 codes).
    dx = _norm(claim.diagnosis_code)
    covered = any(dx.startswith(_norm(p)) for p in rules.covered_diagnosis_prefixes)
    if not covered:
        return {
            "decision": "DENY",
            "reasons": [
                f"Diagnosis {claim.diagnosis_code} is not within the covered set "
                f"{rules.covered_diagnosis_prefixes}."
            ],
        }

    # 3. Frequency limitation.
    if rules.frequency_limit_days and claim.last_service_date:
        gap = _days_between(claim.service_date, claim.last_service_date)
        if gap < rules.frequency_limit_days:
            return {
                "decision": "DENY",
                "reasons": [
                    f"Frequency limit not met: {gap} days since the last covered "
                    f"device, policy requires at least {rules.frequency_limit_days}."
                ],
            }

    # 4. Prior authorization.
    if rules.prior_auth_required and not claim.prior_auth_obtained:
        return {
            "decision": "REVIEW",
            "reasons": ["Prior authorization is required but not on file."],
        }

    return {
        "decision": "PAY",
        "reasons": [
            "Meets procedure, diagnosis, frequency, and authorization criteria."
        ],
    }
