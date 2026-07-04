"""Deterministic tests. Run with: pytest -q  (no API key required)."""

from policy_agent.rule_engine import CoverageRules, Claim, adjudicate
from policy_agent.vectorstore import chunk_policy, VectorStore

RULES = CoverageRules(
    procedure_code="E2103",
    covered_diagnosis_prefixes=["E10", "E11"],
    frequency_limit_days=365,
    prior_auth_required=True,
)


def _claim(**kw):
    base = dict(
        claim_id="C",
        procedure_code="E2103",
        diagnosis_code="E11.9",
        service_date="2026-06-01",
        last_service_date=None,
        prior_auth_obtained=True,
    )
    base.update(kw)
    return Claim(**base)


def test_pay():
    c = _claim(last_service_date="2025-04-10")
    assert adjudicate(c, RULES)["decision"] == "PAY"


def test_deny_diagnosis():
    c = _claim(diagnosis_code="Z00.00")
    assert adjudicate(c, RULES)["decision"] == "DENY"


def test_deny_frequency():
    c = _claim(last_service_date="2026-03-01")  # ~92 days < 365
    assert adjudicate(c, RULES)["decision"] == "DENY"


def test_review_prior_auth():
    c = _claim(prior_auth_obtained=False)
    assert adjudicate(c, RULES)["decision"] == "REVIEW"


def test_review_wrong_procedure():
    c = _claim(procedure_code="99999")
    assert adjudicate(c, RULES)["decision"] == "REVIEW"


def test_retrieval_finds_frequency_section():
    policy = (
        "# Header\nIntro text about the policy.\n\n"
        "## Section 3. Frequency Limitations\n"
        "One device per member per twelve month period, at least 365 days apart.\n\n"
        "## Section 4. Prior Authorization\n"
        "Prior authorization is required for initial claims."
    )

    def toy_embed(texts):
        vocab = ["frequency", "device", "month", "prior", "authorization", "initial"]
        return [[float(t.lower().count(w)) for w in vocab] for t in texts]

    store = VectorStore(toy_embed).build(chunk_policy(policy))
    top = store.search("frequency limitation how often", k=1)[0]
    assert "Frequency" in top["heading"]
