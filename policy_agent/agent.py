"""A minimal ReAct-style loop built on OpenAI function calling.

The loop is deliberately small: send messages + tool schemas, if the model asks
for a tool call, run it and feed the result back, repeat until the model returns
a final answer or we hit the step cap. That is the whole "agent" -- planning and
tool selection are the model's job; execution and the final decision are ours.
"""

import json
import os

from .tools import TOOLS_SCHEMA, make_tool_dispatch

MODEL = os.environ.get("POLICY_AGENT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a healthcare payment-integrity assistant.

You are given one coverage policy (searchable via a tool) and a list of claims.
For the batch of claims:
1. Use search_policy to locate the covered diagnoses, frequency limitation,
   prior-authorization requirement, and exclusions.
2. From the retrieved text, extract a single structured rules object.
3. Call adjudicate_claim once per claim, passing the same extracted rules and, in the citation field, the policy section id you relied on for that claim.
4. Produce a short final summary: for each claim give the decision
   (PAY / DENY / REVIEW), a one-line reason, and the policy section id
   (e.g. Section 2) you relied on.

Rules you must follow:
- Never state coverage criteria that you did not find in the policy text.
- Never decide a claim yourself; the decision must come from adjudicate_claim.
- Cite the section id for every decision.
"""


def run_agent(store, claims, model=MODEL, max_steps=8, verbose=True):
    """Run the loop. Returns (final_text, full_message_transcript)."""
    from openai import OpenAI

    client = OpenAI()
    dispatch = make_tool_dispatch(store)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Adjudicate these claims against the policy:\n"
            + json.dumps(claims, indent=2),
        },
    ]

    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content, messages

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            fn = dispatch[name]
            try:
                result = fn(args["query"]) if name == "search_policy" else fn(**args)
            except Exception as exc:  # surface tool errors back to the model
                result = {"error": str(exc)}
            if verbose:
                short = args.get("query") or args.get("claim_id") or ""
                print(f"  [tool] {name}({short}) -> {json.dumps(result)[:120]}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    return "Reached max steps without a final answer.", messages
