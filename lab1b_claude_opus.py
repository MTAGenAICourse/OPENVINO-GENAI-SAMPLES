"""
Lab 1b - Same task on Claude Opus 5.5 (Anthropic SDK + Messages API)

Compare it with Lab 1a. Opus 5.5 always reasons: tune depth with effort
(low -> max), not temperature. The default effort on Opus 5.5 is "medium".

Setup (once):
    pip install anthropic python-dotenv
    copy .env.example .env      # then fill in ANTHROPIC_API_KEY

Run:
    python lab1b_claude_opus.py
    python lab1b_claude_opus.py --effort high

Try it: same prompt on Sol (Lab 1a) and Opus 5.5 - latency, cost
($2/$10 vs $4/$20 per 1M tokens) and style.

Server-side fallback: if a safety check declines the request, the API reroutes
it to a suitable model inside the same call (fallbacks="default").

Also in Microsoft Foundry: deploy Claude there and use
    anthropic.AnthropicFoundry(api_key=..., resource="<your-resource>")
and drop the betas/fallbacks lines (server-side fallback is not available there).
"""
import argparse
import time

import anthropic
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="Lab 1b: Claude Opus 5.5 via the Messages API")
    parser.add_argument("--prompt", default="Explain RAG in two sentences.")
    parser.add_argument("--effort", default="low",
                        choices=["low", "medium", "high", "xhigh", "max"])
    args = parser.parse_args()

    load_dotenv()                    # reads ANTHROPIC_API_KEY
    client = anthropic.Anthropic()

    start = time.time()
    resp = client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        system="You are a concise tutor.",
        messages=[{"role": "user", "content": args.prompt}],
        output_config={"effort": args.effort},   # default on Opus 5.5: medium
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                     # reroute if a safety check declines
    )
    elapsed = time.time() - start

    if resp.stop_reason == "refusal":
        category = resp.stop_details.category if resp.stop_details else None
        print("Declined:", category)
    else:
        print(next(b.text for b in resp.content if b.type == "text"))

    print(f"\n--- effort={args.effort} | {elapsed:.1f}s | served by {resp.model} | "
          f"input={resp.usage.input_tokens} output={resp.usage.output_tokens} tokens")


if __name__ == "__main__":
    main()
