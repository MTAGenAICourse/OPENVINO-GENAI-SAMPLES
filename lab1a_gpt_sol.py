"""
Lab 1a - First cloud call: GPT-6.1 Sol on Microsoft Foundry (Azure OpenAI v1 API)

Uses the standard OpenAI Python client and the Responses API, pointed at your
Microsoft Foundry deployment. No api_version to manage with the v1 API.

Setup (once):
    pip install openai python-dotenv
    copy .env.example .env      # then fill in AZURE_OPENAI_* values

Run:
    python lab1a_gpt_sol.py
    python lab1a_gpt_sol.py --effort high
    python lab1a_gpt_sol.py --effort low --prompt "Explain LoRA in two sentences."

Try it: run effort low vs high. Compare time, tokens and answer quality.
Note: reasoning models reject temperature / top_p - steer with instructions and effort.
Same code on OpenAI directly: drop base_url and use OPENAI_API_KEY.
"""
import argparse
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="Lab 1a: GPT-6.1 Sol via the Responses API")
    parser.add_argument("--prompt", default="Explain RAG in two sentences.")
    parser.add_argument("--effort", default="low",
                        choices=["low", "medium", "high", "xhigh", "max"])
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/",
    )

    start = time.time()
    resp = client.responses.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],  # e.g. gpt-6.1-sol
        instructions="You are a concise tutor.",
        input=args.prompt,
        reasoning={"effort": args.effort},
    )
    elapsed = time.time() - start

    print(resp.output_text)
    print(f"\n--- effort={args.effort} | {elapsed:.1f}s | "
          f"input={resp.usage.input_tokens} output={resp.usage.output_tokens} tokens")


if __name__ == "__main__":
    main()
