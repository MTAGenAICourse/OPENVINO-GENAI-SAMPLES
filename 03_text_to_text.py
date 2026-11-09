"""
Lab 2 - Chat with a local LLM (OpenVINO GenAI LLMPipeline).

Run:
    python 03_text_to_text.py ./TinyLlama-1.1B-Chat-v1.0/ "Sun is yellow because"
    python 03_text_to_text.py ./TinyLlama-1.1B-Chat-v1.0/ "Why is the sky blue?" --device GPU

Exercises:
    - Time CPU vs GPU with the same prompt (the script prints tokens/sec).
    - Compare INT4 vs FP16 exports: quality and speed.
    - Turn it into a multi-turn chat with pipe.start_chat() / pipe.finish_chat().
"""
import argparse
import time

import openvino_genai as ov_genai


def main():
    parser = argparse.ArgumentParser(description="Lab 2: text generation with a local LLM")
    parser.add_argument("model_dir", help="OpenVINO model folder, e.g. ./TinyLlama-1.1B-Chat-v1.0/")
    parser.add_argument("prompt")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU", "NPU"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading {args.model_dir} on {args.device}...\n")
    pipe = ov_genai.LLMPipeline(args.model_dir, args.device)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    # The model's own chat template is applied automatically
    # (config.apply_chat_template is True by default).

    def streamer(subword):          # print tokens as they arrive
        print(subword, end="", flush=True)
        return False                # False = keep generating

    print(f"Prompt: {args.prompt}\n")
    start = time.time()
    result = pipe.generate([args.prompt], config, streamer)
    elapsed = time.time() - start

    tokens = len(result.perf_metrics.raw_metrics.m_new_token_times) if result.perf_metrics else 0
    rate = result.perf_metrics.get_throughput().mean if result.perf_metrics else 0
    print(f"\n\n--- {args.device} | {elapsed:.1f}s | {tokens} tokens | {rate:.1f} tokens/s")


if __name__ == "__main__":
    main()
