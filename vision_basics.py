"""
Lab 3 - See the world: ask a local vision-language model about an image.

VLMPipeline with Phi-3.5-vision (INT4, OpenVINO). Runs fully offline.

Run:
    python vision_basics.py Data/Staff.jpeg "How many people are in this photo?"
    python vision_basics.py Data/Robot.jpg "What is the robot doing?" --device GPU

Start on CPU, then try GPU and compare the time.
"""
import argparse
import time

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Lab 3: image question answering with a local VLM")
    parser.add_argument("image", help="path to an image file")
    parser.add_argument("question", help="question about the image")
    parser.add_argument("--model", default="./Phi-3.5-vision-instruct-int4-ov")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    args = parser.parse_args()

    pipe = ov_genai.VLMPipeline(args.model, args.device)

    img = Image.open(args.image).convert("RGB")
    tensor = ov.Tensor(np.array(img)[None])      # shape [1, H, W, 3], uint8

    config = pipe.get_generation_config()
    config.max_new_tokens = 150
    # Phi-3.5 ends its turn with <|end|> (token 32007), but the exported
    # generation_config.json only lists </s>. Without this the model answers,
    # then keeps writing unrelated text until max_new_tokens.
    config.stop_token_ids = {32007}

    start = time.time()
    answer = pipe.generate(args.question, image=tensor, generation_config=config)
    print(answer)
    print(f"\n--- {args.device} | {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
