# Reproduce every lab

Step-by-step commands for every lab in the PAGA-2 hands-on deck (`PAGA-2_HandsOn_v5.pptx`).
All timings were measured on 2026-11-09 on a Windows 11 laptop with an Intel CPU, an integrated GPU and an NPU,
using Python 3.14 and OpenVINO GenAI 2025.4. Your numbers will differ. Record them in the homework table.

## 1. Setup (once)

```bash
git clone https://github.com/MTAGenAICourse/OPENVINO-GENAI-SAMPLES.git
cd OPENVINO-GENAI-SAMPLES
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # Lab 1 only: fill in your keys. Never commit .env
```

Download the models. They are already converted to OpenVINO format, so you don't need PyTorch.
The total is about 7 GB.

```bash
hf download OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov --local-dir TinyLlama-1.1B-Chat-v1.0
hf download OpenVINO/Phi-3.5-vision-instruct-int4-ov  --local-dir Phi-3.5-vision-instruct-int4-ov
hf download OpenVINO/whisper-base-fp16-ov             --local-dir whisper-base
hf download OpenVINO/Qwen3-1.7B-int4-ov               --local-dir Qwen3-1.7B-int4-ov
hf download OpenVINO/stable-diffusion-v1-5-int8-ov    --local-dir stable-diffusion-v1-5-int8-ov
```

> If you want to export models yourself with `optimum-cli`, do it in a **separate** venv
> (`pip install "optimum-intel[openvino]"`). On 2026-11-09, exporting Whisper failed with
> `NormalizedConfig ... allow_new`, but the downloads above worked.

## 2. Labs

| Lab | Command | Expected | Measured |
|---|---|---|---|
| 1a | `python lab1a_gpt_sol.py --effort low` | short RAG explanation, plus timing and token counts | needs Azure keys |
| 1b | `python lab1b_claude_opus.py --effort low` | same task on Claude Opus 5.5 | needs `ANTHROPIC_API_KEY` |
| 2 | `python 03_text_to_text.py ./TinyLlama-1.1B-Chat-v1.0/ "Sun is yellow because" --device CPU` | streamed answer, then tokens/s | CPU 33 tok/s · GPU 51–57 tok/s |
| 3 | `python vision_basics.py Data/Robot.jpg "What is the robot doing?"` | "The robot is holding a cup of coffee." | CPU 14 s · GPU 2 s |
| 3 | `python 02_video_analyzer_cli.py --source Data/all_animals.mp4 --interval 2` | 4 frames: lion, elephant, zebra | CPU 75 s · GPU 26 s per clip |
| 3 | `streamlit run 01_camera_to_text.py` | webcam Q&A app | — |
| 4 | `python speech_basics.py --record 5` (or `--wav file.wav`, 16 kHz mono) | your sentence as text | 0.4 s for 5 s of audio (CPU) |
| 4 | `python image_gen_ov.py ./stable-diffusion-v1-5-int8-ov "starry night over Jaffa port"` | `image.bmp`, 512×512 | 66 s on CPU |
| 4 | `streamlit run 04_microphone_to_text.py` · `streamlit run image_generation_app.py` | apps | — |
| 5 | `streamlit run 05_agentic_visual_analysis_langgraph.py` | 3-step workflow, 3 VLM calls | 8 s on GPU |
| 6 | `streamlit run 07_mcp_agentic_tutorial.py` · `streamlit run 13_mcp_skills_client.py` | MCP server lists 7 tools; calculator returns results | — |
| 7 | `streamlit run 08_skills_agentic_tutorial.py` · `streamlit run 09_skills_agentic_tutorial_langgraph.py` | skill routing and chaining; 09 uses Qwen3-1.7B by default | ~2.6 s per request (GPU) |
| 8 | `streamlit run 06_agentic_multimodal_demo.py` · `streamlit run 11_skills_agentic_demo.py` | image analysis plus a generated image prompt; tool routing | 29 s per image (GPU) |

## 3. Known limits

- **GPU vs CPU.** Start on CPU. The GPU is about 3× faster for the vision model, but a small model like
  Whisper-base takes about 15 s to compile on the GPU, so the CPU wins overall there.
- **Phi-3.5-vision stop token.** The exported model doesn't stop at `<|end|>`. The samples set
  `stop_token_ids = {32007}`; without it the model rambles after its answer.
- **Local brain (Lab 7).** Qwen3-1.7B routed 4/6 test requests correctly in fast `/no_think` mode
  (6/6 with thinking on, but 5× slower). TinyLlama routed 1/6. Try `Qwen3-4B-int4-ov` on a bigger laptop.
- **Intel-only tools.** The `chai_*` tools in 10–13 need Intel-internal ChAI credentials. Students outside
  Intel can ignore them; the calculator, web search and text tools work without them.
- **Simulated data.** The weather skill returns simulated data, not a live forecast.
