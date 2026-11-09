# Workshop live demo guide (PAGA-2 hands-on, deck v5)

A run sheet for demoing every lab from `PAGA-2_HandsOn_v5.pptx` live, in slide order.
Each step gives the command, what to show, what students should see and how long it takes.
All timings were measured on the instructor laptop on 2026-11-09.

For student self-study, use [REPRODUCE.md](REPRODUCE.md) instead.

---

## Before class (15 minutes)

1. **Make sure Azure OpenAI bypasses the corporate proxy.** The Azure resource only accepts traffic from its
   private network. If requests go through the corporate proxy (`HTTPS_PROXY`), Lab 1a fails with
   `403 Public access is disabled`. Add `.openai.azure.com` to `NO_PROXY`:

   Permanent, for new terminals (run once):
   ```cmd
   setx NO_PROXY "%NO_PROXY%,.openai.azure.com"
   ```

   For a terminal that is already open:

   | Shell | Command |
   |---|---|
   | PowerShell | `$env:NO_PROXY = "$env:NO_PROXY,.openai.azure.com"` |
   | Command Prompt | `set NO_PROXY=%NO_PROXY%,.openai.azure.com` |
   | Git Bash | `export NO_PROXY="$NO_PROXY,.openai.azure.com" no_proxy="$no_proxy,.openai.azure.com"` |

   To check (PowerShell): `$env:NO_PROXY -split ',' | Select-String openai` should print `.openai.azure.com`.
   Putting `NO_PROXY` in `.env` doesn't work, because `load_dotenv()` doesn't override a value already set in the shell.
   `setx` cuts values off at 1024 characters; if your list is longer, edit it in Windows Settings → Environment variables.
   Students on their own Azure for Students resources don't need this, because those resources allow public access.

2. **Open a new terminal** after running `setx`. Terminals that were already open keep the old `NO_PROXY`.
3. **Activate the environment** and confirm the devices:
   ```bash
   cd C:\Users\gkamhi\OPENVINO-GENAI-SAMPLES
   .venv\Scripts\activate
   python -c "import openvino as ov; print(ov.Core().available_devices)"
   ```
   Expected: `['CPU', 'GPU.0', 'GPU.1', 'NPU']`
4. **Check the cloud keys** (no values are printed):
   ```bash
   python -c "from dotenv import dotenv_values as d; v=d('.env'); print({k: bool(v.get(k)) for k in ['AZURE_OPENAI_ENDPOINT','AZURE_OPENAI_API_KEY','AZURE_OPENAI_DEPLOYMENT','ANTHROPIC_API_KEY']})"
   ```
   All four should be `True`. On this laptop, `AZURE_OPENAI_DEPLOYMENT=gpt-5.4`; the resource has no `gpt-6.1-sol` deployment.
5. **Pre-generate the Lab 4 image.** It takes 1–2.5 minutes on CPU, too long to wait for live:
   ```bash
   python image_gen_ov.py ./stable-diffusion-v1-5-int8-ov "starry night over Jaffa port"
   ```
   Keep the resulting `image.bmp` open in a viewer.
6. **Warm up the vision model once**, so the first live run isn't slowed by GPU compilation:
   ```bash
   python vision_basics.py Data/Robot.jpg "What is the robot doing?" --device GPU
   ```

---

## Part 1: Cloud models (slides 14–15)

### Lab 1a: GPT on Microsoft Foundry (slide 14)
```bash
python lab1a_gpt_sol.py --effort low
python lab1a_gpt_sol.py --effort high
```
- **Show:** the same answer at both efforts, then the token line at the bottom.
- **Expect:** a two-sentence RAG explanation. low: 3.8 s, 66 output tokens · high: 4.5 s, 117 output tokens.
- **Talking point:** effort changes the tokens you pay for more than the answer itself.

### Lab 1b: Claude Opus 5.5 (slide 15)
```bash
python lab1b_claude_opus.py --effort low
python lab1b_claude_opus.py --effort high
```
- **Expect:** a note that server-side fallback isn't enabled for this workspace, then the answer.
  low: 10 s, 810 output tokens · high: 12 s, 955 output tokens. The output count includes Opus's thinking tokens.
- **Talking point:** compare latency, cost ($2/$10 vs $4/$20 per 1M tokens) and writing style with Lab 1a.

---

## Part 2: Local models with OpenVINO (slides 18–20)

### Lab 2: Local LLM, CPU vs GPU (slide 18)
```bash
python 03_text_to_text.py ./TinyLlama-1.1B-Chat-v1.0/ "Sun is yellow because" --device CPU
python 03_text_to_text.py ./TinyLlama-1.1B-Chat-v1.0/ "Sun is yellow because" --device GPU
```
- **Show:** the tokens streaming in, then the `tokens/s` line.
- **Expect:** CPU about 33 tokens/s, GPU about 51–57 tokens/s.
- **Talking point:** the answer is fluent but scientifically wrong. Ask the class to compare it with Lab 1.

### Lab 3: Vision-language model (slide 19)
```bash
python vision_basics.py Data/Robot.jpg "What is the robot doing?" --device GPU
python 02_video_analyzer_cli.py --source Data/all_animals.mp4 --interval 2 --device GPU
```
- **Expect:** "The robot is holding a cup of coffee." in about 2 s on GPU (14 s on CPU).
  The video analysis covers 4 frames, each naming a lion, elephant and zebra, at about 3 s per frame (26 s in total).
- **Talking point:** point to the `stop_token_ids = {32007}` line on the slide. Without it the model keeps writing after its answer.
- **Optional:** `streamlit run 01_camera_to_text.py` for a live webcam demo.

### Lab 4: Speech and images (slide 20)
```bash
python speech_basics.py --record 5
```
- **Show:** say a sentence, e.g. "Starry night over Jaffa port, in a dreamy anime style."
- **Expect:** the transcript in about 0.4 s on CPU. Stay on CPU here: on GPU, Whisper first spends about 15 s compiling.
- **Talking point:** Whisper-base makes small mistakes (earlier test: "Story Knight over Jeffa Port"). Bigger models fix that.
- **Then:** show the pre-generated `image.bmp` and explain the generation command. Don't run it live.

---

## Part 3: Agents (slides 22–25)

### Lab 5: LangGraph agent (slide 22)
```bash
streamlit run 05_agentic_visual_analysis_langgraph.py
```
1. In the sidebar, choose **GPU** and click **🚀 Initialize LangGraph Agent**.
2. Upload `Data/Robot.jpg` and select the **Safety Inspection** workflow.
3. Click **🚀 Execute LangGraph Agent Workflow**.
- **Expect:** 3 steps and 3 VLM calls, in about 8 s on GPU.
- **Talking point:** the graph on the slide, with its conditional edge that loops back to retry a low-quality answer.

### Lab 6: MCP tools (slide 23)
```bash
streamlit run 13_mcp_skills_client.py
```
- **Show:** the server advertising its tools (7, over stdio), then type `calculate (17 + 25) * 2`. The request is routed to the calculator tool, which returns 84.
- **Avoid:** the `chai_*` tools. They need Intel-internal credentials and fail outside that network.
- **Alternative:** `streamlit run 07_mcp_agentic_tutorial.py` for the step-by-step tutorial view.

### Lab 7: Skills with a local brain (slide 24)
```bash
streamlit run 09_skills_agentic_tutorial_langgraph.py
```
1. In the sidebar, select **Qwen3 1.7B (Local, recommended)** and click **🚀 Initialize Agent**.
2. Ask the following. These routed correctly in testing:
   - `What's the weather in Jerusalem?` → weather skill
   - `calculate 12 * 7` → calculator, answer 84
   - `What is 15% of 240?` → calculator, answer 36
- **Expect:** about 2.6 s per request on GPU.
- **Avoid live:** "Remember that…" and sentiment requests. Qwen3 sent those to the wrong skill.
- **Talking point:** select **TinyLlama 1.1B**, click **🔄 Reload**, and ask `calculate 12 * 7` again. It picks the wrong skill (1 of 6 correct in testing), which shows why model size matters for routing.
- **Note:** weather data is simulated.

### Lab 8: Multimodal agent (slide 25)
```bash
streamlit run 06_agentic_multimodal_demo.py
```
1. In the sidebar, click **🚀 Load Model** (choose GPU) and select the **🔄 Full Agentic Pipeline** demo mode.
2. Upload `Data/Robot.jpg` and click **▶️ Run Full Pipeline**.
- **Expect:** 3 analysis steps plus a generated image prompt, in about 29 s on GPU.
- **Talking point:** this covers the "See" stage. The other stages are in Labs 1 and 4, and wiring the full chain is the team design challenge.
- **Optional:** `streamlit run 11_skills_agentic_demo.py`: try `calculate (3 + 4) * 6` (answer 42) and a web search.

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| Lab 1a: `403 Public access is disabled` | The terminal is still using the corporate proxy. Open a **new** terminal (`NO_PROXY` must contain `.openai.azure.com`). |
| Lab 1a: `404 DeploymentNotFound` | `AZURE_OPENAI_DEPLOYMENT` must name a deployment that exists on the resource (here: `gpt-5.4`). |
| Lab 1b: `server_side_fallback not supported` | Expected on this workspace. The script retries automatically without fallback. |
| First GPU run takes 15+ s | GPU compilation on first load. Warm the model up before class, or use CPU. |
| Vision answer rambles past the question | The model is missing `stop_token_ids = {32007}`. All samples in the repo set it. |
| `chai_*` tool fails | Intel-internal service. Skip it in a student demo. |
| Port already in use (Streamlit) | Add `--server.port 8502`, or close the previous app. |
