"""
🎓 Agentic Multimodal AI Demo - College Course Walkthrough
==========================================================
An educational demonstration showcasing agentic AI capabilities for a 
college course on Generative AI Applications.

This demo combines:
- 🖼️ Image Analysis (Vision-Language Models)
- 🎬 Video Analysis (Frame-by-frame understanding)
- 🎨 Image Generation (Text-to-Image synthesis)
- 🤖 Agentic Workflows (LangGraph orchestration)

Author: Educational Demo for GenAI Course
Date: January 2026

Learning Objectives:
1. Understand how AI agents orchestrate multiple capabilities
2. See practical applications of vision-language models
3. Learn about stateful workflow management with LangGraph
4. Explore the full cycle: Analyze → Understand → Create
"""

import streamlit as st
import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image
import cv2
import time
import os
import json
from typing import Dict, List, Any, TypedDict, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

# ============================================================================
# 📚 EDUCATIONAL SECTION: Core Concepts for the Course
# ============================================================================

COURSE_INTRO = """
## 🎓 Welcome to the Agentic AI Demonstration

This application demonstrates key concepts in **Generative AI** and **Agentic Systems**:

### What You'll Learn:

1. **Vision-Language Models (VLMs)**
   - How AI can "see" and "understand" images
   - Multimodal processing combining vision and language

2. **Agentic AI Architecture**
   - Autonomous decision-making systems
   - Multi-step reasoning and planning
   - State management with LangGraph

3. **The Generative AI Pipeline**
   - Analysis → Understanding → Creation cycle
   - How to chain AI capabilities together

4. **Practical Applications**
   - Quality inspection systems
   - Content creation workflows
   - Visual Q&A systems

### 🔄 The Demo Flow:
```
[Image/Video Input] → [AI Analysis] → [Understanding] → [Creative Generation]
```
"""

# ============================================================================
# 🎯 Task Types and Workflow Definitions
# ============================================================================

class DemoMode(Enum):
    """Demonstration modes available"""
    IMAGE_ANALYSIS = "🖼️ Image Analysis"
    VIDEO_ANALYSIS = "🎬 Video Analysis"
    IMAGE_GENERATION = "🎨 Image Generation"
    FULL_PIPELINE = "🔄 Full Agentic Pipeline"

class TaskType(Enum):
    """Types of visual analysis tasks"""
    DESCRIBE = "describe"
    DETECT_OBJECTS = "detect_objects"
    ANALYZE_SCENE = "analyze_scene"
    EXTRACT_TEXT = "extract_text"
    QUALITY_CHECK = "quality_check"
    CREATIVE_PROMPT = "creative_prompt"

@dataclass
class AnalysisStep:
    """Represents a single step in the agent's workflow"""
    step_number: int
    task_type: TaskType
    prompt: str
    result: str = ""
    confidence: str = ""
    timestamp: str = ""
    processing_time: float = 0.0
    retry_count: int = 0  # 📚 NEW: Track retry attempts for quality validation

@dataclass
class AgenticState:
    """State maintained throughout the agentic workflow"""
    mode: str
    current_step: int
    total_steps: int
    image_analysis: List[AnalysisStep]
    video_analysis: List[Dict]
    generated_images: List[Dict]
    final_summary: str
    start_time: float
    end_time: float

# ============================================================================
# 🧠 Educational Agentic Workflow Class
# ============================================================================

class EducationalAgenticDemo:
    """
    Educational demonstration of agentic AI capabilities.
    
    This class showcases how modern AI systems can:
    1. Process visual information (images/video)
    2. Understand and reason about content
    3. Generate new creative content
    4. Orchestrate complex multi-step workflows
    """
    
    def __init__(self, vlm_pipeline=None, image_gen_service=None):
        """
        Initialize the educational demo agent.
        
        Args:
            vlm_pipeline: Vision-Language Model pipeline (OpenVINO GenAI)
            image_gen_service: Image generation service (optional)
        """
        self.vlm_pipeline = vlm_pipeline
        self.image_gen_service = image_gen_service
        self.analysis_history = []
        self.workflow_log = []
        
    def log_step(self, step_name: str, details: str):
        """Log a workflow step for educational purposes"""
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "step": step_name,
            "details": details
        }
        self.workflow_log.append(entry)
        return entry
    
    def process_image_for_model(self, image: Image.Image) -> ov.Tensor:
        """
        📚 EDUCATIONAL: Convert PIL Image to OpenVINO Tensor
        
        This step is crucial for feeding images into AI models:
        1. Resize to prevent memory issues
        2. Convert to numpy array
        3. Reshape to expected dimensions [batch, height, width, channels]
        4. Create OpenVINO tensor for GPU acceleration
        """
        self.log_step("Image Preprocessing", 
                     f"Converting image {image.size} to model tensor")
        
        # Resize large images
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        
        # Convert to numpy and create tensor
        image_data = np.array(image.getdata()).reshape(
            1, image.size[1], image.size[0], 3
        ).astype(np.uint8)
        
        return ov.Tensor(image_data)
    
    def analyze_with_vlm(self, image_tensor, prompt: str, 
                         max_tokens: int = 200) -> tuple:
        """
        📚 EDUCATIONAL: Vision-Language Model Analysis
        
        This demonstrates how VLMs process images:
        1. System prompt defines the AI's role and behavior
        2. User prompt contains the specific question
        3. The model generates text based on visual understanding
        """
        if self.vlm_pipeline is None:
            return "Model not loaded", 0.0
        
        system_prompt = """You are an expert visual analysis AI assistant for an educational demonstration.
Provide clear, accurate, and educational explanations of what you observe.
Focus on details that would help students understand AI vision capabilities."""
        
        template = "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n"
        formatted_prompt = template.format(system_prompt, prompt)
        
        self.log_step("VLM Analysis", f"Prompt: {prompt[:50]}...")
        
        start_time = time.time()
        try:
            result = str(self.vlm_pipeline.generate(
                formatted_prompt, 
                image=image_tensor, 
                max_new_tokens=max_tokens
            ))
            
            # Clean up response
            result = result.split("User:")[0].split("Assistant:")[0].split("#")[0].strip()
            processing_time = time.time() - start_time
            
            return result, processing_time
            
        except Exception as e:
            return f"Error: {str(e)}", 0.0
    
    def execute_analysis_workflow(self, image: Image.Image, 
                                  workflow_type: str = "comprehensive") -> List[AnalysisStep]:
        """
        📚 EDUCATIONAL: Multi-Step Agentic Workflow
        
        This demonstrates the core concept of agentic AI:
        1. Define a sequence of analysis steps
        2. Execute each step autonomously
        3. Build understanding incrementally
        4. Each step can inform the next (chaining)
        """
        workflows = {
            "comprehensive": [
                AnalysisStep(1, TaskType.DESCRIBE, 
                           "Describe this image in detail. What is the main subject?"),
                AnalysisStep(2, TaskType.DETECT_OBJECTS, 
                           "List all distinct objects you can identify in this image."),
                AnalysisStep(3, TaskType.ANALYZE_SCENE, 
                           "What is the context or setting? What story does this image tell?"),
                AnalysisStep(4, TaskType.QUALITY_CHECK, 
                           "Assess the image quality: lighting, composition, clarity."),
                AnalysisStep(5, TaskType.CREATIVE_PROMPT, 
                           "Based on this image, suggest a creative prompt to generate a similar or enhanced version.")
            ],
            "quick": [
                AnalysisStep(1, TaskType.DESCRIBE, 
                           "Briefly describe what you see in this image."),
                AnalysisStep(2, TaskType.CREATIVE_PROMPT, 
                           "Create a detailed image generation prompt based on this image.")
            ],
            "educational": [
                AnalysisStep(1, TaskType.DESCRIBE, 
                           "As an AI vision system, explain what visual features you detect in this image."),
                AnalysisStep(2, TaskType.ANALYZE_SCENE, 
                           "Demonstrate your understanding by explaining the relationships between objects."),
                AnalysisStep(3, TaskType.CREATIVE_PROMPT, 
                           "Show generative AI capabilities by creating a detailed prompt to recreate this scene.")
            ]
        }
        
        selected_workflow = workflows.get(workflow_type, workflows["quick"])
        image_tensor = self.process_image_for_model(image)
        results = []
        max_retries = 2  # Maximum retry attempts per step
        
        for step in selected_workflow:
            retry_count = 0
            step_successful = False
            
            while not step_successful and retry_count <= max_retries:
                if retry_count > 0:
                    self.log_step(f"🔄 Retry {retry_count}/{max_retries}", 
                                 f"Step {step.step_number} - Quality validation failed, retrying...")
                else:
                    self.log_step(f"Step {step.step_number}", step.prompt)
                
                result, proc_time = self.analyze_with_vlm(image_tensor, step.prompt)
                
                # 📚 EDUCATIONAL: Agentic Quality Validation
                # The agent evaluates its own output and decides whether to retry
                if self._is_low_quality_result(result):
                    retry_count += 1
                    if retry_count <= max_retries:
                        self.log_step("⚠️ Quality Check", 
                                     f"Result quality below threshold. Attempt {retry_count}/{max_retries}")
                        continue  # Retry the step
                    else:
                        self.log_step("⚠️ Max Retries", 
                                     f"Step {step.step_number} reached max retries. Proceeding with best result.")
                
                step_successful = True
                step.result = result
                step.processing_time = proc_time
                step.timestamp = datetime.now().strftime("%H:%M:%S")
                step.retry_count = retry_count  # Track how many retries were needed
                results.append(step)
            
        return results
    
    def _is_low_quality_result(self, result: str) -> bool:
        """
        📚 EDUCATIONAL: Quality Validation for Agentic Retry Logic
        
        This method evaluates if a VLM result is low-quality and needs retry.
        This is a KEY AGENTIC PATTERN - the agent evaluates its own outputs.
        
        Low-quality indicators:
        - Very short responses (< 10 chars)
        - Error messages or "I cannot"/"I don't" phrases
        - Empty or whitespace-only results
        - Generic non-answers
        """
        if not result or len(result.strip()) < 10:
            return True
        
        low_quality_phrases = [
            "i cannot", "i can't", "i don't", "i am unable",
            "sorry", "error", "unable to", "not able to",
            "cannot determine", "unclear", "not visible",
            "no image", "image not"
        ]
        
        result_lower = result.lower()
        return any(phrase in result_lower for phrase in low_quality_phrases)
    
    def analyze_video_frames(self, video_path: str, 
                            frame_interval: int = 30,
                            max_frames: int = 5,
                            prompt: str = "Describe what's happening in this frame.") -> List[Dict]:
        """
        📚 EDUCATIONAL: Video Analysis Through Frame Sampling
        
        Videos are analyzed frame-by-frame:
        1. Extract frames at regular intervals
        2. Analyze each frame independently
        3. Build temporal understanding
        4. Summarize the video content
        """
        if not os.path.exists(video_path):
            return [{"error": f"Video file not found: {video_path}"}]
        
        self.log_step("Video Analysis", f"Processing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [{"error": "Could not open video file"}]
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        results = []
        frame_count = 0
        analyzed_count = 0
        
        while cap.isOpened() and analyzed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert OpenCV BGR to PIL RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Analyze frame
                image_tensor = self.process_image_for_model(pil_image)
                result, proc_time = self.analyze_with_vlm(image_tensor, prompt)
                
                timestamp_sec = frame_count / fps if fps > 0 else 0
                
                results.append({
                    "frame_number": frame_count,
                    "timestamp": f"{timestamp_sec:.2f}s",
                    "analysis": result,
                    "processing_time": proc_time
                })
                
                analyzed_count += 1
                self.log_step(f"Frame {frame_count}", 
                            f"Analyzed at {timestamp_sec:.2f}s")
            
            frame_count += 1
        
        cap.release()
        
        return results
    
    def generate_creative_prompt(self, analysis_results: List[AnalysisStep]) -> str:
        """
        📚 EDUCATIONAL: Prompt Engineering from Analysis
        
        Transform analysis into a creative generation prompt:
        1. Extract key visual elements
        2. Identify style and mood
        3. Construct detailed prompt for image generation
        """
        prompt_parts = []
        
        for step in analysis_results:
            if step.task_type == TaskType.CREATIVE_PROMPT:
                return step.result  # Use directly generated prompt
            elif step.task_type == TaskType.DESCRIBE:
                prompt_parts.append(f"Scene: {step.result}")
            elif step.task_type == TaskType.ANALYZE_SCENE:
                prompt_parts.append(f"Context: {step.result}")
        
        # Construct prompt from gathered information
        if prompt_parts:
            base_prompt = " | ".join(prompt_parts)
            enhanced_prompt = f"{base_prompt}, highly detailed, professional photography, 8k resolution"
            return enhanced_prompt
        
        return "A detailed, high-quality image"
    
    def get_workflow_summary(self) -> Dict:
        """Generate a summary of the entire workflow for educational purposes"""
        return {
            "total_steps": len(self.workflow_log),
            "workflow_log": self.workflow_log,
            "analysis_count": len(self.analysis_history),
            "concepts_demonstrated": [
                "Vision-Language Model (VLM) inference",
                "Multi-step agentic reasoning",
                "State management across steps",
                "Cross-modal understanding (vision → language)",
                "Creative generation from analysis"
            ]
        }


# ============================================================================
# 🎨 Streamlit UI Components
# ============================================================================

def display_course_intro():
    """Display educational introduction"""
    st.markdown(COURSE_INTRO)
    
    with st.expander("📖 Key AI Concepts Explained"):
        st.markdown("""
        ### Vision-Language Models (VLMs)
        VLMs are AI systems that can process both images and text together. 
        They use transformer architectures to understand visual content and 
        generate natural language descriptions.
        
        **Example Models**: Phi-3.5-Vision, LLaVA, GPT-4V
        
        ### Agentic AI
        Agentic AI systems can:
        - **Plan**: Break down complex tasks into steps
        - **Execute**: Perform actions autonomously  
        - **Reason**: Make decisions based on observations
        - **Learn**: Improve from feedback
        
        ### LangGraph for Orchestration
        LangGraph provides a framework for building stateful, multi-step AI workflows.
        Key concepts:
        - **Nodes**: Individual processing steps
        - **Edges**: Connections between steps
        - **State**: Information that flows through the graph
        - **Conditional routing**: Dynamic workflow paths
        """)

def display_workflow_visualization():
    """Show the agentic workflow visualization"""
    st.markdown("""
    ### 🔄 Agentic Workflow Architecture
    
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                   AGENTIC MULTIMODAL DEMO                    │
    └─────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
           ┌────────────────┐            ┌────────────────┐
           │  Image Input   │            │  Video Input   │
           │  (Camera/File) │            │  (File/Stream) │
           └────────────────┘            └────────────────┘
                    │                              │
                    └──────────────┬──────────────┘
                                   ▼
                         ┌─────────────────┐
                         │   PREPROCESSING │
                         │  (Resize/Tensor) │
                         └─────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │    VLM ANALYSIS LOOP     │◄──┐
                    │  (Vision-Language Model) │   │
                    └──────────────────────────┘   │
                                   │               │
                                   ▼               │
                        ┌───────────────────┐     │
                        │ Step Complete?    │─No──┘
                        └───────────────────┘
                                   │ Yes
                                   ▼
                    ┌──────────────────────────┐
                    │   DECISION & SUMMARY     │
                    │  (Aggregate findings)    │
                    └──────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
           ┌─────────────────┐          ┌─────────────────┐
           │  Analysis Report │          │ Generate Prompt │
           │  (Understanding)  │          │ (For Creation)  │
           └─────────────────┘          └─────────────────┘
                                               │
                                               ▼
                                   ┌─────────────────────┐
                                   │  IMAGE GENERATION   │
                                   │  (Text-to-Image)    │
                                   └─────────────────────┘
                                               │
                                               ▼
                                        [OUTPUT IMAGE]
    ```
    """)

def initialize_session_state():
    """Initialize Streamlit session state"""
    defaults = {
        "model_loaded": False,
        "vlm_pipeline": None,
        "demo_agent": None,
        "analysis_results": [],
        "generated_images": [],
        "workflow_log": [],
        "current_mode": DemoMode.IMAGE_ANALYSIS.value
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_vlm_model(model_path: str, device: str = "GPU"):
    """Load the Vision-Language Model"""
    with st.spinner("🧠 Loading Vision-Language Model... This may take a moment."):
        try:
            pipeline = ov_genai.VLMPipeline(str(model_path), device)
            # Phi-3.5 ends its turn with <|end|> (token 32007), but the exported
            # generation_config.json only lists </s>: stop there, or it keeps writing.
            _cfg = pipeline.get_generation_config()
            _cfg.stop_token_ids = {32007}
            pipeline.set_generation_config(_cfg)
            st.session_state["vlm_pipeline"] = pipeline
            st.session_state["demo_agent"] = EducationalAgenticDemo(vlm_pipeline=pipeline)
            st.session_state["model_loaded"] = True
            return True
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            return False

def render_sidebar():
    """Render the sidebar with model settings and mode selection"""
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/64/Intel_logo_%282006-2020%29.svg", width=100)
        st.title("🎓 Demo Settings")
        
        st.markdown("---")
        
        # Mode selection
        st.subheader("📌 Demo Mode")
        mode = st.selectbox(
            "Select demonstration mode:",
            [m.value for m in DemoMode],
            help="Choose which AI capabilities to demonstrate"
        )
        st.session_state["current_mode"] = mode
        
        st.markdown("---")
        
        # Model settings
        st.subheader("🤖 Model Configuration")
        
        model_path = st.text_input(
            "VLM Model Path:",
            value="./Phi-3.5-vision-instruct-int4-ov",
            help="Path to OpenVINO Vision-Language Model"
        )
        
        device = st.selectbox(
            "Inference Device:",
            ["GPU", "CPU", "NPU"],
            help="Hardware for AI inference"
        )
        
        if st.button("🚀 Load Model", width='stretch'):
            load_vlm_model(model_path, device)
        
        # Status indicator
        if st.session_state["model_loaded"]:
            st.success("✅ Model Ready")
        else:
            st.warning("⚠️ Model Not Loaded")
        
        st.markdown("---")
        
        # Educational info
        st.subheader("📚 Learning Resources")
        st.markdown("""
        - [OpenVINO GenAI Docs](https://docs.openvino.ai)
        - [LangGraph Tutorial](https://langchain-ai.github.io/langgraph/)
        - [Hugging Face Models](https://huggingface.co/models)
        """)
        
        st.markdown("---")
        st.caption("🎓 GenAI Course Demo v1.0")

def render_image_analysis_demo():
    """Render the image analysis demonstration"""
    st.header("🖼️ Image Analysis Demonstration")
    
    st.markdown("""
    **Learning Objective:** Understand how Vision-Language Models analyze images
    through multi-step agentic workflows.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Input Image")
        
        input_method = st.radio(
            "Image Source:",
            ["📁 Upload File", "📷 Camera Capture", "🌐 Sample Image"]
        )
        
        image = None
        
        if input_method == "📁 Upload File":
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Upload any image for analysis"
            )
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                
        elif input_method == "📷 Camera Capture":
            camera_image = st.camera_input("Take a photo")
            if camera_image:
                image = Image.open(camera_image).convert("RGB")
                
        elif input_method == "🌐 Sample Image":
            # Create a sample gradient image for demo
            sample = np.zeros((256, 256, 3), dtype=np.uint8)
            sample[:, :, 0] = np.linspace(0, 255, 256).astype(np.uint8)  # Red gradient
            sample[:, :, 1] = np.linspace(255, 0, 256).reshape(-1, 1).astype(np.uint8)  # Green
            sample[:, :, 2] = 128  # Blue constant
            image = Image.fromarray(sample)
            st.info("📌 Using sample gradient image. Upload your own for better results!")
        
        if image:
            st.image(image, caption="Input Image", use_container_width=True)
    
    with col2:
        st.subheader("🔬 Analysis Settings")
        
        workflow_type = st.selectbox(
            "Analysis Workflow:",
            ["educational", "comprehensive", "quick"],
            help="Select the depth of analysis"
        )
        
        with st.expander("📖 Workflow Details"):
            if workflow_type == "educational":
                st.markdown("""
                **Educational Workflow** (3 steps):
                1. Visual feature detection explanation
                2. Object relationship analysis
                3. Generative prompt creation
                """)
            elif workflow_type == "comprehensive":
                st.markdown("""
                **Comprehensive Workflow** (5 steps):
                1. Detailed description
                2. Object detection
                3. Scene context analysis
                4. Quality assessment
                5. Creative prompt generation
                """)
            else:
                st.markdown("""
                **Quick Workflow** (2 steps):
                1. Brief description
                2. Generation prompt creation
                """)
        
        if st.button("🚀 Run Analysis", width='stretch', 
                    disabled=not st.session_state["model_loaded"] or image is None):
            if st.session_state["demo_agent"] and image:
                with st.spinner("🤖 Agent analyzing image..."):
                    results = st.session_state["demo_agent"].execute_analysis_workflow(
                        image, workflow_type
                    )
                    st.session_state["analysis_results"] = results
    
    # Display results
    if st.session_state.get("analysis_results"):
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        for step in st.session_state["analysis_results"]:
            with st.expander(f"Step {step.step_number}: {step.task_type.value.replace('_', ' ').title()}", 
                           expanded=True):
                st.markdown(f"**Prompt:** {step.prompt}")
                st.markdown(f"**Result:** {step.result}")
                st.caption(f"⏱️ Processing time: {step.processing_time:.2f}s | 🕐 {step.timestamp}")
        
        # Show generated prompt
        if st.session_state["demo_agent"]:
            creative_prompt = st.session_state["demo_agent"].generate_creative_prompt(
                st.session_state["analysis_results"]
            )
            st.markdown("---")
            st.subheader("🎨 Generated Creative Prompt")
            st.info(creative_prompt)
            st.caption("This prompt can be used for image generation!")

def render_video_analysis_demo():
    """Render the video analysis demonstration"""
    st.header("🎬 Video Analysis Demonstration")
    
    st.markdown("""
    **Learning Objective:** Understand how AI processes video through frame-by-frame 
    analysis and temporal understanding.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Video Input")
        
        video_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a video for frame-by-frame analysis"
        )
        
        if video_file:
            # Save temporarily
            temp_path = f"temp_video_{int(time.time())}.mp4"
            with open(temp_path, "wb") as f:
                f.write(video_file.read())
            st.video(video_file)
            st.session_state["temp_video_path"] = temp_path
    
    with col2:
        st.subheader("⚙️ Analysis Settings")
        
        frame_interval = st.slider(
            "Frame Interval:",
            min_value=10, max_value=100, value=30,
            help="Analyze every Nth frame"
        )
        
        max_frames = st.slider(
            "Maximum Frames:",
            min_value=1, max_value=10, value=5,
            help="Maximum number of frames to analyze"
        )
        
        analysis_prompt = st.text_area(
            "Analysis Prompt:",
            value="Describe what is happening in this frame. Note any important actions or changes.",
            help="Question to ask about each frame"
        )
        
        if st.button("🎬 Analyze Video", width='stretch',
                    disabled=not st.session_state["model_loaded"] or "temp_video_path" not in st.session_state):
            if st.session_state["demo_agent"]:
                with st.spinner("🤖 Analyzing video frames..."):
                    progress_bar = st.progress(0)
                    results = st.session_state["demo_agent"].analyze_video_frames(
                        st.session_state["temp_video_path"],
                        frame_interval=frame_interval,
                        max_frames=max_frames,
                        prompt=analysis_prompt
                    )
                    progress_bar.progress(100)
                    st.session_state["video_analysis_results"] = results
    
    # Display video analysis results
    if st.session_state.get("video_analysis_results"):
        st.markdown("---")
        st.subheader("📊 Frame Analysis Results")
        
        for result in st.session_state["video_analysis_results"]:
            if "error" in result:
                st.error(result["error"])
            else:
                with st.expander(f"Frame {result['frame_number']} @ {result['timestamp']}", expanded=True):
                    st.markdown(result["analysis"])
                    st.caption(f"⏱️ Processing: {result['processing_time']:.2f}s")

def render_image_generation_demo():
    """Render the image generation demonstration"""
    st.header("🎨 Image Generation Demonstration")
    
    st.markdown("""
    **Learning Objective:** Understand text-to-image generation and prompt engineering.
    
    ⚠️ **Note:** Image generation requires additional model setup. 
    See README_IMAGE_GENERATION.md for instructions.
    """)
    
    st.subheader("✍️ Prompt Engineering")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Generation Prompt:",
            value="A friendly robot teaching students in a futuristic classroom, digital art, vibrant colors, highly detailed",
            height=100,
            help="Describe the image you want to generate"
        )
        
        negative_prompt = st.text_input(
            "Negative Prompt (what to avoid):",
            value="blurry, low quality, distorted",
            help="Elements to exclude from the generated image"
        )
    
    with col2:
        st.markdown("**Generation Settings**")
        width = st.select_slider("Width:", options=[256, 384, 512, 640, 768], value=512)
        height = st.select_slider("Height:", options=[256, 384, 512, 640, 768], value=512)
        steps = st.slider("Inference Steps:", 10, 50, 20)
    
    st.markdown("---")
    st.subheader("💡 Prompt Engineering Tips")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **Do include:**
        - Subject description
        - Art style (digital art, oil painting, photo)
        - Lighting (soft, dramatic, natural)
        - Quality keywords (detailed, 8k, masterpiece)
        """)
    
    with tips_col2:
        st.markdown("""
        **Structure:**
        ```
        [Subject] + [Action] + [Setting] + 
        [Style] + [Lighting] + [Quality]
        ```
        
        **Example:**
        "A curious cat exploring a magical library, 
        watercolor style, soft ambient lighting, 
        highly detailed, professional illustration"
        """)
    
    # Generation button (placeholder - needs image gen model)
    if st.button("🎨 Generate Image", width='stretch'):
        st.warning("""
        ⚠️ Image generation model not configured in this demo.
        
        To enable:
        1. Download Stable Diffusion OpenVINO model
        2. Place in ../stable-diffusion-ov or ../dreamlike_anime_1_0_ov/FP16
        3. See README_IMAGE_GENERATION.md for details
        """)

def render_full_pipeline_demo():
    """Render the full agentic pipeline demonstration"""
    st.header("🔄 Full Agentic Pipeline Demonstration")
    
    st.markdown("""
    **Learning Objective:** Experience the complete agentic AI workflow from 
    visual analysis to creative generation.
    
    This demonstrates the **Analyze → Understand → Create** cycle.
    """)
    
    # Workflow visualization
    display_workflow_visualization()
    
    st.markdown("---")
    
    # Pipeline steps
    st.subheader("🚀 Execute Full Pipeline")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_image = st.file_uploader(
            "📥 Upload starting image:",
            type=["jpg", "jpeg", "png"],
            key="pipeline_image"
        )
        
        if uploaded_image:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Input Image", use_container_width=True)
    
    with col2:
        st.markdown("**Pipeline Steps:**")
        st.markdown("""
        1. ✅ **Preprocess** - Prepare image for analysis
        2. 🔍 **Analyze** - Multi-step visual understanding
        3. 💭 **Reason** - Extract key concepts
        4. ✍️ **Generate Prompt** - Create image prompt
        5. 🎨 **Create** - Generate new image (if available)
        """)
        
        if st.button("▶️ Run Full Pipeline", width='stretch',
                    disabled=not st.session_state["model_loaded"] or uploaded_image is None):
            if st.session_state["demo_agent"] and uploaded_image:
                st.session_state["pipeline_running"] = True
                
                # Progress tracking
                progress = st.progress(0)
                status = st.empty()
                
                # Step 1: Preprocess
                status.info("🔄 Step 1/4: Preprocessing image...")
                progress.progress(25)
                time.sleep(0.5)
                
                # Step 2: Analyze
                status.info("🔄 Step 2/4: Running visual analysis...")
                with st.spinner("Analyzing..."):
                    results = st.session_state["demo_agent"].execute_analysis_workflow(
                        image, "educational"
                    )
                progress.progress(50)
                
                # Step 3: Generate prompt
                status.info("🔄 Step 3/4: Generating creative prompt...")
                creative_prompt = st.session_state["demo_agent"].generate_creative_prompt(results)
                progress.progress(75)
                
                # Step 4: Summary
                status.info("🔄 Step 4/4: Generating summary...")
                summary = st.session_state["demo_agent"].get_workflow_summary()
                progress.progress(100)
                
                status.success("✅ Pipeline complete!")
                
                st.session_state["pipeline_results"] = {
                    "analysis": results,
                    "prompt": creative_prompt,
                    "summary": summary
                }
    
    # Display pipeline results
    if st.session_state.get("pipeline_results"):
        st.markdown("---")
        st.subheader("📊 Pipeline Results")
        
        tabs = st.tabs(["📝 Analysis", "🎨 Generated Prompt", "📈 Workflow Summary"])
        
        with tabs[0]:
            for step in st.session_state["pipeline_results"]["analysis"]:
                st.markdown(f"**Step {step.step_number}:** {step.result}")
        
        with tabs[1]:
            st.success(st.session_state["pipeline_results"]["prompt"])
            st.caption("Copy this prompt to use with any text-to-image generator!")
        
        with tabs[2]:
            summary = st.session_state["pipeline_results"]["summary"]
            st.json(summary)
            
            st.markdown("**Concepts Demonstrated:**")
            for concept in summary["concepts_demonstrated"]:
                st.markdown(f"- ✓ {concept}")

# ============================================================================
# 🚀 Main Application
# ============================================================================

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="🎓 Agentic AI Demo - GenAI Course",
        page_icon="🤖",  # Note: Emoji icons may not display in some browsers; use a .ico/.png file for reliable display
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        /* Main app gradient background */
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        
        /* Main content text - beige/cream */
        .stApp h1 {
            color: #F5DEB3 !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .stApp h2, .stApp h3, .stApp h4 {
            color: #FAEBD7 !important;
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(90deg, #4a90d9 0%, #67b26f 100%);
            color: white !important;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #67b26f 0%, #4a90d9 100%);
        }
        
        /* Sidebar headers gold */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #FFD700 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.title("🤖 Agentic Multimodal AI Demo")
    st.markdown("### 🎓 College Course on Generative AI Applications")
    
    # Course introduction
    with st.expander("📚 Course Introduction & Learning Objectives", expanded=False):
        display_course_intro()
    
    st.markdown("---")
    
    # Render appropriate demo based on mode
    current_mode = st.session_state.get("current_mode", DemoMode.IMAGE_ANALYSIS.value)
    
    if current_mode == DemoMode.IMAGE_ANALYSIS.value:
        render_image_analysis_demo()
    elif current_mode == DemoMode.VIDEO_ANALYSIS.value:
        render_video_analysis_demo()
    elif current_mode == DemoMode.IMAGE_GENERATION.value:
        render_image_generation_demo()
    elif current_mode == DemoMode.FULL_PIPELINE.value:
        render_full_pipeline_demo()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🎓 Educational Demo for Generative AI Course | Powered by OpenVINO GenAI</p>
        <p>Demonstrating: Vision-Language Models • Agentic Workflows • Image Generation</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
