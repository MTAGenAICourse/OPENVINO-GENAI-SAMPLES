import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image
import streamlit as st
import io
import time
import json
from typing import Dict, List, Any, TypedDict, Annotated
from dataclasses import dataclass, asdict, replace
from enum import Enum
from langgraph.graph import StateGraph, END
import operator

class TaskType(Enum):
    """Types of visual analysis tasks the agent can perform"""
    OBJECT_DETECTION = "object_detection"
    QUALITY_INSPECTION = "quality_inspection"
    SAFETY_CHECK = "safety_check"
    COMPARISON = "comparison"
    COUNTING = "counting"
    TEXT_EXTRACTION = "text_extraction"

@dataclass
class AnalysisStep:
    """Represents a single step in the agent's workflow"""
    step_number: int
    task_type: TaskType
    prompt: str
    result: str = ""
    confidence: str = ""
    timestamp: str = ""

# LangGraph State Definition
class AgentState(TypedDict):
    """State that flows through the LangGraph workflow"""
    image_tensor: Any
    workflow: List[AnalysisStep]
    current_step: int
    results: Annotated[List[AnalysisStep], operator.add]
    decision: Dict[str, Any]
    pipeline: Any
    error: str
    retry_count: int  # 📚 NEW: Track retry attempts for quality validation

class LangGraphVisionAgent:
    """LangGraph-powered autonomous vision agent"""
    
    def __init__(self, pipeline, model_name="Phi-3.5-Vision"):
        self.pipeline = pipeline
        self.model_name = model_name
        self.analysis_history = []
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each stage of analysis
        workflow.add_node("preprocess_image", self.preprocess_image_node)
        workflow.add_node("execute_analysis_step", self.execute_analysis_step_node)
        workflow.add_node("check_workflow_complete", self.check_workflow_complete_node)
        workflow.add_node("make_decision", self.make_decision_node)
        workflow.add_node("finalize", self.finalize_node)
        
        # Define edges
        workflow.set_entry_point("preprocess_image")
        workflow.add_edge("preprocess_image", "execute_analysis_step")
        workflow.add_conditional_edges(
            "execute_analysis_step",
            self.should_continue,
            {
                "continue": "check_workflow_complete",
                "complete": "make_decision"
            }
        )
        workflow.add_edge("check_workflow_complete", "execute_analysis_step")
        workflow.add_edge("make_decision", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    # 📚 EDUCATIONAL: every node returns ONLY the keys it changes.
    # LangGraph merges that partial update into the state. "results" has an
    # operator.add reducer, so returning {"results": [step]} APPENDS one item.
    # Returning the whole state would re-append every earlier result each time.

    def preprocess_image_node(self, state: AgentState) -> dict:
        """Node: Preprocess image for model input"""
        # Image is already passed as a tensor; nothing to change
        return {}

    def execute_analysis_step_node(self, state: AgentState) -> dict:
        """Node: Execute a single analysis step, retrying it if the answer is low quality"""
        current_step_idx = state["current_step"]
        workflow = state["workflow"]
        if current_step_idx >= len(workflow):
            return {}
        try:
            step = workflow[current_step_idx]
            image_tensor = state["image_tensor"]
            pipeline = state["pipeline"]
            
            # Format prompt
            system_prompt = """You are a precise visual analysis AI agent. Provide concise, accurate answers.
Focus only on observable facts. Format your response as a single clear statement."""
            
            template = "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n"
            formatted_prompt = template.format(system_prompt, step.prompt)
            
            # Execute analysis
            start_time = time.time()
            result = str(pipeline.generate(formatted_prompt, image=image_tensor, max_new_tokens=200))
            
            # Clean up result
            result = result.split("User:")[0].split("Assistant:")[0].split("#")[0].strip()
            
            # 📚 EDUCATIONAL: Quality Validation & Retry -- the KEY agentic pattern.
            # The agent checks its own output. A low-quality answer is NOT
            # recorded and the step counter does not advance, so the graph
            # loops back and retries this step (up to max_retries times).
            retry_count = state.get("retry_count", 0)
            max_retries = 2
            if self._is_low_quality_result(result) and retry_count < max_retries:
                print(f"🔄 RETRY: Step {step.step_number} produced a low-quality result. "
                      f"Attempt {retry_count + 1}/{max_retries}")
                return {"retry_count": retry_count + 1}
            if self._is_low_quality_result(result):
                print(f"⚠️ WARNING: Step {step.step_number} still low quality after "
                      f"{max_retries} retries. Continuing...")

            # Record a copy of the step (keeps the workflow definition unchanged)
            done = replace(step, result=result,
                           timestamp=time.strftime("%H:%M:%S"),
                           confidence=f"{time.time() - start_time:.2f}s")
            return {"results": [done],                     # appended by the reducer
                    "current_step": current_step_idx + 1,  # move to the next step
                    "retry_count": 0}

        except Exception as e:
            # Record the error and move on, so a failing step cannot loop forever
            return {"error": f"Analysis error: {str(e)}",
                    "current_step": current_step_idx + 1,
                    "retry_count": 0}

    def check_workflow_complete_node(self, state: AgentState) -> dict:
        """Node: Check if workflow is complete"""
        # This is just a pass-through node for clarity
        return {}

    def should_continue(self, state: AgentState) -> str:
        """
        📚 EDUCATIONAL: Conditional Edge

        A routing function only READS the state and returns the name of the
        next branch. LangGraph does not save changes made here, which is why
        the retry bookkeeping lives in execute_analysis_step_node.
          - "complete": every step is done -> make_decision
          - "continue": run the next step, or retry the current one
        """
        if state.get("current_step", 0) >= len(state["workflow"]):
            return "complete"
        return "continue"

    def make_decision_node(self, state: AgentState) -> dict:
        """Node: Make final decision based on all results"""
        try:
            results = state.get("results", [])
            
            decision = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_steps": len(results),
                "findings": [
                    {
                        "step": r.step_number, 
                        "task": r.task_type.value, 
                        "result": r.result
                    } for r in results
                ],
                "summary": self._generate_summary(results),
                "agent_type": "LangGraph Multi-Step Agent"
            }
            
            return {"decision": decision}

        except Exception as e:
            return {"error": f"Decision error: {str(e)}"}

    def finalize_node(self, state: AgentState) -> dict:
        """Node: Finalize and cleanup"""
        # Add results to history
        self.analysis_history.extend(state.get("results", []))
        return {}
    
    def _is_low_quality_result(self, result: str) -> bool:
        """
        📚 EDUCATIONAL: Quality Validation for Agentic Retry Logic
        
        This method evaluates if a VLM result is low-quality and needs retry.
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
    
    def _generate_summary(self, results: List[AnalysisStep]) -> str:
        """Generate a summary of all analysis steps"""
        summary_parts = []
        for r in results:
            summary_parts.append(f"Step {r.step_number} ({r.task_type.value}): {r.result[:100]}")
        return " | ".join(summary_parts)
    
    def process_image(self, image):
        """Process image for model input"""
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        
        image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)
        return ov.Tensor(image_data)
    
    def execute_workflow(self, image, workflow: List[AnalysisStep], progress_callback=None) -> Dict[str, Any]:
        """Execute a complete analysis workflow using LangGraph"""
        image_tensor = self.process_image(image)
        
        # Initialize state
        initial_state = {
            "image_tensor": image_tensor,
            "workflow": workflow,
            "current_step": 0,
            "results": [],
            "decision": {},
            "pipeline": self.pipeline,
            "error": "",
            "retry_count": 0
        }
        
        # Execute the graph
        try:
            # Run the graph once. stream_mode="values" yields the FULL state
            # after every node, so the last item is the final state.
            # (Running the graph a second time would repeat every VLM call.)
            final_state = initial_state
            total = len(workflow)
            for state in self.graph.stream(initial_state, stream_mode="values"):
                final_state = state
                current = state.get("current_step", 0)
                if progress_callback and 0 < current <= total:
                    progress_callback(current - 1, total, workflow[current - 1].prompt)

            return {
                "results": final_state.get("results", []),
                "decision": final_state.get("decision", {}),
                "error": final_state.get("error", "")
            }
            
        except Exception as e:
            return {
                "results": [],
                "decision": {},
                "error": f"Workflow execution error: {str(e)}"
            }


def create_predefined_workflows():
    """Create predefined agent workflows for common tasks"""
    workflows = {
        "Quality Inspection": [
            AnalysisStep(1, TaskType.OBJECT_DETECTION, "What is the main object or product in this image? Describe it briefly."),
            AnalysisStep(2, TaskType.QUALITY_INSPECTION, "Are there any visible defects, damages, or quality issues in the image?"),
            AnalysisStep(3, TaskType.QUALITY_INSPECTION, "Rate the overall quality on a scale of 1-10 and explain why."),
        ],
        "Safety Inspection": [
            AnalysisStep(1, TaskType.OBJECT_DETECTION, "List all people and objects visible in this image."),
            AnalysisStep(2, TaskType.SAFETY_CHECK, "Are there any safety hazards visible? Describe them."),
            AnalysisStep(3, TaskType.SAFETY_CHECK, "Is proper safety equipment being used? Yes or no, and explain."),
        ],
        "Inventory Count": [
            AnalysisStep(1, TaskType.COUNTING, "How many distinct items or objects can you count in this image?"),
            AnalysisStep(2, TaskType.OBJECT_DETECTION, "Categorize the items by type. List the categories."),
            AnalysisStep(3, TaskType.COUNTING, "For each category, provide the count."),
        ],
        "Document Analysis": [
            AnalysisStep(1, TaskType.TEXT_EXTRACTION, "What type of document is this? (form, receipt, letter, etc.)"),
            AnalysisStep(2, TaskType.TEXT_EXTRACTION, "Extract any visible text, numbers, or important information."),
            AnalysisStep(3, TaskType.QUALITY_INSPECTION, "Is the document complete and legible?"),
        ],
        "Scene Understanding": [
            AnalysisStep(1, TaskType.OBJECT_DETECTION, "Describe the overall scene. What is happening?"),
            AnalysisStep(2, TaskType.OBJECT_DETECTION, "List all significant objects and their positions."),
            AnalysisStep(3, TaskType.COMPARISON, "What is the most prominent or important element in this scene?"),
        ],
    }
    return workflows


def initialize_session():
    """Initialize session state"""
    if "model_loaded" not in st.session_state:
        st.session_state["model_loaded"] = False
    if "pipe" not in st.session_state:
        st.session_state["pipe"] = None
    if "agent" not in st.session_state:
        st.session_state["agent"] = None
    if "workflow_results" not in st.session_state:
        st.session_state["workflow_results"] = []


def load_model(model_path, device="GPU"):
    """Load the vision model"""
    with st.spinner("🤖 Loading LangGraph Vision Agent Model..."):
        try:
            pipeline = ov_genai.VLMPipeline(str(model_path), device)
            # Phi-3.5 ends its turn with <|end|> (token 32007), but the exported
            # generation_config.json only lists </s>: stop there, or it keeps writing.
            _cfg = pipeline.get_generation_config()
            _cfg.stop_token_ids = {32007}
            pipeline.set_generation_config(_cfg)
            st.session_state["model_loaded"] = True
            st.session_state["pipe"] = pipeline
            st.session_state["agent"] = LangGraphVisionAgent(pipeline)
            return True
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return False


def main():
    st.set_page_config(page_title="LangGraph Agentic Visual Analysis", layout="wide")
    
    st.title("🤖 LangGraph Agentic Visual Analysis System")
    st.markdown("""
    This application demonstrates an **autonomous AI agent powered by LangGraph** that performs multi-step visual analysis tasks.
    The agent uses a stateful graph workflow to execute complex analyses, make decisions, and provide comprehensive reports.
    
    🔗 **LangGraph Features:**
    - Stateful multi-step workflow execution
    - Conditional routing between analysis steps
    - Structured decision-making pipeline
    - Error handling and recovery
    """)
    
    initialize_session()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ LangGraph Agent Configuration")
        model_path = st.text_input("Model Path", value="./Phi-3.5-vision-instruct-int4-ov/")
        device = st.radio("Device", ["GPU", "CPU"], index=0)
        
        if not st.session_state["model_loaded"]:
            if st.button("🚀 Initialize LangGraph Agent", type="primary"):
                load_model(model_path, device)
        else:
            st.success("✅ LangGraph Agent Ready")
            if st.button("🔄 Reload"):
                st.session_state["model_loaded"] = False
                st.rerun()
        
        st.markdown("---")
        st.markdown("### LangGraph Workflow Nodes")
        st.markdown("""
        1. **Preprocess Image** - Prepare image data
        2. **Execute Analysis Step** - Run vision analysis
        3. **Check Completion** - Evaluate progress
        4. **Make Decision** - Generate insights
        5. **Finalize** - Complete workflow
        """)
        
        st.markdown("---")
        st.markdown("### Agent Capabilities")
        st.markdown("""
        - 🔍 Object Detection
        - ✅ Quality Inspection
        - 🛡️ Safety Checks
        - 📊 Counting & Inventory
        - 📄 Text Extraction
        - 🔄 Comparison Analysis
        """)
    
    if not st.session_state["model_loaded"]:
        st.warning("⚠️ Please initialize the LangGraph agent using the sidebar.")
        return
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📸 Live Analysis", "🔄 Batch Processing", "📊 Results Dashboard"])
    
    with tab1:
        st.header("Live Visual Analysis with LangGraph")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Image Input")
            input_method = st.radio("Input Method", ["Camera", "Upload File"], horizontal=True)
            
            if input_method == "Camera":
                image_data = st.camera_input("Capture Image")
            else:
                image_data = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            
            if image_data:
                image = Image.open(io.BytesIO(image_data.getvalue() if hasattr(image_data, 'getvalue') else image_data.read()))
                st.image(image, caption="Input Image", width="auto")
        
        with col2:
            st.subheader("🎯 Workflow Selection")
            
            workflows = create_predefined_workflows()
            workflow_type = st.selectbox("Select Analysis Workflow", list(workflows.keys()))
            
            # Display workflow steps
            st.markdown("**LangGraph Workflow Steps:**")
            for step in workflows[workflow_type]:
                st.markdown(f"**{step.step_number}.** {step.task_type.value}: *{step.prompt[:80]}...*")
            
            # Custom workflow option
            if st.checkbox("🛠️ Customize Workflow"):
                st.markdown("**Add Custom Steps:**")
                custom_prompt = st.text_input("Custom analysis prompt")
                custom_task = st.selectbox("Task Type", [t.value for t in TaskType])
                if st.button("Add Step"):
                    new_step = AnalysisStep(
                        len(workflows[workflow_type]) + 1,
                        TaskType(custom_task),
                        custom_prompt
                    )
                    workflows[workflow_type].append(new_step)
        
        # Execute workflow
        if image_data and st.button("🚀 Execute LangGraph Agent Workflow", type="primary", use_container_width=True):
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            def update_progress(current, total, message):
                progress_placeholder.progress(current / total, text=f"Step {current + 1}/{total}")
                status_placeholder.info(f"🔄 {message}")
            
            with st.spinner("🤖 LangGraph Agent is analyzing..."):
                workflow_output = st.session_state["agent"].execute_workflow(
                    image, 
                    workflows[workflow_type],
                    update_progress
                )
                
                results = workflow_output.get("results", [])
                decision = workflow_output.get("decision", {})
                error = workflow_output.get("error", "")
                
                if error:
                    st.error(f"Error during workflow execution: {error}")
                
                st.session_state["workflow_results"].append({
                    "image": image_data,
                    "workflow": workflow_type,
                    "results": results,
                    "decision": decision
                })
            
            progress_placeholder.empty()
            status_placeholder.empty()
            
            # Display results
            st.success("✅ LangGraph Analysis Complete!")
            
            st.markdown("---")
            st.markdown("### 📋 Analysis Results")
            
            for result in results:
                with st.expander(f"Step {result.step_number}: {result.task_type.value}", expanded=True):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**Prompt:** {result.prompt}")
                        st.markdown(f"**Result:** {result.result}")
                    with col_b:
                        st.metric("Time", result.confidence)
                        st.caption(f"⏰ {result.timestamp}")
            
            # Agent decision
            st.markdown("---")
            st.markdown("### 🎯 LangGraph Agent Decision Summary")
            st.json(decision)
    
    with tab2:
        st.header("Batch Processing")
        st.info("📦 Upload multiple images for batch analysis with LangGraph (Coming soon)")
        uploaded_files = st.file_uploader("Upload Multiple Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files and st.button("Process Batch"):
            st.info("Batch processing feature will analyze multiple images sequentially using LangGraph workflows")
    
    with tab3:
        st.header("Results Dashboard")
        
        if not st.session_state["workflow_results"]:
            st.info("📊 No analysis results yet. Run an analysis in the Live Analysis tab.")
        else:
            st.markdown(f"### Total Analyses: {len(st.session_state['workflow_results'])}")
            
            for idx, result_data in enumerate(reversed(st.session_state["workflow_results"])):
                with st.expander(f"Analysis #{len(st.session_state['workflow_results']) - idx}: {result_data['workflow']}", expanded=False):
                    col_x, col_y = st.columns([1, 2])
                    
                    with col_x:
                        st.image(result_data["image"], caption="Analyzed Image", width="auto")
                    
                    with col_y:
                        st.markdown("**LangGraph Workflow Results:**")
                        for r in result_data["results"]:
                            st.markdown(f"- **{r.task_type.value}**: {r.result[:150]}...")
                        
                        st.markdown("**Decision:**")
                        st.json(result_data["decision"])
            
            # Export results
            if st.button("📥 Export All Results"):
                export_data = []
                for rd in st.session_state["workflow_results"]:
                    export_data.append({
                        "workflow": rd["workflow"],
                        "decision": rd["decision"],
                        "steps": [asdict(r) for r in rd["results"]]
                    })
                
                st.download_button(
                    label="Download JSON Report",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"langgraph_agent_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()
