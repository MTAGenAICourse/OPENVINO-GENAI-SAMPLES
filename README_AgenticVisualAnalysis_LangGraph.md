# 🤖 LangGraph Agentic Visual Analysis System

An advanced autonomous AI agent powered by **LangGraph** and OpenVINO GenAI that performs sophisticated multi-step visual analysis using state-of-the-art graph-based workflow orchestration. This application demonstrates the power of LangGraph for building robust, scalable, and maintainable agentic AI systems.

![LangGraph Agentic Visual Analysis](images/langgraph_visual_analysis.png)

## 🌟 Why LangGraph?

### **The LangGraph Advantage**

LangGraph transforms traditional sequential workflows into intelligent, stateful agent systems with several key advantages:

#### **1. 🔄 Stateful Workflow Management**
- **Traditional Approach**: Manual state tracking with global variables or class attributes
- **LangGraph Approach**: Built-in state management with typed state objects that flow through the graph
- **Benefit**: Eliminates state management bugs, makes debugging easier, and enables better testing

#### **2. 🎯 Conditional Routing & Decision Making**
- **Traditional Approach**: Complex if/else chains or switch statements scattered throughout code
- **LangGraph Approach**: Declarative conditional edges that route execution based on state
- **Benefit**: Clear separation of logic, easier to visualize workflows, and simpler to modify decision paths

#### **3. 🧩 Modular Node Architecture**
- **Traditional Approach**: Monolithic functions that are hard to test and reuse
- **LangGraph Approach**: Each processing step is an independent, testable node
- **Benefit**: Nodes can be reused across workflows, tested in isolation, and easily swapped or extended

#### **4. 📊 Workflow Visualization & Debugging**
- **Traditional Approach**: Mental model of execution flow, scattered logging
- **LangGraph Approach**: Visual graph representation with clear node transitions
- **Benefit**: Easier onboarding, faster debugging, and better documentation through visualization

#### **5. 🔁 Cyclical Workflows & Iterative Processing**
- **Traditional Approach**: Manual loop management with complex exit conditions
- **LangGraph Approach**: Native support for cycles with clean termination conditions
- **Benefit**: Perfect for multi-step reasoning, refinement loops, and iterative analysis

#### **6. ⚡ Streaming & Intermediate Results**
- **Traditional Approach**: Wait for entire workflow to complete before seeing results
- **LangGraph Approach**: Stream intermediate states and results in real-time
- **Benefit**: Better user experience, ability to interrupt/modify mid-execution, and progressive enhancement

#### **7. 🛡️ Error Handling & Recovery**
- **Traditional Approach**: Try-catch blocks scattered throughout code
- **LangGraph Approach**: Centralized error handling at node boundaries with recovery paths
- **Benefit**: Graceful degradation, retry logic, and fallback mechanisms built into the graph structure

#### **8. 🔧 Extensibility & Composition**
- **Traditional Approach**: Modifying workflows requires code refactoring
- **LangGraph Approach**: Add/remove nodes and edges without touching existing logic
- **Benefit**: Rapidly prototype new workflows, A/B test different approaches, and compose sub-graphs

## 🆚 Comparison: Standard vs. LangGraph Implementation

### **Standard Implementation** (`05_agentic_visual_analysis.py`)
```python
# Sequential execution with manual state management
class VisionAgent:
    def execute_workflow(self, image, workflow):
        results = []
        for step in workflow:
            result = self.analyze_step(image, step)
            results.append(result)  # Manual state tracking
        return results
```

**Limitations:**
- ❌ No built-in state management
- ❌ Hard to add conditional logic
- ❌ Sequential-only execution
- ❌ Difficult to visualize workflow
- ❌ Error handling is scattered
- ❌ Testing requires full workflow execution

### **LangGraph Implementation** (`05_agentic_visual_analysis_langgraph.py`)
```python
# Graph-based execution with automatic state management
class LangGraphVisionAgent:
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("preprocess_image", self.preprocess_image_node)
        workflow.add_node("execute_analysis_step", self.execute_analysis_step_node)
        workflow.add_conditional_edges("execute_analysis_step", 
                                      self.should_continue,
                                      {"continue": "check_workflow_complete",
                                       "complete": "make_decision"})
        return workflow.compile()
```

**Advantages:**
- ✅ Automatic state propagation
- ✅ Declarative conditional routing
- ✅ Support for parallel and cyclical execution
- ✅ Visual graph representation
- ✅ Centralized error handling
- ✅ Individual nodes can be tested independently

## 🏗️ LangGraph Architecture

### **Graph Workflow Structure**

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Execution Flow                      │
│                   (with Quality Validation)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Preprocess Image │
                    │   (Entry Point)  │
                    └──────────────────┘
                              │
                              ▼
         ┌───────────────────────────────────────┐
         │                                       │
         │  ┌────────────────────────┐           │
  Retry  │  │ Execute Analysis Step  │◄──────┐   │
  Loop   │  │  (Vision Processing)   │       │   │
         │  └────────────────────────┘       │   │
         │            │                       │   │
         │            ▼                       │   │
         │  ┌──────────────────┐             │   │
         │  │ Quality Check    │             │   │
         │  │ (Low Quality?)   │──── Yes ────┘   │
         │  └──────────────────┘  (max 2 retries)│
         │            │ No                        │
         └────────────┼───────────────────────────┘
                      ▼
            ┌──────────────────┐
            │ Conditional Edge │
            │ (More Steps?)    │
            └──────────────────┘
                  │       │
        Complete  │       │  Continue
                  │       └─────────────────┐
                  ▼                         │
         ┌──────────────────┐               │
         │  Make Decision   │               │
         │ (Aggregate & LLM)│               │
         └──────────────────┘               │
                  │                         │
                  ▼                         │
          ┌──────────────┐     ┌───────────┴───────────┐
          │   Finalize   │     │ Check Workflow Complete │
          │   (Cleanup)  │     └───────────────────────┘
          └──────────────┘                   │
                  │                          │
                  ▼                          │
                [END]         (loops back to Execute Step)
```

### **State Management**

The `AgentState` TypedDict flows through every node:

```python
class AgentState(TypedDict):
    image_tensor: Any              # Preprocessed image
    workflow: List[AnalysisStep]   # Workflow definition
    current_step: int              # Progress tracker
    results: List[AnalysisStep]    # Accumulated results
    decision: Dict[str, Any]       # Final decision
    pipeline: Any                  # Model pipeline
    error: str                     # Error tracking
    retry_count: int               # 📚 NEW: Track retry attempts for quality validation
```

**Benefits:**
- Type-safe state management
- Clear contract between nodes
- Automatic state propagation
- Easy to add new state fields
- **NEW: Retry tracking enables quality-aware self-correction**

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenVINO toolkit with GenAI support
- LangGraph and LangChain libraries

### Installation

1. **Install Dependencies**
```bash
pip install openvino openvino-genai streamlit pillow numpy langgraph langchain
```

2. **Download the Vision Model**
```bash
# Place Phi-3.5-vision-instruct model in: ./Phi-3.5-vision-instruct-int4-ov/
```

3. **Run the LangGraph Application**
```bash
streamlit run 05_agentic_visual_analysis_langgraph.py
```

The application will open at `http://localhost:8501`

## 📖 Usage Guide

### 1. Initialize the LangGraph Agent

1. Open the application in your browser
2. In the sidebar, configure:
   - Model path: `./Phi-3.5-vision-instruct-int4-ov/`
   - Device: GPU (recommended) or CPU
3. Click **"🚀 Initialize LangGraph Agent"**
4. Wait for the graph to compile and model to load

### 2. Execute Visual Analysis Workflows

#### **Tab 1: Live Analysis**

1. **Capture or Upload Image**
   - Use Camera or Upload File

2. **Select Workflow**
   - Choose from 5 predefined LangGraph workflows
   - View the node execution sequence

3. **Execute LangGraph Agent**
   - Click **"🚀 Execute LangGraph Agent Workflow"**
   - Watch real-time node execution with state updates
   - See results stream in as each node completes

4. **View Results**
   - Detailed results for each analysis step
   - LangGraph agent decision summary with metadata
   - Execution timing and confidence scores

#### **Tab 2: Results Dashboard**

- Review all previous LangGraph analyses
- Compare workflows and decisions
- Export comprehensive JSON reports

## 🎯 LangGraph Use Cases

### **1. Multi-Step Quality Control**
**Workflow:** Quality Inspection
- **Node 1**: Detect product
- **Node 2**: Identify defects
- **Node 3**: Rate quality
- **Conditional**: If quality < 7, route to additional inspection node

### **2. Adaptive Safety Analysis**
**Workflow:** Safety Inspection
- **Node 1**: Detect people and equipment
- **Node 2**: Check for hazards
- **Conditional**: If hazards found, route to compliance check
- **Node 3**: Verify PPE usage

### **3. Iterative Document Processing**
**Workflow:** Document Analysis
- **Node 1**: Classify document type
- **Conditional**: Route to type-specific extraction node
- **Node 2**: Extract relevant fields
- **Node 3**: Validate completeness
- **Cycle**: If incomplete, request specific missing fields

### **4. Parallel Multi-Device Analysis**
**Advanced:** Run analysis on CPU and GPU simultaneously
- LangGraph supports parallel node execution
- Compare results and select best output
- Optimize for speed vs. accuracy

## 🔍 Technical Deep Dive

### **Node Implementation Pattern**

Each LangGraph node follows a consistent pattern:

```python
def node_name(self, state: AgentState) -> AgentState:
    """
    Node description
    
    Input: Previous state
    Output: Updated state
    """
    try:
        # 1. Extract needed data from state
        data = state["key"]
        
        # 2. Process data
        result = self.process(data)
        
        # 3. Update state
        state["new_key"] = result
        
        # 4. Return updated state
        return state
        
    except Exception as e:
        state["error"] = str(e)
        return state
```

### **Conditional Edge Pattern**

Conditional edges enable dynamic routing with **quality validation and retry logic**:

```python
def should_continue(self, state: AgentState) -> str:
    """
    📚 EDUCATIONAL: Conditional Edge with Quality Validation & Retry
    
    This is the KEY agentic pattern - the agent evaluates its own output
    and decides whether to:
    1. Continue to next step
    2. Retry the current step (if quality is low)
    3. Complete the workflow
    """
    results = state.get("results", [])
    
    # Quality validation - check if last result needs retry
    if results:
        last_result = results[-1]
        if self._is_low_quality_result(last_result.result):
            retry_count = state.get("retry_count", 0)
            max_retries = 2
            
            if retry_count < max_retries:
                # Go back one step and retry
                state["current_step"] -= 1
                state["retry_count"] = retry_count + 1
                state["results"] = state["results"][:-1]  # Remove failed result
                return "continue"  # Retry the step
    
    # Normal flow - check if workflow complete
    if state["current_step"] >= len(state["workflow"]):
        return "complete"
    
    state["retry_count"] = 0  # Reset for new step
    return "continue"
```

#### Quality Validation Helper

```python
def _is_low_quality_result(self, result: str) -> bool:
    """
    📚 EDUCATIONAL: Quality Validation for Agentic Retry Logic
    
    Low-quality indicators:
    - Very short responses (< 10 chars)
    - Error messages or "I cannot"/"I don't" phrases
    - Empty or whitespace-only results
    """
    if not result or len(result.strip()) < 10:
        return True
    
    low_quality_phrases = [
        "i cannot", "i can't", "i don't", "unable to",
        "sorry", "error", "unclear", "not visible"
    ]
    
    return any(phrase in result.lower() for phrase in low_quality_phrases)
```

### **Graph Compilation**

The graph is compiled once and reused:

```python
workflow = StateGraph(AgentState)
workflow.set_entry_point("preprocess_image")
workflow.add_node("node1", func1)
workflow.add_node("node2", func2)
workflow.add_edge("node1", "node2")
workflow.add_conditional_edges("node2", decision_func, routes)
workflow.add_edge("final", END)

graph = workflow.compile()  # Compile once
```

### **Streaming Execution**

Stream intermediate results for real-time updates:

```python
for step_state in self.graph.stream(initial_state):
    # Access intermediate state
    current_step = step_state.get("current_step")
    # Update UI progressively
    update_progress(current_step)
```

## 📊 Performance & Scalability

### **LangGraph Performance Benefits**

| Aspect | Standard | LangGraph | Improvement |
|--------|----------|-----------|-------------|
| **Modularity** | Low | High | 🚀 Easier to test & maintain |
| **Extensibility** | Medium | Very High | 🎯 Add nodes without refactoring |
| **Debuggability** | Low | High | 🔍 Visual inspection & logging |
| **State Management** | Manual | Automatic | ✅ Fewer bugs |
| **Conditional Logic** | Scattered | Declarative | 📊 Clear routing |
| **Error Recovery** | Basic | Advanced | 🛡️ Retry & fallback built-in |
| **Streaming** | Not supported | Native | ⚡ Real-time updates |

### **Scalability Advantages**

1. **Horizontal Scaling**: Add more nodes for parallel processing
2. **Vertical Scaling**: Optimize individual nodes independently
3. **Workflow Versioning**: Run multiple graph versions simultaneously
4. **A/B Testing**: Compare different graph structures with same data

## 🔧 Advanced Configuration

### **Custom Node Creation**

Add new analysis capabilities:

```python
def custom_analysis_node(self, state: AgentState) -> AgentState:
    """Custom analysis logic"""
    # Access state
    image = state["image_tensor"]
    
    # Run custom analysis
    custom_result = my_custom_function(image)
    
    # Update state
    state["custom_data"] = custom_result
    return state

# Add to graph
workflow.add_node("custom_analysis", self.custom_analysis_node)
workflow.add_edge("preprocess_image", "custom_analysis")
```

### **Parallel Node Execution**

Execute multiple analyses simultaneously:

```python
# Add parallel nodes
workflow.add_node("analyze_objects", self.object_node)
workflow.add_node("analyze_text", self.text_node)
workflow.add_node("analyze_quality", self.quality_node)

# Run in parallel
workflow.add_edge("preprocess_image", "analyze_objects")
workflow.add_edge("preprocess_image", "analyze_text")
workflow.add_edge("preprocess_image", "analyze_quality")

# Merge results
workflow.add_node("merge_results", self.merge_node)
workflow.add_edge("analyze_objects", "merge_results")
workflow.add_edge("analyze_text", "merge_results")
workflow.add_edge("analyze_quality", "merge_results")
```

### **Retry Logic & Fallbacks**

Handle errors gracefully:

```python
def retry_node(self, state: AgentState) -> AgentState:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = risky_operation(state)
            state["result"] = result
            return state
        except Exception as e:
            if attempt == max_retries - 1:
                # Fallback logic
                state["result"] = fallback_operation(state)
            time.sleep(2 ** attempt)  # Exponential backoff
    return state
```

## 🐛 Troubleshooting

### **LangGraph-Specific Issues**

#### **State Type Errors**
```
Error: State key not found
```
**Solution**: Ensure all nodes return the complete `AgentState` dict with required keys.

#### **Circular Dependencies**
```
Error: Cycle detected in graph
```
**Solution**: Use conditional edges to break cycles or add proper termination conditions.

#### **Node Execution Order**
```
Warning: Unexpected execution order
```
**Solution**: Verify edge definitions and check for unintended parallel execution.

### **Model Loading Issues**

See main README for standard OpenVINO troubleshooting.

## 📦 Dependencies

### **Core Dependencies**
- `openvino` >= 2024.0.0
- `openvino-genai` >= 2024.0.0
- `streamlit` >= 1.28.0
- `pillow` >= 10.0.0
- `numpy` >= 1.24.0

### **LangGraph Dependencies**
- `langgraph` >= 0.0.50
- `langchain` >= 0.1.0
- `langchain-core` >= 0.1.0

Install all dependencies:
```bash
pip install openvino openvino-genai streamlit pillow numpy langgraph langchain langchain-core
```

## 🎓 Learning Resources

### **LangGraph Documentation**
- [LangGraph Official Docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [Building Agentic Workflows](https://python.langchain.com/docs/langgraph)

### **Related Technologies**
- [OpenVINO GenAI](https://docs.openvino.ai/latest/openvino_docs_GenAI_Introduction.html)
- [Phi-3.5-Vision Model](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

Contributions to enhance LangGraph integration are welcome!

**Areas for Enhancement:**
- Additional conditional routing strategies
- Parallel node execution patterns
- Graph visualization tools
- Custom node templates
- Workflow optimization techniques
- Integration with LangChain tools

## 📄 License

This project is part of the OpenVINO GenAI Samples collection.

## 🙏 Acknowledgments

- Built with [OpenVINO](https://docs.openvino.ai/)
- Powered by [LangGraph](https://langchain-ai.github.io/langgraph/)
- Orchestrated with [LangChain](https://python.langchain.com/)
- Vision model: [Phi-3.5-Vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
- UI framework: [Streamlit](https://streamlit.io/)

## 🔮 Future Enhancements

With LangGraph, the following features are easier to implement:

1. **Human-in-the-Loop**: Add approval nodes for critical decisions
2. **Multi-Agent Collaboration**: Multiple specialized agents working together
3. **Memory & Context**: Long-term memory across workflow executions
4. **Dynamic Workflow Generation**: AI generates optimal workflows for tasks
5. **Workflow Optimization**: A/B test different graph structures automatically
6. **Real-time Adaptation**: Modify graph structure based on runtime feedback

---

**Why Choose LangGraph?** If you're building production-ready agentic AI systems that need to be maintainable, testable, and extensible, LangGraph provides the architectural foundation that traditional sequential approaches lack. The initial learning curve pays dividends in code quality, debugging speed, and system reliability.

**Start with LangGraph when:**
- Building complex multi-step workflows
- Needing conditional logic and dynamic routing
- Requiring state management across steps
- Planning to iterate and extend functionality
- Wanting better debugging and visualization
- Building production systems that need to scale
