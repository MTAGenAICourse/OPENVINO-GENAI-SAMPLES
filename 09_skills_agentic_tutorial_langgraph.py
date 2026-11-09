"""
🎓 Skills-Based Agentic AI with LangGraph & LLM Integration
============================================================
An educational demonstration of the Skills pattern for agentic AI
with LangGraph workflow orchestration and actual LLM invocation.

This tutorial demonstrates:
- 🎯 Skills: Composable, reusable units of AI capability
- 🔗 LangGraph: State-based workflow for skill orchestration
- 🧠 LLM Integration: Actual LLM calls for intelligent skill selection & response
- 📦 Skill Composition: Building complex behaviors from simple skills

Key Differences from 08_skills_agentic_tutorial.py:
- Uses LangGraph for stateful workflow management
- Actually calls LLM for skill selection and response generation
- Implements conditional routing based on LLM decisions
- Includes retry logic and quality validation

Learning Objectives:
1. Understand LangGraph workflow patterns for skill orchestration
2. Learn how LLMs can intelligently select and chain skills
3. See how LangGraph enables sophisticated agentic behaviors
4. Compare with simple skill selection vs LLM-powered selection

Author: Educational Demo for GenAI Course
Date: January 2026
"""

import streamlit as st
import json
import re
from typing import Dict, List, Any, Optional, Callable, TypedDict, Annotated
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
import random
import time
import operator
from pathlib import Path

# LangGraph imports
from langgraph.graph import StateGraph, END

# OpenVINO GenAI imports (optional - graceful fallback)
try:
    import openvino_genai as ov_genai
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("⚠️ OpenVINO GenAI not available. Running in simulation mode.")

# Azure OpenAI imports (optional - graceful fallback)
try:
    from openai import AzureOpenAI
    import dotenv
    dotenv.load_dotenv()
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    print("⚠️ Azure OpenAI not available.")

# ============================================================================
# 📚 EDUCATIONAL SECTION: LangGraph + Skills Architecture
# ============================================================================

SKILLS_LANGGRAPH_INTRO = """
## 🎯 Skills + LangGraph: The Best of Both Worlds

This tutorial combines **Skills** (composable AI capabilities) with **LangGraph** 
(stateful workflow orchestration) and **LLM integration** (intelligent decision making).

### Architecture Overview:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH WORKFLOW                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│  │   ANALYZE   │────▶│   SELECT    │────▶│   EXECUTE   │               │
│  │   (LLM)     │     │   SKILL     │     │   SKILL     │               │
│  └─────────────┘     └─────────────┘     └─────────────┘               │
│         │                  │                    │                        │
│         │                  │                    ▼                        │
│         │            ┌─────────────┐     ┌─────────────┐               │
│         │            │   CHAIN?    │◀────│  VALIDATE   │               │
│         │            │   (LLM)     │     │   RESULT    │               │
│         │            └─────────────┘     └─────────────┘               │
│         │                  │                    │                        │
│         │                  │ Yes                │ Retry?                 │
│         │                  ▼                    │                        │
│         │            ┌─────────────┐            │                        │
│         │            │   NEXT      │◀───────────┘                       │
│         │            │   SKILL     │                                     │
│         │            └─────────────┘                                     │
│         │                  │                                             │
│         │                  │ No more skills                              │
│         ▼                  ▼                                             │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │              GENERATE RESPONSE (LLM)                        │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       USER RESPONSE                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key LangGraph Concepts Used:

| Concept | Description |
|---------|-------------|
| **StateGraph** | Defines workflow as a directed graph with nodes and edges |
| **Nodes** | Functions that transform state (analyze, execute, respond) |
| **Conditional Edges** | LLM-powered decisions for routing |
| **State Persistence** | Context flows through entire workflow |
| **Error Handling** | Retry logic with quality validation |

### LLM Integration Points:

1. **Intent Analysis**: LLM analyzes user input to understand intent
2. **Skill Selection**: LLM chooses which skill(s) to use
3. **Response Generation**: LLM formats final response naturally
4. **Chain Decision**: LLM decides if more skills should be invoked
"""

# ============================================================================
# 🎯 SKILL DEFINITIONS (Same as 08, for consistency)
# ============================================================================

class SkillCategory(Enum):
    """Categories of skills for organization"""
    INFORMATION = "information"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    UTILITY = "utility"
    MEMORY = "memory"


@dataclass
class SkillContext:
    """
    Shared context that flows between skills in LangGraph workflow.
    """
    user_input: str = ""
    previous_results: List[Dict] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)
    
    def add_result(self, skill_name: str, result: Any):
        """Add a skill result to the context"""
        self.previous_results.append({
            "skill": skill_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_last_result(self) -> Optional[Dict]:
        """Get the most recent skill result"""
        return self.previous_results[-1] if self.previous_results else None
    
    def set_variable(self, name: str, value: Any):
        """Set a context variable for use by other skills"""
        self.variables[name] = value
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a context variable"""
        return self.variables.get(name, default)


class Skill(ABC):
    """Abstract base class for all Skills"""
    
    def __init__(self, name: str, description: str, category: SkillCategory):
        self.name = name
        self.description = description
        self.category = category
        self.prerequisites: List[str] = []
        self.triggers: List[str] = []
    
    @abstractmethod
    def can_execute(self, context: SkillContext) -> bool:
        """Check if this skill can be executed given the current context"""
        pass
    
    @abstractmethod
    def execute(self, context: SkillContext, **kwargs) -> Dict[str, Any]:
        """Execute the skill and return results"""
        pass
    
    def post_process(self, result: Dict[str, Any], context: SkillContext) -> Dict[str, Any]:
        """Optional post-processing of results"""
        return result
    
    def get_confidence(self, user_input: str) -> float:
        """Calculate confidence score for this skill"""
        input_lower = user_input.lower()
        matches = sum(1 for trigger in self.triggers if trigger in input_lower)
        if not self.triggers:
            return 0.0
        return min(matches / len(self.triggers) * 2, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "triggers": self.triggers,
            "prerequisites": self.prerequisites
        }


# ============================================================================
# 🛠️ CONCRETE SKILLS
# ============================================================================

class WeatherSkill(Skill):
    """Skill for getting weather information"""
    
    def __init__(self):
        super().__init__(
            name="weather",
            description="Get current weather information for any location worldwide",
            category=SkillCategory.INFORMATION
        )
        self.triggers = ["weather", "temperature", "forecast", "rain", "sunny", "cold", "hot", "climate"]
    
    def can_execute(self, context: SkillContext) -> bool:
        return any(trigger in context.user_input.lower() for trigger in self.triggers)
    
    def execute(self, context: SkillContext, **kwargs) -> Dict[str, Any]:
        location = kwargs.get("location") or self._extract_location(context.user_input)
        unit = kwargs.get("unit", "celsius")
        
        conditions = ["sunny ☀️", "cloudy ☁️", "rainy 🌧️", "partly cloudy ⛅", "windy 💨"]
        temp_c = random.randint(10, 35)
        temp = temp_c if unit == "celsius" else int(temp_c * 9/5 + 32)
        unit_symbol = "°C" if unit == "celsius" else "°F"
        
        result = {
            "location": location,
            "temperature": temp,
            "unit": unit_symbol,
            "condition": random.choice(conditions),
            "humidity": random.randint(30, 80),
            "wind_speed": random.randint(5, 30)
        }
        
        context.set_variable("current_weather", result)
        context.set_variable("current_location", location)
        
        return result
    
    def _extract_location(self, text: str) -> str:
        patterns = [
            r"(?:weather|temperature|forecast)\s+(?:in|for|at)\s+([A-Za-z\s]+)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+weather",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "New York"


class CalculatorSkill(Skill):
    """Skill for mathematical calculations"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations and evaluations",
            category=SkillCategory.UTILITY
        )
        self.triggers = ["calculate", "compute", "math", "sum", "multiply", "divide", "add", "subtract", "what is", "+", "-", "*", "/", "x", "times"]
    
    def can_execute(self, context: SkillContext) -> bool:
        text = context.user_input.lower()
        has_trigger = any(trigger in text for trigger in self.triggers)
        # Match patterns like "5 + 3", "5 * 3", "5 x 3", "5x3"
        has_numbers = bool(re.search(r'\d+\s*[\+\-\*\/x]\s*\d+', context.user_input, re.IGNORECASE))
        return has_trigger or has_numbers
    
    def execute(self, context: SkillContext, **kwargs) -> Dict[str, Any]:
        expression = kwargs.get("expression") or self._extract_expression(context.user_input)
        
        try:
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return {"error": "Invalid characters in expression", "expression": expression}
            
            result = eval(expression)
            context.set_variable("last_calculation", result)
            
            return {
                "expression": expression,
                "result": result,
                "formatted": f"{expression} = {result}"
            }
        except Exception as e:
            return {"error": str(e), "expression": expression}
    
    def _extract_expression(self, text: str) -> str:
        # First, normalize 'x' to '*' for multiplication
        text = re.sub(r'(\d+)\s*x\s*(\d+)', r'\1 * \2', text, flags=re.IGNORECASE)
        
        match = re.search(r'[\d\s\+\-\*\/\.\(\)]+', text)
        if match:
            return match.group().strip()
        
        text_lower = text.lower()
        text = text_lower.replace("plus", "+").replace("minus", "-")
        text = text.replace("times", "*").replace("multiplied by", "*")
        text = text.replace("divided by", "/")
        
        match = re.search(r'[\d\s\+\-\*\/\.\(\)]+', text)
        return match.group().strip() if match else "0"


class TextAnalysisSkill(Skill):
    """Skill for analyzing text content"""
    
    def __init__(self):
        super().__init__(
            name="text_analysis",
            description="Analyze text for sentiment, keywords, summary, and statistics",
            category=SkillCategory.ANALYSIS
        )
        self.triggers = ["analyze", "sentiment", "keywords", "summarize", "summary", "count words", "statistics"]
    
    def can_execute(self, context: SkillContext) -> bool:
        return any(trigger in context.user_input.lower() for trigger in self.triggers)
    
    def execute(self, context: SkillContext, **kwargs) -> Dict[str, Any]:
        text = kwargs.get("text") or context.get_variable("text_to_analyze") or self._extract_text(context.user_input)
        analysis_type = kwargs.get("analysis_type") or self._detect_analysis_type(context.user_input)
        
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        sentence_count = len(re.split(r'[.!?]+', text.strip()))
        
        result = {
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "word_count": word_count,
            "character_count": char_count,
            "sentence_count": sentence_count,
            "analysis_type": analysis_type
        }
        
        if analysis_type in ["sentiment", "all"]:
            positive = ["good", "great", "excellent", "happy", "love", "amazing", "wonderful", "fantastic", "best"]
            negative = ["bad", "terrible", "hate", "awful", "sad", "poor", "worst", "horrible", "disappointing"]
            text_lower = text.lower()
            pos_score = sum(1 for w in positive if w in text_lower)
            neg_score = sum(1 for w in negative if w in text_lower)
            
            if pos_score > neg_score:
                result["sentiment"] = {"label": "positive 😊", "score": pos_score}
            elif neg_score > pos_score:
                result["sentiment"] = {"label": "negative 😞", "score": neg_score}
            else:
                result["sentiment"] = {"label": "neutral 😐", "score": 0}
        
        if analysis_type in ["keywords", "all"]:
            stop_words = {"the", "a", "an", "is", "are", "was", "were", "it", "this", "that", "to", "of", "and", "for", "in", "on"}
            words_clean = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text)]
            words_filtered = [w for w in words_clean if w not in stop_words]
            word_freq = {}
            for w in words_filtered:
                word_freq[w] = word_freq.get(w, 0) + 1
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            result["keywords"] = [{"word": w, "count": c} for w, c in top_keywords]
        
        context.set_variable("text_analysis_result", result)
        
        return result
    
    def _extract_text(self, user_input: str) -> str:
        patterns = [
            r'analyze[:\s]+["\'](.+)["\']',
            r'analyze[:\s]+(.+)',
            r'text[:\s]+["\'](.+)["\']',
        ]
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return match.group(1)
        return user_input
    
    def _detect_analysis_type(self, text: str) -> str:
        text_lower = text.lower()
        if "sentiment" in text_lower:
            return "sentiment"
        if "keyword" in text_lower:
            return "keywords"
        return "all"


class MemorySkill(Skill):
    """Skill for remembering and recalling information"""
    
    def __init__(self):
        super().__init__(
            name="memory",
            description="Remember information and recall it later. Can store notes, facts, and user preferences.",
            category=SkillCategory.MEMORY
        )
        self.triggers = ["remember", "recall", "forget", "what did", "store", "save", "note"]
    
    def can_execute(self, context: SkillContext) -> bool:
        return any(trigger in context.user_input.lower() for trigger in self.triggers)
    
    def execute(self, context: SkillContext, **kwargs) -> Dict[str, Any]:
        if "agent_memory" not in st.session_state:
            st.session_state.agent_memory = {}
        
        action = self._detect_action(context.user_input)
        
        if action == "remember":
            match = re.search(r'remember\s+(?:that\s+)?(.+)', context.user_input, re.IGNORECASE)
            if match:
                memory_content = match.group(1)
                memory_key = f"memory_{len(st.session_state.agent_memory)}"
                st.session_state.agent_memory[memory_key] = {
                    "content": memory_content,
                    "timestamp": datetime.now().isoformat()
                }
                return {
                    "action": "stored",
                    "message": f"I'll remember: {memory_content}",
                    "key": memory_key
                }
        
        elif action == "recall":
            if st.session_state.agent_memory:
                memories = list(st.session_state.agent_memory.values())
                return {
                    "action": "recalled",
                    "memories": memories,
                    "count": len(memories)
                }
            return {"action": "recall", "message": "I don't have any memories stored yet."}
        
        elif action == "forget":
            count = len(st.session_state.agent_memory)
            st.session_state.agent_memory = {}
            return {"action": "forgot", "message": f"I've forgotten {count} memories."}
        
        return {"action": "unknown", "message": "I'm not sure what you want me to do with memory."}
    
    def _detect_action(self, text: str) -> str:
        text_lower = text.lower()
        if "remember" in text_lower or "store" in text_lower or "save" in text_lower:
            return "remember"
        if "recall" in text_lower or "what did" in text_lower or "what do you" in text_lower:
            return "recall"
        if "forget" in text_lower or "clear" in text_lower:
            return "forget"
        return "unknown"


class RecommendationSkill(Skill):
    """Skill that uses context from other skills to make recommendations"""
    
    def __init__(self):
        super().__init__(
            name="recommendation",
            description="Make recommendations based on context and previous skill results",
            category=SkillCategory.GENERATION
        )
        self.triggers = ["recommend", "suggest", "what should", "advice", "best"]
        self.prerequisites = []
    
    def can_execute(self, context: SkillContext) -> bool:
        return any(trigger in context.user_input.lower() for trigger in self.triggers)
    
    def execute(self, context: SkillContext, **kwargs) -> Dict[str, Any]:
        recommendations = []
        
        weather = context.get_variable("current_weather")
        if weather:
            temp = weather.get("temperature", 20)
            condition = weather.get("condition", "")
            
            if "rain" in condition.lower():
                recommendations.append("☔ Bring an umbrella today!")
            if temp > 25:
                recommendations.append("🧴 Don't forget sunscreen!")
                recommendations.append("💧 Stay hydrated!")
            if temp < 10:
                recommendations.append("🧥 Dress warmly!")
        
        text_analysis = context.get_variable("text_analysis_result")
        if text_analysis and "sentiment" in text_analysis:
            sentiment = text_analysis["sentiment"]["label"]
            if "negative" in sentiment:
                recommendations.append("💡 The text seems negative. Consider a more positive tone?")
            if "positive" in sentiment:
                recommendations.append("✨ Great positive tone! Keep it up!")
        
        last_calc = context.get_variable("last_calculation")
        if last_calc is not None:
            if last_calc > 1000:
                recommendations.append(f"📊 That's a large number ({last_calc})! Double-check your calculation.")
        
        if not recommendations:
            recommendations = [
                "💡 Try asking about the weather first, then ask for recommendations!",
                "📝 Analyze some text, then I can recommend improvements!",
                "🔢 Do a calculation, and I'll provide context about the result!"
            ]
        
        return {
            "recommendations": recommendations,
            "context_used": list(context.variables.keys()),
            "message": "Based on our conversation, here are my recommendations:"
        }


# ============================================================================
# 🤖 LANGGRAPH AGENT STATE
# ============================================================================

class AgentState(TypedDict):
    """State that flows through the LangGraph workflow"""
    # Input
    user_input: str
    
    # Skill management
    available_skills: List[Dict]
    selected_skill: Optional[str]
    skill_result: Optional[Dict]
    executed_skills: List[str]  # Track executed skills to prevent loops
    
    # LLM integration
    llm_analysis: str
    llm_response: str
    
    # Context
    context: Dict[str, Any]
    conversation_history: List[Dict]
    
    # Workflow control
    should_chain: bool
    chain_count: int  # Limit number of chains
    
    # Error handling
    error: str
    retry_count: int
    
    # Pipeline reference
    pipeline: Any


# ============================================================================
# 🧠 LANGGRAPH SKILLS AGENT
# ============================================================================

class LangGraphSkillsAgent:
    """
    📚 EDUCATIONAL: LangGraph-powered Skills Agent with LLM Integration
    
    This agent uses LangGraph to orchestrate skills and actually calls
    an LLM for:
    1. Analyzing user intent
    2. Selecting the best skill
    3. Deciding on skill chaining
    4. Generating natural language responses
    """
    
    def __init__(self, pipeline=None, model_name="TinyLlama"):
        self.pipeline = pipeline
        self.model_name = model_name
        self.skills: Dict[str, Skill] = {}
        self.context = SkillContext()
        self.execution_history = []
        
        # Register all skills
        self._register_skills()
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
    
    def _register_skills(self):
        """Register all available skills"""
        skills = [
            WeatherSkill(),
            CalculatorSkill(),
            TextAnalysisSkill(),
            MemorySkill(),
            RecommendationSkill(),
        ]
        for skill in skills:
            self.skills[skill.name] = skill
    
    def _build_graph(self) -> StateGraph:
        """
        📚 EDUCATIONAL: Build the LangGraph Workflow
        
        This creates a stateful graph with the following nodes:
        1. analyze_intent - LLM analyzes user input
        2. select_skill - LLM chooses skill based on analysis
        3. execute_skill - Run the selected skill
        4. validate_result - Check result quality
        5. decide_chain - LLM decides if more skills needed
        6. generate_response - LLM creates final response
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_intent", self._analyze_intent_node)
        workflow.add_node("select_skill", self._select_skill_node)
        workflow.add_node("execute_skill", self._execute_skill_node)
        workflow.add_node("validate_result", self._validate_result_node)
        workflow.add_node("decide_chain", self._decide_chain_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Define edges
        workflow.set_entry_point("analyze_intent")
        workflow.add_edge("analyze_intent", "select_skill")
        
        # Conditional edge after skill selection
        workflow.add_conditional_edges(
            "select_skill",
            self._route_after_selection,
            {
                "execute": "execute_skill",
                "no_skill": "generate_response"
            }
        )
        
        workflow.add_edge("execute_skill", "validate_result")
        
        # Conditional edge after validation
        workflow.add_conditional_edges(
            "validate_result",
            self._route_after_validation,
            {
                "retry": "execute_skill",
                "continue": "decide_chain"
            }
        )
        
        # Conditional edge for chaining
        workflow.add_conditional_edges(
            "decide_chain",
            self._route_after_chain_decision,
            {
                "chain": "execute_skill",
                "finish": "generate_response"
            }
        )
        
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    # ========================================================================
    # 🔗 LANGGRAPH NODES
    # ========================================================================
    
    def _analyze_intent_node(self, state: AgentState) -> AgentState:
        """
        📚 NODE 1: Analyze user intent using LLM
        
        The LLM analyzes the user's input to understand:
        - What the user wants to accomplish
        - Which skills might be relevant
        - Any specific parameters mentioned
        """
        user_input = state["user_input"]
        pipeline = state.get("pipeline")
        
        # Build skills description for LLM
        skills_desc = "\n".join([
            f"- {name}: {skill.description} (triggers: {', '.join(skill.triggers[:3])})"
            for name, skill in self.skills.items()
        ])
        
        prompt = f"""Analyze the following user request and identify the intent.

Available skills:
{skills_desc}

User request: "{user_input}"

Provide a brief analysis of what the user wants and which skill(s) would be most helpful.
Keep your response under 50 words.

Analysis:"""
        
        analysis = self._call_llm(pipeline, prompt)
        
        state["llm_analysis"] = analysis
        state["available_skills"] = [s.to_dict() for s in self.skills.values()]
        
        return state
    
    def _select_skill_node(self, state: AgentState) -> AgentState:
        """
        📚 NODE 2: Select the best skill using LLM
        
        Based on the intent analysis, the LLM selects which skill to use.
        This is more intelligent than simple keyword matching!
        """
        user_input = state["user_input"]
        analysis = state.get("llm_analysis", "")
        pipeline = state.get("pipeline")
        
        skill_names = list(self.skills.keys())
        
        prompt = f"""Based on the user request and analysis, select the BEST skill to use.

User request: "{user_input}"
Analysis: {analysis}

Available skills: {', '.join(skill_names)}

Respond with ONLY the skill name (one word), or "none" if no skill is appropriate.

Selected skill:"""
        
        selected = self._call_llm(pipeline, prompt).strip().lower()
        
        # Clean up LLM response to extract skill name
        # First, try to find an exact match at the start or as a standalone word
        for skill_name in skill_names:
            # Check for exact match (skill name is the only word or at start)
            if selected == skill_name or selected.startswith(skill_name + " ") or selected.startswith(skill_name + ".") or selected.startswith(skill_name + ","):
                state["selected_skill"] = skill_name
                return state
        
        # Second pass: check if skill name appears as a word boundary match
        for skill_name in skill_names:
            # Use word boundary to avoid partial matches (e.g., "analysis" matching "text_analysis")
            if re.search(r'\b' + re.escape(skill_name) + r'\b', selected):
                state["selected_skill"] = skill_name
                return state
        
        # Fallback to confidence-based selection
        self.context.user_input = user_input
        best_skill = None
        best_confidence = 0.0
        
        for skill in self.skills.values():
            if skill.can_execute(self.context):
                confidence = skill.get_confidence(user_input)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_skill = skill.name
        
        state["selected_skill"] = best_skill
        return state
    
    def _execute_skill_node(self, state: AgentState) -> AgentState:
        """
        📚 NODE 3: Execute the selected skill
        
        This node runs the actual skill and captures the result.
        """
        skill_name = state.get("selected_skill")
        
        if not skill_name or skill_name not in self.skills:
            state["error"] = f"Skill '{skill_name}' not found"
            return state
        
        skill = self.skills[skill_name]
        
        # Update context
        self.context.user_input = state["user_input"]
        
        try:
            start_time = time.time()
            result = skill.execute(self.context)
            execution_time = time.time() - start_time
            
            # Store result
            self.context.add_result(skill_name, result)
            
            state["skill_result"] = {
                "skill": skill_name,
                "result": result,
                "execution_time": f"{execution_time:.2f}s",
                "context_variables": list(self.context.variables.keys())
            }
            
            # Track executed skills to prevent infinite loops
            executed = state.get("executed_skills", [])
            if skill_name not in executed:
                state["executed_skills"] = executed + [skill_name]
            
            # Track execution
            self.execution_history.append({
                "skill": skill_name,
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
        except Exception as e:
            state["error"] = f"Skill execution error: {str(e)}"
            state["skill_result"] = {"error": str(e)}
        
        return state
    
    def _validate_result_node(self, state: AgentState) -> AgentState:
        """
        📚 NODE 4: Validate result quality
        
        Check if the skill result is valid and useful.
        Implements retry logic for poor results.
        """
        result = state.get("skill_result", {})
        retry_count = state.get("retry_count", 0)
        
        # Check for errors
        if "error" in result and result["error"]:
            if retry_count < 2:
                state["retry_count"] = retry_count + 1
                return state
        
        # Reset retry count on success
        state["retry_count"] = 0
        return state
    
    def _decide_chain_node(self, state: AgentState) -> AgentState:
        """
        📚 NODE 5: Decide if skills should be chained
        
        The LLM analyzes the result and decides if additional
        skills should be invoked for a complete response.
        """
        user_input = state["user_input"]
        skill_result = state.get("skill_result", {})
        executed_skills = state.get("executed_skills", [])
        chain_count = state.get("chain_count", 0)
        pipeline = state.get("pipeline")
        
        # Limit chaining to prevent infinite loops (max 2 chains)
        if chain_count >= 2:
            state["should_chain"] = False
            return state
        
        # Get remaining skills (exclude already executed ones)
        remaining_skills = [s for s in self.skills.keys() if s not in executed_skills]
        
        # No more skills to chain
        if not remaining_skills:
            state["should_chain"] = False
            return state
        
        prompt = f"""Should we use another skill to complete this request?

User request: "{user_input}"
Skills already executed: {', '.join(executed_skills)}
Result: {json.dumps(skill_result.get('result', {}), default=str)[:200]}

Remaining skills: {', '.join(remaining_skills)}

Answer "yes" and the skill name, or "no" if the request is complete.
Response:"""
        
        decision = self._call_llm(pipeline, prompt).lower()
        
        if "yes" in decision:
            # Find which skill was suggested
            for skill_name in remaining_skills:
                if skill_name in decision:
                    state["should_chain"] = True
                    state["selected_skill"] = skill_name
                    state["chain_count"] = chain_count + 1
                    return state
        
        state["should_chain"] = False
        return state
    
    def _generate_response_node(self, state: AgentState) -> AgentState:
        """
        📚 NODE 6: Generate natural language response
        
        The LLM creates a friendly, informative response
        based on all the skill results.
        """
        user_input = state["user_input"]
        skill_result = state.get("skill_result", {})
        selected_skill = state.get("selected_skill")
        pipeline = state.get("pipeline")
        
        if not selected_skill:
            # No skill was selected
            available = ", ".join(self.skills.keys())
            state["llm_response"] = f"I'm not sure how to help with that. I can help with: {available}. Try asking about weather, calculations, text analysis, memory, or recommendations!"
            return state
        
        # Format result for LLM
        result_str = json.dumps(skill_result.get("result", {}), default=str, indent=2)
        
        prompt = f"""Generate a friendly, helpful response based on this skill execution.

User request: "{user_input}"
Skill used: {selected_skill}
Skill result:
{result_str}

Create a natural, conversational response that presents this information clearly.
Keep it concise (under 100 words).

Response:"""
        
        response = self._call_llm(pipeline, prompt)
        state["llm_response"] = response
        
        return state
    
    # ========================================================================
    # 🔀 ROUTING FUNCTIONS
    # ========================================================================
    
    def _route_after_selection(self, state: AgentState) -> str:
        """Route based on whether a skill was selected"""
        if state.get("selected_skill"):
            return "execute"
        return "no_skill"
    
    def _route_after_validation(self, state: AgentState) -> str:
        """Route based on validation result"""
        if state.get("retry_count", 0) > 0:
            return "retry"
        return "continue"
    
    def _route_after_chain_decision(self, state: AgentState) -> str:
        """Route based on chaining decision"""
        if state.get("should_chain", False):
            return "chain"
        return "finish"
    
    # ========================================================================
    # 🤖 LLM HELPER
    # ========================================================================
    
    def _call_llm(self, pipeline, prompt: str) -> str:
        """
        📚 EDUCATIONAL: Call the LLM
        
        This method handles the actual LLM invocation using OpenVINO GenAI.
        Falls back to simulation mode if no pipeline is available.
        """
        if pipeline is None:
            # Simulation mode - generate reasonable responses
            return self._simulate_llm_response(prompt)
        
        # Handle Azure OpenAI mode
        if pipeline == "openai" or self.model_name.startswith("gpt"):
            return self._call_openai(prompt)
        
        try:
            # Select template based on model type
            if self.model_name == "qwen3":
                # 📚 Qwen3: OpenVINO GenAI applies the model's own chat template
                # automatically. "/no_think" switches off its step-by-step
                # reasoning, so it answers directly (faster, no <think> text).
                system = "You are a helpful AI assistant. Be concise and direct. Only output the requested answer, nothing else."
                result = str(pipeline.generate(system + "\n\n" + prompt + " /no_think", max_new_tokens=150))
                result = re.sub(r"<think>.*?</think>", "", result, flags=re.S).strip()
                return result if result else self._simulate_llm_response(prompt)

            if self.model_name == "phi3":
                # Phi-3.5 chat template
                template = "<|system|>\nYou are a helpful AI assistant. Be concise and direct. Only output the requested answer, nothing else.<|end|>\n<|user|>\n{}<|end|>\n<|assistant|>\n"
            else:
                # TinyLlama chat template
                template = "<|system|>\nYou are a helpful AI assistant. Be concise and direct. Only output the requested answer, nothing else.</s>\n<|user|>\n{}</s>\n<|assistant|>\n"
            
            formatted_prompt = template.format(prompt)
            
            # Generate response
            result = str(pipeline.generate(formatted_prompt, max_new_tokens=150))
            
            # Remove the prompt from the response if it's echoed back
            if formatted_prompt in result:
                result = result.replace(formatted_prompt, "")
            
            # Clean up response - remove special tokens and prompt artifacts
            result = result.split("</s>")[0].strip()
            result = result.split("<|end|>")[0].strip()  # Phi-3 end token
            result = result.split("<|user|>")[0].strip()
            result = result.split("<|system|>")[0].strip()
            result = result.split("<|assistant|>")[0].strip()
            result = result.split("<|endoftext|>")[0].strip()  # Phi-3 end token
            result = result.split("User:")[0].strip()
            result = result.split("Human:")[0].strip()
            
            # Remove common prompt echo patterns
            patterns_to_remove = [
                "Sure, here's a friendly, helpful response based on the skill execution:",
                "Sure, here's a friendly, helpful response:",
                "Here's a friendly response:",
            ]
            for pattern in patterns_to_remove:
                if result.startswith(pattern):
                    result = result[len(pattern):].strip()
            
            return result if result else self._simulate_llm_response(prompt)
        except Exception as e:
            return f"LLM Error: {str(e)}"
    
    def _call_openai(self, prompt: str) -> str:
        """
        📚 EDUCATIONAL: Call Azure OpenAI API
        
        This method calls Azure OpenAI for high-quality LLM responses.
        Supports GPT-4o, GPT-4, and other Azure OpenAI models.
        """
        try:
            if not AZURE_OPENAI_AVAILABLE:
                return self._simulate_llm_response(prompt)
            
            import os
            
            # Get Azure OpenAI configuration from environment
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
            deployment = os.environ.get("AZURE_OPENAI_LLM_4", "gpt-4o")
            
            if not api_key or not endpoint:
                return self._simulate_llm_response(prompt)
            
            # Create Azure OpenAI client
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            
            # Call the API
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Be concise and direct. Only output the requested answer, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                return self._simulate_llm_response(prompt)
                
        except Exception as e:
            return f"OpenAI Error: {str(e)}"
    
    def _simulate_llm_response(self, prompt: str) -> str:
        """Simulate LLM responses for demo mode"""
        prompt_lower = prompt.lower()
        
        # Check for math patterns in prompt (numbers with operators)
        has_math = bool(re.search(r'\d+\s*[\+\-\*\/x]\s*\d+', prompt_lower))
        
        if "analyze" in prompt_lower and "intent" in prompt_lower:
            if "weather" in prompt_lower or "temperature" in prompt_lower or "forecast" in prompt_lower:
                return "The user wants weather information for a specific location."
            if "calculate" in prompt_lower or "math" in prompt_lower or has_math:
                return "The user wants to perform a mathematical calculation."
            if "remember" in prompt_lower:
                return "The user wants to store information in memory."
            if "recommend" in prompt_lower or "suggestion" in prompt_lower:
                return "The user wants recommendations."
            if "sentiment" in prompt_lower or "keywords" in prompt_lower or "summarize" in prompt_lower:
                return "The user wants to analyze some text."
            if has_math:
                return "The user wants to perform a mathematical calculation."
            return "The user has a general question."
        
        if "select" in prompt_lower and "skill" in prompt_lower:
            # Check for weather keywords in the user request embedded in the prompt
            if "weather" in prompt_lower or "temperature" in prompt_lower or "forecast" in prompt_lower:
                return "weather"
            if "calculate" in prompt_lower or "math" in prompt_lower or has_math or "x" in prompt_lower:
                return "calculator"
            if "remember" in prompt_lower:
                return "memory"
            if "sentiment" in prompt_lower or "keywords" in prompt_lower or "summarize" in prompt_lower:
                return "text_analysis"
            if "recommend" in prompt_lower or "suggest" in prompt_lower:
                return "recommendation"
            # Check the user request embedded in the prompt for math
            if has_math:
                return "calculator"
            return "none"
        
        if "should we use another skill" in prompt_lower:
            return "no, the request is complete"
        
        if "generate" in prompt_lower and "response" in prompt_lower:
            # Generate skill-specific responses
            if "weather" in prompt_lower:
                # Extract info from result if available
                return "Based on the weather data, I found the current conditions for your requested location. The temperature and forecast details are shown in the results above."
            if "calculator" in prompt_lower:
                return "I've completed the calculation for you. The result is shown above."
            if "text_analysis" in prompt_lower:
                return "I've analyzed the text you provided. The sentiment, keywords, and other insights are shown above."
            if "memory" in prompt_lower:
                return "I've stored that information in memory. You can recall it later by asking me to remember."
            if "recommendation" in prompt_lower:
                return "Based on the available context, here are my recommendations for you."
            return "Here's the information you requested. The skill executed successfully and produced the results shown above."
        
        return "I understand. Let me help you with that."
    
    # ========================================================================
    # 📤 PUBLIC INTERFACE
    # ========================================================================
    
    def process_input(self, user_input: str, progress_callback=None) -> Dict[str, Any]:
        """
        Process user input through the LangGraph workflow.
        
        Returns a dictionary with:
        - success: bool
        - skill_used: str
        - result: Dict
        - llm_response: str
        - workflow_trace: List
        """
        # Initialize state
        initial_state: AgentState = {
            "user_input": user_input,
            "available_skills": [],
            "selected_skill": None,
            "skill_result": None,
            "executed_skills": [],
            "llm_analysis": "",
            "llm_response": "",
            "context": {},
            "conversation_history": [],
            "should_chain": False,
            "chain_count": 0,
            "error": "",
            "retry_count": 0,
            "pipeline": self.pipeline
        }
        
        workflow_trace = []
        final_state = initial_state
        
        try:
            # Execute the graph and track nodes
            for step_output in self.graph.stream(initial_state):
                if isinstance(step_output, dict):
                    for node_name, node_state in step_output.items():
                        workflow_trace.append({
                            "node": node_name,
                            "timestamp": datetime.now().isoformat()
                        })
                        final_state = node_state
                        
                        if progress_callback:
                            progress_callback(node_name, len(workflow_trace))
            
            return {
                "success": True,
                "skill_used": final_state.get("selected_skill"),
                "skill_result": final_state.get("skill_result", {}),
                "llm_analysis": final_state.get("llm_analysis", ""),
                "llm_response": final_state.get("llm_response", ""),
                "workflow_trace": workflow_trace,
                "context_variables": list(self.context.variables.keys()),
                "error": final_state.get("error", "")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "workflow_trace": workflow_trace
            }
    
    def get_skills(self) -> List[Skill]:
        """Get all registered skills"""
        return list(self.skills.values())
    
    def reset(self):
        """Reset agent context"""
        self.context = SkillContext()
        self.execution_history = []


# ============================================================================
# 🎨 STREAMLIT UI
# ============================================================================

def initialize_session():
    """Initialize session state"""
    if "model_loaded" not in st.session_state:
        st.session_state["model_loaded"] = False
    if "pipe" not in st.session_state:
        st.session_state["pipe"] = None
    if "agent" not in st.session_state:
        st.session_state["agent"] = None
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []


def load_model(model_path: str, device: str = "CPU", model_type: str = "openvino") -> bool:
    """Load the LLM model"""
    # Handle Azure OpenAI mode
    if model_type == "openai":
        import os
        deployment = os.environ.get("AZURE_OPENAI_LLM_4", "gpt-4o")
        st.session_state["pipe"] = "openai"  # Special marker for OpenAI
        st.session_state["model_type"] = "openai"
        st.session_state["agent"] = LangGraphSkillsAgent(pipeline="openai", model_name=deployment)
        st.session_state["model_loaded"] = True
        st.success(f"✅ Connected to Azure OpenAI ({deployment})")
        return True
    
    # Handle simulation mode
    if model_type == "simulation" or model_path is None:
        st.session_state["agent"] = LangGraphSkillsAgent(pipeline=None, model_name="simulation")
        st.session_state["model_loaded"] = True
        st.info("🎭 Running in simulation mode")
        return True
    
    with st.spinner("🤖 Loading LangGraph Skills Agent with LLM..."):
        try:
            if OPENVINO_AVAILABLE and Path(model_path).exists():
                # Detect model type based on path
                model_path_lower = model_path.lower()
                if "qwen" in model_path_lower:
                    llm_type = "qwen3"
                elif "phi" in model_path_lower:
                    llm_type = "phi3"
                elif "tinyllama" in model_path_lower or "llama" in model_path_lower:
                    llm_type = "tinyllama"
                else:
                    llm_type = "generic"
                
                pipeline = ov_genai.LLMPipeline(str(model_path), device)
                st.session_state["pipe"] = pipeline
                st.session_state["model_type"] = llm_type
                st.session_state["agent"] = LangGraphSkillsAgent(pipeline, model_name=llm_type)
                st.session_state["model_loaded"] = True
                st.success(f"✅ Loaded model: {llm_type}")
                return True
            else:
                # Demo mode without actual LLM
                st.session_state["agent"] = LangGraphSkillsAgent(pipeline=None)
                st.session_state["model_loaded"] = True
                st.warning("⚠️ Running in simulation mode (no LLM loaded)")
                return True
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            # Fall back to simulation mode
            st.session_state["agent"] = LangGraphSkillsAgent(pipeline=None)
            st.session_state["model_loaded"] = True
            return True


def display_skill_card(skill: Skill):
    """Display a skill as a card"""
    category_colors = {
        SkillCategory.INFORMATION: "🔵",
        SkillCategory.ANALYSIS: "🟢",
        SkillCategory.GENERATION: "🟣",
        SkillCategory.UTILITY: "🟡",
        SkillCategory.MEMORY: "🔴"
    }
    
    color = category_colors.get(skill.category, "⚪")
    
    st.markdown(f"""
    **{color} {skill.name}** - {skill.category.value}
    
    {skill.description}
    
    *Triggers:* {', '.join(skill.triggers[:5])}{'...' if len(skill.triggers) > 5 else ''}
    """)


def display_workflow_trace(trace: List[Dict]):
    """Display the LangGraph workflow execution trace"""
    st.markdown("### 🔗 LangGraph Workflow Trace")
    
    node_icons = {
        "analyze_intent": "🔍",
        "select_skill": "🎯",
        "execute_skill": "⚡",
        "validate_result": "✅",
        "decide_chain": "🔀",
        "generate_response": "💬"
    }
    
    for i, step in enumerate(trace):
        node = step["node"]
        icon = node_icons.get(node, "📍")
        st.markdown(f"{icon} **{node}**")


def format_skill_result(skill_name: str, result: Dict[str, Any]) -> None:
    """Format and display skill result"""
    data = result.get("result", {})
    
    if skill_name == "weather":
        st.success(f"🌍 Weather in {data.get('location', 'Unknown')}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Temperature", f"{data.get('temperature')}°")
        with col2:
            st.metric("Humidity", f"{data.get('humidity')}%")
        with col3:
            st.metric("Wind", f"{data.get('wind_speed')} km/h")
        st.write(f"Condition: {data.get('condition')}")
    
    elif skill_name == "calculator":
        if data.get("error"):
            st.error(f"Error: {data['error']}")
        else:
            st.success(f"🧮 {data.get('formatted')}")
    
    elif skill_name == "text_analysis":
        st.success("📊 Text Analysis Complete")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Words", data.get("word_count"))
        with col2:
            st.metric("Characters", data.get("character_count"))
        with col3:
            st.metric("Sentences", data.get("sentence_count"))
        
        if "sentiment" in data:
            st.write(f"**Sentiment:** {data['sentiment']['label']}")
        
        if "keywords" in data:
            keywords = [k["word"] for k in data["keywords"]]
            st.write(f"**Keywords:** {', '.join(keywords)}")
    
    elif skill_name == "memory":
        st.info(f"🧠 {data.get('message', data.get('action'))}")
        if "memories" in data:
            for mem in data["memories"]:
                st.write(f"- {mem['content']}")
    
    elif skill_name == "recommendation":
        st.success("💡 Recommendations")
        for rec in data.get("recommendations", []):
            st.write(f"• {rec}")
    
    else:
        st.json(result)


def main():
    st.set_page_config(
        page_title="🎓 LangGraph Skills Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        .stApp h1 {
            color: #F5DEB3 !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .stApp h2, .stApp h3, .stApp h4 {
            color: #FAEBD7 !important;
        }
        .stButton>button {
            background: linear-gradient(90deg, #9b59b6 0%, #3498db 100%);
            color: white !important;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #3498db 0%, #9b59b6 100%);
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #FFD700 !important;
        }
        .workflow-node {
            padding: 10px;
            margin: 5px;
            border-radius: 5px;
            background: rgba(255,255,255,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    initialize_session()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🤖 LangGraph Skills Agent")
        st.markdown("---")
        
        st.markdown("### ⚙️ Model Configuration")
        
        # Model selection - text-only LLMs
        model_options = {
            "Qwen3 1.7B (Local, recommended)": {"type": "openvino", "path": "./Qwen3-1.7B-int4-ov"},
            "TinyLlama 1.1B (Local)": {"type": "openvino", "path": "./TinyLlama-1.1B-Chat-v1.0/"},
            "Azure OpenAI GPT-4o (Cloud)": {"type": "openai", "path": None},
            "Simulation Mode": {"type": "simulation", "path": None},
        }
        
        # Filter out Azure OpenAI if not available
        if not AZURE_OPENAI_AVAILABLE:
            model_options = {k: v for k, v in model_options.items() if v["type"] != "openai"}
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0,
            help="Choose the LLM model to use. Azure OpenAI uses cloud-based GPT-4o."
        )
        model_config = model_options[selected_model]
        model_path = model_config["path"]
        model_type = model_config["type"]
        
        if model_type == "openvino":
            st.caption(f"Path: `{model_path}`")
        elif model_type == "openai":
            st.caption("☁️ Using Azure OpenAI GPT-4o")
        else:
            st.caption("Running without LLM - using rule-based simulation")
        
        device = st.radio("Device", ["CPU", "GPU"], index=0, disabled=(model_type == "openai"))
        
        if not st.session_state["model_loaded"]:
            if st.button("🚀 Initialize Agent", type="primary"):
                load_model(model_path, device, model_type)
        else:
            st.success("✅ Agent Ready")
            current_type = st.session_state.get("model_type", "simulation")
            if current_type == "openai":
                mode = "Azure OpenAI GPT-4o Mode"
            elif st.session_state.get("pipe"):
                mode = f"LLM Mode ({current_type})"
            else:
                mode = "Simulation Mode"
            st.caption(f"Running in: {mode}")
            if st.button("🔄 Reload"):
                st.session_state["model_loaded"] = False
                st.session_state["agent"] = None
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔗 LangGraph Workflow")
        st.markdown("""
        1. 🔍 **Analyze Intent** (LLM)
        2. 🎯 **Select Skill** (LLM)
        3. ⚡ **Execute Skill**
        4. ✅ **Validate Result**
        5. 🔀 **Decide Chain** (LLM)
        6. 💬 **Generate Response** (LLM)
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        
        examples = [
            "What's the weather in Paris?",
            "Calculate 42 * 17 + 100",
            "Analyze: This is an excellent product!",
            "Remember my name is Alice",
            "What do you remember?",
            "Give me recommendations",
        ]
        
        for ex in examples:
            if st.button(f"📝 {ex[:25]}...", key=f"ex_{ex}"):
                st.session_state.example_input = ex
        
        st.markdown("---")
        if st.button("🔄 Reset Context"):
            if st.session_state.get("agent"):
                st.session_state["agent"].reset()
            st.session_state["conversation_history"] = []
            st.rerun()
    
    # Main content
    st.title("🤖 LangGraph Skills Agent with LLM")
    st.markdown("### Intelligent skill orchestration powered by LangGraph and LLM")
    
    # Introduction
    with st.expander("📚 How This Works (Click to Learn)", expanded=False):
        st.markdown(SKILLS_LANGGRAPH_INTRO)
    
    if not st.session_state["model_loaded"]:
        st.warning("⚠️ Please initialize the agent using the sidebar.")
        st.info("""
        ### Getting Started
        
        1. Click **Initialize Agent** in the sidebar
        2. The agent will load in simulation mode (or with LLM if model exists)
        3. Try the example queries to see LangGraph in action!
        
        ### What's Different from 08_skills_tutorial?
        
        - **LangGraph Workflow**: Instead of simple function calls, this uses a stateful graph
        - **LLM Integration**: The agent actually calls an LLM for intelligent decisions
        - **Conditional Routing**: LLM decides which skills to use and whether to chain them
        - **Workflow Visibility**: You can see exactly which nodes executed
        """)
        return
    
    agent = st.session_state["agent"]
    
    # Show available skills
    with st.expander("🎯 Available Skills", expanded=False):
        cols = st.columns(2)
        for i, skill in enumerate(agent.get_skills()):
            with cols[i % 2]:
                display_skill_card(skill)
                st.markdown("---")
    
    st.markdown("---")
    
    # Chat interface
    st.markdown("## 💬 Chat with the Agent")
    
    # Check for example input
    initial_value = ""
    if "example_input" in st.session_state:
        initial_value = st.session_state.example_input
        del st.session_state.example_input
    
    user_input = st.text_input(
        "Your message:",
        value=initial_value,
        placeholder="Try: What's the weather in Tokyo?",
        key="user_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        send_button = st.button("🚀 Execute", type="primary")
    with col2:
        show_trace = st.checkbox("Show Workflow", value=True)
    
    if send_button and user_input:
        # Process through LangGraph
        with st.spinner("🤖 LangGraph Agent Processing..."):
            progress_placeholder = st.empty()
            
            def update_progress(node_name: str, step_count: int):
                progress_placeholder.info(f"⚙️ Executing node: **{node_name}** (step {step_count})")
            
            result = agent.process_input(user_input, progress_callback=update_progress)
            progress_placeholder.empty()
        
        st.markdown("---")
        
        if result["success"]:
            # Show workflow trace
            if show_trace and result.get("workflow_trace"):
                with st.expander("🔗 LangGraph Execution Trace", expanded=True):
                    display_workflow_trace(result["workflow_trace"])
            
            # LLM Analysis
            if result.get("llm_analysis"):
                st.markdown("### 🔍 LLM Intent Analysis")
                st.info(result["llm_analysis"])
            
            # Skill execution
            if result.get("skill_used"):
                st.markdown("### ⚡ Skill Execution")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.metric("Skill Used", result["skill_used"])
                with col2:
                    if result.get("skill_result"):
                        st.metric("Execution Time", result["skill_result"].get("execution_time", "N/A"))
                
                # Display formatted result
                if result.get("skill_result"):
                    format_skill_result(result["skill_used"], result["skill_result"])
            
            # LLM Response
            st.markdown("### 💬 Agent Response")
            st.success(result.get("llm_response", "No response generated"))
            
            # Context state
            if result.get("context_variables"):
                st.markdown("### 📦 Context Variables")
                st.caption(f"Available for skill chaining: {', '.join(result['context_variables'])}")
            
            # Add to conversation history
            st.session_state["conversation_history"].append({
                "user": user_input,
                "response": result,
                "timestamp": datetime.now().isoformat()
            })
        
        else:
            st.error(f"Error: {result.get('error', 'Unknown error')}")
    
    # Conversation history
    if st.session_state["conversation_history"]:
        st.markdown("---")
        st.markdown("### 📜 Conversation History")
        
        for i, entry in enumerate(reversed(st.session_state["conversation_history"][-5:])):
            with st.expander(f"💬 {entry['user'][:50]}...", expanded=False):
                st.markdown(f"**You:** {entry['user']}")
                st.markdown(f"**Agent:** {entry['response'].get('llm_response', 'N/A')}")
                st.caption(f"Skill: {entry['response'].get('skill_used', 'None')}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🤖 LangGraph Skills Agent - GenAI Course Tutorial</p>
        <p>This demo shows LangGraph + LLM integration for intelligent skill orchestration</p>
        <p>Compare with 08_skills_tutorial.py to see the difference!</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
