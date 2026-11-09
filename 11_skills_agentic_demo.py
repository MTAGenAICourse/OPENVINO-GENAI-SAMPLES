"""
🎯 Skills-Based Agentic Demo
============================
A demonstration of agentic AI that routes user requests to real skills
from the .claude/skills directory.

This demo integrates:
- 🖼️ Image2Mermaid: Convert diagram images to Mermaid code
- 💬 ChAI Assistant: Query Intel AI Assistants (WiFi_QnA, BT_QnA, etc.)
- 🔧 Built-in utilities: Calculator, text analysis, etc.

Author: Skills Integration Demo
Date: February 2026
"""

import streamlit as st
import json
import re
import os
import sys
import subprocess
import tempfile
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# 📁 SKILLS CONFIGURATION
# ============================================================================

# Path to skills directory
SKILLS_DIR = Path(__file__).parent / ".claude" / "skills"
MINIPYTHON = r"C:\MiniPython\python.exe"

# Fallback to system Python if MiniPython not available
if not os.path.exists(MINIPYTHON):
    MINIPYTHON = sys.executable

# ============================================================================
# 📚 EDUCATIONAL INTRO
# ============================================================================

INTRO_TEXT = """
## 🎯 Skills-Based Agentic AI

This demo shows how an AI agent can **autonomously choose** which skill to use
based on your natural language request.

### Available Skills:

| Skill | Description | Example Request |
|-------|-------------|-----------------|
| 🖼️ **Image2Mermaid** | Convert diagrams to Mermaid code | "Convert this image to mermaid" |
| 💬 **ChAI WiFi QnA** | Ask about Intel WiFi products | "What is MLD in WiFi?" |
| 💬 **ChAI BT QnA** | Ask about Intel Bluetooth | "Explain LE Audio" |
| 🔢 **Calculator** | Mathematical calculations | "Calculate 125 * 4" |
| 📊 **Text Analyzer** | Analyze text properties | "Analyze this text for sentiment" |

### How It Works:
```
┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Your Request    │────▶│  Agent Router   │────▶│  Execute Skill   │
│  (Natural Lang)  │     │  (Decides)      │     │  (Real Tools!)   │
└──────────────────┘     └─────────────────┘     └──────────────────┘
```
"""

# ============================================================================
# 🔧 TOOL DEFINITIONS
# ============================================================================

@dataclass
class ToolParameter:
    """Defines a parameter for a tool"""
    name: str
    type: str
    description: str
    required: bool = True
    enum: List[str] = field(default_factory=list)

@dataclass 
class Tool:
    """Represents a skill/tool that the AI agent can use"""
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable
    category: str = "general"
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert tool to JSON schema format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

# ============================================================================
# 🛠️ SKILL IMPLEMENTATIONS
# ============================================================================

def image2mermaid_skill(image_path: str, quick_mode: bool = True, diagram_type: str = "") -> Dict[str, Any]:
    """
    Convert an image to Mermaid diagram code using the Doc2Md-image2mermaid skill.
    
    This calls the actual skill from .claude/skills/Doc2Md-image2mermaid/
    """
    skill_script = SKILLS_DIR / "Doc2Md-image2mermaid" / "image2mermaid.py"
    
    if not skill_script.exists():
        return {
            "error": f"Skill not found at {skill_script}",
            "success": False
        }
    
    if not os.path.exists(image_path):
        return {
            "error": f"Image file not found: {image_path}",
            "success": False
        }
    
    # Create output directory
    output_dir = tempfile.mkdtemp(prefix="mermaid_")
    
    # Build command
    cmd = [MINIPYTHON, str(skill_script), image_path, "-o", output_dir]
    
    if quick_mode:
        cmd.append("-q")
    
    if diagram_type:
        cmd.extend(["-t", diagram_type])
    
    try:
        # Run the skill
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(SKILLS_DIR / "Doc2Md-image2mermaid")
        )
        
        # Look for generated mermaid files
        mermaid_files = list(Path(output_dir).glob("*.mermaid")) + list(Path(output_dir).glob("*.md"))
        
        if mermaid_files:
            # Read the first mermaid file
            with open(mermaid_files[0], 'r', encoding='utf-8') as f:
                mermaid_content = f.read()
            
            return {
                "success": True,
                "mermaid_code": mermaid_content,
                "output_file": str(mermaid_files[0]),
                "image_path": image_path,
                "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
            }
        else:
            return {
                "success": False,
                "error": "No mermaid output generated",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        return {"error": "Skill execution timed out (120s)", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


def chai_assistant_skill(query: str, usecase: str = "WiFi_QnA") -> Dict[str, Any]:
    """
    Query the ChAI Assistant using the ChAI-Assistant skill.
    
    Available usecases: WiFi_QnA, BT_QnA, TOP_QnA, ICA_QnA, QnA
    """
    skill_script = SKILLS_DIR / "ChAI-Assistant" / "ask_chai_tool.py"
    
    if not skill_script.exists():
        return {
            "error": f"Skill not found at {skill_script}",
            "success": False
        }
    
    # Check for client secret
    client_secret = os.environ.get("CHAI_SDK_CLIENT_SECRET")
    if not client_secret:
        return {
            "error": "CHAI_SDK_CLIENT_SECRET environment variable not set. Please configure your ChAI credentials.",
            "success": False
        }
    
    # Build command
    cmd = [
        MINIPYTHON, str(skill_script),
        "--query", query,
        "--usecase", usecase
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=90,
            cwd=str(Path(__file__).parent),
            env={**os.environ, "CHAI_SDK_CLIENT_SECRET": client_secret}
        )
        
        if result.returncode == 0:
            # Parse the response
            response_text = result.stdout.strip()
            return {
                "success": True,
                "response": response_text,
                "usecase": usecase,
                "query": query
            }
        else:
            return {
                "success": False,
                "error": result.stderr or "Unknown error",
                "stdout": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {"error": "ChAI request timed out (90s)", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


def calculator_skill(expression: str) -> Dict[str, Any]:
    """Safe mathematical expression evaluator"""
    try:
        # Only allow safe math operations
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression", "result": None, "success": False}
        
        result = eval(expression)
        return {
            "success": True,
            "expression": expression,
            "result": result,
            "type": type(result).__name__
        }
    except Exception as e:
        return {"error": str(e), "result": None, "success": False}


def text_analyzer_skill(text: str, analysis_type: str = "summary") -> Dict[str, Any]:
    """Analyze text in various ways"""
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = len(re.split(r'[.!?]+', text.strip())) if text.strip() else 0
    
    result = {
        "success": True,
        "word_count": word_count,
        "character_count": char_count,
        "sentence_count": sentence_count,
        "analysis_type": analysis_type
    }
    
    if analysis_type == "summary":
        result["summary"] = f"Text contains {word_count} words in {sentence_count} sentences."
    elif analysis_type == "sentiment":
        positive_words = ["good", "great", "excellent", "happy", "love", "amazing", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "hate", "awful", "sad", "poor", "horrible", "worst"]
        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        if pos_count > neg_count:
            result["sentiment"] = "positive"
            result["confidence"] = "high" if pos_count > neg_count + 1 else "medium"
        elif neg_count > pos_count:
            result["sentiment"] = "negative"
            result["confidence"] = "high" if neg_count > pos_count + 1 else "medium"
        else:
            result["sentiment"] = "neutral"
            result["confidence"] = "medium"
    elif analysis_type == "keywords":
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        result["keywords"] = [w[0] for w in top_words]
    
    return result


def web_search_skill(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Search the web for general knowledge questions.
    
    Tries multiple search methods:
    1. Weather API (for weather queries)
    2. DuckDuckGo (more reliable than Google scraping)
    3. Google search via requests
    4. Opens browser as last resort
    """
    query_clean = query.strip()
    query_lower = query_clean.lower()
    
    # Method 0: Check if this is a weather query - use wttr.in API
    if any(word in query_lower for word in ["weather", "temperature", "rain", "sunny", "forecast", "wetaher"]):
        try:
            import requests
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # Extract location - use word boundaries to avoid matching within words
            location = "Tel Aviv"  # Default
            for word in [" in ", " at ", " for "]:
                if word in query_lower:
                    parts = query_lower.split(word)
                    if len(parts) > 1:
                        # Clean up the location
                        loc = parts[-1].strip().strip("?").strip()
                        # Remove trailing words like "today", "this month", etc.
                        loc = re.sub(r'\s+(today|tomorrow|this\s+\w+|next\s+\w+|now).*$', '', loc, flags=re.IGNORECASE)
                        if loc and len(loc) > 1:
                            location = loc
                        break
            
            # Use wttr.in for weather (free, no API key needed)
            weather_url = f"https://wttr.in/{location}?format=j1"
            # Use proxy for external sites (Intel corporate network)
            proxy = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
            proxies = {"http": proxy, "https": proxy} if proxy else None
            response = requests.get(
                weather_url, 
                timeout=15, 
                headers={'User-Agent': 'curl/7.68.0'},
                proxies=proxies,
                verify=False  # Skip SSL verification for proxy
            )
            
            if response.status_code == 200:
                data = response.json()
                current = data.get("current_condition", [{}])[0]
                area = data.get("nearest_area", [{}])[0]
                
                temp_c = current.get("temp_C", "N/A")
                temp_f = current.get("temp_F", "N/A")
                condition = current.get("weatherDesc", [{}])[0].get("value", "N/A")
                humidity = current.get("humidity", "N/A")
                wind = current.get("windspeedKmph", "N/A")
                city = area.get("areaName", [{}])[0].get("value", location)
                country = area.get("country", [{}])[0].get("value", "")
                
                weather_info = f"🌡️ **Weather in {city}, {country}**\n\n"
                weather_info += f"- Temperature: {temp_c}°C ({temp_f}°F)\n"
                weather_info += f"- Condition: {condition}\n"
                weather_info += f"- Humidity: {humidity}%\n"
                weather_info += f"- Wind: {wind} km/h"
                
                return {
                    "success": True,
                    "query": query,
                    "source": "wttr.in Weather API",
                    "num_results": 1,
                    "results": [{
                        "title": f"Weather in {city}, {country}",
                        "snippet": weather_info,
                        "url": f"https://wttr.in/{location}"
                    }],
                    "weather_data": {
                        "location": f"{city}, {country}",
                        "temp_c": temp_c,
                        "temp_f": temp_f,
                        "condition": condition,
                        "humidity": humidity,
                        "wind_kmph": wind
                    }
                }
        except Exception as e:
            pass  # Fall through to other methods
    
    # Method 1: Try DuckDuckGo (more reliable)
    try:
        from duckduckgo_search import DDGS
        
        # Use proxy for external sites (Intel corporate network)
        proxy = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
        with DDGS(proxy=proxy) as ddgs:
            results = list(ddgs.text(query_clean, max_results=num_results))
        
        if results:
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", "")
                })
            
            return {
                "success": True,
                "query": query,
                "source": "DuckDuckGo",
                "num_results": len(formatted_results),
                "results": formatted_results
            }
    except:
        pass  # Try next method
    
    # Method 2: Try Google search via requests
    try:
        import requests
        from urllib.parse import quote_plus
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        search_url = f"https://www.google.com/search?q={quote_plus(query_clean)}&num={num_results}&hl=en"
        # Use proxy for external sites (Intel corporate network)
        proxy = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
        proxies = {"http": proxy, "https": proxy} if proxy else None
        response = requests.get(search_url, headers=headers, timeout=10, proxies=proxies)
        
        if response.status_code == 200:
            # Multiple patterns to try
            patterns = [
                r'<h3[^>]*>([^<]+)</h3>',
                r'class="[^"]*LC20lb[^"]*"[^>]*>([^<]+)<',
                r'<div class="[^"]*BNeawe[^"]*">([^<]{20,})</div>',
            ]
            
            for pattern in patterns:
                titles = re.findall(pattern, response.text)
                if titles:
                    formatted_results = [{"title": t.strip(), "snippet": "", "url": search_url} for t in titles[:num_results] if len(t.strip()) > 5]
                    if formatted_results:
                        return {
                            "success": True,
                            "query": query,
                            "source": "Google",
                            "num_results": len(formatted_results),
                            "results": formatted_results
                        }
    except Exception as e:
        pass  # Try next method
    
    # Method 3: Open browser and provide search URL
    try:
        import webbrowser
        from urllib.parse import quote_plus
        
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        webbrowser.open(search_url)
        
        return {
            "success": True,
            "query": query,
            "source": "Browser",
            "message": f"Opened Google search in your browser",
            "url": search_url,
            "results": [{
                "title": f"Google Search: {query}",
                "snippet": "Search opened in your default browser. Please check the browser window.",
                "url": search_url
            }]
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"All search methods failed. Last error: {str(e)}",
            "query": query
        }


# ============================================================================
# 📋 TOOL REGISTRY
# ============================================================================

def create_tool_registry() -> Dict[str, Tool]:
    """Create and register all available skills/tools"""
    
    tools = {
        "image2mermaid": Tool(
            name="image2mermaid",
            description="Convert a diagram image (PNG, JPG, etc.) to Mermaid diagram code. Use this when the user wants to convert an image to mermaid, create a mermaid diagram from an image, or turn a picture/diagram/flowchart into code.",
            parameters=[
                ToolParameter("image_path", "string", "Full path to the image file to convert"),
                ToolParameter("quick_mode", "boolean", "Use quick single-step conversion (faster)", required=False),
                ToolParameter("diagram_type", "string", "Hint for diagram type (flowchart TB, sequenceDiagram, etc.)", required=False)
            ],
            function=image2mermaid_skill,
            category="conversion"
        ),
        
        "chai_wifi": Tool(
            name="chai_wifi",
            description="Ask questions about Intel WiFi products using the ChAI WiFi_QnA assistant. Use this for WiFi-related questions, MLD, 802.11, WiFi 6/7, wireless networking.",
            parameters=[
                ToolParameter("query", "string", "The question to ask about Intel WiFi products")
            ],
            function=lambda query: chai_assistant_skill(query, "WiFi_QnA"),
            category="intel_assistant"
        ),
        
        "chai_bluetooth": Tool(
            name="chai_bluetooth", 
            description="Ask questions about Intel Bluetooth products using the ChAI BT_QnA assistant. Use this for Bluetooth-related questions, LE Audio, BT SIG specifications.",
            parameters=[
                ToolParameter("query", "string", "The question to ask about Intel Bluetooth products")
            ],
            function=lambda query: chai_assistant_skill(query, "BT_QnA"),
            category="intel_assistant"
        ),
        
        "chai_general": Tool(
            name="chai_general",
            description="Ask general questions about Intel Wireless products (WiFi, Bluetooth, TOP) using the ChAI QnA assistant.",
            parameters=[
                ToolParameter("query", "string", "The question to ask about Intel Wireless products")
            ],
            function=lambda query: chai_assistant_skill(query, "QnA"),
            category="intel_assistant"
        ),
        
        "calculator": Tool(
            name="calculator", 
            description="Perform mathematical calculations. Use this when the user wants to calculate, compute, or do math operations.",
            parameters=[
                ToolParameter("expression", "string", "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')")
            ],
            function=calculator_skill,
            category="utility"
        ),
        
        "text_analyzer": Tool(
            name="text_analyzer",
            description="Analyze text for various properties like word count, sentiment, or keywords. Use this when the user wants to analyze text.",
            parameters=[
                ToolParameter("text", "string", "The text to analyze"),
                ToolParameter("analysis_type", "string", "Type of analysis", required=False, 
                            enum=["summary", "sentiment", "keywords"])
            ],
            function=text_analyzer_skill,
            category="analysis"
        ),
        
        "web_search": Tool(
            name="web_search",
            description="Search the web for general knowledge questions. Use this when the user asks about topics NOT related to Intel products, WiFi, or Bluetooth - such as history, science, current events, definitions, 'what is', 'who is', 'how does', general facts, etc.",
            parameters=[
                ToolParameter("query", "string", "The search query for web search"),
                ToolParameter("num_results", "number", "Number of results to return (default 5)", required=False)
            ],
            function=web_search_skill,
            category="search"
        ),
    }
    
    return tools

# ============================================================================
# 🧠 AGENTIC ROUTER
# ============================================================================

class SkillsRouter:
    """
    The Skills Router analyzes user requests and routes them to the appropriate skill.
    """
    
    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools
        self.routing_patterns = self._build_routing_patterns()
    
    def _build_routing_patterns(self) -> List[Dict]:
        """Build patterns for routing natural language to skills"""
        return [
            # Image to Mermaid - HIGH PRIORITY
            {
                "tool": "image2mermaid",
                "patterns": [
                    r"(?:convert|turn|transform).*(?:image|picture|diagram|png|jpg|jpeg|gif).*(?:to|into)\s*mermaid",
                    r"mermaid.*(?:from|of).*(?:image|picture|diagram)",
                    r"(?:image|picture|diagram).*(?:to|into).*mermaid",
                    r"(?:create|generate|make).*mermaid.*(?:from|of).*(?:image|picture|diagram)",
                    r"image2mermaid",
                    r"(?:diagram|flowchart).*(?:to|into).*(?:code|text)",
                ],
                "extract": lambda m, text: self._extract_image_path(text)
            },
            
            # WiFi questions
            {
                "tool": "chai_wifi",
                "patterns": [
                    r"(?:what|explain|describe|tell me about).*(?:wifi|wi-fi|802\.11|mld|wireless|wlan)",
                    r"(?:wifi|wi-fi|802\.11|mld|wireless|wlan).*(?:what|how|why|when)",
                    r"wifi\s*\d+",  # WiFi 6, WiFi 7
                ],
                "extract": lambda m, text: {"query": text}
            },
            
            # Bluetooth questions
            {
                "tool": "chai_bluetooth",
                "patterns": [
                    r"(?:what|explain|describe|tell me about).*(?:bluetooth|bt|le audio|ble)",
                    r"(?:bluetooth|bt|le audio|ble).*(?:what|how|why|when)",
                ],
                "extract": lambda m, text: {"query": text}
            },
            
            # General Intel wireless
            {
                "tool": "chai_general",
                "patterns": [
                    r"(?:intel|wireless).*(?:product|specification|requirement)",
                    r"(?:what|explain).*(?:intel|wreq|dcr)",
                ],
                "extract": lambda m, text: {"query": text}
            },
            
            # Calculator
            {
                "tool": "calculator",
                "patterns": [
                    r"calculate\s+(.+)",
                    r"compute\s+(.+)",
                    r"what\s+is\s+(\d+[\s\d\+\-\*\/\.\(\)]+)",
                    r"(\d+\s*[\+\-\*\/]\s*\d+)",
                ],
                "extract": lambda m, text: {"expression": m.group(1).strip() if m else self._extract_math(text)}
            },
            
            # Text analyzer
            {
                "tool": "text_analyzer",
                "patterns": [
                    r"analyze\s+(?:this\s+)?(?:text|sentence|paragraph)",
                    r"(?:word|character)\s+count",
                    r"sentiment\s+(?:of|for|analysis)",
                    r"(?:find|extract)\s+keywords",
                ],
                "extract": lambda m, text: {"text": text, "analysis_type": self._detect_analysis_type(text)}
            },
            
            # Web search - for general knowledge (should be last as fallback)
            {
                "tool": "web_search",
                "patterns": [
                    r"search\s+(?:for|the\s+web|online|google|internet)",
                    r"(?:who|what|when|where|why|how)\s+(?:is|are|was|were|did|does|do)\s+(?!.*(?:wifi|bluetooth|intel|mld|802\.11))",
                    r"(?:tell me about|what do you know about|look up|find information)",
                    r"(?:definition|meaning)\s+of",
                    r"(?:history|origin)\s+of",
                    r"(?:capital|population|president|ceo)\s+of",
                ],
                "extract": lambda m, text: {"query": text, "num_results": 5}
            },
        ]
    
    def _extract_image_path(self, text: str) -> Dict:
        """Extract image path from text"""
        # Look for file paths
        path_patterns = [
            r'["\']([^"\']+\.(?:png|jpg|jpeg|gif|bmp|emf|webp))["\']',  # Quoted paths
            r'(\S+\.(?:png|jpg|jpeg|gif|bmp|emf|webp))',  # Unquoted paths
            r'([A-Za-z]:\\[^\s]+\.(?:png|jpg|jpeg|gif|bmp|emf|webp))',  # Windows paths
            r'(/[^\s]+\.(?:png|jpg|jpeg|gif|bmp|emf|webp))',  # Unix paths
        ]
        
        for pattern in path_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {"image_path": match.group(1), "quick_mode": True}
        
        return {"image_path": "", "quick_mode": True}
    
    def _extract_math(self, text: str) -> str:
        """Extract mathematical expression from text"""
        match = re.search(r'[\d\s\+\-\*\/\.\(\)]+', text)
        return match.group().strip() if match else "0"
    
    def _detect_analysis_type(self, text: str) -> str:
        """Detect what type of text analysis is requested"""
        text_lower = text.lower()
        if "sentiment" in text_lower or "feeling" in text_lower:
            return "sentiment"
        elif "keyword" in text_lower:
            return "keywords"
        return "summary"
    
    def route(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Route natural language input to the appropriate skill"""
        user_lower = user_input.lower()
        
        for route in self.routing_patterns:
            for pattern in route["patterns"]:
                match = re.search(pattern, user_lower)
                if match:
                    params = route["extract"](match, user_input)
                    return {
                        "tool": route["tool"],
                        "parameters": params,
                        "confidence": "high",
                        "matched_pattern": pattern
                    }
        
        # Keyword fallback
        keyword_mapping = {
            "image2mermaid": ["mermaid", "diagram", "flowchart", "convert image"],
            "chai_wifi": ["wifi", "wi-fi", "802.11", "wireless", "mld"],
            "chai_bluetooth": ["bluetooth", "bt", "le audio", "ble"],
            "calculator": ["calculate", "compute", "math", "+", "-", "*", "/"],
            "text_analyzer": ["analyze", "sentiment", "count", "keywords"],
            "web_search": ["search", "look up", "find", "google", "what is", "who is", "tell me about"],
        }
        
        for tool_name, keywords in keyword_mapping.items():
            if any(kw in user_lower for kw in keywords):
                return {
                    "tool": tool_name,
                    "parameters": self._get_default_params(tool_name, user_input),
                    "confidence": "low",
                    "matched_pattern": "keyword_fallback"
                }
        
        # Ultimate fallback: use web search for any question
        if "?" in user_input or user_lower.startswith(("what", "who", "when", "where", "why", "how")):
            return {
                "tool": "web_search",
                "parameters": {"query": user_input, "num_results": 5},
                "confidence": "low",
                "matched_pattern": "question_fallback"
            }
        
        return None
    
    def _get_default_params(self, tool_name: str, text: str) -> Dict:
        """Get default parameters for a tool based on text"""
        if tool_name == "image2mermaid":
            return self._extract_image_path(text)
        elif tool_name in ["chai_wifi", "chai_bluetooth", "chai_general"]:
            return {"query": text}
        elif tool_name == "calculator":
            return {"expression": self._extract_math(text)}
        elif tool_name == "text_analyzer":
            return {"text": text, "analysis_type": "summary"}
        elif tool_name == "web_search":
            return {"query": text, "num_results": 5}
        return {}
    
    def execute(self, routing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the routed skill with the extracted parameters"""
        tool_name = routing_result["tool"]
        params = routing_result["parameters"]
        
        if tool_name not in self.tools:
            return {"error": f"Skill '{tool_name}' not found", "success": False}
        
        tool = self.tools[tool_name]
        
        try:
            result = tool.function(**params)
            return {
                "success": result.get("success", True),
                "tool_used": tool_name,
                "parameters": params,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "tool_used": tool_name,
                "error": str(e)
            }

# ============================================================================
# 🎨 STREAMLIT UI
# ============================================================================

def display_skill_cards(tools: Dict[str, Tool]):
    """Display skills as cards"""
    st.markdown("### 🛠️ Available Skills")
    
    cols = st.columns(3)
    skill_icons = {
        "image2mermaid": "🖼️",
        "chai_wifi": "📶",
        "chai_bluetooth": "🔵",
        "chai_general": "💬",
        "calculator": "🔢",
        "text_analyzer": "📊"
    }
    
    for i, (name, tool) in enumerate(tools.items()):
        with cols[i % 3]:
            icon = skill_icons.get(name, "🔧")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #2d3436 0%, #000000 100%); 
                        padding: 15px; border-radius: 10px; margin: 5px 0;
                        border: 1px solid #4a90d9;'>
                <h4 style='color: #F5DEB3; margin: 0;'>{icon} {name}</h4>
                <p style='color: #FAEBD7; font-size: 0.85em; margin: 5px 0;'>{tool.description[:80]}...</p>
                <span style='color: #67b26f; font-size: 0.75em;'>Category: {tool.category}</span>
            </div>
            """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="🎯 WCS Skills Based Agentic Flows",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Dark theme CSS
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
            background: linear-gradient(90deg, #4a90d9 0%, #67b26f 100%);
            color: white !important;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #67b26f 0%, #4a90d9 100%);
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #FFD700 !important;
        }
        .mermaid-output {
            background: #1e1e1e;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #4a90d9;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize tools and router
    tools = create_tool_registry()
    router = SkillsRouter(tools)
    
    # Initialize session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Skills Demo")
        st.markdown("---")
        
        st.markdown("### ⚙️ Options")
        show_routing = st.checkbox("Show Routing Logic", value=True)
        show_raw_output = st.checkbox("Show Raw JSON", value=False)
        
        st.markdown("---")
        st.markdown("### 💡 Example Requests:")
        
        examples = [
            ("🖼️", "Convert image to mermaid: C:\\path\\to\\diagram.png"),
            ("📶", "What is MLD in WiFi 7?"),
            ("🔵", "Explain LE Audio in Bluetooth"),
            ("🔢", "Calculate 256 * 4 + 128"),
            ("📊", "Analyze sentiment: This product is amazing!"),
            ("🌐", "What is the capital of France?"),
            ("🌐", "Who invented the telephone?"),
        ]
        
        for icon, prompt in examples:
            if st.button(f"{icon} {prompt[:35]}...", key=f"ex_{prompt[:20]}"):
                st.session_state.example_prompt = prompt
                st.rerun()
        
        st.markdown("---")
        
        # File uploader for images
        st.markdown("### 📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Upload a diagram to convert",
            type=["png", "jpg", "jpeg", "gif", "bmp"],
            key="image_uploader"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = Path(tempfile.gettempdir()) / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_image_path = str(temp_path)
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            
            if st.button("🔄 Convert to Mermaid"):
                st.session_state.example_prompt = f"Convert this image to mermaid: {temp_path}"
                st.rerun()
        
        st.markdown("---")
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Main content
    st.title("🎯 WCS Skills Based Agentic Flows")
    st.markdown("### Real skills, intelligent routing, natural language")
    
    # Introduction
    with st.expander("📚 About This Demo (Click to Learn)", expanded=False):
        st.markdown(INTRO_TEXT)
    
    # Display skill cards
    display_skill_cards(tools)
    
    st.markdown("---")
    
    # Chat interface
    st.markdown("## 💬 Ask Me Anything")
    st.markdown("*I'll choose the right skill to answer your request!*")
    
    # Check for example prompt
    initial_value = ""
    if "example_prompt" in st.session_state:
        initial_value = st.session_state.example_prompt
        del st.session_state.example_prompt
    
    # User input
    user_input = st.text_area(
        "Your request:",
        value=initial_value,
        placeholder="e.g., Convert this image to mermaid: C:\\diagrams\\flowchart.png\nor: What is MLD in WiFi 7?",
        height=80,
        key="user_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        send_button = st.button("🚀 Execute", type="primary")
    with col2:
        if st.button("🔍 Analyze Only"):
            if user_input:
                routing_result = router.route(user_input)
                st.session_state.last_routing_result = routing_result
                st.session_state.last_user_input = user_input
                st.session_state.last_execution_result = None
    
    if send_button and user_input:
        with st.spinner("🔄 Processing your request..."):
            # Route the request
            routing_result = router.route(user_input)
            st.session_state.last_routing_result = routing_result
            st.session_state.last_user_input = user_input
            
            if routing_result:
                execution_result = router.execute(routing_result)
                st.session_state.last_execution_result = execution_result
            else:
                st.session_state.last_execution_result = None
    
    # Display results
    if st.session_state.get("last_routing_result") or st.session_state.get("last_user_input"):
        routing_result = st.session_state.get("last_routing_result")
        user_input_display = st.session_state.get("last_user_input", "")
        execution_result = st.session_state.get("last_execution_result")
        
        # Show routing explanation
        if show_routing and routing_result:
            st.markdown("### 🧠 Routing Decision")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Input:** {user_input_display[:100]}...")
            with col2:
                st.success(f"**Selected Skill:** {routing_result['tool']}")
                st.caption(f"Confidence: {routing_result['confidence']} | Pattern: {routing_result.get('matched_pattern', 'N/A')[:30]}")
        
        st.markdown("---")
        
        if routing_result and execution_result:
            st.markdown("### 📤 Result")
            
            result_data = execution_result.get("result", {})
            
            # Format based on skill type
            if routing_result['tool'] == "image2mermaid":
                if result_data.get("success"):
                    st.success("✅ Image converted to Mermaid!")
                    
                    # Display mermaid code
                    mermaid_code = result_data.get("mermaid_code", "")
                    st.markdown("**Generated Mermaid Code:**")
                    st.code(mermaid_code, language="mermaid")
                    
                    # Try to render it
                    st.markdown("**Preview:**")
                    try:
                        st.markdown(f"```mermaid\n{mermaid_code}\n```")
                    except:
                        st.info("Preview not available in this environment")
                else:
                    st.error(f"❌ Conversion failed: {result_data.get('error')}")
                    if result_data.get('stderr'):
                        with st.expander("Error details"):
                            st.text(result_data.get('stderr'))
            
            elif routing_result['tool'].startswith("chai_"):
                if result_data.get("success"):
                    st.success(f"✅ Response from {result_data.get('usecase', 'ChAI')}")
                    st.markdown("**Answer:**")
                    st.markdown(result_data.get("response", "No response"))
                else:
                    st.error(f"❌ ChAI Error: {result_data.get('error')}")
            
            elif routing_result['tool'] == "calculator":
                if result_data.get("success"):
                    st.success(f"🔢 {result_data.get('expression')} = **{result_data.get('result')}**")
                else:
                    st.error(f"❌ {result_data.get('error')}")
            
            elif routing_result['tool'] == "text_analyzer":
                if result_data.get("success"):
                    st.success("📊 Text Analysis Complete")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Words", result_data.get('word_count'))
                    with col2:
                        st.metric("Characters", result_data.get('character_count'))
                    with col3:
                        st.metric("Sentences", result_data.get('sentence_count'))
                    
                    if 'sentiment' in result_data:
                        sentiment = result_data['sentiment']
                        emoji = "😊" if sentiment == "positive" else "😐" if sentiment == "neutral" else "😞"
                        st.write(f"**Sentiment:** {emoji} {sentiment}")
                    
                    if 'keywords' in result_data:
                        st.write(f"**Keywords:** {', '.join(result_data['keywords'])}")
            
            elif routing_result['tool'] == "web_search":
                if result_data.get("success"):
                    st.success(f"🌐 Web Search: Found {result_data.get('num_results', 0)} results")
                    st.markdown("**Search Results:**")
                    
                    # Display formatted results
                    for i, r in enumerate(result_data.get('results', [])[:5], 1):
                        with st.container():
                            st.markdown(f"**{i}. [{r.get('title', 'No title')}]({r.get('url', '#')})**")
                            st.markdown(f"_{r.get('snippet', 'No description')}_")
                            st.markdown("---")
                else:
                    st.error(f"❌ Web Search Error: {result_data.get('error')}")
            
            if show_raw_output:
                with st.expander("Raw JSON Output"):
                    st.json(result_data)
            
            # Add to history
            if send_button and user_input:
                st.session_state.conversation_history.append({
                    "user": user_input,
                    "tool": routing_result['tool'],
                    "success": execution_result.get("success", False),
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
        
        elif routing_result is None:
            st.warning("🤔 I couldn't determine which skill to use for that request.")
            st.info("Try being more specific, or mention the skill type (e.g., 'convert to mermaid', 'wifi question')")
    
    # Conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("### 📜 Recent Activity")
        
        for entry in reversed(st.session_state.conversation_history[-5:]):
            status = "✅" if entry.get('success') else "❌"
            st.markdown(f"**{entry['timestamp']}** | {status} `{entry['tool']}`: {entry['user'][:50]}...")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🎯 Skills-Based Agentic Demo - Powered by .claude/skills</p>
        <p>Skills: Image2Mermaid | ChAI Assistants | Calculator | Text Analysis</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
