"""
🔌 MCP Skills Server
====================
A Model Context Protocol (MCP) server that exposes WCS skills as tools.

This server implements the MCP specification, allowing any MCP-compatible
client to discover and execute skills.

Run this server with:
    python 12_mcp_skills_server.py

Or use with MCP stdio transport:
    python -m mcp run 12_mcp_skills_server.py

Author: WCS Skills MCP Server
Date: February 2026
"""

import asyncio
import json
import os
import sys
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Any, Sequence
from dotenv import load_dotenv

# Load environment variables from the correct location
# (important when running as MCP subprocess)
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Also try loading from current working directory
load_dotenv()

# Suppress SSL warnings (Intel proxy intercepts SSL)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Debug: Print if CHAI_SDK_CLIENT_SECRET is set (to stderr so it doesn't interfere with MCP)
if os.environ.get("CHAI_SDK_CLIENT_SECRET"):
    print("✓ CHAI_SDK_CLIENT_SECRET is set", file=sys.stderr)
else:
    print("⚠ CHAI_SDK_CLIENT_SECRET is NOT set", file=sys.stderr)

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
)

# ============================================================================
# 📁 CONFIGURATION
# ============================================================================

SKILLS_DIR = Path(__file__).parent / ".claude" / "skills"
MINIPYTHON = r"C:\MiniPython\python.exe"

# Fallback to system Python if MiniPython not available
if not os.path.exists(MINIPYTHON):
    MINIPYTHON = sys.executable

# ============================================================================
# 🛠️ SKILL IMPLEMENTATIONS
# ============================================================================

async def image2mermaid_execute(image_path: str, quick_mode: bool = True, diagram_type: str = "") -> dict:
    """Convert an image to Mermaid diagram code"""
    skill_script = SKILLS_DIR / "Doc2Md-image2mermaid" / "image2mermaid.py"
    
    if not skill_script.exists():
        return {"error": f"Skill not found at {skill_script}", "success": False}
    
    if not os.path.exists(image_path):
        return {"error": f"Image file not found: {image_path}", "success": False}
    
    output_dir = tempfile.mkdtemp(prefix="mermaid_")
    cmd = [MINIPYTHON, str(skill_script), image_path, "-o", output_dir]
    
    if quick_mode:
        cmd.append("-q")
    if diagram_type:
        cmd.extend(["-t", diagram_type])
    
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
                cwd=str(Path(__file__).parent)
            )
        )
        
        mermaid_files = list(Path(output_dir).glob("*.mermaid")) + list(Path(output_dir).glob("*.md"))
        
        if mermaid_files:
            with open(mermaid_files[0], 'r', encoding='utf-8') as f:
                mermaid_content = f.read()
            return {
                "success": True,
                "mermaid_code": mermaid_content,
                "output_file": str(mermaid_files[0]),
                "image_path": image_path
            }
        else:
            return {"success": False, "error": "No mermaid output generated", "stderr": result.stderr}
            
    except subprocess.TimeoutExpired:
        return {"error": "Skill execution timed out (120s)", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


async def chai_assistant_execute(query: str, usecase: str = "WiFi_QnA") -> dict:
    """Query the ChAI Assistant"""
    skill_script = SKILLS_DIR / "ChAI-Assistant" / "ask_chai_tool.py"
    
    if not skill_script.exists():
        return {"error": f"Skill not found at {skill_script}", "success": False}
    
    client_secret = os.environ.get("CHAI_SDK_CLIENT_SECRET")
    if not client_secret:
        return {"error": "CHAI_SDK_CLIENT_SECRET not set", "success": False}
    
    cmd = [MINIPYTHON, str(skill_script), "--query", query, "--usecase", usecase]
    
    try:
        env = {**os.environ, "CHAI_SDK_CLIENT_SECRET": client_secret}
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                cmd, capture_output=True, text=True, timeout=90,
                cwd=str(Path(__file__).parent), env=env
            )
        )
        
        if result.returncode == 0:
            return {"success": True, "response": result.stdout.strip(), "usecase": usecase, "query": query}
        else:
            return {"success": False, "error": result.stderr or "Unknown error"}
            
    except subprocess.TimeoutExpired:
        return {"error": "ChAI request timed out (90s)", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


def calculator_execute(expression: str) -> dict:
    """Safe mathematical expression evaluator"""
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression", "success": False}
        result = eval(expression)
        return {"success": True, "expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e), "success": False}


def text_analyzer_execute(text: str, analysis_type: str = "summary") -> dict:
    """Analyze text properties"""
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
    
    if analysis_type == "sentiment":
        positive = ["good", "great", "excellent", "happy", "love", "amazing"]
        negative = ["bad", "terrible", "hate", "awful", "sad", "poor"]
        text_lower = text.lower()
        pos = sum(1 for w in positive if w in text_lower)
        neg = sum(1 for w in negative if w in text_lower)
        result["sentiment"] = "positive" if pos > neg else "negative" if neg > pos else "neutral"
    elif analysis_type == "keywords":
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        top = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        result["keywords"] = [w[0] for w in top]
    else:
        result["summary"] = f"Text contains {word_count} words in {sentence_count} sentences."
    
    return result


def web_search_execute(query: str, num_results: int = 5) -> dict:
    """Search the web - Weather API first, then DuckDuckGo, then Google, then browser"""
    query_clean = query.strip()
    query_lower = query_clean.lower()
    
    # Debug
    print(f"[DEBUG] web_search_execute: query='{query_clean}'", file=sys.stderr)
    
    # Method 0: Weather API for weather queries
    if any(word in query_lower for word in ["weather", "temperature", "rain", "sunny", "forecast", "wetaher"]):
        print(f"[DEBUG] Detected weather query, using wttr.in", file=sys.stderr)
        try:
            import requests
            
            # Extract location - be careful not to match within words
            location = "Tel Aviv"  # Default
            # Use word boundaries for matching
            import re
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
            
            print(f"[DEBUG] Extracted location: {location}", file=sys.stderr)
            
            weather_url = f"https://wttr.in/{location}?format=j1"
            # Use proxy for external sites (Intel corporate network)
            proxy = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
            print(f"[DEBUG] Proxy: {proxy}", file=sys.stderr)
            print(f"[DEBUG] Weather URL: {weather_url}", file=sys.stderr)
            proxies = {"http": proxy, "https": proxy} if proxy else None
            
            # Disable SSL verification for proxy (Intel proxy intercepts SSL)
            response = requests.get(
                weather_url, 
                timeout=15, 
                headers={'User-Agent': 'curl/7.68.0'},
                proxies=proxies,
                verify=False  # Skip SSL verification for proxy
            )
            print(f"[DEBUG] Weather response status: {response.status_code}", file=sys.stderr)
            
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
            print(f"[DEBUG] wttr.in failed: {e}", file=sys.stderr)
    
    # Method 1: Try DuckDuckGo (more reliable)
    try:
        from duckduckgo_search import DDGS
        print(f"[DEBUG] Trying DuckDuckGo...", file=sys.stderr)
        
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
        pass
    
    # Method 2: Try Google via requests
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
            patterns = [
                r'<h3[^>]*>([^<]+)</h3>',
                r'class="[^"]*LC20lb[^"]*"[^>]*>([^<]+)<',
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
    except:
        pass
    
    # Method 3: Open browser
    try:
        import webbrowser
        from urllib.parse import quote_plus
        
        search_url = f"https://www.google.com/search?q={quote_plus(query_clean)}"
        webbrowser.open(search_url)
        
        return {
            "success": True,
            "query": query,
            "source": "Browser",
            "message": "Opened Google search in browser",
            "url": search_url,
            "results": [{"title": f"Google Search: {query}", "snippet": "Opened in browser", "url": search_url}]
        }
    except Exception as e:
        return {"success": False, "error": str(e), "query": query}


# ============================================================================
# 🔌 MCP SERVER SETUP
# ============================================================================

# Create the MCP server
server = Server("wcs-skills-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    List all available tools (MCP tools/list handler).
    This is called by MCP clients to discover available tools.
    """
    return [
        Tool(
            name="image2mermaid",
            description="Convert a diagram image (PNG, JPG, etc.) to Mermaid diagram code using AI vision. Use this when the user wants to convert an image to mermaid, create a mermaid diagram from an image, or turn a picture/diagram/flowchart into code.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Full path to the image file to convert"
                    },
                    "quick_mode": {
                        "type": "boolean",
                        "description": "Use quick single-step conversion (faster)",
                        "default": True
                    },
                    "diagram_type": {
                        "type": "string",
                        "description": "Hint for diagram type (flowchart TB, sequenceDiagram, etc.)",
                        "default": ""
                    }
                },
                "required": ["image_path"]
            }
        ),
        Tool(
            name="chai_wifi_qna",
            description="Ask questions about Intel WiFi products using the ChAI WiFi_QnA assistant. Use for WiFi-related questions, MLD, 802.11, WiFi 6/7, wireless networking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to ask about Intel WiFi products"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="chai_bluetooth_qna",
            description="Ask questions about Intel Bluetooth products using the ChAI BT_QnA assistant. Use for Bluetooth-related questions, LE Audio, BT SIG specifications.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to ask about Intel Bluetooth products"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="chai_general_qna",
            description="Ask general questions about Intel Wireless products (WiFi, Bluetooth, TOP) using the ChAI QnA assistant.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to ask about Intel Wireless products"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="calculator",
            description="Perform mathematical calculations. Use when the user wants to calculate, compute, or do math operations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="text_analyzer",
            description="Analyze text for various properties like word count, sentiment, or keywords.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze"
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis: summary, sentiment, or keywords",
                        "enum": ["summary", "sentiment", "keywords"],
                        "default": "summary"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="web_search",
            description="Search the web for general knowledge questions. Use this when the user asks about topics NOT related to Intel products - such as history, science, current events, definitions, 'what is', 'who is', 'how does', general facts, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """
    Execute a tool (MCP tools/call handler).
    This is called by MCP clients to execute a specific tool.
    """
    try:
        if name == "image2mermaid":
            result = await image2mermaid_execute(
                image_path=arguments["image_path"],
                quick_mode=arguments.get("quick_mode", True),
                diagram_type=arguments.get("diagram_type", "")
            )
        elif name == "chai_wifi_qna":
            result = await chai_assistant_execute(arguments["query"], "WiFi_QnA")
        elif name == "chai_bluetooth_qna":
            result = await chai_assistant_execute(arguments["query"], "BT_QnA")
        elif name == "chai_general_qna":
            result = await chai_assistant_execute(arguments["query"], "QnA")
        elif name == "calculator":
            result = calculator_execute(arguments["expression"])
        elif name == "text_analyzer":
            result = text_analyzer_execute(
                text=arguments["text"],
                analysis_type=arguments.get("analysis_type", "summary")
            )
        elif name == "web_search":
            result = web_search_execute(
                query=arguments["query"],
                num_results=arguments.get("num_results", 5)
            )
        else:
            result = {"error": f"Unknown tool: {name}", "success": False}
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e), "success": False}))]


# ============================================================================
# 🚀 SERVER ENTRY POINT
# ============================================================================

async def main():
    """Run the MCP server using stdio transport"""
    print("🔌 Starting WCS Skills MCP Server...", file=sys.stderr)
    print(f"📁 Skills directory: {SKILLS_DIR}", file=sys.stderr)
    print(f"🐍 Python executable: {MINIPYTHON}", file=sys.stderr)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
