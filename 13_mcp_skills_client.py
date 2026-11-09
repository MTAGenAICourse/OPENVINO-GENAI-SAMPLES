"""
🎯 WCS Skills Based Agentic Flows - MCP Edition
================================================
A demonstration of agentic AI using the Model Context Protocol (MCP)
to discover and execute skills.

This version uses a HYBRID approach:
- ChAI skills: Direct subprocess call (more reliable for long-running calls)
- Other tools: MCP server via stdio transport

Run this client:
    streamlit run 13_mcp_skills_client.py

Author: WCS Skills MCP Client
Date: February 2026
"""

import streamlit as st
import json
import asyncio
import re
import os
import sys
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MCP SDK imports
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# ============================================================================
# 📁 CONFIGURATION
# ============================================================================

MCP_SERVER_SCRIPT = Path(__file__).parent / "12_mcp_skills_server.py"
SKILLS_DIR = Path(__file__).parent / ".claude" / "skills"
PYTHON_EXECUTABLE = sys.executable

# Try to find MiniPython for skill execution
MINIPYTHON = r"C:\MiniPython\python.exe" if Path(r"C:\MiniPython\python.exe").exists() else sys.executable

# ============================================================================
# 💬 DIRECT CHAI SKILL (bypasses MCP for reliability)
# ============================================================================

def chai_assistant_direct(query: str, usecase: str = "WiFi_QnA") -> Dict[str, Any]:
    """
    Query the ChAI Assistant DIRECTLY via subprocess (not MCP).
    This is more reliable for long-running ChAI calls.
    """
    skill_script = SKILLS_DIR / "ChAI-Assistant" / "ask_chai_tool.py"
    
    if not skill_script.exists():
        return {"error": f"Skill not found at {skill_script}", "success": False}
    
    client_secret = os.environ.get("CHAI_SDK_CLIENT_SECRET")
    if not client_secret:
        return {"error": "CHAI_SDK_CLIENT_SECRET not set", "success": False}
    
    cmd = [MINIPYTHON, str(skill_script), "--query", query, "--usecase", usecase]
    
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
            return {"success": True, "response": result.stdout.strip(), "usecase": usecase, "query": query}
        else:
            return {"success": False, "error": result.stderr or "Unknown error", "stdout": result.stdout}
            
    except subprocess.TimeoutExpired:
        return {"error": "ChAI request timed out (90s)", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}

# ============================================================================
# 📚 EDUCATIONAL INTRO
# ============================================================================

INTRO_TEXT = """
## 🔌 Model Context Protocol (MCP) Edition

This demo uses the **real MCP protocol** to communicate with skills:

### MCP Architecture:
```
┌──────────────────┐                      ┌──────────────────┐
│   Streamlit UI   │                      │   MCP Server     │
│   (MCP Client)   │◄────── stdio ──────►│   (Skills)       │
└──────────────────┘                      └──────────────────┘
        │                                         │
        │  1. tools/list                          │
        │────────────────────────────────────────►│
        │◄─── [image2mermaid, chai_wifi, ...]────│
        │                                         │
        │  2. tools/call {name, arguments}        │
        │────────────────────────────────────────►│
        │◄─── {result}───────────────────────────│
```

### MCP Benefits:
- **Standardized Protocol**: Any MCP client can use these skills
- **Tool Discovery**: Client discovers available tools at runtime
- **Type Safety**: JSON Schema validation for parameters
- **Extensibility**: Add new skills without changing the client
"""

# ============================================================================
# 🔌 MCP CLIENT
# ============================================================================

class MCPSkillsClient:
    """MCP Client that connects to the WCS Skills Server"""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.tools: List[Dict] = []
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to the MCP server and discover tools"""
        try:
            server_params = StdioServerParameters(
                command=PYTHON_EXECUTABLE,
                args=[str(MCP_SERVER_SCRIPT)],
                env={**os.environ}
            )
            
            # Create the client connection
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    
                    # Initialize the connection
                    await session.initialize()
                    
                    # Discover tools
                    tools_response = await session.list_tools()
                    self.tools = [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.inputSchema
                        }
                        for tool in tools_response.tools
                    ]
                    
                    self._connected = True
                    return True
                    
        except Exception as e:
            st.error(f"Failed to connect to MCP server: {e}")
            return False
    
    async def call_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """
        Call a tool - uses DIRECT call for ChAI, MCP for others.
        
        ChAI tools are called directly via subprocess for reliability
        (MCP can timeout on long-running ChAI calls).
        """
        # DIRECT CALL for ChAI tools (more reliable)
        if tool_name in ["chai_wifi_qna", "chai_bluetooth_qna", "chai_general_qna"]:
            usecase_map = {
                "chai_wifi_qna": "WiFi_QnA",
                "chai_bluetooth_qna": "BT_QnA",
                "chai_general_qna": "QnA"
            }
            usecase = usecase_map.get(tool_name, "WiFi_QnA")
            return chai_assistant_direct(arguments.get("query", ""), usecase)
        
        # MCP CALL for all other tools
        try:
            server_params = StdioServerParameters(
                command=PYTHON_EXECUTABLE,
                args=[str(MCP_SERVER_SCRIPT)],
                env={**os.environ}
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Call the tool via MCP
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments),
                        timeout=120.0
                    )
                    
                    # Parse the result
                    if result.content:
                        for content in result.content:
                            if hasattr(content, 'text'):
                                return json.loads(content.text)
                    
                    return {"error": "No result returned", "success": False}
                    
        except asyncio.TimeoutError:
            return {"error": "MCP call timed out (120s)", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def discover_tools(self) -> List[Dict]:
        """Discover available tools from the MCP server"""
        try:
            server_params = StdioServerParameters(
                command=PYTHON_EXECUTABLE,
                args=[str(MCP_SERVER_SCRIPT)],
                env={**os.environ}
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_response = await session.list_tools()
                    
                    return [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.inputSchema
                        }
                        for tool in tools_response.tools
                    ]
                    
        except Exception as e:
            st.error(f"Failed to discover tools: {e}")
            return []


# ============================================================================
# 🧠 AGENTIC ROUTER (Routes to MCP tools)
# ============================================================================

class MCPRouter:
    """Routes natural language to MCP tools"""
    
    def __init__(self, tools: List[Dict]):
        self.tools = {t["name"]: t for t in tools}
        self.routing_patterns = self._build_patterns()
    
    def _build_patterns(self) -> List[Dict]:
        """Build routing patterns for natural language"""
        return [
            {
                "tool": "image2mermaid",
                "patterns": [
                    r"(?:convert|turn|transform).*(?:image|picture|diagram).*(?:to|into)\s*mermaid",
                    r"mermaid.*(?:from|of).*(?:image|picture|diagram)",
                    r"(?:image|picture|diagram).*(?:to|into).*mermaid",
                    r"image2mermaid",
                ],
                "extract": lambda text: self._extract_image_path(text)
            },
            {
                "tool": "chai_wifi_qna",
                "patterns": [
                    r"(?:what|explain|describe).*(?:wifi|wi-fi|802\.11|mld|wireless|wlan)",
                    r"(?:wifi|wi-fi|802\.11|mld).*(?:what|how|why)",
                ],
                "extract": lambda text: {"query": text}
            },
            {
                "tool": "chai_bluetooth_qna",
                "patterns": [
                    r"(?:what|explain|describe).*(?:bluetooth|bt|le audio|ble)",
                    r"(?:bluetooth|bt|le audio|ble).*(?:what|how|why)",
                ],
                "extract": lambda text: {"query": text}
            },
            {
                "tool": "chai_general_qna",
                "patterns": [
                    r"(?:intel|wireless).*(?:product|specification)",
                    r"(?:wreq|dcr)",
                ],
                "extract": lambda text: {"query": text}
            },
            {
                "tool": "calculator",
                "patterns": [
                    r"calculate\s+(.+)",
                    r"compute\s+(.+)",
                    r"(\d+\s*[\+\-\*\/]\s*\d+)",
                ],
                "extract": lambda text: {"expression": self._extract_math(text)}
            },
            {
                "tool": "text_analyzer",
                "patterns": [
                    r"analyze.*(?:text|sentence)",
                    r"sentiment",
                    r"(?:word|character)\s+count",
                ],
                "extract": lambda text: {"text": text, "analysis_type": self._detect_analysis(text)}
            },
            {
                "tool": "web_search",
                "patterns": [
                    r"search\s+(?:for|the\s+web|online|google|internet)",
                    r"(?:who|what|when|where|why|how)\s+(?:is|are|was|were|did|does|do)\s+(?!.*(?:wifi|bluetooth|intel|mld|802\.11))",
                    r"(?:tell me about|look up|find information)",
                    r"(?:definition|meaning|history|capital|population)\s+of",
                ],
                "extract": lambda text: {"query": text, "num_results": 5}
            },
        ]
    
    def _extract_image_path(self, text: str) -> Dict:
        """Extract image path from text"""
        patterns = [
            r'["\']([^"\']+\.(?:png|jpg|jpeg|gif|bmp))["\']',
            r'(\S+\.(?:png|jpg|jpeg|gif|bmp))',
            r'([A-Za-z]:\\[^\s]+\.(?:png|jpg|jpeg|gif|bmp))',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {"image_path": match.group(1), "quick_mode": True}
        return {"image_path": "", "quick_mode": True}
    
    def _extract_math(self, text: str) -> str:
        """Extract math expression"""
        match = re.search(r'[\d\s\+\-\*\/\.\(\)]+', text)
        return match.group().strip() if match else "0"
    
    def _detect_analysis(self, text: str) -> str:
        """Detect analysis type"""
        if "sentiment" in text.lower():
            return "sentiment"
        elif "keyword" in text.lower():
            return "keywords"
        return "summary"
    
    def route(self, user_input: str) -> Optional[Dict]:
        """Route input to appropriate tool"""
        user_lower = user_input.lower()
        
        for route in self.routing_patterns:
            for pattern in route["patterns"]:
                if re.search(pattern, user_lower):
                    return {
                        "tool": route["tool"],
                        "arguments": route["extract"](user_input),
                        "confidence": "high"
                    }
        
        # Fallback: use web search for any question not matched
        if "?" in user_input or user_lower.startswith(("what", "who", "when", "where", "why", "how")):
            return {
                "tool": "web_search",
                "arguments": {"query": user_input, "num_results": 5},
                "confidence": "low"
            }
        
        return None


# ============================================================================
# 🎨 STREAMLIT UI
# ============================================================================

def display_mcp_tools(tools: List[Dict]):
    """Display MCP tools with their schemas"""
    st.markdown("### 🔌 MCP Tools (Discovered from Server)")
    
    for tool in tools:
        with st.expander(f"🔧 {tool['name']}", expanded=False):
            st.markdown(f"**Description:** {tool['description']}")
            st.markdown("**Input Schema:**")
            st.json(tool['inputSchema'])


def run_async(coro):
    """Run async function in Streamlit"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def main():
    st.set_page_config(
        page_title="🔌 WCS Skills - MCP Edition",
        page_icon="🔌",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Dark theme CSS
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        .stApp h1 { color: #F5DEB3 !important; }
        .stApp h2, .stApp h3, .stApp h4 { color: #FAEBD7 !important; }
        .stButton>button {
            background: linear-gradient(90deg, #9b59b6 0%, #3498db 100%);
            color: white !important;
            border: none;
        }
        .mcp-badge {
            background: linear-gradient(90deg, #9b59b6, #3498db);
            padding: 5px 15px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize MCP client
    mcp_client = MCPSkillsClient()
    
    # Initialize session state
    if "mcp_tools" not in st.session_state:
        st.session_state.mcp_tools = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔌 MCP Edition")
        st.markdown('<span class="mcp-badge">Real MCP Protocol</span>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Discover tools button
        if st.button("🔍 Discover MCP Tools"):
            with st.spinner("Connecting to MCP server..."):
                tools = run_async(mcp_client.discover_tools())
                if tools:
                    st.session_state.mcp_tools = tools
                    st.success(f"✅ Found {len(tools)} tools!")
                else:
                    st.error("Failed to discover tools")
        
        if st.session_state.mcp_tools:
            st.markdown(f"**Tools discovered:** {len(st.session_state.mcp_tools)}")
            for tool in st.session_state.mcp_tools:
                st.markdown(f"- `{tool['name']}`")
        
        st.markdown("---")
        st.markdown("### 💡 Examples:")
        
        examples = [
            "Convert image to mermaid: C:\\path\\diagram.png",
            "What is MLD in WiFi 7?",
            "Explain LE Audio in Bluetooth",
            "Calculate 256 * 4 + 128",
            "What is the capital of France?",
            "Who invented the light bulb?",
        ]
        
        for ex in examples:
            if st.button(f"📝 {ex[:30]}...", key=f"ex_{ex[:15]}"):
                st.session_state.example_prompt = ex
                st.rerun()
        
        st.markdown("---")
        if st.button("🗑️ Clear"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Main content
    st.title("🔌 WCS Skills Based Agentic Flows")
    st.markdown("### MCP Edition - Real Model Context Protocol")
    
    # MCP indicator
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 10px; 
                    background: linear-gradient(90deg, #9b59b6, #3498db);
                    border-radius: 10px; color: white;'>
            <strong>🔌 MCP Protocol Active</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Introduction
    with st.expander("📚 About MCP (Click to Learn)", expanded=False):
        st.markdown(INTRO_TEXT)
    
    # Display discovered tools
    if st.session_state.mcp_tools:
        display_mcp_tools(st.session_state.mcp_tools)
    else:
        st.info("👆 Click 'Discover MCP Tools' in the sidebar to connect to the MCP server")
    
    st.markdown("---")
    
    # Chat interface
    st.markdown("## 💬 Chat with MCP Agent")
    
    initial_value = st.session_state.pop("example_prompt", "")
    
    user_input = st.text_area(
        "Your request:",
        value=initial_value,
        placeholder="e.g., What is MLD in WiFi 7?",
        height=80
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        execute_btn = st.button("🚀 Execute via MCP", type="primary")
    
    if execute_btn and user_input and st.session_state.mcp_tools:
        # Create router with discovered tools
        router = MCPRouter(st.session_state.mcp_tools)
        routing = router.route(user_input)
        
        if routing:
            st.markdown("### 🧠 MCP Routing")
            st.info(f"**Tool:** `{routing['tool']}` | **Arguments:** {json.dumps(routing['arguments'])}")
            
            with st.spinner(f"🔌 Calling MCP tool: {routing['tool']}..."):
                result = run_async(mcp_client.call_tool(routing['tool'], routing['arguments']))
            
            st.markdown("### 📤 MCP Response")
            
            if result.get("success"):
                # Format based on tool
                if routing['tool'] == "image2mermaid":
                    st.success("✅ Converted to Mermaid!")
                    st.code(result.get("mermaid_code", ""), language="mermaid")
                    
                elif routing['tool'].startswith("chai_"):
                    st.success(f"✅ Response from {result.get('usecase', 'ChAI')}")
                    st.markdown(result.get("response", ""))
                    
                elif routing['tool'] == "calculator":
                    st.success(f"🔢 {result.get('expression')} = **{result.get('result')}**")
                    
                elif routing['tool'] == "text_analyzer":
                    st.success("📊 Analysis Complete")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Words", result.get('word_count'))
                    with col2:
                        st.metric("Characters", result.get('character_count'))
                    if 'sentiment' in result:
                        st.write(f"**Sentiment:** {result['sentiment']}")
                
                elif routing['tool'] == "web_search":
                    st.success(f"🌐 Web Search: Found {result.get('num_results', 0)} results")
                    st.markdown("**Search Results:**")
                    
                    for i, r in enumerate(result.get('results', [])[:5], 1):
                        with st.container():
                            st.markdown(f"**{i}. [{r.get('title', 'No title')}]({r.get('url', '#')})**")
                            st.markdown(f"_{r.get('snippet', 'No description')}_")
                            st.markdown("---")
                
                with st.expander("Raw MCP Response"):
                    st.json(result)
            else:
                st.error(f"❌ Error: {result.get('error')}")
            
            # Add to history
            st.session_state.conversation_history.append({
                "input": user_input,
                "tool": routing['tool'],
                "success": result.get("success", False),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        else:
            st.warning("🤔 Could not route to any MCP tool")
    
    elif execute_btn and not st.session_state.mcp_tools:
        st.warning("Please discover MCP tools first (sidebar button)")
    
    # History
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("### 📜 MCP Call History")
        for entry in reversed(st.session_state.conversation_history[-5:]):
            status = "✅" if entry['success'] else "❌"
            st.markdown(f"**{entry['timestamp']}** | {status} `{entry['tool']}`: {entry['input'][:50]}...")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🔌 WCS Skills - MCP Edition | Real Model Context Protocol Implementation</p>
        <p>Server: 12_mcp_skills_server.py | Client: 13_mcp_skills_client.py</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
