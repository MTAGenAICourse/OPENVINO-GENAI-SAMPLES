# 🎓 MCP Agentic Tutorial

## Model Context Protocol (MCP) Educational Demo

This tutorial demonstrates how AI agents use the **Model Context Protocol (MCP)** to interact with tools through natural language. It's designed for a college course on Generative AI Applications.

---

## 🎯 Learning Objectives

By completing this tutorial, you will:

1. **Understand MCP Concepts**
   - What tools, resources, and prompts are in MCP
   - How AI agents decide which tool to use
   - The role of schemas in tool definitions

2. **Learn Tool Design Patterns**
   - How to define tools with clear descriptions
   - Parameter schemas and validation
   - Handling tool responses

3. **Experience Agentic Workflows**
   - Natural language to tool routing
   - Multi-step reasoning
   - Error handling and fallbacks

---

## 🚀 Running the Demo

```bash
# Activate the virtual environment
.\venv_agentic_demo\Scripts\activate  # Windows
source venv_agentic_demo/bin/activate  # Linux/Mac

# Run the Streamlit app
streamlit run 07_mcp_agentic_tutorial.py
```

---

## 🔧 Available Tools

The demo includes 5 educational tools:

| Tool | Description | Example Query |
|------|-------------|---------------|
| 🌤️ **Weather** | Get weather for a location | "What's the weather in Paris?" |
| 🧮 **Calculator** | Perform math calculations | "Calculate 125 * 4 + 50" |
| 📊 **Text Analyzer** | Analyze text (sentiment, keywords) | "Analyze this text for sentiment: I love it!" |
| 📏 **Unit Converter** | Convert between units | "Convert 100 km to miles" |
| ✅ **Task Manager** | Manage a todo list | "Add a task: Complete homework" |

---

## 📚 Key Concepts Explained

### What is MCP?

The **Model Context Protocol** is a standardized way for AI models to:
- Discover available tools
- Understand tool capabilities through schemas
- Execute tools with proper parameters
- Handle responses consistently

### Agentic Routing

The demo shows how an "agent" routes natural language to tools:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────┐
│  "What's the    │────▶│  Intent         │────▶│  weather    │
│   weather in    │     │  Recognition    │     │  tool       │
│   Tokyo?"       │     │                 │     │             │
└─────────────────┘     └─────────────────┘     └─────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  Extract params │
                        │  location=Tokyo │
                        └─────────────────┘
```

### Tool Schema Format

Each tool is defined with a JSON schema:

```json
{
  "name": "weather",
  "description": "Get current weather information for a location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or location"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location"]
  }
}
```

---

## 🧪 Exercises

### Exercise 1: Try Different Queries
Test how the agent handles various phrasings:
- "Weather in London"
- "Is it raining in Seattle?"
- "Tell me the temperature in Tokyo"

### Exercise 2: Chain Tools
Try using multiple tools in sequence:
1. Add a task: "Study for exam"
2. List your tasks
3. Complete the task

### Exercise 3: Edge Cases
See how the agent handles:
- Ambiguous requests
- Missing parameters
- Invalid inputs

---

## 🏗️ Code Structure

```
07_mcp_agentic_tutorial.py
├── Tool Definitions
│   ├── @dataclass ToolParameter
│   ├── @dataclass Tool
│   └── Tool functions (weather, calculator, etc.)
│
├── Tool Registry
│   └── create_tool_registry() - Creates all available tools
│
├── Agentic Router
│   ├── Pattern matching for intent detection
│   ├── Parameter extraction
│   └── Tool execution
│
└── Streamlit UI
    ├── Interactive chat interface
    ├── Tool schema display
    └── Routing explanation
```

---

## 🔗 Connecting to Real MCP

In production, you would:

1. **Use an MCP Server**
   ```python
   from mcp import Server
   server = Server("my-server")
   
   @server.tool()
   def weather(location: str) -> str:
       # Real API call here
       pass
   ```

2. **Connect to MCP Clients**
   - Claude Desktop
   - VS Code extensions
   - Custom applications

3. **Add More Tools**
   - Database queries
   - File operations
   - API integrations

---

## 📖 Further Reading

- [MCP Specification](https://modelcontextprotocol.io)
- [Building MCP Servers](https://modelcontextprotocol.io/docs/building-servers)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

---

## 🎓 Course Integration

This demo is part of the **GenAI Applications** course curriculum:

1. **Week 1-2**: Vision-Language Models (see `01_camera_to_text.py`)
2. **Week 3-4**: Agentic AI (see `05_agentic_visual_analysis.py`)
3. **Week 5-6**: MCP & Tool Use (this tutorial)
4. **Week 7-8**: Full Stack GenAI Applications

---

*Created for educational purposes - January 2026*
