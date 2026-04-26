# MCP
#### Example usage
As client to send request: 
create a session -> initilize the session -> check list tools -> use tool by call tool name -> more info by call read_resource / get_prompt

As server provide mcp:
import FastMCP -> define the tool by @mcp.tool() -> define the resource by @mcp.resource("url") -> design the prompts by @mcp.prompt()

Abstract MCP server should use with the robust Agent workflow


```json
// 1. User Request
  "User: Read the file config.yaml"

  // 2. LLM Tool Call (JSON-RPC format)
  {
    "method": "tools/call",
    "params": {
      "name": "read_file",
      "arguments": {"file_path": "config.yaml"}
    }
  }

  // 3. MCP Server Response  
  {
    "result": {
      "content": [{"type": "text", "text": "api_key: abc123\nmodel: claude-3"}],
      "isError": false
    }
  }

  // 4. Back to LLM
  "Here's the content of config.yaml: api_key: abc123..."
```

### Key Technical Details to Highlight - refer to MiniAgent mcp_loader.py
#### Connection Types:
  - STDIO: Launch subprocess, communicate via stdin/stdout
  - HTTP/SSE: Connect to web-based MCP servers
  - Streamable HTTP: For real-time streaming responses

#### Timeout Architecture:
  connect_timeout: 10s    # Server connection  
  execute_timeout: 60s    # Tool execution
  sse_read_timeout: 120s  # Streaming reads

#### Error Handling:
```python
  try:
      result = await session.call_tool(name, arguments)
  except TimeoutError:
      return ToolResult(success=False, error="Server timeout")
  except Exception as e:
      return ToolResult(success=False, error=f"Tool failed: {e}")
```

### related questions
  Q: "How is this different from OpenAI function calling?"
  A: "OpenAI function calling is LLM-side only - it generates function calls but can't execute them. MCP provides the execution layer plus
  standardized server discovery. It's like the difference between generating SQL vs. actually querying a database."

  Q: "What if an MCP server is slow or crashes?"A: "We implement three-layer timeout protection and graceful degradation. If a server fails,
  we return a structured error to the LLM so it can adapt or try alternatives."

  Q: "How does this scale with many tools?"
  A: "Tools are discovered dynamically, so adding new capabilities is just configuration. The agent doesn't need code changes. Plus, each MCP
  server runs independently, so they don't interfere with each other."

  ✅ MCP = Universal adapter for LLM capabilities
  ✅ Three components: Client (loader), Server (tools), Protocol (JSON-RPC)
  ✅ Dynamic discovery vs. hardcoded integration
  ✅ Timeout protection at connection, execution, and streaming levels
  ✅ Configuration-driven tool loading (mcp.json)
  ✅ Structured error handling and graceful degradation
  ✅ Multiple connection types (STDIO, HTTP, SSE)


# Skills
Fundamental Architecture Differences

  | Aspect       | Skill Loader                       | MCP Loader                    |
  |--------------|------------------------------------|-------------------------------|
  | Purpose      | Load static documentation/guidance | Connect to external services  |
  | Data Source  | Local markdown files (SKILL.md)    | Remote MCP servers            |
  | Content Type | Text-based instructions/knowledge  | Executable functions          |
  | Connection   | File system reading                | Network/process communication |
  | Execution    | Returns documentation              | Executes actual operations    |

A. Data Loading Pattern

```python
  Skill Loader (skill_loader.py:60-117):
  # Loads static files from filesystem
  def load_skill(self, skill_path: Path) -> Optional[Skill]:
      content = skill_path.read_text(encoding="utf-8")  # Read markdown file
      frontmatter_match = re.match(r"^---\n(.*?)\n---\n(.*)$", content)
      return Skill(name=..., description=..., content=processed_content)

  MCP Loader (mcp_loader.py:194-208):
  # Connects to live servers and discovers capabilities
  async def connect(self) -> bool:
      session = await ClientSession(read_stream, write_stream)
      tools_list = await session.list_tools()  # Ask server what it can do
      return MCPTool(name=tool.name, session=session)  # Wrap for execution

  B. Tool Behavior

  Skill Tool (skill_tool.py:40-54):
  # Returns documentation text
  async def execute(self, skill_name: str) -> ToolResult:
      skill = self.skill_loader.get_skill(skill_name)
      result = skill.to_prompt()  # Convert to text instruction
      return ToolResult(success=True, content=result)

  MCP Tool (mcp_loader.py:89-96):
  # Executes actual operations
  async def execute(self, **kwargs) -> ToolResult:
      result = await self._session.call_tool(self._name, arguments=kwargs)
      return ToolResult(success=not result.isError, content=result.content)
```

#### The Skill Loader implements a clever Progressive Disclosure pattern:
```python
# Level 1 - Metadata Only (skill_loader.py:237-256):

  def get_skills_metadata_prompt(self) -> str:
      # Shows only skill names + descriptions in system prompt
      for skill in self.loaded_skills.values():
          prompt_parts.append(f"- `{skill.name}`: {skill.description}")

# Level 2 - Full Content On-Demand (skill_tool.py:40-54):

  # LLM can request full skill content when needed
  async def execute(self, skill_name: str) -> ToolResult:
      return ToolResult(content=skill.to_prompt())  # Full documentation

# Level 3 - Resource References (skill_loader.py:119-192):

  def _process_skill_paths(self, content: str, skill_dir: Path) -> str:
      # Converts relative paths to absolute paths
      # "see reference.md" → "see `/full/path/reference.md` (use read_file to access)"
```


# A/B Test
ab test refer to the online/offline test that to test if the LLM or the agent is good enough to deploy. refer to this link: https://mlsavvy.substack.com/p/ab-testing-for-machine-learning-models
