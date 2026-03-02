# Repository Research Methodology for Agentic Systems

## Step-by-Step Exploration Process

### Phase 1: Repository Overview (5-10 minutes)
1. **README.md first** - Get high-level understanding
2. **Repository structure** - Understand organization
3. **Package.json/requirements.txt** - Identify dependencies
4. **Configuration files** - Find default settings

### Phase 2: Entry Points Discovery (10-15 minutes)
1. **Main entry points**: 
   - `main.py`, `app.py`, `cli.py`
   - `__init__.py` files in root directories
   - Setup.py or package configuration

2. **Command-line interfaces**:
   - Look for CLI argument parsing
   - Entry point definitions
   - Usage examples

### Phase 3: Core Architecture Identification (20-30 minutes)

#### For Mini-Agent specifically, I found:
```
Key Files to Examine in Order:
1. mini_agent/cli.py        # Entry point - HOW users interact
2. mini_agent/agent.py      # Core logic - THE MAIN LOOP  
3. mini_agent/config.py     # Configuration - ACTUAL DEFAULTS
4. mini_agent/schema.py     # Data structures - MESSAGE FORMATS
5. mini_agent/llm.py        # LLM integration - API CALLS
6. tools/ directory         # Tool system - CAPABILITIES
```

#### Critical Code Patterns to Look For:

**1. Main Execution Loop:**
```python
# In agent.py - THE CORE LOOP
async def run(self, user_input: str):
    while not done and step_count < self.config.max_steps:
        # LLM reasoning
        response = await self.llm.generate(messages)
        
        # Tool execution
        if response.tool_calls:
            results = await self.execute_tools(response.tool_calls)
            messages.append(tool_results)
        else:
            done = True
```

**2. Context Management:**
```python
# In agent.py - MEMORY HANDLING
def manage_context(self, messages):
    if self.token_count > self.config.context_threshold:
        summarized = self.summarize_history(messages)
        return summarized
```

**3. Tool Integration:**
```python
# In tools/base.py - TOOL ARCHITECTURE  
class Tool(ABC):
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        pass
```

### Phase 4: Configuration Deep-Dive (10-15 minutes)

#### Real Configuration Analysis:
```yaml
# config/config-example.yaml - ACTUAL DEFAULTS
max_steps: 100                    # NOT "often 100" - it IS 100
model: "MiniMax-M2.5"            # Specific model version
context_threshold: 0.3           # 30% not "80k tokens"
workspace_dir: "./workspace"     # Sandboxed execution
retry:
  max_retries: 3                 # Built-in retry logic
  initial_delay: 1.0
```

### Phase 5: Tool/Skill System Analysis (15-20 minutes)

#### I found the actual implementation:
```python
# tools/skill_loader.py - SKILLS ARE GIT SUBMODULE
def load_skills():
    skills_dir = "./skills"  # External git submodule
    return discover_skills(skills_dir)

# tools/mcp_loader.py - MCP INTEGRATION  
def load_mcp_tools(config_path):
    mcp_config = json.load(config_path)
    return initialize_mcp_clients(mcp_config)
```

### Phase 6: Memory System Investigation (10-15 minutes)

#### Real Memory Implementation:
```python
# tools/basic/note.py - SESSION MEMORY
class SessionNoteTool:
    def record_note(self, content: str):
        # Persistent storage across sessions
        
    def recall_notes(self) -> List[str]:
        # Cross-session memory retrieval
```

## Key Differences: Documentation vs Implementation

### What Documentation Often Misses:
1. **Actual default values** (claims "often X" instead of "IS X")
2. **Implementation complexity** (simplifies sophisticated systems)
3. **Error handling mechanisms** (focuses on happy path)
4. **Performance considerations** (ignores optimization details)
5. **Security boundaries** (workspace sandboxing, command restrictions)

### What Code Reveals:
1. **Precise configuration values**
2. **Real error handling strategies**
3. **Performance optimizations** (context summarization triggers)
4. **Security measures** (workspace isolation)
5. **Extensibility patterns** (tool loading, MCP integration)

## Red Flags in Documentation vs Reality:

❌ **Vague language**: "often", "typically", "around"
✅ **Specific values**: "max_steps: 100", "threshold: 0.3"

❌ **Conceptual descriptions**: "asynchronous event-driven"  
✅ **Implementation reality**: "synchronous orchestration of async operations"

❌ **Marketing claims**: "infinitely long context"
✅ **Technical truth**: "automatic summarization at 30% threshold"

## Tools for Systematic Code Analysis:

### Search Patterns to Use:
```bash
# Find configuration defaults
grep -r "max_steps\|token.*limit\|context.*window" .

# Find main loops
grep -r "while.*not.*done\|for.*in.*range.*steps" .

# Find tool implementations  
find . -name "*tool*" -o -name "*skill*" 

# Find memory/persistence code
grep -r "save\|load\|persist\|memory\|session" .

# Find error handling
grep -r "try.*except\|error.*handling\|retry" .
```

### Critical Questions to Answer:
1. **How does the system actually start?** (entry points)
2. **What's the real execution loop?** (core agent logic)  
3. **How are tools really loaded?** (plugin architecture)
4. **What are the actual limits?** (not examples, real config)
5. **How does memory actually work?** (persistence mechanisms)
6. **What security exists?** (sandboxing, validation)

This methodology ensures you understand the **real architecture**, not just the marketing description.