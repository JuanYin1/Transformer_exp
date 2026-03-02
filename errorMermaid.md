```mermaid
sequenceDiagram
    participant User
    participant CLI/Editor as CLI/Zed Editor
    participant Agent as Mini-Agent Core
    participant Context as Context Manager
    participant Memory as Persistent Memory
    participant API as MiniMax API
    participant Tools as Tool Handlers
    participant FS as File System
    participant Shell as Bash/PowerShell
    participant MCP as MCP Servers
    
    User->>CLI/Editor: Submit prompt/task
    CLI/Editor->>Agent: Forward user input
    
    Agent->>Context: Check token count
    alt Token limit approaching (80k)
        Context->>Context: Auto-summarize older history
        Context->>Context: Preserve system prompt + recent messages
    end
    
    Agent->>Memory: Load session memory
    Memory-->>Agent: Return .agent_memory.json data
    
    Agent->>Context: Assemble complete context
    Context-->>Agent: Complete context package
    
    Agent->>API: Send context to MiniMax M2/M2.5
    
    loop Execution Loop (max_steps = 100)
        API->>API: Interleaved thinking & reasoning
        API-->>Agent: Decision + potential tool calls
        
        alt Need to call tools
            Agent->>Tools: Route tool call
            
            alt File operations
                Tools->>FS: Execute file operations
                FS-->>Tools: File contents/status
            else Shell commands  
                Tools->>Shell: Execute bash/PowerShell
                Shell-->>Tools: stdout/stderr/status
            else Memory operations
                Tools->>Memory: record_note/recall_notes
                Memory-->>Tools: Memory data
            else Skill activation
                Tools->>Tools: Load specialized prompts
                Tools-->>Agent: Enhanced context
            else MCP integration
                Tools->>MCP: External server calls
                MCP-->>Tools: External data
            end
            
            Tools-->>Agent: Tool result
            Agent->>Agent: Append tool_result to history
            Agent->>API: Continue with updated context
            
        else Task complete
            break
        else Error occurred
            Agent->>Agent: Retry logic with exponential backoff
            alt Max retries exceeded
                break
            end
        end
    end
    
    API-->>Agent: Final response
    Agent->>CLI/Editor: Stream response
    CLI/Editor->>User: Display result
```