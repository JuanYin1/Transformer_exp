# Mini-Agent System Architecture: Complete Flow Diagram

## System Overview Flowchart

```mermaid
flowchart TD
    A[User Input via CLI/Zed Editor] --> B{Check Token Count}
    B -->|Approaching Limit 80k| C[Auto-Summarization Process]
    B -->|Within Limits| D[Load Persistent Memory]
    
    C --> C1[Compress Older History]
    C1 --> C2[Preserve System Prompt & Recent Messages]
    C2 --> D
    
    D --> D1[Load .agent_memory.json via Session Note Tool]
    D1 --> E[Assemble Complete Context]
    
    E --> F[Send to MiniMax API M2/M2.5]
    F --> G[Interleaved Thinking & Reasoning Block]
    
    G --> H{Model Decision}
    H -->|Need More Info/Action| I[Generate Tool Call]
    H -->|Task Complete| P[Final Synthesis]
    H -->|Error/Cannot Proceed| Q[Error Handling & Retries]
    
    I --> J[Route to Tool Handler]
    J --> K{Tool Type}
    
    K -->|File Operations| L[File System Tools]
    K -->|Shell Commands| M[Bash/PowerShell Execution]
    K -->|Memory Operations| N[Persistent Memory Tools]
    K -->|Skills| O[Get Skill - Dynamic Context Injection]
    K -->|MCP| O1[Model Context Protocol Server]
    
    L --> L1[read_file/write_file/edit_file]
    M --> M1[Execute in Local Shell]
    N --> N1[record_note/recall_notes]
    O --> O1[Load Specialized Prompts & Workflows]
    O1 --> O2[External Servers: GitHub/Web/Knowledge Graphs]
    
    L1 --> R[Tool Result/Output]
    M1 --> R
    N1 --> R
    O2 --> R
    
    R --> S[Append to Message History as tool_result]
    S --> T{Check Loop Constraints}
    T -->|Steps < max_steps| G
    T -->|Steps >= max_steps| U[Force Termination]
    
    Q --> Q1{Retry Attempts}
    Q1 -->|Can Retry| G
    Q1 -->|Max Retries| U
    
    P --> V[Stream Response to User Interface]
    U --> V
    
    V --> W[End Session/Wait for Next Input]
    
    style A fill:#e1f5fe
    style V fill:#c8e6c9
    style G fill:#fff3e0
    style H fill:#fce4ec
    style U fill:#ffebee
```

## Detailed Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant CLI as CLI/Zed Editor
    participant Agent as Mini-Agent Core
    participant Context as Context Manager
    participant Memory as Persistent Memory
    participant API as MiniMax API
    participant Tools as Tool Handlers
    participant FS as File System
    participant Shell as Bash/PowerShell
    participant MCP as MCP Servers
    
    User->>CLI: Submit prompt/task
    CLI->>Agent: Forward user input
    
    Agent->>Context: Check token count
    alt Token limit approaching (30% threshold)
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
            Note over Tools: Tool type determined and executed
            Tools->>FS: File operations (if needed)
            FS-->>Tools: File contents/status
            Tools->>Shell: Shell commands (if needed)
            Shell-->>Tools: stdout/stderr/status
            Tools->>Memory: Memory operations (if needed)
            Memory-->>Tools: Memory data
            Tools->>MCP: MCP integration (if needed)
            MCP-->>Tools: External data
            Tools-->>Agent: Tool result
            Agent->>Agent: Append tool_result to history
            Agent->>API: Continue with updated context
        else Task complete
            Note over Agent: Task successfully completed
        else Error occurred
            Agent->>Agent: Retry logic with exponential backoff
            Note over Agent: Check if max retries exceeded
        end
    end
    
    API-->>Agent: Final response
    Agent->>CLI: Stream response
    CLI->>User: Display result
```

## Key System Components Detail

### 1. Context Management System
```mermaid
graph LR
    A[Raw Context] --> B{Token Counter}
    B -->|< 80k tokens| C[Direct Processing]
    B -->|≥ 80k tokens| D[Summarization Engine]
    D --> E[Compressed Context]
    E --> F[Merge with Recent Messages]
    F --> G[Final Context Package]
    C --> G
```

### 2. Tool Execution Pipeline
```mermaid
graph TD
    A[Tool Call Generated] --> B[Tool Router]
    B --> C{Tool Type Classification}
    
    C -->|File| D[File System Handler]
    C -->|Bash| E[Shell Executor]
    C -->|Memory| F[Session Manager]
    C -->|Skill| G[Dynamic Context Loader]
    C -->|MCP| H[External Protocol Handler]
    
    D --> I[Workspace Scoped Operations]
    E --> J[Local Command Execution]
    F --> K[Persistent Storage I/O]
    G --> L[Specialized Prompt Injection]
    H --> M[External Server Communication]
    
    I --> N[Result Aggregation]
    J --> N
    K --> N
    L --> N
    M --> N
    
    N --> O[Tool Result Package]
    O --> P[Append to Message History]
```

### 3. Memory & State Management
```mermaid
stateDiagram-v2
    [*] --> SessionStart
    SessionStart --> LoadMemory
    LoadMemory --> .agent_memory.json
    .agent_memory.json --> ContextAssembly
    
    ContextAssembly --> Reasoning
    Reasoning --> ToolExecution
    ToolExecution --> UpdateMemory
    UpdateMemory --> record_note
    record_note --> Reasoning
    
    Reasoning --> TaskComplete
    TaskComplete --> SaveSession
    SaveSession --> [*]
    
    ToolExecution --> ErrorHandling
    ErrorHandling --> Reasoning
    ErrorHandling --> SessionEnd
    SessionEnd --> [*]
```

## Architecture Bottlenecks & Risk Points

```mermaid
graph TD
    A[System Risks] --> B[Context Summarization Loss]
    A --> C[Local Execution Security]
    A --> D[Model Dependency]
    A --> E[Debugging Limitations]
    
    B --> B1[Critical Details Lost in Compression]
    B --> B2[Variable Names & Microscopic Context]
    
    C --> C1[No Default Sandboxing]
    C --> C2[AI Hallucination Risk - rm -rf]
    
    D --> D1[MiniMax M2/M2.5 Optimized]
    D --> D2[Smaller Models Break Reasoning Loop]
    
    E --> E1[M2 Less Powerful at Code Debugging]
    E --> E2[vs Claude 3.5 Sonnet Performance Gap]
    
    style B fill:#ffebee
    style C fill:#ffebee
    style D fill:#fff3e0
    style E fill:#fff3e0
```