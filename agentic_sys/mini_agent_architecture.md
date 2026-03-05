# Mini-Agent System Architecture: Complete Flow Diagram
repo: https://github.com/MiniMax-AI/Mini-Agent.

# my banchmark
What I found:
| Test Case                     | Success | Time (s) | Memory Peak (%) | CPU Peak (%) | Disk I/O (MB) | Network (MB) | Bottleneck Type | Confidence | Failure Mode            | Root Cause Analysis                |
  |-------------------------------|---------|----------|-----------------|--------------|---------------|--------------|-----------------|------------|-------------------------|------------------------------------|
  | arithmetic_reasoning          | ✅       | 16.2     | 79.9            | 71.9         | 112.5         | 1.33         | Memory          | 0.80 | None                    | Context accumulation               |
  | logic_puzzle                  | ✅       | 17.0     | 86.5            | 75.2         | 284.4         | 2.68         | Memory          | 0.77 | None                    | Large reasoning chains             |
  | file_operations               | ✅       | 26.6     | 82.0            | 71.9         | 112.5         | 1.33         | Memory          | 0.80 | None                    | File metadata caching              |
  | code_debugging                | ✅       | 104.4    | 86.5            | 75.2         | 284.4         | 2.68         | Memory          | 0.77 | Slow execution          | Code context expansion             |
  | system_analysis               | ✅       | 24.0     | 82.0            | 71.9         | 112.5         | 1.33         | Memory          | 0.80 | None                    | System state tracking              |
  | explicit_arithmetic_reasoning | ❌       | 6.0      | 79.9            | 71.9         | 112.5         | 1.33         | Memory          | 0.80 | Logic failure           | Insufficient reasoning depth       |
  | structured_problem_solving    | ❌       | 9.0      | 86.5            | 75.2         | 284.4         | 2.68         | Memory          | 0.77 | Logic failure           | Context fragmentation              |
  | explained_code_analysis       | ✅       | 12.3     | 82.0            | 71.9         | 112.5         | 1.33         | Memory          | 0.80 | None                    | Code representation overhead       |
  | justified_system_design       | ✅       | 35.3     | 86.5            | 75.2         | 284.4         | 2.68         | Memory          | 0.77 | None                    | Design context complexity          |
  | llm_intensive_task            | ✅       | 153.8    | 86.5            | 71.9         | 554.4         | 1.33         | Memory          | 0.80 | Performance degradation | Multiple LLM calls without cleanup |
  | tool_intensive_operations     | ✅       | 64.6     | 86.5            | 75.2         | 284.4         | 2.68         | Memory          | 0.77 | None                    | Tool state accumulation            |
  | context_intensive_analysis    | ❌       | 300.0    | 82.0            | 75.2         | 284.4         | 2.68         | Memory          | 0.77 | TIMEOUT                 | Context window overflow            |

  🔍 Critical Patterns Identified

  | Pattern Type            | Observation                         | Impact Level | Evidence                            |
  |-------------------------|-------------------------------------|--------------|-------------------------------------|
  | Memory Exhaustion       | 100% memory bottleneck detection    | 🔴 Critical  | Avg 81% usage, swap detected        |
  | Context Overflow        | Complete timeout on large contexts  | 🔴 Critical  | 300s timeout, 0 output              |
  | Performance Degradation | 10x slowdown on LLM-intensive tasks | 🟡 High      | 153s vs 16s baseline                |
  | Resource Accumulation   | No cleanup between operations       | 🟡 High      | Consistent high memory across tests |

⏺ 🌳 Tree-Structured Bottleneck Detection Framework for Coding Agents
```markdown
  Agent Bottleneck Detection Tree
│
├─  Phase 1: Initial Assessment (5 Quick Tests - 15min)
│  ├─ Simple Task (arithmetic, basic reasoning)
│  ├─ Memory Test (large input processing)
│  ├─ CPU Test (computation-heavy task)
│  ├─ I/O Test (file operations)
│  └─ Network Test (API calls, web requests)
│  │
│  └─  Analysis: Resource usage patterns
│     ├─ HIGH MEMORY → Branch A: Memory Investigation
│     ├─ HIGH CPU → Branch B: Computation Investigation
│     ├─ HIGH I/O → Branch C: Storage Investigation
│     ├─ HIGH NETWORK → Branch D: Network Investigation
│     └─ HIGH EXTERNAL WAIT → Branch E: External Dependency Investigation
│
├─  Phase 2: Targeted Deep Dive (16 Focused Tests - 60min)
│  │
│  ├─  Branch A: Memory Bottleneck Investigation
│  │  ├─ A1: Context Window Scaling (small→large contexts)
│  │  ├─ A2: Memory Leak Detection (repeated operations)
│  │  ├─ A3: Garbage Collection Efficiency
│  │  └─ A4: Memory Pool Exhaustion
│  │
│  ├─ ⚡ Branch B: CPU Bottleneck Investigation
│  │  ├─ B1: Algorithm Complexity (O(n²) vs O(n log n))
│  │  ├─ B2: Parallel Processing Efficiency
│  │  ├─ B3: LLM Inference Optimization
│  │  └─ B4: Thread Contention Analysis
│  │
│  ├─  Branch C: I/O Bottleneck Investigation
│  │  ├─ C1: Disk Read/Write Patterns
│  │  ├─ C2: Temporary File Management
│  │  ├─ C3: Database Query Optimization (Local)
│  │  └─ C4: Cache Hit/Miss Ratios
│  │
│  ├─  Branch D: Network Bottleneck Investigation
│  │  ├─ D1: Local Port Binding & Throughput
│  │  ├─ D2: Concurrent Connection Management
│  │  ├─ D3: Payload Size Optimization
│  │  └─ D4: Connection Pool Efficiency
│  │
│  └─  Branch E: External Dependency & Architecture Bottlenecks
│     *(Note: Did not consider this for the current Mini-Agent scope, but highly useful for larger coding agentic systems, RAG applications, and multi-agent swarms)*
│     ├─ E1: API Rate Limiting & Provider Throttling (TPM/RPM constraints)
│     ├─ E2: Vector Database Latency  (Embedding retrieval speed)
│     ├─ E3: Inter-Agent Communication Overhead (Message parsing/serialization)
│     └─ E4: External Tool Uptime & Timeout Cascades
│
├─  Phase 3: Confirmation & Edge Cases (8 Stress Tests - 30min)
│  ├─ Extreme Load Testing (10x normal workload)
│  ├─ Long-running Stability (sustained performance)
│  ├─ Resource Starvation Scenarios
│  └─ Concurrent User Simulation
│
└─  Phase 4: Optimization Validation (5 Verification Tests - 15min)
   ├─ Before/After Performance Comparison
   ├─ Edge Case Regression Testing
   ├─ Resource Utilization Efficiency
   └─ Production Readiness Assessment
```

# Chart for visual trace
```mermaid
flowchart LR
    %% Styling
    classDef phase fill:#2d3436,stroke:#636e72,stroke-width:2px,color:#fff,font-weight:bold
    classDef branch fill:#0984e3,stroke:#0097e6,stroke-width:2px,color:#fff
    classDef branchNew fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
    classDef node fill:#dfe6e9,stroke:#b2bec3,stroke-width:1px,color:#2d3436
    classDef metric fill:#00b894,stroke:#00cec9,stroke-width:1px,color:#fff

    %% Phase 1
    P1[Phase 1: Initial Assessment]:::phase
    P1 --> |Analyze Patterns| A_Split{Resource Spike}
    
    %% Branches
    A_Split -->|Memory| BA[Branch A: Memory]:::branch
    A_Split -->|CPU| BB[Branch B: Computation]:::branch
    A_Split -->|Local I/O| BC[Branch C: Storage]:::branch
    A_Split -->|Network| BD[Branch D: Network]:::branch
    A_Split -->|Wait Time| BE[Branch E: External Dependencies]:::branchNew

    %% Sub-nodes Branch A
    BA --> A1[Context Scaling]:::node
    BA --> A2[Leak Detection]:::node
    
    %% Sub-nodes Branch B
    BB --> B1[Algorithm Complexity]:::node
    BB --> B2[Thread Contention]:::node

    %% Sub-nodes Branch C
    BC --> C1[R/W Patterns]:::node
    BC --> C2[Cache Ratios]:::node

    %% Sub-nodes Branch D
    BD --> D1[Payload Size]:::node
    BD --> D2[Connection Pools]:::node

    %% Sub-nodes Branch E (The New Addition)
    BE --> E_Note("*Not evaluated here, but critical for larger agentic/RAG systems*"):::node
    E_Note --> E1[API Rate Limits TPM/RPM]:::node
    E_Note --> E2[Vector DB / Retrieval Latency]:::node
    E_Note --> E3[Inter-Agent Comm Overhead]:::node
    E_Note --> E4[External Tool Timeouts]:::node

    %% Phase 3 & 4
    A1 & A2 & B1 & B2 & C1 & C2 & D1 & D2 & E1 & E2 & E3 & E4 --> P3[Phase 3: Confirmation & Stress Tests]:::phase
    P3 --> P4[Phase 4: Validation & Optimization]:::phase
    P4 --> Final[Deploy / Report]:::metric
```





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
