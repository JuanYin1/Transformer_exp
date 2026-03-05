#!/usr/bin/env python3
"""
Reasoning-Focused Test Cases for Mini-Agent
==========================================
Enhanced test cases specifically designed to encourage explicit reasoning steps
and improve the reasoning quality scores.
"""

from advanced_evaluation_system import EvaluationCriteria
from integrated_mini_agent_evaluation import TestCaseDefinition

def create_reasoning_enhanced_test_suite():
    """Create test cases with explicit reasoning prompts to improve step-by-step thinking"""
    
    reasoning_tests = []
    
    # Enhanced Arithmetic with Explicit Step Requirements
    reasoning_tests.append(TestCaseDefinition(
        name="explicit_arithmetic_reasoning",
        category="reasoning",
        complexity="simple",
        description="Arithmetic with mandatory step-by-step breakdown",
        task_prompt="""Calculate 18% of 350, then subtract 25, then multiply by 2.

IMPORTANT: You MUST show your work using this exact format:
Step 1: [describe what you're doing]
Step 2: [describe what you're doing] 
Step 3: [describe what you're doing]
Final Answer: [result]

Do not just give the final answer - show each step clearly.""",
        evaluation_criteria=EvaluationCriteria(
            evaluation_type="hybrid",
            expected_keywords=["step 1", "step 2", "step 3", "18%", "350", "63", "38", "76"],
            correctness_weight=0.3,    # Lower weight on correctness
            reasoning_weight=0.5,      # Higher weight on reasoning process
            completeness_weight=0.2,
            efficiency_weight=0.0,
            execution_weight=0.0
        ),
        max_time_seconds=30
    ))
    
    # Problem-Solving with Required Analysis
    reasoning_tests.append(TestCaseDefinition(
        name="structured_problem_solving",
        category="reasoning", 
        complexity="medium",
        description="Problem-solving with required analytical structure",
        task_prompt="""A company has 120 employees. 40% work remotely, 35% work in the office full-time, and the rest work hybrid. 
If remote workers get a $100/month internet stipend and hybrid workers get $50/month, how much does the company spend monthly on internet stipends?

Use this required structure:
1. UNDERSTAND: Restate the problem in your own words
2. IDENTIFY: What information do we have?
3. CALCULATE: Show each calculation step
4. VERIFY: Check your answer makes sense
5. CONCLUDE: State the final answer clearly

Follow this structure exactly.""",
        evaluation_criteria=EvaluationCriteria(
            evaluation_type="hybrid",
            expected_keywords=["understand", "identify", "calculate", "verify", "conclude", "120", "40%", "35%", "25%", "4800"],
            correctness_weight=0.25,
            reasoning_weight=0.55,     # Very high weight on reasoning structure
            completeness_weight=0.2,
            efficiency_weight=0.0,
            execution_weight=0.0
        ),
        max_time_seconds=60
    ))
    
    # Code Analysis with Mandatory Explanation
    reasoning_tests.append(TestCaseDefinition(
        name="explained_code_analysis",
        category="programming",
        complexity="medium",
        description="Code analysis requiring explicit reasoning for each finding",
        task_prompt="""Analyze this Python code and find ALL bugs. For EACH bug you find, use this format:

BUG #1:
- Location: [where in the code]
- Problem: [what's wrong]  
- Why it's wrong: [explain the issue]
- How to fix: [specific solution]
- Test case that breaks it: [example]

```python
def process_data(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
        elif item = 0:
            result.append(0)
        else:
            result.append(item / 0)
    return result

data = [1, 2, 0, -1, 3]
print(process_data(data))
```

Find ALL bugs and explain each one using the format above.""",
        evaluation_criteria=EvaluationCriteria(
            evaluation_type="hybrid",
            expected_keywords=["bug", "location", "problem", "why", "how to fix", "test case", "assignment", "division by zero"],
            correctness_weight=0.3,
            reasoning_weight=0.5,      # High reasoning weight
            completeness_weight=0.2,
            efficiency_weight=0.0,
            execution_weight=0.0
        ),
        max_time_seconds=90
    ))
    
    # System Design with Required Justification
    reasoning_tests.append(TestCaseDefinition(
        name="justified_system_design",
        category="analysis",
        complexity="complex",
        description="System design requiring justification for each decision", 
        task_prompt="""Design a scalable chat application architecture.

For EACH component you propose, use this format:
COMPONENT: [name]
PURPOSE: [what it does]
WHY NEEDED: [justification]
ALTERNATIVES CONSIDERED: [other options]
TRADE-OFFS: [pros and cons]
SCALING STRATEGY: [how it scales]

Required components to address:
1. User authentication
2. Message storage
3. Real-time messaging
4. File sharing
5. Load balancing

Use the format above for each component.""",
        evaluation_criteria=EvaluationCriteria(
            evaluation_type="hybrid",
            expected_keywords=["component", "purpose", "why needed", "alternatives", "trade-offs", "scaling", "authentication", "storage", "real-time", "load"],
            correctness_weight=0.2,
            reasoning_weight=0.6,      # Very high reasoning weight
            completeness_weight=0.2,
            efficiency_weight=0.0,
            execution_weight=0.0
        ),
        max_time_seconds=180
    ))
    
    return reasoning_tests

def create_system_stress_tests():
    """Create tests specifically designed to stress different system components"""
    
    stress_tests = []
    
    # LLM Call Intensity Test
    stress_tests.append(TestCaseDefinition(
        name="llm_intensive_task",
        category="performance",
        complexity="complex",
        description="Task requiring multiple LLM inference rounds",
        task_prompt="""Write a comprehensive analysis of the pros and cons of 5 different programming languages (Python, JavaScript, Rust, Go, Java) for web backend development.

For EACH language, provide:
1. 3 major strengths with examples
2. 3 major weaknesses with examples  
3. Best use cases
4. Performance characteristics
5. Learning curve assessment
6. Community/ecosystem evaluation

Then provide an overall comparison and recommendation matrix.

This should be thorough and detailed.""",
        evaluation_criteria=EvaluationCriteria(
            evaluation_type="hybrid",
            expected_keywords=["python", "javascript", "rust", "go", "java", "strengths", "weaknesses", "performance", "learning"],
            correctness_weight=0.3,
            reasoning_weight=0.4,
            completeness_weight=0.3,
            efficiency_weight=0.0,
            execution_weight=0.0
        ),
        max_time_seconds=300  # Longer timeout for complex task
    ))
    
    # Tool Execution Intensive Test  
    stress_tests.append(TestCaseDefinition(
        name="tool_intensive_operations",
        category="performance",
        complexity="complex", 
        description="Task requiring heavy file operations and bash commands",
        task_prompt="""Create a data processing pipeline:

1. Create 5 CSV files with sample data (employees_1.csv to employees_5.csv)
2. Each file should have: Name, Department, Salary, Years_Experience (10 rows each)
3. Write a Python script to merge all files into 'combined_data.csv'
4. Use the script to calculate department-wise statistics
5. Generate a summary report 'analysis_report.txt' 
6. Create a bash script that runs the entire pipeline
7. Execute the bash script and show all results
8. Verify all files were created correctly

Show the contents of all created files.""",
        evaluation_criteria=EvaluationCriteria(
            evaluation_type="execution",
            execution_tests=[
                lambda response: "employees_1.csv" in response.lower(),
                lambda response: "combined_data.csv" in response.lower(), 
                lambda response: "analysis_report.txt" in response.lower(),
                lambda response: "python" in response.lower(),
                lambda response: "bash" in response.lower(),
                lambda response: response.count(".csv") >= 6  # At least 6 CSV file references
            ],
            correctness_weight=0.2,
            completeness_weight=0.3,
            execution_weight=0.4,      # High execution weight
            reasoning_weight=0.1,
            efficiency_weight=0.0
        ),
        max_time_seconds=180
    ))
    
    # Memory/Context Intensive Test
    stress_tests.append(TestCaseDefinition(
        name="context_intensive_analysis", 
        category="performance",
        complexity="complex",
        description="Task requiring large context and memory usage",
        task_prompt="""Analyze this large dataset description and create a comprehensive data processing plan:

DATASET INFO:
- Customer table: 2M records (ID, Name, Email, SignupDate, Plan, Status)
- Orders table: 15M records (OrderID, CustomerID, Amount, Date, Products)  
- Products table: 50K records (ProductID, Name, Category, Price, Stock)
- Reviews table: 8M records (ReviewID, CustomerID, ProductID, Rating, Text)
- Logs table: 100M records (Timestamp, UserID, Action, Details)

REQUIREMENTS:
1. Design ETL pipeline for this data
2. Identify 10 key business metrics to calculate
3. Propose data warehouse schema
4. Design real-time analytics architecture
5. Create data quality monitoring strategy
6. Estimate infrastructure requirements
7. Plan for data backup and disaster recovery
8. Design user access control system
9. Create data lineage tracking
10. Propose cost optimization strategies

Provide detailed analysis for each requirement.""",
        evaluation_criteria=EvaluationCriteria(
            evaluation_type="hybrid",
            expected_keywords=["etl", "pipeline", "metrics", "warehouse", "real-time", "quality", "infrastructure", "backup", "access", "lineage"],
            correctness_weight=0.3,
            reasoning_weight=0.4,
            completeness_weight=0.3,
            efficiency_weight=0.0,
            execution_weight=0.0
        ),
        max_time_seconds=300
    ))
    
    return stress_tests

if __name__ == "__main__":
    reasoning_tests = create_reasoning_enhanced_test_suite()
    stress_tests = create_system_stress_tests()
    
    print("🧠 Reasoning-Enhanced Test Suite:")
    for i, test in enumerate(reasoning_tests, 1):
        print(f"{i}. {test.name} ({test.complexity}) - {test.description}")
    
    print(f"\n⚡ System Stress Test Suite:")
    for i, test in enumerate(stress_tests, 1):
        print(f"{i}. {test.name} ({test.complexity}) - {test.description}")
    
    print(f"\n📊 Total: {len(reasoning_tests + stress_tests)} additional tests")
    print("These tests focus on explicit reasoning and system performance stress testing.")