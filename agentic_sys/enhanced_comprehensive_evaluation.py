#!/usr/bin/env python3
"""
Enhanced Comprehensive Mini-Agent Evaluation
===========================================
Integrates:
1. Reasoning-focused tests to improve step-by-step demonstration
2. Real-time system monitoring for bottleneck identification  
3. Expanded test suite for comprehensive performance analysis
4. Advanced multi-methodology evaluation
"""

import asyncio
import json
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

# Import our components
from advanced_evaluation_system import AdvancedEvaluator, EvaluationResult
from realtime_system_monitor import RealTimeSystemMonitor, PerformanceAnalysis
from reasoning_focused_tests import create_reasoning_enhanced_test_suite, create_system_stress_tests
from integrated_mini_agent_evaluation import TestCaseDefinition, ComprehensiveTestResult

@dataclass
class EnhancedTestResult:
    """Enhanced test result with system monitoring data"""
    # Original test result data
    test_case: TestCaseDefinition
    total_execution_time: float
    mini_agent_output: str
    mini_agent_error: Optional[str]
    execution_success: bool
    evaluation_result: EvaluationResult
    overall_success: bool
    confidence_score: float
    
    # Enhanced monitoring data
    performance_analysis: PerformanceAnalysis
    resource_bottleneck: str
    bottleneck_confidence: float
    
    # Additional metrics
    llm_inference_pattern: str  # "heavy", "moderate", "light"
    tool_usage_intensity: str   # "intensive", "moderate", "minimal"
    reasoning_quality_category: str  # "excellent", "good", "fair", "poor"

class EnhancedMiniAgentEvaluator:
    """Enhanced evaluator with reasoning focus and real-time monitoring"""
    
    def __init__(self, results_dir: str = "enhanced_evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.evaluator = AdvancedEvaluator(use_llm_judge=False)
        
    def create_enhanced_test_suite(self, include_stress_tests: bool = True) -> List[TestCaseDefinition]:
        """Create comprehensive test suite with reasoning and stress tests"""
        
        # Get original tests
        original_tests = self._get_original_tests()
        
        # Get reasoning-enhanced tests
        reasoning_tests = create_reasoning_enhanced_test_suite()
        
        # Get system stress tests
        stress_tests = create_system_stress_tests() if include_stress_tests else []
        
        all_tests = original_tests + reasoning_tests + stress_tests
        
        print(f"📊 Enhanced Test Suite Created:")
        print(f"   • Original tests: {len(original_tests)}")
        print(f"   • Reasoning-enhanced tests: {len(reasoning_tests)}")
        print(f"   • System stress tests: {len(stress_tests)}")
        print(f"   • Total: {len(all_tests)} tests")
        
        return all_tests
    
    def _get_original_tests(self) -> List[TestCaseDefinition]:
        """Get the original 5 baseline tests for comparison"""
        from integrated_mini_agent_evaluation import IntegratedMiniAgentEvaluator
        
        baseline_evaluator = IntegratedMiniAgentEvaluator()
        original_tests = baseline_evaluator.create_comprehensive_test_suite()
        return original_tests
    
    async def run_enhanced_evaluation(self, test_categories: Optional[List[str]] = None) -> List[EnhancedTestResult]:
        """Run enhanced evaluation with real-time monitoring"""
        
        print("🚀 Enhanced Mini-Agent Comprehensive Evaluation")
        print("=" * 80)
        print("🧠 Focus: Explicit reasoning + Real-time bottleneck analysis")
        print("📊 Monitoring: CPU, Memory, Disk, Network, Process metrics")
        print("🎯 Goal: Identify reasoning patterns and system bottlenecks")
        print("=" * 80)
        
        # Create enhanced test suite
        all_tests = self.create_enhanced_test_suite()
        
        # Filter tests if categories specified
        if test_categories:
            filtered_tests = []
            for test in all_tests:
                if any(category.lower() in test.category.lower() for category in test_categories):
                    filtered_tests.append(test)
            all_tests = filtered_tests
            print(f"🔍 Filtered to {len(all_tests)} tests matching categories: {test_categories}")
        
        results = []
        
        for i, test_case in enumerate(all_tests, 1):
            print(f"\n[{i}/{len(all_tests)}] {test_case.category.upper()}: {test_case.name}")
            print(f"📋 {test_case.description}")
            
            result = await self._run_enhanced_single_test(test_case)
            results.append(result)
            
            # Print immediate summary
            self._print_immediate_summary(result)
            
            # Save individual result
            await self._save_enhanced_result(result)
        
        # Generate comprehensive analysis
        await self._generate_enhanced_analysis(results)
        
        return results
    
    async def _run_enhanced_single_test(self, test_case: TestCaseDefinition) -> EnhancedTestResult:
        """Run single test with enhanced monitoring and analysis"""
        
        # Initialize real-time monitoring
        monitor = RealTimeSystemMonitor(sample_interval=0.2)  # High-frequency sampling
        
        with tempfile.TemporaryDirectory() as temp_workspace:
            
            # Start enhanced monitoring
            monitor.start_monitoring("python")
            start_time = time.time()
            
            mini_agent_output = ""
            mini_agent_error = None
            execution_success = False
            
            try:
                print(f"🔍 Starting real-time monitoring...")
                print(f"🚀 Executing: mini-agent --workspace {temp_workspace} --task")
                
                # Execute Mini-Agent
                process = subprocess.run([
                    "mini-agent",
                    "--workspace", temp_workspace,
                    "--task", test_case.task_prompt
                ], 
                capture_output=True,
                text=True,
                timeout=test_case.max_time_seconds
                )
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                mini_agent_output = process.stdout
                mini_agent_error = process.stderr if process.stderr else None
                execution_success = process.returncode == 0
                
            except subprocess.TimeoutExpired:
                end_time = time.time()
                execution_time = end_time - start_time
                mini_agent_error = f"Test timed out after {test_case.max_time_seconds} seconds"
                execution_success = False
                
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                mini_agent_error = str(e)
                execution_success = False
            
            # Stop monitoring and get analysis
            performance_analysis = monitor.stop_monitoring()
            
            print(f"📊 Execution: {execution_time:.1f}s | Monitoring: {performance_analysis.sample_count} samples")
            
            # Run advanced evaluation
            evaluation_result = self.evaluator.evaluate_response(
                task_prompt=test_case.task_prompt,
                agent_response=mini_agent_output,
                criteria=test_case.evaluation_criteria,
                execution_time=execution_time,
                workspace_path=temp_workspace
            )
            
            # Enhanced analysis
            enhanced_metrics = self._analyze_enhanced_metrics(
                mini_agent_output, performance_analysis, evaluation_result
            )
            
            # Create enhanced result
            result = EnhancedTestResult(
                test_case=test_case,
                total_execution_time=execution_time,
                mini_agent_output=mini_agent_output,
                mini_agent_error=mini_agent_error,
                execution_success=execution_success,
                evaluation_result=evaluation_result,
                overall_success=execution_success and evaluation_result.passed,
                confidence_score=evaluation_result.confidence,
                performance_analysis=performance_analysis,
                resource_bottleneck=performance_analysis.primary_bottleneck,
                bottleneck_confidence=performance_analysis.bottleneck_confidence,
                **enhanced_metrics
            )
            
            return result
    
    def _analyze_enhanced_metrics(self, output: str, perf_analysis: PerformanceAnalysis, eval_result: EvaluationResult) -> dict:
        """Analyze enhanced metrics for patterns"""
        
        # LLM Inference Pattern Analysis
        if perf_analysis.monitoring_duration > 60:  # Long duration
            llm_pattern = "heavy"
        elif perf_analysis.monitoring_duration > 30:
            llm_pattern = "moderate"
        else:
            llm_pattern = "light"
        
        # Tool Usage Intensity
        disk_network_total = perf_analysis.total_disk_read_mb + perf_analysis.total_disk_write_mb + perf_analysis.total_network_mb
        if disk_network_total > 10:  # >10MB of I/O
            tool_intensity = "intensive"
        elif disk_network_total > 1:
            tool_intensity = "moderate"
        else:
            tool_intensity = "minimal"
        
        # Reasoning Quality Categorization
        reasoning_score = eval_result.reasoning_score
        if reasoning_score >= 0.8:
            reasoning_category = "excellent"
        elif reasoning_score >= 0.6:
            reasoning_category = "good"
        elif reasoning_score >= 0.4:
            reasoning_category = "fair"
        else:
            reasoning_category = "poor"
        
        return {
            "llm_inference_pattern": llm_pattern,
            "tool_usage_intensity": tool_intensity,
            "reasoning_quality_category": reasoning_category
        }
    
    def _print_immediate_summary(self, result: EnhancedTestResult):
        """Print immediate test result summary"""
        
        status = "✅ PASS" if result.overall_success else "❌ FAIL"
        bottleneck = result.resource_bottleneck.upper()
        reasoning = result.reasoning_quality_category.upper()
        
        print(f"   {status} | Score: {result.evaluation_result.overall_score:.2f} | Reasoning: {reasoning} | Bottleneck: {bottleneck}")
        print(f"   ⏱️  Time: {result.total_execution_time:.1f}s | CPU: {result.performance_analysis.avg_cpu_percent:.1f}% | Memory: {result.performance_analysis.avg_memory_percent:.1f}%")
        print(f"   🔍 Pattern: {result.llm_inference_pattern} LLM, {result.tool_usage_intensity} tools")
    
    async def _save_enhanced_result(self, result: EnhancedTestResult):
        """Save enhanced result with full monitoring data"""
        
        timestamp = int(time.time())
        filename = f"enhanced_{result.test_case.name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Create comprehensive result data
        result_data = {
            "test_info": {
                "name": result.test_case.name,
                "category": result.test_case.category,
                "complexity": result.test_case.complexity,
                "description": result.test_case.description,
                "max_time_seconds": result.test_case.max_time_seconds
            },
            "execution": {
                "total_time": result.total_execution_time,
                "success": result.execution_success,
                "error": result.mini_agent_error,
                "output_length": len(result.mini_agent_output)
            },
            "evaluation": asdict(result.evaluation_result),
            "performance_monitoring": asdict(result.performance_analysis),
            "enhanced_analysis": {
                "overall_success": result.overall_success,
                "confidence_score": result.confidence_score,
                "resource_bottleneck": result.resource_bottleneck,
                "bottleneck_confidence": result.bottleneck_confidence,
                "llm_inference_pattern": result.llm_inference_pattern,
                "tool_usage_intensity": result.tool_usage_intensity,
                "reasoning_quality_category": result.reasoning_quality_category
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"   📁 Enhanced result saved: {filename}")
    
    async def _generate_enhanced_analysis(self, results: List[EnhancedTestResult]):
        """Generate comprehensive enhanced analysis report"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"enhanced_analysis_{timestamp}.md"
        
        # Calculate comprehensive statistics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.overall_success)
        
        # Performance metrics
        avg_execution_time = sum(r.total_execution_time for r in results) / total_tests
        avg_score = sum(r.evaluation_result.overall_score for r in results) / total_tests
        avg_reasoning_score = sum(r.evaluation_result.reasoning_score for r in results) / total_tests
        
        # System performance metrics
        avg_cpu = sum(r.performance_analysis.avg_cpu_percent for r in results) / total_tests
        avg_memory = sum(r.performance_analysis.avg_memory_percent for r in results) / total_tests
        total_disk_io = sum(r.performance_analysis.total_disk_read_mb + r.performance_analysis.total_disk_write_mb for r in results)
        total_network_io = sum(r.performance_analysis.total_network_mb for r in results)
        
        # Bottleneck analysis
        bottleneck_counts = {}
        reasoning_counts = {}
        llm_pattern_counts = {}
        tool_intensity_counts = {}
        
        for result in results:
            # Count bottlenecks
            bottleneck = result.resource_bottleneck
            bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
            
            # Count reasoning quality
            reasoning = result.reasoning_quality_category
            reasoning_counts[reasoning] = reasoning_counts.get(reasoning, 0) + 1
            
            # Count LLM patterns
            llm_pattern = result.llm_inference_pattern
            llm_pattern_counts[llm_pattern] = llm_pattern_counts.get(llm_pattern, 0) + 1
            
            # Count tool intensity
            tool_intensity = result.tool_usage_intensity
            tool_intensity_counts[tool_intensity] = tool_intensity_counts.get(tool_intensity, 0) + 1
        
        # Generate enhanced report
        report = f"""# Enhanced Mini-Agent Comprehensive Evaluation Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This enhanced evaluation combines explicit reasoning assessment with real-time system monitoring to identify both response quality patterns and performance bottlenecks in Mini-Agent.

### 🎯 Overall Performance

| Metric | Value | Analysis |
|--------|-------|----------|
| **Success Rate** | {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%) | Overall task completion |
| **Response Quality** | {avg_score:.3f}/1.000 | Multi-criteria evaluation |
| **Reasoning Quality** | {avg_reasoning_score:.3f}/1.000 | {'🎉 Improved!' if avg_reasoning_score > 0.6 else '⚠️ Needs work'} |
| **Avg Execution Time** | {avg_execution_time:.1f}s | Time efficiency |
| **System CPU Usage** | {avg_cpu:.1f}% | Resource utilization |
| **System Memory Usage** | {avg_memory:.1f}% | Memory efficiency |

### 🔍 System Performance Analysis

| Resource | Usage | Status |
|----------|-------|--------|
| **Total Disk I/O** | {total_disk_io:.1f} MB | {'High' if total_disk_io > 100 else 'Moderate' if total_disk_io > 10 else 'Low'} usage |
| **Total Network I/O** | {total_network_io:.1f} MB | {'High' if total_network_io > 50 else 'Moderate' if total_network_io > 5 else 'Low'} usage |
| **Primary Bottlenecks** | {max(bottleneck_counts, key=bottleneck_counts.get)} | Most common limitation |

---

## 🧠 Reasoning Quality Analysis

### Reasoning Quality Distribution:
"""
        
        for quality, count in reasoning_counts.items():
            percentage = (count / total_tests) * 100
            report += f"- **{quality.title()}**: {count} tests ({percentage:.1f}%)\n"
        
        report += f"""
{'🎉 **Success!** Reasoning quality has improved significantly!' if avg_reasoning_score > 0.6 else '⚠️ **Action Needed:** Reasoning quality still needs improvement.'}

---

## 🔧 System Bottleneck Analysis

### Resource Bottleneck Distribution:
"""
        
        for bottleneck, count in bottleneck_counts.items():
            percentage = (count / total_tests) * 100
            report += f"- **{bottleneck.title()}**: {count} tests ({percentage:.1f}%)\n"
        
        report += f"""

### LLM Inference Patterns:
"""
        
        for pattern, count in llm_pattern_counts.items():
            percentage = (count / total_tests) * 100
            report += f"- **{pattern.title()}**: {count} tests ({percentage:.1f}%)\n"
        
        report += f"""

### Tool Usage Intensity:
"""
        
        for intensity, count in tool_intensity_counts.items():
            percentage = (count / total_tests) * 100
            report += f"- **{intensity.title()}**: {count} tests ({percentage:.1f}%)\n"
        
        report += f"""

---

## 📊 Detailed Test Results

| Test Name | Success | Score | Reasoning | Time | Bottleneck | LLM Pattern | Tool Use |
|-----------|---------|-------|-----------|------|------------|-------------|----------|
"""
        
        for result in results:
            success = "✅" if result.overall_success else "❌"
            report += f"| {result.test_case.name} | {success} | {result.evaluation_result.overall_score:.2f} | {result.reasoning_quality_category} | {result.total_execution_time:.1f}s | {result.resource_bottleneck} | {result.llm_inference_pattern} | {result.tool_usage_intensity} |\n"
        
        report += f"""

---

## 🔬 Key Findings & Recommendations

### Reasoning Quality Improvements:
"""
        
        if avg_reasoning_score > 0.6:
            report += "- ✅ **Significant improvement** in explicit reasoning demonstration\n"
            report += "- ✅ **Structured prompts** successfully encourage step-by-step thinking\n"
        else:
            report += "- ⚠️ **Still needs work** - consider more explicit reasoning requirements\n"
            report += "- 💡 **Recommendation**: Add even more structured prompt templates\n"
        
        report += f"""

### System Performance Insights:
"""
        
        primary_bottleneck = max(bottleneck_counts, key=bottleneck_counts.get)
        if primary_bottleneck == "cpu":
            report += "- 🔥 **CPU Bottleneck Detected** - Consider optimizing LLM inference\n"
            report += "- 💡 Recommendation: Implement model caching or request batching\n"
        elif primary_bottleneck == "memory":
            report += "- 🧠 **Memory Bottleneck Detected** - Large context usage identified\n"
            report += "- 💡 Recommendation: Implement context compression strategies\n"
        elif primary_bottleneck == "disk":
            report += "- 💾 **Disk I/O Bottleneck** - Heavy file operations detected\n"
            report += "- 💡 Recommendation: Optimize file operation patterns\n"
        elif primary_bottleneck == "network":
            report += "- 🌐 **Network Bottleneck** - API call patterns need optimization\n"
            report += "- 💡 Recommendation: Implement API request batching\n"
        else:
            report += "- ⚖️ **Balanced Performance** - No major bottlenecks identified\n"
            report += "- ✅ System is well-optimized for current workload\n"
        
        report += f"""

### Production Readiness Assessment:
- **Reasoning Quality**: {'Ready' if avg_reasoning_score > 0.7 else 'Needs improvement'}
- **Performance**: {'Scalable' if primary_bottleneck == 'balanced' else 'Optimization needed'}  
- **Reliability**: {'High' if successful_tests/total_tests > 0.9 else 'Moderate'}

---

*Report generated by Enhanced Mini-Agent Evaluation System v2.0*
*Combining reasoning analysis with real-time performance monitoring*
"""
        
        # Save comprehensive report
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📊 Enhanced analysis complete!")
        print(f"📄 Full report: {report_path}")
        print(f"📁 Individual results: {self.results_dir}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("ENHANCED EVALUATION SUMMARY") 
        print("=" * 80)
        print(f"Success Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
        print(f"Reasoning Quality: {avg_reasoning_score:.3f}/1.000 ({'Improved!' if avg_reasoning_score > 0.6 else 'Needs work'})")
        print(f"Primary Bottleneck: {primary_bottleneck}")
        print(f"System Efficiency: CPU {avg_cpu:.1f}%, Memory {avg_memory:.1f}%")
        print("=" * 80)

async def main():
    """Main execution function for enhanced evaluation"""
    
    evaluator = EnhancedMiniAgentEvaluator()
    
    print("🚀 Enhanced Mini-Agent Comprehensive Evaluation")
    print("Combining reasoning assessment with real-time performance monitoring")
    
    # Run enhanced evaluation (you can filter by categories)
    # results = await evaluator.run_enhanced_evaluation(test_categories=["reasoning"])  # Just reasoning tests
    results = await evaluator.run_enhanced_evaluation()  # All tests
    
    print(f"\n🎯 Enhanced evaluation complete! Analyzed {len(results)} tests with real-time monitoring.")

if __name__ == "__main__":
    asyncio.run(main())