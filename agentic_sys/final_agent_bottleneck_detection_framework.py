#!/usr/bin/env python3
"""
General Agent Bottleneck Detection Framework
Tree-structured evaluation methodology for coding agent systems
"""

import asyncio
import time
import psutil
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging


class BottleneckType(Enum):
    MEMORY = "memory"
    CPU = "cpu"
    IO = "io"
    NETWORK = "network"
    UNKNOWN = "unknown"


class TestPhase(Enum):
    INITIAL_ASSESSMENT = "initial_assessment"
    TARGETED_INVESTIGATION = "targeted_investigation"
    CONFIRMATION_STRESS = "confirmation_stress"
    OPTIMIZATION_VALIDATION = "optimization_validation"


@dataclass
class ResourceMetrics:
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    execution_time: float = 0.0


@dataclass
class BottleneckResult:
    test_name: str
    phase: TestPhase
    bottleneck_type: BottleneckType
    confidence: float
    metrics: ResourceMetrics
    success: bool
    failure_mode: Optional[str] = None
    recommendations: List[str] = None


class AgentBottleneckDetector:
    """
    Tree-structured bottleneck detection for coding agent systems
    
    Usage:
        detector = AgentBottleneckDetector()
        results = await detector.run_full_detection(agent_executor_func)
    """
    
    def __init__(self):
        self.results: List[BottleneckResult] = []
        self.detected_bottlenecks: Dict[BottleneckType, float] = {}
        
    async def run_full_detection(self, agent_executor) -> Dict[str, Any]:
        """
        Run complete tree-structured bottleneck detection
        
        Args:
            agent_executor: Async function that executes agent tasks
                           Should accept (task_name: str, task_params: dict)
        """
        logging.info("🌳 Starting Tree-Structured Bottleneck Detection")
        
        # Phase 1: Initial Assessment (5 tests)
        phase1_results = await self._phase1_initial_assessment(agent_executor)
        primary_bottleneck = self._analyze_primary_bottleneck(phase1_results)
        
        # Phase 2: Targeted Investigation (12 tests)
        phase2_results = await self._phase2_targeted_investigation(
            agent_executor, primary_bottleneck
        )
        
        # Phase 3: Confirmation & Stress Testing (8 tests)
        phase3_results = await self._phase3_confirmation_stress(
            agent_executor, primary_bottleneck
        )
        
        # Phase 4: Optimization Validation (5 tests)
        phase4_results = await self._phase4_optimization_validation(
            agent_executor
        )
        
        # Final analysis
        final_analysis = self._generate_final_analysis()
        
        return {
            "detection_tree_results": {
                "phase1_initial": phase1_results,
                "phase2_targeted": phase2_results,
                "phase3_confirmation": phase3_results,
                "phase4_validation": phase4_results
            },
            "bottleneck_analysis": final_analysis,
            "recommendations": self._generate_recommendations()
        }
    
    async def _phase1_initial_assessment(self, agent_executor) -> List[BottleneckResult]:
        """Phase 1: Quick 5-test assessment to identify primary bottleneck category"""
        
        tests = [
            ("simple_arithmetic", {"complexity": "low", "context_size": "small"}),
            ("memory_stress", {"complexity": "medium", "context_size": "large"}),
            ("cpu_intensive", {"complexity": "high", "context_size": "medium"}),
            ("file_operations", {"complexity": "medium", "file_count": 50}),
            ("network_requests", {"complexity": "medium", "request_count": 10})
        ]
        
        results = []
        for test_name, params in tests:
            result = await self._execute_monitored_test(
                agent_executor, test_name, params, TestPhase.INITIAL_ASSESSMENT
            )
            results.append(result)
            
        return results
    
    async def _phase2_targeted_investigation(self, agent_executor, 
                                           primary_bottleneck: BottleneckType) -> List[BottleneckResult]:
        """Phase 2: Deep dive into identified bottleneck category"""
        
        if primary_bottleneck == BottleneckType.MEMORY:
            return await self._investigate_memory_bottleneck(agent_executor)
        elif primary_bottleneck == BottleneckType.CPU:
            return await self._investigate_cpu_bottleneck(agent_executor)
        elif primary_bottleneck == BottleneckType.IO:
            return await self._investigate_io_bottleneck(agent_executor)
        elif primary_bottleneck == BottleneckType.NETWORK:
            return await self._investigate_network_bottleneck(agent_executor)
        else:
            return await self._investigate_general_bottleneck(agent_executor)
    
    async def _investigate_memory_bottleneck(self, agent_executor) -> List[BottleneckResult]:
        """Branch A: Memory bottleneck investigation"""
        tests = [
            ("context_scaling_small", {"context_tokens": 1000}),
            ("context_scaling_medium", {"context_tokens": 10000}),
            ("context_scaling_large", {"context_tokens": 50000}),
            ("memory_leak_detection", {"iterations": 20}),
            ("garbage_collection_test", {"objects": 10000}),
            ("memory_pool_exhaustion", {"allocations": 1000}),
            ("concurrent_memory_usage", {"threads": 5}),
            ("memory_fragmentation_test", {"pattern": "random"}),
            ("context_window_overflow", {"context_tokens": 100000}),
            ("memory_cleanup_verification", {"cycles": 10}),
            ("swap_usage_analysis", {"memory_pressure": "high"}),
            ("memory_growth_pattern", {"duration": 60})
        ]
        
        results = []
        for test_name, params in tests:
            result = await self._execute_monitored_test(
                agent_executor, test_name, params, TestPhase.TARGETED_INVESTIGATION
            )
            results.append(result)
            
        return results
    
    async def _investigate_cpu_bottleneck(self, agent_executor) -> List[BottleneckResult]:
        """Branch B: CPU bottleneck investigation"""
        tests = [
            ("algorithm_complexity_linear", {"data_size": 1000}),
            ("algorithm_complexity_quadratic", {"data_size": 1000}),
            ("parallel_processing_efficiency", {"threads": 4}),
            ("llm_inference_optimization", {"batch_size": 5}),
            ("thread_contention_analysis", {"concurrent_tasks": 10}),
            ("cpu_core_utilization", {"cores": "all"}),
            ("computation_intensive_task", {"operations": 1000000}),
            ("async_processing_efficiency", {"async_tasks": 20}),
            ("cpu_bound_vs_io_bound", {"ratio": "80/20"}),
            ("process_vs_thread_overhead", {"workers": 8}),
            ("cpu_cache_efficiency", {"data_pattern": "sequential"}),
            ("instruction_pipeline_analysis", {"complexity": "high"})
        ]
        
        results = []
        for test_name, params in tests:
            result = await self._execute_monitored_test(
                agent_executor, test_name, params, TestPhase.TARGETED_INVESTIGATION
            )
            results.append(result)
            
        return results
    
    async def _phase3_confirmation_stress(self, agent_executor, 
                                        primary_bottleneck: BottleneckType) -> List[BottleneckResult]:
        """Phase 3: Stress testing and edge case confirmation"""
        
        stress_tests = [
            ("extreme_load_test", {"multiplier": 10}),
            ("sustained_performance_test", {"duration": 300}),
            ("resource_starvation_scenario", {"limit": "aggressive"}),
            ("concurrent_user_simulation", {"users": 20}),
            ("edge_case_large_input", {"size": "maximum"}),
            ("edge_case_rapid_requests", {"rate": "high"}),
            ("failure_recovery_test", {"failure_mode": "resource_exhaustion"}),
            ("degradation_analysis", {"load_pattern": "gradual_increase"})
        ]
        
        results = []
        for test_name, params in tests:
            result = await self._execute_monitored_test(
                agent_executor, test_name, params, TestPhase.CONFIRMATION_STRESS
            )
            results.append(result)
            
        return results
    
    async def _phase4_optimization_validation(self, agent_executor) -> List[BottleneckResult]:
        """Phase 4: Post-optimization validation"""
        
        validation_tests = [
            ("baseline_performance_retest", {}),
            ("optimization_effectiveness", {"comparison": "before_after"}),
            ("regression_testing", {"previous_passed": True}),
            ("production_readiness_check", {"criteria": "strict"}),
            ("scalability_verification", {"scale_factor": 5})
        ]
        
        results = []
        for test_name, params in tests:
            result = await self._execute_monitored_test(
                agent_executor, test_name, params, TestPhase.OPTIMIZATION_VALIDATION
            )
            results.append(result)
            
        return results
    
    async def _execute_monitored_test(self, agent_executor, test_name: str, 
                                    params: dict, phase: TestPhase) -> BottleneckResult:
        """Execute a single test with comprehensive resource monitoring"""
        
        # Get initial resource baseline
        initial_metrics = self._capture_resource_metrics()
        start_time = time.time()
        
        try:
            # Execute the agent task
            success = await agent_executor(test_name, params)
            failure_mode = None
        except Exception as e:
            success = False
            failure_mode = str(e)
        
        # Capture final metrics
        end_time = time.time()
        final_metrics = self._capture_resource_metrics()
        
        # Calculate resource deltas
        execution_metrics = ResourceMetrics(
            cpu_percent=final_metrics.cpu_percent,
            memory_percent=final_metrics.memory_percent,
            disk_read_mb=final_metrics.disk_read_mb - initial_metrics.disk_read_mb,
            disk_write_mb=final_metrics.disk_write_mb - initial_metrics.disk_write_mb,
            network_sent_mb=final_metrics.network_sent_mb - initial_metrics.network_sent_mb,
            network_recv_mb=final_metrics.network_recv_mb - initial_metrics.network_recv_mb,
            execution_time=end_time - start_time
        )
        
        # Analyze bottleneck type and confidence
        bottleneck_type, confidence = self._classify_bottleneck(execution_metrics)
        
        return BottleneckResult(
            test_name=test_name,
            phase=phase,
            bottleneck_type=bottleneck_type,
            confidence=confidence,
            metrics=execution_metrics,
            success=success,
            failure_mode=failure_mode
        )
    
    def _capture_resource_metrics(self) -> ResourceMetrics:
        """Capture current system resource utilization"""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
        disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
        
        # Network metrics
        network_io = psutil.net_io_counters()
        network_sent_mb = network_io.bytes_sent / (1024 * 1024) if network_io else 0
        network_recv_mb = network_io.bytes_recv / (1024 * 1024) if network_io else 0
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb
        )
    
    def _classify_bottleneck(self, metrics: ResourceMetrics) -> Tuple[BottleneckType, float]:
        """Classify bottleneck type based on resource metrics"""
        
        # Thresholds for bottleneck detection
        memory_threshold = 75.0
        cpu_threshold = 80.0
        io_threshold = 50.0  # MB/s
        
        scores = {
            BottleneckType.MEMORY: min(metrics.memory_percent / memory_threshold, 1.0),
            BottleneckType.CPU: min(metrics.cpu_percent / cpu_threshold, 1.0),
            BottleneckType.IO: min((metrics.disk_read_mb + metrics.disk_write_mb) / 
                                 (io_threshold * metrics.execution_time), 1.0),
            BottleneckType.NETWORK: min((metrics.network_sent_mb + metrics.network_recv_mb) / 
                                      (10.0 * metrics.execution_time), 1.0)
        }
        
        # Find primary bottleneck
        primary_bottleneck = max(scores.items(), key=lambda x: x[1])
        return primary_bottleneck[0], primary_bottleneck[1]
    
    def _analyze_primary_bottleneck(self, phase1_results: List[BottleneckResult]) -> BottleneckType:
        """Analyze Phase 1 results to determine primary bottleneck for Phase 2"""
        
        bottleneck_votes = {}
        total_confidence = {}
        
        for result in phase1_results:
            if result.bottleneck_type not in bottleneck_votes:
                bottleneck_votes[result.bottleneck_type] = 0
                total_confidence[result.bottleneck_type] = 0
            
            bottleneck_votes[result.bottleneck_type] += 1
            total_confidence[result.bottleneck_type] += result.confidence
        
        # Weight votes by confidence
        weighted_scores = {}
        for bottleneck_type, votes in bottleneck_votes.items():
            avg_confidence = total_confidence[bottleneck_type] / votes
            weighted_scores[bottleneck_type] = votes * avg_confidence
        
        return max(weighted_scores.items(), key=lambda x: x[1])[0]
    
    def _generate_final_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive bottleneck analysis from all phases"""
        
        all_results = self.results
        
        # Count bottleneck types across all phases
        bottleneck_distribution = {}
        confidence_scores = {}
        
        for result in all_results:
            bt = result.bottleneck_type
            if bt not in bottleneck_distribution:
                bottleneck_distribution[bt] = 0
                confidence_scores[bt] = []
            
            bottleneck_distribution[bt] += 1
            confidence_scores[bt].append(result.confidence)
        
        # Calculate final bottleneck classification
        final_bottlenecks = {}
        for bt, count in bottleneck_distribution.items():
            avg_confidence = sum(confidence_scores[bt]) / len(confidence_scores[bt])
            final_bottlenecks[bt.value] = {
                "occurrence_count": count,
                "average_confidence": avg_confidence,
                "weighted_score": count * avg_confidence
            }
        
        # Primary bottleneck
        primary = max(final_bottlenecks.items(), key=lambda x: x[1]["weighted_score"])
        
        return {
            "primary_bottleneck": primary[0],
            "primary_confidence": primary[1]["average_confidence"],
            "bottleneck_distribution": final_bottlenecks,
            "total_tests_executed": len(all_results),
            "success_rate": sum(1 for r in all_results if r.success) / len(all_results),
            "critical_failures": [r.test_name for r in all_results if not r.success]
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate specific optimization recommendations based on detected bottlenecks"""
        
        recommendations = []
        
        if BottleneckType.MEMORY in [r.bottleneck_type for r in self.results]:
            recommendations.extend([
                "Implement context window sliding or compression",
                "Add explicit garbage collection after task completion", 
                "Set hard memory limits with graceful degradation",
                "Implement memory pooling for large objects",
                "Add memory profiling and monitoring"
            ])
        
        if BottleneckType.CPU in [r.bottleneck_type for r in self.results]:
            recommendations.extend([
                "Implement parallel processing for independent operations",
                "Optimize LLM inference with batching and caching",
                "Profile algorithm complexity and optimize hot paths",
                "Add async processing for I/O bound operations"
            ])
        
        if BottleneckType.IO in [r.bottleneck_type for r in self.results]:
            recommendations.extend([
                "Implement disk I/O caching strategies",
                "Optimize temporary file usage and cleanup",
                "Add connection pooling for database operations",
                "Implement streaming for large file processing"
            ])
        
        return recommendations


# Example usage and agent executor template
async def example_agent_executor(task_name: str, params: dict) -> bool:
    """
    Example agent executor function - replace with your agent system
    
    Args:
        task_name: Name of the test task to execute
        params: Task-specific parameters
        
    Returns:
        bool: True if task succeeded, False otherwise
    """
    
    # Simulate different types of agent tasks
    if "memory" in task_name:
        # Simulate memory-intensive operation
        data = [i for i in range(params.get("context_tokens", 1000))]
        await asyncio.sleep(0.1)
        
    elif "cpu" in task_name:
        # Simulate CPU-intensive operation
        result = sum(i**2 for i in range(params.get("operations", 1000)))
        
    elif "file" in task_name:
        # Simulate file operations
        await asyncio.sleep(params.get("duration", 0.2))
        
    elif "network" in task_name:
        # Simulate network operations
        await asyncio.sleep(params.get("latency", 0.1))
    
    # Random success rate for demonstration
    import random
    return random.random() > 0.2  # 80% success rate


async def main():
    """Example usage of the bottleneck detection framework"""
    
    detector = AgentBottleneckDetector()
    results = await detector.run_full_detection(example_agent_executor)
    
    # Save results
    with open("bottleneck_detection_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("🌳 Bottleneck Detection Complete!")
    print(f"Primary Bottleneck: {results['bottleneck_analysis']['primary_bottleneck']}")
    print(f"Confidence: {results['bottleneck_analysis']['primary_confidence']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())