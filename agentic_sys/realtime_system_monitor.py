#!/usr/bin/env python3
"""
Real-Time System Monitoring for Mini-Agent Bottleneck Analysis
=============================================================
Comprehensive system monitoring to identify performance bottlenecks during Mini-Agent execution.

Monitors:
- CPU usage per core
- Memory (RAM) usage and allocation patterns
- Disk I/O (read/write operations and throughput)
- Network I/O (bytes sent/received)
- Process-specific metrics for Mini-Agent
- GPU usage (if available)
- LLM API call timing patterns
"""

import asyncio
import json
import logging
import platform
import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import psutil
import os
import signal

# Try to import GPU monitoring
try:
    import GPUtil
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

logger = logging.getLogger(__name__)

@dataclass
class SystemSnapshot:
    """Single point-in-time system resource snapshot"""
    timestamp: float
    
    # CPU Metrics
    cpu_percent_total: float
    cpu_percent_per_core: List[float]
    cpu_freq_current: float
    cpu_count_logical: int
    cpu_count_physical: int
    
    # Memory Metrics
    memory_total_gb: float
    memory_used_gb: float
    memory_available_gb: float
    memory_percent: float
    swap_total_gb: float
    swap_used_gb: float
    swap_percent: float
    
    # Disk I/O Metrics
    disk_read_bytes: int
    disk_write_bytes: int
    disk_read_count: int
    disk_write_count: int
    disk_usage_percent: float
    
    # Network I/O Metrics  
    network_bytes_sent: int
    network_bytes_recv: int
    network_packets_sent: int
    network_packets_recv: int
    
    # Process-Specific Metrics (for Mini-Agent process)
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0
    process_memory_percent: float = 0.0
    process_num_threads: int = 0
    process_open_files: int = 0
    
    # GPU Metrics (if available)
    gpu_usage_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_temperature: float = 0.0

@dataclass
class PerformanceAnalysis:
    """Analysis of performance patterns and bottlenecks"""
    monitoring_duration: float
    sample_count: int
    
    # CPU Analysis
    avg_cpu_percent: float
    max_cpu_percent: float
    cpu_bottleneck_duration: float  # Time spent above 80%
    cpu_core_imbalance: float       # Variance across cores
    
    # Memory Analysis
    avg_memory_percent: float
    max_memory_percent: float
    memory_growth_rate: float       # MB per second growth
    swap_usage_detected: bool
    
    # Disk Analysis  
    total_disk_read_mb: float
    total_disk_write_mb: float
    avg_disk_throughput_mbps: float
    disk_bottleneck_detected: bool
    
    # Network Analysis
    total_network_mb: float
    avg_network_throughput_mbps: float
    network_pattern: str            # "bursty", "steady", "idle"
    
    # Process Analysis
    process_peak_memory_mb: float
    process_avg_cpu: float
    process_thread_growth: int      # Change in thread count
    
    # Bottleneck Identification
    primary_bottleneck: str         # "cpu", "memory", "disk", "network", "balanced"
    bottleneck_confidence: float    # 0.0-1.0
    bottleneck_details: Dict[str, any] = field(default_factory=dict)

class RealTimeSystemMonitor:
    """Real-time system resource monitoring with bottleneck detection"""
    
    def __init__(self, sample_interval: float = 0.1, max_samples: int = 10000):
        self.sample_interval = sample_interval
        self.max_samples = max_samples
        self.snapshots: deque = deque(maxlen=max_samples)
        self.monitoring = False
        self.monitor_thread = None
        self.target_process = None
        self.start_time = None
        
        # Initialize baseline metrics
        self._initial_disk_io = psutil.disk_io_counters()
        self._initial_network_io = psutil.net_io_counters()
        
    def start_monitoring(self, target_process_name: str = "python"):
        """Start real-time monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        self.snapshots.clear()
        
        # Find target process (Mini-Agent)
        self._find_target_process(target_process_name)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Started real-time system monitoring (interval: {self.sample_interval}s)")
    
    def stop_monitoring(self) -> PerformanceAnalysis:
        """Stop monitoring and return performance analysis"""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        analysis = self._analyze_performance()
        logger.info(f"Stopped monitoring. Collected {len(self.snapshots)} samples over {analysis.monitoring_duration:.1f}s")
        
        return analysis
    
    def _find_target_process(self, process_name: str):
        """Find the target process to monitor"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if process_name.lower() in proc.info['name'].lower():
                    # Look for Mini-Agent specifically
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'mini-agent' in cmdline.lower() or 'mini_agent' in cmdline.lower():
                        self.target_process = psutil.Process(proc.info['pid'])
                        logger.info(f"Found target process: {proc.info['name']} (PID: {proc.info['pid']})")
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread"""
        while self.monitoring:
            try:
                snapshot = self._capture_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sample_interval)
    
    def _capture_snapshot(self) -> SystemSnapshot:
        """Capture a single system snapshot"""
        current_time = time.time()
        
        # CPU Metrics
        cpu_percent = psutil.cpu_percent()
        cpu_percents = psutil.cpu_percent(percpu=True)
        cpu_freq = psutil.cpu_freq()
        
        # Memory Metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk I/O Metrics
        disk_io = psutil.disk_io_counters()
        disk_usage = psutil.disk_usage('/')
        
        # Network I/O Metrics
        network_io = psutil.net_io_counters()
        
        # Process-specific metrics
        process_cpu = 0.0
        process_memory_mb = 0.0
        process_memory_percent = 0.0
        process_threads = 0
        process_files = 0
        
        if self.target_process:
            try:
                process_cpu = self.target_process.cpu_percent()
                process_memory = self.target_process.memory_info()
                process_memory_mb = process_memory.rss / (1024 * 1024)
                process_memory_percent = self.target_process.memory_percent()
                process_threads = self.target_process.num_threads()
                process_files = len(self.target_process.open_files())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.target_process = None
        
        # GPU Metrics
        gpu_usage = 0.0
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        gpu_temp = 0.0
        
        if HAS_GPU:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Primary GPU
                    gpu_usage = gpu.load * 100
                    gpu_memory_used = gpu.memoryUsed
                    gpu_memory_total = gpu.memoryTotal
                    gpu_temp = gpu.temperature
            except Exception:
                pass  # GPU monitoring optional
        
        return SystemSnapshot(
            timestamp=current_time,
            cpu_percent_total=cpu_percent,
            cpu_percent_per_core=cpu_percents,
            cpu_freq_current=cpu_freq.current if cpu_freq else 0.0,
            cpu_count_logical=psutil.cpu_count(),
            cpu_count_physical=psutil.cpu_count(logical=False),
            memory_total_gb=memory.total / (1024**3),
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            memory_percent=memory.percent,
            swap_total_gb=swap.total / (1024**3),
            swap_used_gb=swap.used / (1024**3),
            swap_percent=swap.percent,
            disk_read_bytes=disk_io.read_bytes - self._initial_disk_io.read_bytes if disk_io and self._initial_disk_io else 0,
            disk_write_bytes=disk_io.write_bytes - self._initial_disk_io.write_bytes if disk_io and self._initial_disk_io else 0,
            disk_read_count=disk_io.read_count - self._initial_disk_io.read_count if disk_io and self._initial_disk_io else 0,
            disk_write_count=disk_io.write_count - self._initial_disk_io.write_count if disk_io and self._initial_disk_io else 0,
            disk_usage_percent=disk_usage.percent,
            network_bytes_sent=network_io.bytes_sent - self._initial_network_io.bytes_sent if network_io and self._initial_network_io else 0,
            network_bytes_recv=network_io.bytes_recv - self._initial_network_io.bytes_recv if network_io and self._initial_network_io else 0,
            network_packets_sent=network_io.packets_sent - self._initial_network_io.packets_sent if network_io and self._initial_network_io else 0,
            network_packets_recv=network_io.packets_recv - self._initial_network_io.packets_recv if network_io and self._initial_network_io else 0,
            process_cpu_percent=process_cpu,
            process_memory_mb=process_memory_mb,
            process_memory_percent=process_memory_percent,
            process_num_threads=process_threads,
            process_open_files=process_files,
            gpu_usage_percent=gpu_usage,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_temperature=gpu_temp
        )
    
    def _analyze_performance(self) -> PerformanceAnalysis:
        """Analyze collected snapshots to identify performance patterns and bottlenecks"""
        
        if not self.snapshots:
            return PerformanceAnalysis(
                monitoring_duration=0.0,
                sample_count=0,
                avg_cpu_percent=0.0,
                max_cpu_percent=0.0,
                cpu_bottleneck_duration=0.0,
                cpu_core_imbalance=0.0,
                avg_memory_percent=0.0,
                max_memory_percent=0.0,
                memory_growth_rate=0.0,
                swap_usage_detected=False,
                total_disk_read_mb=0.0,
                total_disk_write_mb=0.0,
                avg_disk_throughput_mbps=0.0,
                disk_bottleneck_detected=False,
                total_network_mb=0.0,
                avg_network_throughput_mbps=0.0,
                network_pattern="idle",
                process_peak_memory_mb=0.0,
                process_avg_cpu=0.0,
                process_thread_growth=0,
                primary_bottleneck="unknown",
                bottleneck_confidence=0.0
            )
        
        snapshots = list(self.snapshots)
        duration = snapshots[-1].timestamp - snapshots[0].timestamp
        
        # CPU Analysis
        cpu_values = [s.cpu_percent_total for s in snapshots]
        avg_cpu = sum(cpu_values) / len(cpu_values)
        max_cpu = max(cpu_values)
        cpu_bottleneck_time = sum(1 for cpu in cpu_values if cpu > 80) * self.sample_interval
        
        # CPU core variance analysis
        core_variances = []
        for snapshot in snapshots:
            if snapshot.cpu_percent_per_core:
                variance = sum((core - avg_cpu) ** 2 for core in snapshot.cpu_percent_per_core) / len(snapshot.cpu_percent_per_core)
                core_variances.append(variance)
        cpu_core_imbalance = sum(core_variances) / len(core_variances) if core_variances else 0.0
        
        # Memory Analysis
        memory_values = [s.memory_percent for s in snapshots]
        avg_memory = sum(memory_values) / len(memory_values)
        max_memory = max(memory_values)
        
        # Memory growth rate
        memory_mb_values = [s.memory_used_gb * 1024 for s in snapshots]
        memory_growth = (memory_mb_values[-1] - memory_mb_values[0]) / duration if duration > 0 else 0
        
        swap_used = any(s.swap_percent > 1.0 for s in snapshots)
        
        # Disk Analysis
        last_snapshot = snapshots[-1]
        disk_read_mb = last_snapshot.disk_read_bytes / (1024**2)
        disk_write_mb = last_snapshot.disk_write_bytes / (1024**2)
        total_disk_mb = disk_read_mb + disk_write_mb
        disk_throughput = total_disk_mb / duration if duration > 0 else 0
        disk_bottleneck = disk_throughput > 100  # >100 MB/s indicates heavy I/O
        
        # Network Analysis  
        network_sent_mb = last_snapshot.network_bytes_sent / (1024**2)
        network_recv_mb = last_snapshot.network_bytes_recv / (1024**2)
        total_network_mb = network_sent_mb + network_recv_mb
        network_throughput = total_network_mb / duration if duration > 0 else 0
        
        # Network pattern detection
        network_samples = [(s.network_bytes_sent + s.network_bytes_recv) / (1024**2) for s in snapshots]
        if len(network_samples) > 10:
            network_variance = sum((x - total_network_mb/len(network_samples)) ** 2 for x in network_samples) / len(network_samples)
            if network_variance > total_network_mb * 0.5:
                network_pattern = "bursty"
            elif total_network_mb > 1.0:
                network_pattern = "steady"
            else:
                network_pattern = "idle"
        else:
            network_pattern = "idle"
        
        # Process Analysis
        process_memory_values = [s.process_memory_mb for s in snapshots if s.process_memory_mb > 0]
        process_cpu_values = [s.process_cpu_percent for s in snapshots if s.process_cpu_percent > 0]
        process_thread_values = [s.process_num_threads for s in snapshots if s.process_num_threads > 0]
        
        process_peak_memory = max(process_memory_values) if process_memory_values else 0.0
        process_avg_cpu = sum(process_cpu_values) / len(process_cpu_values) if process_cpu_values else 0.0
        process_thread_growth = (process_thread_values[-1] - process_thread_values[0]) if len(process_thread_values) > 1 else 0
        
        # Bottleneck Detection
        bottleneck_scores = {
            "cpu": avg_cpu / 100.0,
            "memory": avg_memory / 100.0,
            "disk": min(disk_throughput / 200.0, 1.0),  # Normalize to 0-1
            "network": min(network_throughput / 50.0, 1.0),  # Normalize to 0-1
        }
        
        primary_bottleneck = max(bottleneck_scores, key=bottleneck_scores.get)
        bottleneck_confidence = bottleneck_scores[primary_bottleneck]
        
        # If no clear bottleneck, mark as balanced
        if bottleneck_confidence < 0.6:
            primary_bottleneck = "balanced"
            bottleneck_confidence = 1.0 - max(bottleneck_scores.values())
        
        # Detailed bottleneck analysis
        bottleneck_details = {
            "cpu_analysis": {
                "avg_usage": avg_cpu,
                "peak_usage": max_cpu,
                "bottleneck_duration_percent": (cpu_bottleneck_time / duration * 100) if duration > 0 else 0,
                "core_imbalance": cpu_core_imbalance
            },
            "memory_analysis": {
                "avg_usage": avg_memory,
                "peak_usage": max_memory,
                "growth_rate_mb_per_sec": memory_growth,
                "swap_usage": swap_used
            },
            "disk_analysis": {
                "total_read_mb": disk_read_mb,
                "total_write_mb": disk_write_mb,
                "avg_throughput_mbps": disk_throughput,
                "bottleneck_detected": disk_bottleneck
            },
            "network_analysis": {
                "total_mb": total_network_mb,
                "avg_throughput_mbps": network_throughput,
                "pattern": network_pattern
            },
            "process_analysis": {
                "peak_memory_mb": process_peak_memory,
                "avg_cpu_percent": process_avg_cpu,
                "thread_growth": process_thread_growth
            }
        }
        
        return PerformanceAnalysis(
            monitoring_duration=duration,
            sample_count=len(snapshots),
            avg_cpu_percent=avg_cpu,
            max_cpu_percent=max_cpu,
            cpu_bottleneck_duration=cpu_bottleneck_time,
            cpu_core_imbalance=cpu_core_imbalance,
            avg_memory_percent=avg_memory,
            max_memory_percent=max_memory,
            memory_growth_rate=memory_growth,
            swap_usage_detected=swap_used,
            total_disk_read_mb=disk_read_mb,
            total_disk_write_mb=disk_write_mb,
            avg_disk_throughput_mbps=disk_throughput,
            disk_bottleneck_detected=disk_bottleneck,
            total_network_mb=total_network_mb,
            avg_network_throughput_mbps=network_throughput,
            network_pattern=network_pattern,
            process_peak_memory_mb=process_peak_memory,
            process_avg_cpu=process_avg_cpu,
            process_thread_growth=process_thread_growth,
            primary_bottleneck=primary_bottleneck,
            bottleneck_confidence=bottleneck_confidence,
            bottleneck_details=bottleneck_details
        )
    
    def save_monitoring_data(self, filepath: Path):
        """Save detailed monitoring data to file"""
        
        data = {
            "monitoring_info": {
                "start_time": self.start_time,
                "end_time": time.time(),
                "sample_interval": self.sample_interval,
                "total_samples": len(self.snapshots)
            },
            "snapshots": [asdict(snapshot) for snapshot in self.snapshots],
            "performance_analysis": asdict(self._analyze_performance())
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Monitoring data saved to {filepath}")

# Example usage and testing
async def test_monitoring():
    """Test the monitoring system"""
    
    monitor = RealTimeSystemMonitor(sample_interval=0.5)
    
    print("🔍 Starting system monitoring test...")
    monitor.start_monitoring("python")
    
    # Simulate some work
    print("💻 Simulating workload...")
    await asyncio.sleep(10)
    
    # Stop monitoring and analyze
    analysis = monitor.stop_monitoring()
    
    print(f"\n📊 Monitoring Results:")
    print(f"Duration: {analysis.monitoring_duration:.1f}s")
    print(f"Samples: {analysis.sample_count}")
    print(f"Average CPU: {analysis.avg_cpu_percent:.1f}%")
    print(f"Average Memory: {analysis.avg_memory_percent:.1f}%")
    print(f"Primary Bottleneck: {analysis.primary_bottleneck} ({analysis.bottleneck_confidence:.2f} confidence)")
    print(f"Disk I/O: {analysis.total_disk_read_mb + analysis.total_disk_write_mb:.1f} MB")
    print(f"Network I/O: {analysis.total_network_mb:.1f} MB")
    
    # Save detailed data
    output_dir = Path("monitoring_test_results")
    output_dir.mkdir(exist_ok=True)
    monitor.save_monitoring_data(output_dir / "test_monitoring_data.json")
    
    return analysis

if __name__ == "__main__":
    asyncio.run(test_monitoring())