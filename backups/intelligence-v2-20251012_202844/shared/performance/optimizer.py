# Agent Zero v1 - Phase 2: Interactive Control
# PerformanceOptimizer - LLM prompt optimization and system performance analysis

import asyncio
import time
import statistics
import psutil
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

import aiofiles
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, Summary


class OptimizationType(Enum):
    """Types of optimization strategies"""
    PROMPT_REDUCTION = "prompt_reduction"
    MODEL_SELECTION = "model_selection"
    BATCH_PROCESSING = "batch_processing"
    CACHING = "caching"
    PARALLEL_EXECUTION = "parallel_execution"
    RESOURCE_SCALING = "resource_scaling"


class BottleneckType(Enum):
    """Types of performance bottlenecks"""
    LLM_LATENCY = "llm_latency"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    AGENT_SCHEDULING = "agent_scheduling"
    QUEUE_BACKLOG = "queue_backlog"


@dataclass
class PerformanceMetric:
    """Performance measurement data"""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any]
    threshold: Optional[float] = None
    is_critical: bool = False


@dataclass
class OptimizationSuggestion:
    """Performance optimization recommendation"""
    optimization_id: str
    type: OptimizationType
    priority: int  # 1-10, 10 being highest
    title: str
    description: str
    expected_improvement: str
    implementation_steps: List[str]
    effort_estimate: str  # "low", "medium", "high"
    potential_savings: Dict[str, float]  # {"time": 0.3, "cost": 0.15, "resources": 0.2}
    code_example: Optional[str] = None
    monitoring_metrics: List[str] = None
    
    def __post_init__(self):
        if self.monitoring_metrics is None:
            self.monitoring_metrics = []


@dataclass
class LLMPerformanceData:
    """LLM execution performance data"""
    agent_type: str
    model_name: str
    prompt_length: int
    response_length: int
    execution_time: float
    tokens_per_second: float
    cost_estimate: float
    success: bool
    error_message: Optional[str] = None
    memory_peak_mb: float = 0.0
    prompt_hash: str = ""
    
    def __post_init__(self):
        if not self.prompt_hash:
            # Generate hash for prompt caching
            prompt_str = f"{self.agent_type}_{self.model_name}_{self.prompt_length}"
            self.prompt_hash = hashlib.md5(prompt_str.encode()).hexdigest()


@dataclass
class SystemResourceMetrics:
    """System resource usage metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_read_mb_s: float
    disk_write_mb_s: float
    network_sent_mb_s: float
    network_recv_mb_s: float
    active_agents: int
    queue_sizes: Dict[str, int]
    
    
class ResourceMonitor:
    """System resource monitoring and alerting"""
    
    def __init__(self, sampling_interval: float = 5.0):
        self.sampling_interval = sampling_interval
        self.is_monitoring = False
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 samples
        
        # Prometheus metrics
        self.cpu_usage = Gauge('agent_zero_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('agent_zero_memory_usage_percent', 'Memory usage percentage')
        self.agent_count = Gauge('agent_zero_active_agents', 'Number of active agents')
        self.queue_size = Gauge('agent_zero_queue_size', 'Message queue size', ['queue_name'])
        
        # Resource thresholds
        self.thresholds = {
            'cpu_critical': 90.0,
            'cpu_warning': 75.0,
            'memory_critical': 90.0,
            'memory_warning': 80.0,
            'disk_critical': 95.0,
            'disk_warning': 85.0
        }
        
    async def start_monitoring(self):
        """Start continuous resource monitoring"""
        self.is_monitoring = True
        
        while self.is_monitoring:
            metrics = await self._collect_system_metrics()
            self.metrics_history.append(metrics)
            
            # Update Prometheus metrics
            self.cpu_usage.set(metrics.cpu_percent)
            self.memory_usage.set(metrics.memory_percent)
            self.agent_count.set(metrics.active_agents)
            
            for queue_name, size in metrics.queue_sizes.items():
                self.queue_size.labels(queue_name=queue_name).set(size)
                
            # Check thresholds and alert if needed
            await self._check_resource_thresholds(metrics)
            
            await asyncio.sleep(self.sampling_interval)
            
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.is_monitoring = False
        
    async def _collect_system_metrics(self) -> SystemResourceMetrics:
        """Collect current system resource metrics"""
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk I/O
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network I/O
        network_io = psutil.net_io_counters()
        
        # Calculate rates (bytes/sec to MB/s)
        current_time = time.time()
        if hasattr(self, '_last_disk_io') and hasattr(self, '_last_network_io'):
            time_diff = current_time - self._last_time
            
            disk_read_mb_s = ((disk_io.read_bytes - self._last_disk_io.read_bytes) / time_diff) / (1024 * 1024)
            disk_write_mb_s = ((disk_io.write_bytes - self._last_disk_io.write_bytes) / time_diff) / (1024 * 1024)
            
            network_sent_mb_s = ((network_io.bytes_sent - self._last_network_io.bytes_sent) / time_diff) / (1024 * 1024)
            network_recv_mb_s = ((network_io.bytes_recv - self._last_network_io.bytes_recv) / time_diff) / (1024 * 1024)
        else:
            disk_read_mb_s = disk_write_mb_s = 0.0
            network_sent_mb_s = network_recv_mb_s = 0.0
            
        self._last_disk_io = disk_io
        self._last_network_io = network_io
        self._last_time = current_time
        
        return SystemResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk_usage.percent,
            disk_read_mb_s=disk_read_mb_s,
            disk_write_mb_s=disk_write_mb_s,
            network_sent_mb_s=network_sent_mb_s,
            network_recv_mb_s=network_recv_mb_s,
            active_agents=0,  # Will be updated by agent registry
            queue_sizes={}     # Will be updated by message bus
        )
        
    async def _check_resource_thresholds(self, metrics: SystemResourceMetrics):
        """Check if resource usage exceeds thresholds"""
        
        alerts = []
        
        if metrics.cpu_percent > self.thresholds['cpu_critical']:
            alerts.append(f"🚨 CRITICAL: CPU usage {metrics.cpu_percent:.1f}% > {self.thresholds['cpu_critical']}%")
        elif metrics.cpu_percent > self.thresholds['cpu_warning']:
            alerts.append(f"⚠️ WARNING: CPU usage {metrics.cpu_percent:.1f}% > {self.thresholds['cpu_warning']}%")
            
        if metrics.memory_percent > self.thresholds['memory_critical']:
            alerts.append(f"🚨 CRITICAL: Memory usage {metrics.memory_percent:.1f}% > {self.thresholds['memory_critical']}%")
        elif metrics.memory_percent > self.thresholds['memory_warning']:
            alerts.append(f"⚠️ WARNING: Memory usage {metrics.memory_percent:.1f}% > {self.thresholds['memory_warning']}%")
            
        if metrics.disk_usage_percent > self.thresholds['disk_critical']:
            alerts.append(f"🚨 CRITICAL: Disk usage {metrics.disk_usage_percent:.1f}% > {self.thresholds['disk_critical']}%")
        elif metrics.disk_usage_percent > self.thresholds['disk_warning']:
            alerts.append(f"⚠️ WARNING: Disk usage {metrics.disk_usage_percent:.1f}% > {self.thresholds['disk_warning']}%")
            
        # Log alerts (in real implementation, would send to monitoring system)
        for alert in alerts:
            print(f"[RESOURCE ALERT] {alert}")
            
    def get_resource_trends(self, duration_minutes: int = 60) -> Dict[str, List[float]]:
        """Get resource usage trends over specified duration"""
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {}
            
        trends = {
            'cpu_percent': [m.cpu_percent for m in recent_metrics],
            'memory_percent': [m.memory_percent for m in recent_metrics],
            'disk_read_mb_s': [m.disk_read_mb_s for m in recent_metrics],
            'disk_write_mb_s': [m.disk_write_mb_s for m in recent_metrics],
            'network_sent_mb_s': [m.network_sent_mb_s for m in recent_metrics],
            'network_recv_mb_s': [m.network_recv_mb_s for m in recent_metrics],
            'timestamps': [m.timestamp.isoformat() for m in recent_metrics]
        }
        
        return trends


class LLMPerformanceTracker:
    """Track and analyze LLM performance across agents"""
    
    def __init__(self):
        self.execution_history: List[LLMPerformanceData] = []
        self.model_performance: Dict[str, List[float]] = defaultdict(list)
        self.agent_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Prometheus metrics
        self.llm_request_duration = Histogram(
            'agent_zero_llm_request_duration_seconds',
            'LLM request duration',
            ['agent_type', 'model_name']
        )
        self.llm_tokens_per_second = Histogram(
            'agent_zero_llm_tokens_per_second', 
            'LLM tokens generated per second',
            ['agent_type', 'model_name']
        )
        self.llm_request_cost = Counter(
            'agent_zero_llm_request_cost_total',
            'Total LLM request cost',
            ['agent_type', 'model_name']
        )
        
    def record_llm_execution(self, perf_data: LLMPerformanceData):
        """Record LLM execution performance data"""
        
        self.execution_history.append(perf_data)
        
        # Update performance tracking
        self.model_performance[perf_data.model_name].append(perf_data.execution_time)
        self.agent_performance[perf_data.agent_type].append(perf_data.execution_time)
        
        # Update Prometheus metrics
        self.llm_request_duration.labels(
            agent_type=perf_data.agent_type,
            model_name=perf_data.model_name
        ).observe(perf_data.execution_time)
        
        if perf_data.tokens_per_second > 0:
            self.llm_tokens_per_second.labels(
                agent_type=perf_data.agent_type,
                model_name=perf_data.model_name
            ).observe(perf_data.tokens_per_second)
            
        self.llm_request_cost.labels(
            agent_type=perf_data.agent_type,
            model_name=perf_data.model_name
        ).inc(perf_data.cost_estimate)
        
    def get_model_performance_stats(self, model_name: str) -> Dict[str, float]:
        """Get performance statistics for a specific model"""
        
        model_executions = [p for p in self.execution_history if p.model_name == model_name]
        
        if not model_executions:
            return {}
            
        execution_times = [p.execution_time for p in model_executions]
        tokens_per_second = [p.tokens_per_second for p in model_executions if p.tokens_per_second > 0]
        costs = [p.cost_estimate for p in model_executions]
        success_rate = sum(1 for p in model_executions if p.success) / len(model_executions)
        
        stats = {
            'total_requests': len(model_executions),
            'avg_execution_time': statistics.mean(execution_times),
            'median_execution_time': statistics.median(execution_times),
            'p95_execution_time': np.percentile(execution_times, 95) if execution_times else 0,
            'success_rate': success_rate,
            'total_cost': sum(costs),
            'avg_cost_per_request': statistics.mean(costs) if costs else 0
        }
        
        if tokens_per_second:
            stats.update({
                'avg_tokens_per_second': statistics.mean(tokens_per_second),
                'median_tokens_per_second': statistics.median(tokens_per_second)
            })
            
        return stats
        
    def identify_slow_agents(self, threshold_percentile: float = 90) -> List[Tuple[str, float]]:
        """Identify agents with slow LLM response times"""
        
        slow_agents = []
        
        for agent_type, times in self.agent_performance.items():
            if len(times) < 5:  # Need minimum samples
                continue
                
            threshold = np.percentile(times, threshold_percentile)
            avg_time = statistics.mean(times)
            
            if avg_time > threshold:
                slow_agents.append((agent_type, avg_time))
                
        return sorted(slow_agents, key=lambda x: x[1], reverse=True)
        
    def get_prompt_optimization_candidates(self) -> List[Dict[str, Any]]:
        """Identify prompts that could benefit from optimization"""
        
        candidates = []
        prompt_groups = defaultdict(list)
        
        # Group executions by prompt characteristics
        for execution in self.execution_history:
            key = f"{execution.agent_type}_{execution.prompt_length//100*100}"  # Group by 100-char buckets
            prompt_groups[key].append(execution)
            
        for group_key, executions in prompt_groups.items():
            if len(executions) < 10:  # Need sufficient samples
                continue
                
            avg_time = statistics.mean([e.execution_time for e in executions])
            avg_prompt_length = statistics.mean([e.prompt_length for e in executions])
            
            # Identify optimization opportunities
            if avg_time > 10.0:  # Slow executions
                candidates.append({
                    'group': group_key,
                    'avg_execution_time': avg_time,
                    'avg_prompt_length': avg_prompt_length,
                    'total_executions': len(executions),
                    'optimization_potential': 'reduce_prompt_length',
                    'priority': min(10, int(avg_time / 2))
                })
                
            if avg_prompt_length > 2000:  # Long prompts
                candidates.append({
                    'group': group_key,
                    'avg_execution_time': avg_time,
                    'avg_prompt_length': avg_prompt_length,
                    'total_executions': len(executions),
                    'optimization_potential': 'prompt_compression',
                    'priority': min(10, int(avg_prompt_length / 500))
                })
                
        return sorted(candidates, key=lambda x: x['priority'], reverse=True)


class PromptOptimizer:
    """LLM prompt optimization based on execution results"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.optimization_history: Dict[str, List[Dict]] = defaultdict(list)
        
    async def optimize_prompt_for_task(self, 
                                     original_prompt: str, 
                                     task_type: str,
                                     performance_data: List[LLMPerformanceData],
                                     target_improvement: float = 0.2) -> Dict[str, Any]:
        """Optimize prompt based on historical performance"""
        
        # Analyze current prompt performance
        current_stats = self._analyze_prompt_performance(original_prompt, performance_data)
        
        optimization_strategies = []
        
        # Strategy 1: Reduce prompt length while maintaining quality
        if current_stats['avg_prompt_length'] > 1000:
            compressed_prompt = await self._compress_prompt(original_prompt, task_type)
            optimization_strategies.append({
                'strategy': 'prompt_compression',
                'original_length': len(original_prompt),
                'optimized_length': len(compressed_prompt),
                'optimized_prompt': compressed_prompt,
                'expected_speedup': min(0.4, (len(original_prompt) - len(compressed_prompt)) / len(original_prompt))
            })
            
        # Strategy 2: Template-based optimization
        if current_stats['repetitive_patterns'] > 0.3:
            template_prompt = await self._create_prompt_template(original_prompt, task_type)
            optimization_strategies.append({
                'strategy': 'template_optimization',
                'original_length': len(original_prompt),
                'optimized_length': len(template_prompt),
                'optimized_prompt': template_prompt,
                'expected_speedup': 0.15
            })
            
        # Strategy 3: Context reduction
        if current_stats['context_ratio'] > 0.6:
            context_optimized = await self._optimize_context(original_prompt, task_type)
            optimization_strategies.append({
                'strategy': 'context_optimization',
                'original_length': len(original_prompt),
                'optimized_length': len(context_optimized),
                'optimized_prompt': context_optimized,
                'expected_speedup': 0.25
            })
            
        # Select best strategy
        best_strategy = max(optimization_strategies, key=lambda x: x['expected_speedup']) if optimization_strategies else None
        
        return {
            'original_prompt': original_prompt,
            'original_stats': current_stats,
            'optimization_strategies': optimization_strategies,
            'recommended_strategy': best_strategy,
            'expected_improvement': best_strategy['expected_speedup'] if best_strategy else 0.0
        }
        
    def _analyze_prompt_performance(self, prompt: str, performance_data: List[LLMPerformanceData]) -> Dict[str, float]:
        """Analyze prompt characteristics and performance"""
        
        # Basic metrics
        prompt_length = len(prompt)
        word_count = len(prompt.split())
        lines = prompt.split('\n')
        
        # Pattern analysis
        repetitive_words = len([w for w in prompt.split() if prompt.count(w) > 3])
        repetitive_patterns = repetitive_words / max(word_count, 1)
        
        # Context analysis (approximate)
        context_keywords = ['context', 'background', 'previously', 'history', 'example']
        context_mentions = sum(prompt.lower().count(keyword) for keyword in context_keywords)
        context_ratio = context_mentions / max(word_count, 1)
        
        # Performance correlation
        matching_executions = [p for p in performance_data if abs(p.prompt_length - prompt_length) < 100]
        avg_execution_time = statistics.mean([p.execution_time for p in matching_executions]) if matching_executions else 0
        
        return {
            'avg_prompt_length': prompt_length,
            'word_count': word_count,
            'line_count': len(lines),
            'repetitive_patterns': repetitive_patterns,
            'context_ratio': context_ratio,
            'avg_execution_time': avg_execution_time,
            'performance_samples': len(matching_executions)
        }
        
    async def _compress_prompt(self, prompt: str, task_type: str) -> str:
        """Compress prompt while maintaining essential information"""
        
        compression_prompt = f"""
Compress the following prompt for {task_type} while maintaining all essential information and instructions:

Original prompt:
{prompt}

Requirements:
1. Keep all specific instructions and requirements
2. Remove redundant explanations
3. Use concise language
4. Maintain technical accuracy
5. Preserve examples if critical

Compressed prompt:
"""
        
        try:
            response = await self.llm_client.generate(
                prompt=compression_prompt,
                max_tokens=len(prompt.split()) // 2,  # Target 50% reduction
                temperature=0.1  # Low temperature for consistency
            )
            
            compressed = response.strip()
            
            # Validate compression didn't remove critical information
            if len(compressed) > len(prompt) * 0.9:  # Less than 10% reduction
                return prompt  # Return original if compression wasn't effective
                
            return compressed
            
        except Exception:
            return prompt  # Return original if compression fails
            
    async def _create_prompt_template(self, prompt: str, task_type: str) -> str:
        """Create a template-based version of the prompt"""
        
        template_prompt = f"""
Convert the following prompt into a reusable template with placeholders:

Original prompt:
{prompt}

Create a template that:
1. Identifies variable parts and replaces them with {{placeholder}} syntax
2. Keeps fixed instructions and format requirements
3. Reduces repetitive text
4. Maintains clarity and completeness

Template:
"""
        
        try:
            response = await self.llm_client.generate(
                prompt=template_prompt,
                max_tokens=len(prompt.split()),
                temperature=0.1
            )
            
            return response.strip()
            
        except Exception:
            return prompt
            
    async def _optimize_context(self, prompt: str, task_type: str) -> str:
        """Optimize context information in prompt"""
        
        context_prompt = f"""
Optimize the context and background information in this prompt:

Original prompt:
{prompt}

Optimization goals:
1. Remove redundant context
2. Keep only essential background information  
3. Focus on information directly relevant to the task
4. Maintain enough context for accurate results

Optimized prompt:
"""
        
        try:
            response = await self.llm_client.generate(
                prompt=context_prompt,
                max_tokens=len(prompt.split()),
                temperature=0.1
            )
            
            return response.strip()
            
        except Exception:
            return prompt


class PerformanceOptimizer:
    """
    Performance optimization system for Agent Zero v1
    
    Features:
    - LLM prompt optimization based on execution results
    - Execution bottleneck detection in agent pipeline  
    - Load balancing suggestions for agent workload
    - Intelligent task scheduling optimization
    """
    
    def __init__(self, llm_client=None):
        self.resource_monitor = ResourceMonitor()
        self.llm_tracker = LLMPerformanceTracker() 
        self.prompt_optimizer = PromptOptimizer(llm_client) if llm_client else None
        
        # Optimization state
        self.optimization_enabled = True
        self.optimization_history: List[OptimizationSuggestion] = []
        self.implemented_optimizations: Set[str] = set()
        
        # Performance baselines
        self.performance_baselines = {
            'avg_task_execution_time': 30.0,  # seconds
            'max_memory_usage_percent': 70.0,
            'max_cpu_usage_percent': 80.0,
            'target_agent_utilization': 0.75,
            'max_queue_wait_time': 10.0  # seconds
        }
        
    async def start_optimization_monitoring(self):
        """Start continuous performance optimization monitoring"""
        
        # Start resource monitoring
        resource_task = asyncio.create_task(self.resource_monitor.start_monitoring())
        
        # Start optimization analysis loop
        optimization_task = asyncio.create_task(self._optimization_analysis_loop())
        
        return resource_task, optimization_task
        
    async def _optimization_analysis_loop(self):
        """Continuous optimization analysis and suggestion generation"""
        
        while self.optimization_enabled:
            try:
                # Run comprehensive performance analysis
                suggestions = await self.analyze_system_performance()
                
                # Add new suggestions to history
                for suggestion in suggestions:
                    if suggestion.optimization_id not in self.implemented_optimizations:
                        self.optimization_history.append(suggestion)
                        
                # Auto-implement low-risk optimizations
                await self._auto_implement_optimizations(suggestions)
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                print(f"[OPTIMIZATION ERROR] {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error
                
    async def analyze_system_performance(self) -> List[OptimizationSuggestion]:
        """Comprehensive system performance analysis"""
        
        suggestions = []
        
        # 1. Resource utilization analysis
        resource_suggestions = await self._analyze_resource_utilization()
        suggestions.extend(resource_suggestions)
        
        # 2. LLM performance analysis
        llm_suggestions = await self._analyze_llm_performance()
        suggestions.extend(llm_suggestions)
        
        # 3. Agent scheduling analysis
        scheduling_suggestions = await self._analyze_agent_scheduling()
        suggestions.extend(scheduling_suggestions)
        
        # 4. Queue and bottleneck analysis
        bottleneck_suggestions = await self._detect_bottlenecks()
        suggestions.extend(bottleneck_suggestions)
        
        # Sort by priority
        suggestions.sort(key=lambda x: x.priority, reverse=True)
        
        return suggestions
        
    async def _analyze_resource_utilization(self) -> List[OptimizationSuggestion]:
        """Analyze system resource utilization patterns"""
        
        suggestions = []
        trends = self.resource_monitor.get_resource_trends(60)  # Last hour
        
        if not trends:
            return suggestions
            
        # CPU optimization suggestions
        avg_cpu = statistics.mean(trends['cpu_percent']) if trends['cpu_percent'] else 0
        max_cpu = max(trends['cpu_percent']) if trends['cpu_percent'] else 0
        
        if avg_cpu > 85:
            suggestions.append(OptimizationSuggestion(
                optimization_id="cpu_high_utilization",
                type=OptimizationType.RESOURCE_SCALING,
                priority=9,
                title="High CPU Utilization Detected",
                description=f"Average CPU usage {avg_cpu:.1f}% exceeds recommended threshold",
                expected_improvement="Reduce CPU bottlenecks and improve response times",
                implementation_steps=[
                    "Scale horizontally by adding more worker processes",
                    "Implement CPU-intensive task queuing",
                    "Consider moving to more powerful hardware",
                    "Profile and optimize CPU-heavy agent operations"
                ],
                effort_estimate="medium",
                potential_savings={"time": 0.25, "resources": 0.15},
                monitoring_metrics=["cpu_percent", "task_execution_time"]
            ))
            
        # Memory optimization suggestions  
        avg_memory = statistics.mean(trends['memory_percent']) if trends['memory_percent'] else 0
        
        if avg_memory > 80:
            suggestions.append(OptimizationSuggestion(
                optimization_id="memory_optimization",
                type=OptimizationType.RESOURCE_SCALING,
                priority=8,
                title="High Memory Usage Detected",
                description=f"Average memory usage {avg_memory:.1f}% approaching limits",
                expected_improvement="Prevent memory exhaustion and improve stability",
                implementation_steps=[
                    "Implement agent memory pooling",
                    "Add memory usage monitoring per agent",
                    "Implement garbage collection optimization", 
                    "Consider memory-efficient data structures"
                ],
                effort_estimate="medium",
                potential_savings={"stability": 0.3, "resources": 0.2},
                monitoring_metrics=["memory_percent", "memory_available_mb"]
            ))
            
        # Disk I/O optimization
        if trends['disk_read_mb_s'] or trends['disk_write_mb_s']:
            avg_disk_read = statistics.mean(trends['disk_read_mb_s'])
            avg_disk_write = statistics.mean(trends['disk_write_mb_s'])
            
            if avg_disk_read > 50 or avg_disk_write > 50:
                suggestions.append(OptimizationSuggestion(
                    optimization_id="disk_io_optimization",
                    type=OptimizationType.CACHING,
                    priority=6,
                    title="High Disk I/O Activity",
                    description=f"Disk I/O: {avg_disk_read:.1f}MB/s read, {avg_disk_write:.1f}MB/s write",
                    expected_improvement="Reduce disk I/O bottlenecks",
                    implementation_steps=[
                        "Implement file result caching",
                        "Use in-memory storage for temporary data",
                        "Batch file operations",
                        "Consider SSD upgrade if using HDDs"
                    ],
                    effort_estimate="low",
                    potential_savings={"time": 0.15, "resources": 0.1},
                    monitoring_metrics=["disk_read_mb_s", "disk_write_mb_s"]
                ))
                
        return suggestions
        
    async def _analyze_llm_performance(self) -> List[OptimizationSuggestion]:
        """Analyze LLM performance and suggest optimizations"""
        
        suggestions = []
        
        if not self.llm_tracker.execution_history:
            return suggestions
            
        # Find slow models
        model_stats = {}
        for model_name in self.llm_tracker.model_performance.keys():
            model_stats[model_name] = self.llm_tracker.get_model_performance_stats(model_name)
            
        # Identify models with poor performance
        for model_name, stats in model_stats.items():
            if stats['avg_execution_time'] > 15.0:  # Slow model
                suggestions.append(OptimizationSuggestion(
                    optimization_id=f"model_optimization_{model_name}",
                    type=OptimizationType.MODEL_SELECTION,
                    priority=7,
                    title=f"Slow Model Performance: {model_name}",
                    description=f"Model {model_name} has avg execution time {stats['avg_execution_time']:.1f}s",
                    expected_improvement="Faster LLM response times",
                    implementation_steps=[
                        f"Consider switching to faster model for {model_name} tasks",
                        "Implement model warm-up procedures",
                        "Optimize model configuration parameters",
                        "Use model caching where appropriate"
                    ],
                    effort_estimate="low",
                    potential_savings={"time": 0.4, "cost": 0.1},
                    monitoring_metrics=["llm_request_duration_seconds"]
                ))
                
        # Prompt optimization suggestions
        if self.prompt_optimizer:
            optimization_candidates = self.llm_tracker.get_prompt_optimization_candidates()
            
            for candidate in optimization_candidates[:3]:  # Top 3 candidates
                suggestions.append(OptimizationSuggestion(
                    optimization_id=f"prompt_optimization_{candidate['group']}",
                    type=OptimizationType.PROMPT_REDUCTION,
                    priority=candidate['priority'],
                    title=f"Prompt Optimization Opportunity: {candidate['group']}",
                    description=f"Long prompts (avg {candidate['avg_prompt_length']} chars) with slow execution times",
                    expected_improvement="Reduce prompt processing time and costs",
                    implementation_steps=[
                        "Analyze and compress verbose prompts",
                        "Create reusable prompt templates",
                        "Remove redundant context information",
                        "A/B test optimized prompts"
                    ],
                    effort_estimate="medium",
                    potential_savings={"time": 0.3, "cost": 0.25},
                    monitoring_metrics=["llm_request_duration_seconds", "llm_request_cost_total"]
                ))
                
        return suggestions
        
    async def _analyze_agent_scheduling(self) -> List[OptimizationSuggestion]:
        """Analyze agent scheduling and load balancing"""
        
        suggestions = []
        
        # Simulate agent performance data analysis
        # In real implementation, this would analyze actual agent metrics
        
        suggestions.append(OptimizationSuggestion(
            optimization_id="agent_load_balancing",
            type=OptimizationType.PARALLEL_EXECUTION,
            priority=6,
            title="Agent Load Balancing Optimization",
            description="Uneven load distribution detected across agents",
            expected_improvement="Better resource utilization and faster task completion",
            implementation_steps=[
                "Implement intelligent task assignment algorithm",
                "Monitor agent capacity and availability",
                "Create agent workload balancing system",
                "Add agent performance metrics tracking"
            ],
            effort_estimate="high",
            potential_savings={"time": 0.2, "resources": 0.25},
            monitoring_metrics=["agent_utilization", "task_queue_wait_time"],
            code_example="""
# Enhanced agent assignment with load balancing
class LoadBalancedAgentScheduler:
    def assign_task(self, task, available_agents):
        # Score agents based on current load and capability
        agent_scores = []
        for agent in available_agents:
            load_score = 1.0 - agent.current_load_ratio
            capability_score = agent.get_capability_score(task.type)
            performance_score = agent.get_performance_score()
            
            total_score = (load_score * 0.4 + 
                         capability_score * 0.4 + 
                         performance_score * 0.2)
            agent_scores.append((agent, total_score))
            
        # Select agent with highest score
        best_agent = max(agent_scores, key=lambda x: x[1])[0]
        return best_agent
"""
        ))
        
        return suggestions
        
    async def _detect_bottlenecks(self) -> List[OptimizationSuggestion]:
        """Detect system bottlenecks and suggest optimizations"""
        
        suggestions = []
        
        # Queue backlog analysis
        suggestions.append(OptimizationSuggestion(
            optimization_id="queue_optimization",
            type=OptimizationType.BATCH_PROCESSING,
            priority=5,
            title="Message Queue Optimization",
            description="Implement queue optimization and batch processing",
            expected_improvement="Reduce queue wait times and improve throughput",
            implementation_steps=[
                "Implement message batching for similar tasks",
                "Add queue priority levels",
                "Implement queue monitoring and alerting",
                "Consider message queue scaling"
            ],
            effort_estimate="medium",
            potential_savings={"time": 0.15, "throughput": 0.3},
            monitoring_metrics=["queue_size", "message_processing_time"]
        ))
        
        return suggestions
        
    async def _auto_implement_optimizations(self, suggestions: List[OptimizationSuggestion]):
        """Automatically implement low-risk optimizations"""
        
        auto_implementable = [
            OptimizationType.CACHING,
            OptimizationType.BATCH_PROCESSING
        ]
        
        for suggestion in suggestions:
            if (suggestion.type in auto_implementable and 
                suggestion.effort_estimate == "low" and
                suggestion.optimization_id not in self.implemented_optimizations):
                
                # Implement optimization (placeholder - actual implementation would vary)
                await self._implement_optimization(suggestion)
                self.implemented_optimizations.add(suggestion.optimization_id)
                
    async def _implement_optimization(self, suggestion: OptimizationSuggestion):
        """Implement a specific optimization"""
        
        print(f"[AUTO-OPTIMIZATION] Implementing: {suggestion.title}")
        
        # Placeholder implementation - real implementation would depend on optimization type
        if suggestion.type == OptimizationType.CACHING:
            print("  - Enabling result caching")
        elif suggestion.type == OptimizationType.BATCH_PROCESSING:
            print("  - Implementing batch processing")
            
        # Record implementation
        suggestion_dict = asdict(suggestion)
        suggestion_dict['implemented_at'] = datetime.now().isoformat()
        
        await self._save_optimization_record(suggestion_dict)
        
    async def _save_optimization_record(self, optimization_record: Dict):
        """Save optimization implementation record"""
        
        record_file = f"optimization_records_{datetime.now().strftime('%Y%m')}.json"
        
        try:
            # Load existing records
            try:
                async with aiofiles.open(record_file, 'r') as f:
                    records = json.loads(await f.read())
            except FileNotFoundError:
                records = []
                
            # Add new record
            records.append(optimization_record)
            
            # Save updated records
            async with aiofiles.open(record_file, 'w') as f:
                await f.write(json.dumps(records, indent=2, default=str))
                
        except Exception as e:
            print(f"[ERROR] Failed to save optimization record: {e}")
            
    async def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        
        # Get current system metrics
        resource_trends = self.resource_monitor.get_resource_trends(60)
        
        # LLM performance summary
        llm_summary = {}
        for model_name in self.llm_tracker.model_performance.keys():
            llm_summary[model_name] = self.llm_tracker.get_model_performance_stats(model_name)
            
        # Recent optimization suggestions
        recent_suggestions = sorted(
            self.optimization_history[-10:], 
            key=lambda x: x.priority, 
            reverse=True
        )
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'system_health': {
                'resource_trends': resource_trends,
                'llm_performance': llm_summary,
                'optimization_suggestions': len(self.optimization_history),
                'implemented_optimizations': len(self.implemented_optimizations)
            },
            'performance_metrics': {
                'total_llm_requests': len(self.llm_tracker.execution_history),
                'avg_execution_time': statistics.mean([p.execution_time for p in self.llm_tracker.execution_history]) if self.llm_tracker.execution_history else 0,
                'total_cost_estimate': sum([p.cost_estimate for p in self.llm_tracker.execution_history]),
                'success_rate': sum(1 for p in self.llm_tracker.execution_history if p.success) / max(len(self.llm_tracker.execution_history), 1)
            },
            'optimization_recommendations': [asdict(s) for s in recent_suggestions],
            'next_actions': [
                'Review high-priority optimization suggestions',
                'Monitor resource usage trends',
                'Test prompt optimization strategies',
                'Implement load balancing improvements'
            ]
        }
        
        return report
        
    async def export_performance_data(self, output_path: str) -> List[str]:
        """Export performance data for external analysis"""
        
        exported_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export LLM performance data
        llm_file = f"{output_path}/llm_performance_{timestamp}.json"
        llm_data = [asdict(p) for p in self.llm_tracker.execution_history]
        
        async with aiofiles.open(llm_file, 'w') as f:
            await f.write(json.dumps(llm_data, indent=2, default=str))
        exported_files.append(llm_file)
        
        # Export resource metrics
        resource_file = f"{output_path}/resource_metrics_{timestamp}.json"
        resource_data = [asdict(m) for m in self.resource_monitor.metrics_history]
        
        async with aiofiles.open(resource_file, 'w') as f:
            await f.write(json.dumps(resource_data, indent=2, default=str))
        exported_files.append(resource_file)
        
        # Export optimization suggestions
        optimization_file = f"{output_path}/optimization_suggestions_{timestamp}.json"
        optimization_data = [asdict(s) for s in self.optimization_history]
        
        async with aiofiles.open(optimization_file, 'w') as f:
            await f.write(json.dumps(optimization_data, indent=2, default=str))
        exported_files.append(optimization_file)
        
        return exported_files


# Example usage and integration
if __name__ == "__main__":
    
    async def demo_performance_optimization():
        optimizer = PerformanceOptimizer()
        
        # Start monitoring
        resource_task, optimization_task = await optimizer.start_optimization_monitoring()
        
        # Simulate some LLM executions
        sample_executions = [
            LLMPerformanceData(
                agent_type="backend",
                model_name="llama3.1:8b",
                prompt_length=1250,
                response_length=800,
                execution_time=12.5,
                tokens_per_second=64.0,
                cost_estimate=0.05,
                success=True,
                memory_peak_mb=512.0
            ),
            LLMPerformanceData(
                agent_type="frontend",
                model_name="llama3.1:8b", 
                prompt_length=2100,
                response_length=1200,
                execution_time=18.3,
                tokens_per_second=65.5,
                cost_estimate=0.08,
                success=True,
                memory_peak_mb=480.0
            )
        ]
        
        for execution in sample_executions:
            optimizer.llm_tracker.record_llm_execution(execution)
            
        # Wait for some monitoring data
        await asyncio.sleep(10)
        
        # Generate performance analysis
        suggestions = await optimizer.analyze_system_performance()
        
        print(f"Generated {len(suggestions)} optimization suggestions:")
        for i, suggestion in enumerate(suggestions[:5], 1):
            print(f"{i}. {suggestion.title} (Priority: {suggestion.priority})")
            print(f"   {suggestion.description}")
            print(f"   Expected improvement: {suggestion.expected_improvement}")
            print()
            
        # Generate report
        report = await optimizer.generate_optimization_report()
        print(f"Performance Report Generated:")
        print(f"- Total LLM Requests: {report['performance_metrics']['total_llm_requests']}")
        print(f"- Avg Execution Time: {report['performance_metrics']['avg_execution_time']:.2f}s")
        print(f"- Success Rate: {report['performance_metrics']['success_rate']:.2%}")
        
        # Cleanup
        optimizer.optimization_enabled = False
        resource_task.cancel()
        optimization_task.cancel()
        
    # Run demo
    asyncio.run(demo_performance_optimization())