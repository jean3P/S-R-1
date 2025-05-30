"""
Adaptive exploration strategies for LeetCode solution generation.
Implements multi-signal branch termination and priority-based exploration.
"""

import heapq
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BranchHistory:
    """Tracks the history and statistics of a branch exploration."""
    node_id: str
    failures: int = 0
    improvements: List[float] = field(default_factory=list)
    error_types: List[str] = field(default_factory=list)
    solution_hashes: List[str] = field(default_factory=list)
    total_time: float = 0.0
    candidates_generated: int = 0
    start_time: float = field(default_factory=time.time)
    test_improvements: List[int] = field(default_factory=list)  # Number of tests fixed

    def add_error(self, error_type: str):
        """Add an error type to the history."""
        self.error_types.append(error_type)

    def add_improvement(self, improvement_rate: float, tests_fixed: int = 0):
        """Add an improvement measurement."""
        self.improvements.append(improvement_rate)
        self.test_improvements.append(tests_fixed)

    def get_recent_errors(self, n: int = 5) -> List[str]:
        """Get the n most recent errors."""
        return self.error_types[-n:] if self.error_types else []

    def calculate_error_diversity(self) -> float:
        """Calculate the diversity of recent errors (0-1, higher is more diverse)."""
        recent = self.get_recent_errors(5)
        if len(recent) < 2:
            return 1.0
        return len(set(recent)) / len(recent)

    def get_improvement_trend(self) -> float:
        """Calculate recent improvement trend (-1 to 1, positive is improving)."""
        if len(self.improvements) < 3:
            return 0.0
        recent = self.improvements[-3:]
        if len(recent) < 2:
            return 0.0
        # Calculate linear trend
        x = np.arange(len(recent))
        y = np.array(recent)
        if np.std(y) == 0:
            return 0.0
        trend = np.polyfit(x, y, 1)[0]
        return np.clip(trend, -1, 1)


@dataclass
class ExplorationNode:
    """Node in the exploration priority queue."""
    node: Dict[str, Any]
    priority: float
    timestamp: float = field(default_factory=time.time)
    parent_score: float = 1.0

    def __lt__(self, other):
        """For heap operations - higher priority first."""
        return self.priority > other.priority


class AdaptiveExplorationStrategy:
    """
    Implements adaptive exploration with multi-signal termination
    and priority-based branch selection.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the adaptive exploration strategy."""
        self.config = config

        # Multi-signal termination weights
        self.termination_weights = {
            'consecutive_failures': 0.25,
            'error_repetition': 0.20,
            'time_budget': 0.15,
            'improvement_rate': 0.15,
            'solution_similarity': 0.15,
            'error_complexity': 0.10
        }

        # Priority calculation weights
        self.priority_weights = {
            'test_improvement': 0.40,
            'error_severity': 0.30,
            'solution_novelty': 0.20,
            'depth_penalty': 0.10
        }

        # Thresholds and limits
        self.termination_threshold = 0.7
        self.base_time_budget = 60  # seconds per depth level
        self.max_total_time = 300  # 5 minutes max per problem
        self.max_candidates = 50  # max candidates per problem

        # Error complexity scores (how hard to fix automatically)
        self.error_complexity_scores = {
            'import_error': 0.8,
            'missing_module': 0.85,
            'recursion_error': 0.7,
            'memory_error': 0.9,
            'syntax_error': 0.3,
            'type_error': 0.4,
            'value_error': 0.45,
            'index_error': 0.5,
            'key_error': 0.5,
            'attribute_error': 0.55,
            'zero_division_error': 0.4,
            'name_error': 0.35,
            'assertion_failure': 0.6,
            'runtime_error': 0.65,
            'other_error': 0.5
        }

        # Error severity scores for priority calculation
        self.error_severity_scores = {
            'assertion_failure': 0.8,  # Close to correct
            'index_error': 0.6,
            'key_error': 0.6,
            'type_error': 0.5,
            'value_error': 0.5,
            'zero_division_error': 0.4,
            'syntax_error': 0.3,
            'name_error': 0.3,
            'import_error': 0.2,
            'recursion_error': 0.2,
            'memory_error': 0.1
        }

        # Statistics tracking
        self.stats = {
            'termination_decisions': [],
            'priority_calculations': [],
            'branch_histories': {},
            'exploration_timeline': []
        }

    def should_terminate_branch(
            self,
            node: Dict[str, Any],
            branch_history: BranchHistory,
            depth: int,
            global_stats: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """
        Multi-signal decision to terminate a branch early.

        Returns:
            Tuple of (should_terminate, confidence_score, primary_reason)
        """
        signals = {}

        # 1. Consecutive failures signal
        failures = branch_history.failures
        threshold = max(1, self.config.get('consecutive_failures_threshold', 3) - depth // 3)
        signals['consecutive_failures'] = min(failures / threshold, 1.0)

        # 2. Error repetition signal
        error_diversity = branch_history.calculate_error_diversity()
        signals['error_repetition'] = 1.0 - error_diversity

        # 3. Time budget signal
        branch_time = time.time() - branch_history.start_time
        max_branch_time = self.base_time_budget * max(1, depth)
        signals['time_budget'] = min(branch_time / max_branch_time, 1.0)

        # 4. Improvement rate signal
        improvement_trend = branch_history.get_improvement_trend()
        # Convert trend to termination signal (negative trend = high signal)
        signals['improvement_rate'] = max(0, -improvement_trend + 0.5)

        # 5. Solution similarity signal
        if len(branch_history.solution_hashes) >= 3:
            recent_hashes = branch_history.solution_hashes[-3:]
            unique_ratio = len(set(recent_hashes)) / len(recent_hashes)
            signals['solution_similarity'] = 1.0 - unique_ratio
        else:
            signals['solution_similarity'] = 0.0

        # 6. Error complexity signal
        if branch_history.error_types:
            recent_error = branch_history.error_types[-1]
            complexity = self.error_complexity_scores.get(recent_error, 0.5)
            signals['error_complexity'] = complexity
        else:
            signals['error_complexity'] = 0.0

        # Calculate weighted confidence score
        confidence_score = sum(
            signals[k] * self.termination_weights[k]
            for k in signals
        )

        # Determine primary reason
        weighted_signals = {
            k: signals[k] * self.termination_weights[k]
            for k in signals
        }
        primary_reason = max(weighted_signals.items(), key=lambda x: x[1])[0]

        # Special case: Always terminate at depth 3 if no improvement
        if depth >= 3 and branch_history.improvements and max(branch_history.improvements) < 0.05:
            return True, 1.0, "depth_limit_no_improvement"

        # Record decision for analysis
        decision_record = {
            'node_id': node['node_id'],
            'depth': depth,
            'signals': signals.copy(),
            'confidence': confidence_score,
            'terminated': confidence_score >= self.termination_threshold,
            'primary_reason': primary_reason if confidence_score >= self.termination_threshold else None,
            'timestamp': time.time()
        }
        self.stats['termination_decisions'].append(decision_record)

        return (
            confidence_score >= self.termination_threshold,
            confidence_score,
            primary_reason
        )

    def calculate_branch_priority(
            self,
            node: Dict[str, Any],
            parent_node: Optional[Dict[str, Any]],
            all_solutions: List[Dict[str, Any]],
            global_stats: Dict[str, Any]
    ) -> float:
        """
        Calculate priority score for exploring a branch.
        Higher scores indicate more promising branches.
        """
        factors = {}

        # 1. Test improvement factor
        if parent_node:
            parent_failed = len(parent_node.get('test_result', {}).get('failed_tests', []))
            current_failed = len(node.get('test_result', {}).get('failed_tests', []))
            if parent_failed > 0:
                factors['test_improvement'] = max(0, (parent_failed - current_failed) / parent_failed)
            else:
                factors['test_improvement'] = 0.0
        else:
            factors['test_improvement'] = 0.0

        # 2. Error severity factor
        error_type = self._categorize_error(node.get('test_result', {}).get('error_message', ''))
        factors['error_severity'] = self.error_severity_scores.get(error_type, 0.5)

        # 3. Solution novelty factor
        node_hash = node.get('solution_hash', '')
        if node_hash and all_solutions:
            # Check similarity to all explored solutions
            similar_count = sum(
                1 for s in all_solutions
                if s.get('solution_hash') == node_hash
            )
            factors['solution_novelty'] = 1.0 / (1.0 + similar_count)
        else:
            factors['solution_novelty'] = 1.0

        # 4. Depth penalty factor
        depth = node.get('depth', 0)
        factors['depth_penalty'] = 1.0 / (1.0 + depth * 0.3)

        # Calculate weighted priority
        priority = sum(
            factors[k] * self.priority_weights[k]
            for k in factors
        )

        # Record calculation for analysis
        calc_record = {
            'node_id': node['node_id'],
            'factors': factors.copy(),
            'priority': priority,
            'timestamp': time.time()
        }
        self.stats['priority_calculations'].append(calc_record)

        return priority

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message (reuse from main pipeline)."""
        if not error_message:
            return "unknown"

        error_patterns = {
            "import_error": ["ModuleNotFoundError", "ImportError"],
            "missing_module": ["No module named"],
            "assertion_failure": ["AssertionError"],
            "index_error": ["IndexError"],
            "type_error": ["TypeError"],
            "value_error": ["ValueError"],
            "key_error": ["KeyError"],
            "attribute_error": ["AttributeError"],
            "zero_division_error": ["ZeroDivisionError"],
            "name_error": ["NameError"],
            "syntax_error": ["SyntaxError"],
            "recursion_error": ["RecursionError", "maximum recursion depth"],
            "memory_error": ["MemoryError", "out of memory"],
            "runtime_error": ["RuntimeError"]
        }

        for error_type, patterns in error_patterns.items():
            if any(pattern in error_message for pattern in patterns):
                return error_type

        return "other_error"

    def get_exploration_summary(self) -> Dict[str, Any]:
        """Generate summary statistics of the exploration."""
        summary = {
            'total_termination_decisions': len(self.stats['termination_decisions']),
            'termination_reasons': defaultdict(int),
            'average_confidence': 0.0,
            'priority_distribution': {},
            'exploration_efficiency': {}
        }

        # Analyze termination decisions
        if self.stats['termination_decisions']:
            terminated = [d for d in self.stats['termination_decisions'] if d['terminated']]
            summary['termination_rate'] = len(terminated) / len(self.stats['termination_decisions'])

            for decision in terminated:
                if decision['primary_reason']:
                    summary['termination_reasons'][decision['primary_reason']] += 1

            confidences = [d['confidence'] for d in self.stats['termination_decisions']]
            summary['average_confidence'] = np.mean(confidences)
            summary['confidence_std'] = np.std(confidences)

        # Analyze priority calculations
        if self.stats['priority_calculations']:
            priorities = [p['priority'] for p in self.stats['priority_calculations']]
            summary['priority_distribution'] = {
                'mean': np.mean(priorities),
                'std': np.std(priorities),
                'min': np.min(priorities),
                'max': np.max(priorities),
                'quartiles': np.percentile(priorities, [25, 50, 75]).tolist()
            }

        return dict(summary)


class BranchPriorityQueue:
    """
    Priority queue for branch exploration with resource budgets.
    """

    def __init__(self, time_budget: float = 300, candidate_budget: int = 50):
        """Initialize the priority queue with resource budgets."""
        self.queue: List[ExplorationNode] = []
        self.time_budget = time_budget
        self.candidate_budget = candidate_budget
        self.start_time = time.time()
        self.candidates_generated = 0
        self.nodes_explored = 0

        # Track exploration order for analysis
        self.exploration_order = []

    def add_node(
            self,
            node: Dict[str, Any],
            priority: float,
            parent_score: float = 1.0
    ):
        """Add a node to the priority queue."""
        exploration_node = ExplorationNode(
            node=node,
            priority=priority * parent_score,  # Decay with parent score
            parent_score=parent_score * 0.9  # Decay factor for children
        )
        heapq.heappush(self.queue, exploration_node)

    def get_next_node(self) -> Optional[Dict[str, Any]]:
        """
        Get the next node to explore, respecting resource budgets.
        """
        # Check time budget
        if time.time() - self.start_time >= self.time_budget:
            logger.info(f"Time budget exhausted ({self.time_budget}s)")
            return None

        # Check candidate budget
        if self.candidates_generated >= self.candidate_budget:
            logger.info(f"Candidate budget exhausted ({self.candidate_budget} candidates)")
            return None

        # Get highest priority node
        if not self.queue:
            return None

        exploration_node = heapq.heappop(self.queue)
        self.nodes_explored += 1

        # Track exploration order
        self.exploration_order.append({
            'node_id': exploration_node.node['node_id'],
            'priority': exploration_node.priority,
            'timestamp': time.time(),
            'queue_size': len(self.queue)
        })

        return exploration_node.node

    def update_stats(self, candidates_generated: int = 1):
        """Update resource usage statistics."""
        self.candidates_generated += candidates_generated

    def get_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        elapsed_time = time.time() - self.start_time
        return {
            'queue_size': len(self.queue),
            'time_elapsed': elapsed_time,
            'time_remaining': max(0, self.time_budget - elapsed_time),
            'candidates_generated': self.candidates_generated,
            'candidates_remaining': max(0, self.candidate_budget - self.candidates_generated),
            'nodes_explored': self.nodes_explored,
            'exploration_order_length': len(self.exploration_order)
        }
