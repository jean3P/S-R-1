"""
Enhanced statistics tracking for adaptive exploration.
Provides detailed analysis and persistence of exploration patterns.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdaptiveStatisticsTracker:
    """
    Tracks and analyzes statistics for adaptive exploration strategies.
    """

    def __init__(self, results_dir: Path):
        """Initialize the statistics tracker."""
        self.results_dir = results_dir
        self.stats_dir = results_dir / "adaptive_stats"
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking structures
        self.problem_stats = {}
        self.global_stats = {
            'total_problems': 0,
            'total_time': 0,
            'total_candidates': 0,
            'termination_patterns': defaultdict(int),
            'error_recovery_patterns': defaultdict(list),
            'depth_effectiveness': defaultdict(lambda: {'attempts': 0, 'successes': 0}),
            'model_comparisons': defaultdict(dict)
        }

    def track_problem_exploration(
            self,
            problem_id: str,
            model_name: str,
            result: Dict[str, Any],
            exploration_strategy: Any,
            branch_histories: Dict[str, Any]
    ):
        """Track statistics for a single problem exploration."""
        problem_key = f"{model_name}_{problem_id}"

        # Extract comprehensive statistics
        stats = {
            'problem_id': problem_id,
            'model': model_name,
            'timestamp': time.time(),
            'success': result.get('status') == 'solved',
            'processing_time': result.get('processing_time', 0),
            'total_candidates': result.get('total_candidates', 0),
            'nodes_explored': result.get('nodes_explored', 0),
            'tree_depth': result.get('tree_depth', 0),

            # Adaptive exploration metrics
            'exploration_summary': exploration_strategy.get_exploration_summary(),
            'termination_decisions': self._analyze_terminations(
                exploration_strategy.stats['termination_decisions']
            ),
            'priority_patterns': self._analyze_priorities(
                exploration_strategy.stats['priority_calculations']
            ),
            'branch_productivity': self._calculate_branch_productivity(
                branch_histories, result
            ),

            # Error and recovery analysis
            'error_patterns': self._extract_error_patterns(result),
            'recovery_analysis': self._analyze_recovery_patterns(result),

            # Efficiency metrics
            'efficiency_metrics': self._calculate_efficiency_metrics(result),

            # Solution characteristics
            'solution_diversity': result.get('stats', {}).get('solution_diversity', {}),
            'successful_patterns': self._extract_successful_patterns(result)
        }

        # Store problem-specific stats
        self.problem_stats[problem_key] = stats

        # Update global statistics
        self._update_global_stats(stats)

        # Save to file
        self._save_problem_stats(problem_key, stats)

        return stats

    def _analyze_terminations(self, termination_decisions: List[Dict]) -> Dict[str, Any]:
        """Analyze termination decision patterns."""
        if not termination_decisions:
            return {}

        terminated = [d for d in termination_decisions if d['terminated']]

        analysis = {
            'total_decisions': len(termination_decisions),
            'termination_rate': len(terminated) / len(termination_decisions),
            'reasons': defaultdict(int),
            'confidence_distribution': {
                'mean': np.mean([d['confidence'] for d in termination_decisions]),
                'std': np.std([d['confidence'] for d in termination_decisions]),
                'percentiles': np.percentile(
                    [d['confidence'] for d in termination_decisions],
                    [10, 25, 50, 75, 90]
                ).tolist()
            },
            'depth_distribution': defaultdict(int),
            'signal_importance': self._calculate_signal_importance(termination_decisions)
        }

        # Count termination reasons and depths
        for decision in terminated:
            if decision.get('primary_reason'):
                analysis['reasons'][decision['primary_reason']] += 1
            analysis['depth_distribution'][decision.get('depth', 0)] += 1

        return dict(analysis)

    def _calculate_signal_importance(self, decisions: List[Dict]) -> Dict[str, float]:
        """Calculate the importance of each signal in termination decisions."""
        if not decisions:
            return {}

        # Extract signal values for terminated branches
        terminated_signals = defaultdict(list)
        continued_signals = defaultdict(list)

        for decision in decisions:
            signals = decision.get('signals', {})
            if decision['terminated']:
                for signal, value in signals.items():
                    terminated_signals[signal].append(value)
            else:
                for signal, value in signals.items():
                    continued_signals[signal].append(value)

        # Calculate importance as difference in means
        importance = {}
        for signal in terminated_signals:
            if signal in continued_signals and continued_signals[signal]:
                term_mean = np.mean(terminated_signals[signal])
                cont_mean = np.mean(continued_signals[signal])
                importance[signal] = term_mean - cont_mean
            else:
                importance[signal] = np.mean(terminated_signals[signal])

        return importance

    def _analyze_priorities(self, priority_calculations: List[Dict]) -> Dict[str, Any]:
        """Analyze priority calculation patterns."""
        if not priority_calculations:
            return {}

        priorities = [p['priority'] for p in priority_calculations]
        factors_df = pd.DataFrame([p['factors'] for p in priority_calculations])

        analysis = {
            'total_calculations': len(priority_calculations),
            'priority_stats': {
                'mean': np.mean(priorities),
                'std': np.std(priorities),
                'min': np.min(priorities),
                'max': np.max(priorities),
                'percentiles': np.percentile(priorities, [10, 25, 50, 75, 90]).tolist()
            },
            'factor_correlations': factors_df.corr().to_dict() if len(factors_df) > 1 else {},
            'factor_importance': {
                col: factors_df[col].mean() for col in factors_df.columns
            } if not factors_df.empty else {}
        }

        return analysis

    def _calculate_branch_productivity(
            self,
            branch_histories: Dict[str, Any],
            result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate productivity metrics for branches."""
        if not branch_histories:
            return {}

        productivity_metrics = []

        for node_id, history in branch_histories.items():
            if hasattr(history, 'candidates_generated') and history.candidates_generated > 0:
                # Calculate productivity score
                improvements = sum(1 for imp in history.improvements if imp > 0)
                time_efficiency = 1.0 / (1.0 + history.total_time / 60)  # Normalize by minutes

                productivity = {
                    'node_id': node_id,
                    'candidates': history.candidates_generated,
                    'improvements': improvements,
                    'improvement_rate': improvements / history.candidates_generated,
                    'time_efficiency': time_efficiency,
                    'overall_score': (improvements / history.candidates_generated) * time_efficiency
                }
                productivity_metrics.append(productivity)

        if not productivity_metrics:
            return {}

        # Aggregate metrics
        scores = [p['overall_score'] for p in productivity_metrics]
        return {
            'branch_count': len(productivity_metrics),
            'avg_productivity': np.mean(scores),
            'std_productivity': np.std(scores),
            'top_branches': sorted(
                productivity_metrics,
                key=lambda x: x['overall_score'],
                reverse=True
            )[:5],
            'improvement_distribution': {
                'zero_improvements': sum(1 for p in productivity_metrics if p['improvements'] == 0),
                'some_improvements': sum(1 for p in productivity_metrics if p['improvements'] > 0)
            }
        }

    def _extract_error_patterns(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and analyze error patterns from the exploration."""
        stats = result.get('stats', {})

        patterns = {
            'error_type_frequency': {},
            'error_transitions': stats.get('feedback_impact', {}).get('error_transitions', {}),
            'hardest_test_cases': stats.get('test_case_analysis', {}).get('hardest_cases', {}),
            'first_failing_tests': stats.get('test_case_analysis', {}).get('first_failing_tests', {}),
            'error_resolution_rate': {}
        }

        # Calculate error type frequencies
        if 'feedback_impact' in stats and 'error_types' in stats['feedback_impact']:
            for error_type, data in stats['feedback_impact']['error_types'].items():
                attempts = data.get('attempts', 0)
                improvements = data.get('improvements', 0)
                patterns['error_type_frequency'][error_type] = attempts
                if attempts > 0:
                    patterns['error_resolution_rate'][error_type] = improvements / attempts

        return patterns

    def _analyze_recovery_patterns(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how errors are recovered through the exploration."""
        stats = result.get('stats', {})
        feedback_impact = stats.get('feedback_impact', {})

        recovery_analysis = {
            'depth_effectiveness': {},
            'error_type_recovery': {},
            'test_case_improvements': {}
        }

        # Analyze recovery by depth
        if 'depths' in feedback_impact:
            for depth, data in feedback_impact['depths'].items():
                attempts = data.get('attempts', 0)
                improvements = data.get('improvements', 0)
                solved = data.get('solved', 0)

                if attempts > 0:
                    recovery_analysis['depth_effectiveness'][depth] = {
                        'improvement_rate': improvements / attempts,
                        'solution_rate': solved / attempts,
                        'attempts': attempts
                    }

        # Analyze recovery by error type
        if 'error_types' in feedback_impact:
            for error_type, data in feedback_impact['error_types'].items():
                attempts = data.get('attempts', 0)
                improvements = data.get('improvements', 0)

                if attempts > 0:
                    recovery_analysis['error_type_recovery'][error_type] = {
                        'recovery_rate': improvements / attempts,
                        'attempts': attempts
                    }

        # Test case improvement analysis
        if 'test_case_improvements' in feedback_impact:
            recovery_analysis['test_case_improvements'] = feedback_impact['test_case_improvements']

        return recovery_analysis

    def _calculate_efficiency_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed efficiency metrics."""
        total_time = result.get('processing_time', 1)  # Avoid division by zero
        total_candidates = result.get('total_candidates', 0)
        nodes_explored = result.get('nodes_explored', 0)
        success = result.get('status') == 'solved'

        metrics = {
            'time_per_candidate': total_time / max(1, total_candidates),
            'time_per_node': total_time / max(1, nodes_explored),
            'candidates_per_node': total_candidates / max(1, nodes_explored),
            'exploration_efficiency': 1.0 if success else 0.0,  # Binary for now
            'resource_utilization': {
                'time': total_time,
                'candidates': total_candidates,
                'nodes': nodes_explored
            }
        }

        # Add success-specific metrics
        if success:
            metrics['time_to_solution'] = total_time
            metrics['candidates_to_solution'] = total_candidates
            metrics['nodes_to_solution'] = nodes_explored

        return metrics

    def _extract_successful_patterns(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patterns from successful solutions."""
        if result.get('status') != 'solved':
            return {}

        patterns = {
            'solution_depth': None,
            'solution_characteristics': {},
            'path_to_solution': []
        }

        # Find the successful solution in the tree
        solution_tree = result.get('solution_tree', [])
        for node in solution_tree:
            if node.get('passed'):
                patterns['solution_depth'] = node.get('depth', 0)
                patterns['solution_node_id'] = node.get('node_id')

                # Trace path to solution
                current = node
                path = [current['node_id']]
                while current.get('parent_id'):
                    parent = next((n for n in solution_tree if n['node_id'] == current['parent_id']), None)
                    if parent:
                        path.append(parent['node_id'])
                        current = parent
                    else:
                        break
                patterns['path_to_solution'] = list(reversed(path))
                break

        return patterns

    def _update_global_stats(self, problem_stats: Dict[str, Any]):
        """Update global statistics with problem results."""
        self.global_stats['total_problems'] += 1
        self.global_stats['total_time'] += problem_stats.get('processing_time', 0)
        self.global_stats['total_candidates'] += problem_stats.get('total_candidates', 0)

        # Update termination patterns
        term_analysis = problem_stats.get('termination_decisions', {})
        if 'reasons' in term_analysis:
            for reason, count in term_analysis['reasons'].items():
                self.global_stats['termination_patterns'][reason] += count

        # Update depth effectiveness
        recovery = problem_stats.get('recovery_analysis', {})
        if 'depth_effectiveness' in recovery:
            for depth, data in recovery['depth_effectiveness'].items():
                self.global_stats['depth_effectiveness'][depth]['attempts'] += data.get('attempts', 0)
                self.global_stats['depth_effectiveness'][depth]['successes'] += data.get('attempts', 0) * data.get(
                    'solution_rate', 0)

        # Update model comparisons
        model = problem_stats.get('model', 'unknown')
        if model not in self.global_stats['model_comparisons']:
            self.global_stats['model_comparisons'][model] = {
                'problems': 0,
                'successes': 0,
                'total_time': 0,
                'total_candidates': 0
            }

        self.global_stats['model_comparisons'][model]['problems'] += 1
        if problem_stats.get('success'):
            self.global_stats['model_comparisons'][model]['successes'] += 1
        self.global_stats['model_comparisons'][model]['total_time'] += problem_stats.get('processing_time', 0)
        self.global_stats['model_comparisons'][model]['total_candidates'] += problem_stats.get('total_candidates', 0)

    def _save_problem_stats(self, problem_key: str, stats: Dict[str, Any]):
        """Save problem statistics to file."""
        stats_file = self.stats_dir / f"{problem_key}_adaptive.json"

        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            logger.info(f"Saved adaptive statistics to {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")

    def save_global_summary(self):
        """Save global summary statistics."""
        summary_file = self.stats_dir / "global_adaptive_summary.json"

        # Convert defaultdicts to regular dicts for JSON serialization
        summary = {
            'timestamp': time.time(),
            'total_problems': self.global_stats['total_problems'],
            'total_time': self.global_stats['total_time'],
            'total_candidates': self.global_stats['total_candidates'],
            'termination_patterns': dict(self.global_stats['termination_patterns']),
            'depth_effectiveness': dict(self.global_stats['depth_effectiveness']),
            'model_comparisons': dict(self.global_stats['model_comparisons']),

            # Calculate aggregated metrics
            'aggregate_metrics': self._calculate_aggregate_metrics()
        }

        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Saved global adaptive summary to {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save global summary: {e}")

    def _calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregated metrics across all problems."""
        if self.global_stats['total_problems'] == 0:
            return {}

        metrics = {
            'avg_time_per_problem': self.global_stats['total_time'] / self.global_stats['total_problems'],
            'avg_candidates_per_problem': self.global_stats['total_candidates'] / self.global_stats['total_problems'],
            'termination_reason_distribution': {},
            'model_performance_comparison': {}
        }

        # Calculate termination reason distribution
        total_terminations = sum(self.global_stats['termination_patterns'].values())
        if total_terminations > 0:
            for reason, count in self.global_stats['termination_patterns'].items():
                metrics['termination_reason_distribution'][reason] = count / total_terminations

        # Calculate model performance metrics
        for model, data in self.global_stats['model_comparisons'].items():
            if data['problems'] > 0:
                metrics['model_performance_comparison'][model] = {
                    'success_rate': data['successes'] / data['problems'],
                    'avg_time': data['total_time'] / data['problems'],
                    'avg_candidates': data['total_candidates'] / data['problems']
                }

        return metrics

    def generate_analysis_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        report = {
            'summary': self._calculate_aggregate_metrics(),
            'termination_analysis': self._analyze_termination_effectiveness(),
            'branch_productivity_analysis': self._analyze_branch_productivity(),
            'error_recovery_analysis': self._analyze_error_recovery(),
            'recommendations': self._generate_recommendations()
        }

        return report

    def _analyze_termination_effectiveness(self) -> Dict[str, Any]:
        """Analyze how effective termination decisions were."""
        # This would analyze whether terminated branches would have led to solutions
        # For now, return a placeholder
        return {
            'analysis': 'Termination effectiveness analysis',
            'findings': []
        }

    def _analyze_branch_productivity(self) -> Dict[str, Any]:
        """Analyze overall branch productivity patterns."""
        productivities = []

        for problem_stats in self.problem_stats.values():
            branch_prod = problem_stats.get('branch_productivity', {})
            if 'avg_productivity' in branch_prod:
                productivities.append(branch_prod['avg_productivity'])

        if productivities:
            return {
                'avg_productivity': np.mean(productivities),
                'std_productivity': np.std(productivities),
                'productivity_range': [np.min(productivities), np.max(productivities)]
            }

        return {}

    def _analyze_error_recovery(self) -> Dict[str, Any]:
        """Analyze error recovery patterns across all problems."""
        recovery_rates = defaultdict(list)

        for problem_stats in self.problem_stats.values():
            error_patterns = problem_stats.get('error_patterns', {})
            for error_type, rate in error_patterns.get('error_resolution_rate', {}).items():
                recovery_rates[error_type].append(rate)

        analysis = {}
        for error_type, rates in recovery_rates.items():
            if rates:
                analysis[error_type] = {
                    'avg_recovery_rate': np.mean(rates),
                    'std_recovery_rate': np.std(rates),
                    'sample_size': len(rates)
                }

        return analysis

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on the analysis."""
        recommendations = []

        # Analyze termination patterns
        if self.global_stats['termination_patterns']:
            most_common_termination = max(
                self.global_stats['termination_patterns'].items(),
                key=lambda x: x[1]
            )[0]

            if most_common_termination == 'time_budget':
                recommendations.append(
                    "Consider increasing time budgets for complex problems or implementing "
                    "more aggressive early termination for clearly unsolvable branches."
                )
            elif most_common_termination == 'consecutive_failures':
                recommendations.append(
                    "The consecutive failure threshold appears effective. Consider "
                    "fine-tuning the threshold based on problem difficulty."
                )

        # Analyze model performance
        model_stats = self.global_stats['model_comparisons']
        if len(model_stats) > 1:
            best_model = max(
                model_stats.items(),
                key=lambda x: x[1]['successes'] / x[1]['problems'] if x[1]['problems'] > 0 else 0
            )[0]
            recommendations.append(
                f"Model '{best_model}' shows the best success rate. Consider using it "
                f"as the primary model for similar problems."
            )

        return recommendations
