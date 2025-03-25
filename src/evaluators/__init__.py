"""
Code evaluators for the AI system.
This package contains different types of evaluators that can be used
to execute and assess generated code. The main evaluator types are:
- PythonExecutor: Executes Python code and captures output
- UnitTester: Runs unit tests on Python code
- CodeAnalyzer: Performs static analysis on Python code
New evaluator types can be added by implementing the BaseEvaluator interface and
registering them with the evaluator registry.
"""
from src.evaluators.base_evaluator import BaseEvaluator
from src.evaluators.python_executor import PythonExecutor
from src.evaluators.unit_tester import UnitTester
from src.evaluators.code_analyzer import CodeAnalyzer
from src.evaluators.swe_bench_evaluator import SWEBenchEvaluator  # Import the SWE-bench evaluator

from src.evaluators.registry import (
    register_evaluator,
    get_evaluator,
    get_evaluator_class,
    list_available_evaluators,
    clear_evaluator_cache,
    get_evaluator_configs
)

# Register the evaluators
register_evaluator("python_executor", PythonExecutor)
register_evaluator("unit_tester", UnitTester)
register_evaluator("code_analyzer", CodeAnalyzer)
register_evaluator("swe_bench_evaluator", SWEBenchEvaluator)  # Register the SWE-bench evaluator

# Version of the evaluators package
__version__ = "0.1.0"
__all__ = [
    'BaseEvaluator',
    'PythonExecutor',
    'UnitTester',
    'CodeAnalyzer',
    'SWEBenchEvaluator',
    'register_evaluator',
    'get_evaluator',
    'get_evaluator_class',
    'list_available_evaluators',
    'clear_evaluator_cache',
    'get_evaluator_configs'
]
