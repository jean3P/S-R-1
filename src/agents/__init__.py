# src/agents/__init__.py
"""
Self-reflection agents for the AI system.

This package contains different types of agents that can perform self-reflection
to improve their outputs. The main agent types are:

- CodeRefinementAgent: Generates and refines code through self-reflection
- ReasoningAgent: Performs multi-step reasoning to solve complex problems

New agent types can be added by implementing the BaseAgent interface and
registering them with the agent registry.
"""

from src.agents.base_agent import BaseAgent
from src.agents.code_refinement_agent import CodeRefinementAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.registry import (
    register_agent,
    get_agent,
    get_agent_class,
    list_available_agents
)

# Version of the agents package
__version__ = "0.1.0"

__all__ = [
    'BaseAgent',
    'CodeRefinementAgent',
    'ReasoningAgent',
    'register_agent',
    'get_agent',
    'get_agent_class',
    'list_available_agents'
]