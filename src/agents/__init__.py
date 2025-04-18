# src/agents/__init__.py

from src.agents.base_agent import BaseAgent
from src.agents.improved_code_refinement_agent import ImprovedCodeRefinementAgent
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
    'register_agent',
    'get_agent',
    'get_agent_class',
    'list_available_agents',
    'ImprovedCodeRefinementAgent',
]