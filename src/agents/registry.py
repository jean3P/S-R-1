# src/agents/registry.py

from typing import Dict, Any, Type, Optional

from src.agents.tree_of_thought_patch_agent import TreeOfThoughtPatchAgent
from src.agents.improved_code_refinement_agent import ImprovedCodeRefinementAgent
from src.agents.tree_of_thought_diagnostic_agent import TreeOfThoughtDiagnosticAgent
from src.agents.base_agent import BaseAgent
from src.utils.logging import get_logger
from src.config.settings import load_config

# Initialize logger
logger = get_logger("agent_registry")

# Registry of available agents
AGENT_REGISTRY = {
    # "patch_refinement": PatchRefinementAgent,
    "improved_code_refinement": ImprovedCodeRefinementAgent,
    "tree_of_thought_diagnostic": TreeOfThoughtDiagnosticAgent,
    "tree_of_thought_patch": TreeOfThoughtPatchAgent,
    # Add more agent types as they are implemented
}


def register_agent(agent_type: str, agent_class: Type[BaseAgent]) -> None:
    """
    Register a new agent type.

    Args:
        agent_type: Type identifier for the agent
        agent_class: Agent class to register
    """
    if agent_type in AGENT_REGISTRY:
        logger.warning(f"Overwriting existing agent type: {agent_type}")

    AGENT_REGISTRY[agent_type] = agent_class
    logger.info(f"Registered agent type: {agent_type}")


def get_agent_class(agent_type: str) -> Optional[Type[BaseAgent]]:
    """
    Get an agent class by type.

    Args:
        agent_type: Type of the agent

    Returns:
        Agent class or None if not found
    """
    if agent_type not in AGENT_REGISTRY:
        logger.error(f"Agent type not found: {agent_type}")
        return None

    return AGENT_REGISTRY[agent_type]


def get_agent(
        agent_id: str,
        model_id: str,
        prompt_id: str,
        evaluator_id: str,
        config: Dict[str, Any] = None
) -> BaseAgent:
    """
    Get an agent instance.

    Args:
        agent_id: ID of the agent
        model_id: ID of the model to use
        prompt_id: ID of the prompt to use
        evaluator_id: ID of the evaluator to use
        config: Agent configuration (optional)

    Returns:
        Agent instance

    Raises:
        ValueError: If agent type is not registered
    """
    # Load agent configuration if not provided
    if config is None:
        try:
            config_path = f"configs/agents/{agent_id}.yaml"
            loaded_config = load_config(config_path)
            config = loaded_config.get("config", {})
        except FileNotFoundError:
            logger.warning(f"Configuration file not found for agent: {agent_id}. Using default configuration.")
            config = {}

    # Get agent type
    agent_type = None

    # Try to get agent type from configuration
    config_path = f"configs/agents/{agent_id}.yaml"
    try:
        loaded_config = load_config(config_path)
        agent_type = loaded_config.get("type")
    except FileNotFoundError:
        # If config file not found, try to infer type from id
        agent_type = agent_id

    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Agent type '{agent_type}' is not registered")

    # Instantiate agent
    agent_class = AGENT_REGISTRY[agent_type]

    logger.info(
        f"Creating agent of type '{agent_type}' with model '{model_id}', prompt '{prompt_id}', and evaluator '{evaluator_id}'")
    agent = agent_class(model_id, prompt_id, evaluator_id, config)

    return agent


def list_available_agents() -> Dict[str, str]:
    """
    List all available agent types with their descriptions.

    Returns:
        Dictionary mapping agent types to their descriptions
    """
    # Get descriptions from docstrings
    descriptions = {}
    for agent_type, agent_class in AGENT_REGISTRY.items():
        doc = agent_class.__doc__ or ""
        # Use the first line of the docstring as the description
        description = doc.split("\n")[0].strip()
        descriptions[agent_type] = description

    return descriptions
