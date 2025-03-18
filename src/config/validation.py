# src/config/validation.py
from typing import Dict, Any, List, Optional, Tuple
from jsonschema import validate, ValidationError
from src.utils.logging import get_logger

# Initialize logger
logger = get_logger("config_validation")

# Schema definitions for different configuration types
AGENT_SCHEMA = {
    "type": "object",
    "required": ["id", "type"],
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string"},
        "config": {"type": "object"}
    }
}

MODEL_SCHEMA = {
    "type": "object",
    "required": ["id", "type"],
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string"},
        "config": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"}
            },
            "required": ["model_name"]
        }
    }
}

PROMPT_SCHEMA = {
    "type": "object",
    "required": ["id", "type"],
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string"},
        "config": {
            "type": "object",
            "properties": {
                "templates": {"type": "object"}
            }
        }
    }
}

EVALUATOR_SCHEMA = {
    "type": "object",
    "required": ["id", "type"],
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string"},
        "config": {"type": "object"}
    }
}

EXPERIMENT_SCHEMA = {
    "type": "object",
    "required": ["name", "agent", "model", "prompt", "evaluator", "task"],
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "agent": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "config": {"type": "object"}
            },
            "required": ["id"]
        },
        "model": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "config": {"type": "object"}
            },
            "required": ["id"]
        },
        "prompt": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "config": {"type": "object"}
            },
            "required": ["id"]
        },
        "evaluator": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "config": {"type": "object"}
            },
            "required": ["id"]
        },
        "task": {
            "type": "object",
            "required": ["name", "initial_prompt"],
            "properties": {
                "name": {"type": "string"},
                "language": {"type": "string"},
                "initial_prompt": {"type": "string"},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "examples": {"type": "array"}
            }
        }
    }
}

# Schema registry
SCHEMA_REGISTRY = {
    "agent": AGENT_SCHEMA,
    "model": MODEL_SCHEMA,
    "prompt": PROMPT_SCHEMA,
    "evaluator": EVALUATOR_SCHEMA,
    "experiment": EXPERIMENT_SCHEMA
}


def validate_config(config: Dict[str, Any], schema_type: str) -> Tuple[bool, List[str]]:
    """
    Validate configuration against a schema.

    Args:
        config: Configuration to validate
        schema_type: Type of schema to validate against

    Returns:
        Tuple of (is_valid, error_messages)
    """
    if schema_type not in SCHEMA_REGISTRY:
        return False, [f"Unknown schema type: {schema_type}"]

    schema = SCHEMA_REGISTRY[schema_type]
    errors = []

    try:
        validate(instance=config, schema=schema)
        return True, []
    except ValidationError as e:
        logger.error(f"Configuration validation error: {str(e)}")
        errors.append(str(e))
        return False, errors


def validate_experiment_config(config: Dict[str, Any]) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Validate a complete experiment configuration.

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (is_valid, errors_by_component)
    """
    errors = {}

    # Validate experiment structure
    is_valid, experiment_errors = validate_config(config, "experiment")
    if not is_valid:
        errors["experiment"] = experiment_errors

    # Validate component references
    for component_type in ["agent", "model", "prompt", "evaluator"]:
        if component_type in config:
            component_config = config[component_type]
            if not isinstance(component_config, dict) or "id" not in component_config:
                errors[component_type] = [f"Invalid {component_type} configuration: missing id"]

    # Validate task
    if "task" in config:
        task = config["task"]
        task_errors = []

        if not isinstance(task, dict):
            task_errors.append("Task must be an object")
        elif "name" not in task:
            task_errors.append("Task must have a name")
        elif "initial_prompt" not in task:
            task_errors.append("Task must have an initial prompt")

        if task_errors:
            errors["task"] = task_errors
    else:
        errors["task"] = ["Task configuration is required"]

    # Overall validation result
    is_valid = len(errors) == 0

    return is_valid, errors


def validate_component_references(config: Dict[str, Any], available_components: Dict[str, List[str]]) -> Dict[
    str, List[str]]:
    """
    Validate component references in an experiment configuration.

    Args:
        config: Experiment configuration
        available_components: Dictionary mapping component types to lists of available component IDs

    Returns:
        Dictionary mapping component types to lists of reference errors
    """
    reference_errors = {}

    for component_type in ["agent", "model", "prompt", "evaluator"]:
        if component_type in config:
            component_id = config[component_type].get("id")

            if component_id:
                available_ids = available_components.get(component_type, [])

                if component_id not in available_ids:
                    reference_errors[component_type] = [
                        f"{component_type.capitalize()} '{component_id}' not found. Available: {', '.join(available_ids)}"
                    ]

    return reference_errors