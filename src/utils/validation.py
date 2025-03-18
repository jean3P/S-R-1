# src/utils/validation.py
import os
from typing import Dict, Any, List, Optional
import jsonschema

from src.utils.logging import get_logger

# Initialize logger
logger = get_logger("validation")


def validate_config(config: Dict[str, Any], schema_type: str) -> Dict[str, List[str]]:
    """
    Validate configuration against a schema.

    Args:
        config: Configuration to validate
        schema_type: Type of schema to validate against

    Returns:
        Dictionary of validation errors (empty if valid)
    """
    # Schema definitions for different configuration types
    schemas = {
        "agent": {
            "type": "object",
            "required": ["id", "type"],
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string"},
                "config": {"type": "object"}
            }
        },
        "model": {
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
        },
        "prompt": {
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
        },
        "evaluator": {
            "type": "object",
            "required": ["id", "type"],
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string"},
                "config": {"type": "object"}
            }
        },
        "experiment": {
            "type": "object",
            "required": ["name", "agent", "model", "prompt", "evaluator", "task"],
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "agent": {"type": "object"},
                "model": {"type": "object"},
                "prompt": {"type": "object"},
                "evaluator": {"type": "object"},
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
    }

    if schema_type not in schemas:
        return {"schema_error": [f"Unknown schema type: {schema_type}"]}

    schema = schemas[schema_type]
    errors = {"validation_errors": []}

    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        errors["validation_errors"].append(str(e))

    return errors


def validate_file_path(file_path: str, must_exist: bool = True,
                       file_type: Optional[str] = None) -> List[str]:
    """
    Validate a file path.

    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        file_type: Optional file extension to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check if path is empty
    if not file_path:
        errors.append("File path cannot be empty")
        return errors

    # Check if path has the correct extension
    if file_type and not file_path.endswith(f".{file_type}"):
        errors.append(f"File must have .{file_type} extension")

    # Check if file exists (if required)
    if must_exist and not os.path.exists(file_path):
        errors.append(f"File does not exist: {file_path}")

    return errors


def validate_model_input(prompt: str, model_constraints: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate model input against constraints.

    Args:
        prompt: Input prompt
        model_constraints: Constraints for the model

    Returns:
        Dictionary with validation result and errors
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }

    # Check max length
    max_length = model_constraints.get("max_input_length")
    if max_length and len(prompt) > max_length:
        result["warnings"].append(f"Prompt exceeds maximum length: {len(prompt)} > {max_length}")

    # Check for required prefixes
    required_prefix = model_constraints.get("required_prefix")
    if required_prefix and not prompt.startswith(required_prefix):
        result["errors"].append(f"Prompt must start with: {required_prefix}")
        result["valid"] = False

    # Check for disallowed content
    disallowed_patterns = model_constraints.get("disallowed_patterns", [])
    for pattern in disallowed_patterns:
        if pattern in prompt:
            result["errors"].append(f"Prompt contains disallowed pattern: {pattern}")
            result["valid"] = False

    return result


def validate_python_code(code: str) -> Dict[str, Any]:
    """
    Validate Python code for syntax errors.

    Args:
        code: Python code to validate

    Returns:
        Dictionary with validation result and errors
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }

    try:
        # Compile the code to check for syntax errors
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        result["errors"].append(f"Syntax error: {str(e)}")
        result["valid"] = False
    except Exception as e:
        result["errors"].append(f"Validation error: {str(e)}")
        result["valid"] = False

    # Check for potentially dangerous functions
    potentially_dangerous = ["eval", "exec", "os.system", "subprocess", "__import__"]
    for func in potentially_dangerous:
        if func in code:
            result["warnings"].append(f"Code contains potentially dangerous function: {func}")

    return result


def validate_required_functions(code: str, required_functions: List[str]) -> Dict[str, Any]:
    """
    Validate that code contains required functions.

    Args:
        code: Python code to validate
        required_functions: List of required function names

    Returns:
        Dictionary with validation result and missing functions
    """
    result = {
        "valid": True,
        "missing_functions": []
    }

    # Simple regex to find function definitions
    import re
    for func_name in required_functions:
        # Look for function definition
        pattern = re.compile(r"def\s+" + re.escape(func_name) + r"\s*\(")
        if not pattern.search(code):
            result["missing_functions"].append(func_name)
            result["valid"] = False

    return result
