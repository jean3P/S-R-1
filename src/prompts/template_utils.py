# src/prompts/template_utils.py

import os
import re
from typing import Dict, Any, Optional, List
from jinja2 import Environment, FileSystemLoader, meta

from src.utils.logging import get_logger

# Initialize logger
logger = get_logger("template_utils")


class TemplateLoader:
    """Utility class for loading templates from files or strings."""

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the template loader.

        Args:
            templates_dir: Optional directory containing template files
        """
        self.templates_dir = templates_dir

        # Set up Jinja environment if templates directory is provided
        if templates_dir and os.path.exists(templates_dir):
            self.jinja_env = Environment(
                loader=FileSystemLoader(templates_dir),
                trim_blocks=True,
                lstrip_blocks=True
            )
        else:
            self.jinja_env = Environment(
                trim_blocks=True,
                lstrip_blocks=True
            )

    def load_template_file(self, template_path: str) -> str:
        """
        Load a template from a file.

        Args:
            template_path: Path to the template file

        Returns:
            Template string

        Raises:
            FileNotFoundError: If the template file does not exist
        """
        if self.templates_dir:
            # Try to load relative to templates directory
            full_path = os.path.join(self.templates_dir, template_path)
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    return f.read()

        # Try with absolute path or relative to current directory
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                return f.read()

        # Template not found
        logger.error(f"Template file not found: {template_path}")
        raise FileNotFoundError(f"Template file not found: {template_path}")

    def render_template(self, template_str: str, variables: Dict[str, Any]) -> str:
        """
        Render a template with variables using Jinja2.

        Args:
            template_str: Template string
            variables: Template variables

        Returns:
            Rendered template
        """
        try:
            template = self.jinja_env.from_string(template_str)
            return template.render(**variables)
        except Exception as e:
            logger.error(f"Error rendering template: {str(e)}")
            return template_str + f"\n\nError rendering template: {str(e)}"

    def get_template_variables(self, template_str: str) -> List[str]:
        """
        Get all variables used in a template.

        Args:
            template_str: Template string

        Returns:
            List of variable names
        """
        try:
            ast = self.jinja_env.parse(template_str)
            return meta.find_undeclared_variables(ast)
        except Exception as e:
            logger.error(f"Error parsing template: {str(e)}")
            return []


def format_string_template(template: str, variables: Dict[str, Any]) -> str:
    """
    Format a template string with variables using Python's string formatting.

    Args:
        template: Template string
        variables: Template variables

    Returns:
        Formatted string
    """
    try:
        return template.format(**variables)
    except KeyError as e:
        logger.warning(f"Missing template variable: {str(e)}")
        # Add placeholder for missing variable
        var_name = str(e).strip("'")
        variables[var_name] = f"[{var_name}]"
        return format_string_template(template, variables)
    except Exception as e:
        logger.error(f"Error formatting template: {str(e)}")
        return template + f"\n\nError formatting template: {str(e)}"


def extract_template_variables(template: str) -> List[str]:
    """
    Extract all variables from a string template.

    Args:
        template: Template string

    Returns:
        List of variable names
    """
    import re

    # Find all placeholders in the template using regex
    placeholders = re.findall(r'{([^{}]*)}', template)

    # Extract variable names (remove formatting specs)
    variables = []
    for placeholder in placeholders:
        # Handle formatting specifiers like {var:02d}
        var_name = placeholder.split(':', 1)[0]
        variables.append(var_name)

    return list(set(variables))


def process_template_includes(template: str, templates_dir: str) -> str:
    """
    Process template include directives.

    Args:
        template: Template string
        templates_dir: Directory containing template files

    Returns:
        Processed template with includes resolved
    """
    include_pattern = re.compile(r'{%\s*include\s+"([^"]+)"\s*%}')

    def replace_include(match):
        include_path = match.group(1)
        full_path = os.path.join(templates_dir, include_path)

        try:
            with open(full_path, 'r') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Error including template {include_path}: {str(e)}")
            return f"[Error including {include_path}: {str(e)}]"

    # Process includes (with recursion)
    while include_pattern.search(template):
        template = include_pattern.sub(replace_include, template)

    return template


def apply_template_filters(text: str, filters: Dict[str, callable]) -> str:
    """
    Apply custom filters to a text.

    Args:
        text: Text to filter
        filters: Dictionary mapping filter names to filter functions

    Returns:
        Filtered text
    """
    # Apply each filter sequentially
    for filter_name, filter_func in filters.items():
        try:
            text = filter_func(text)
        except Exception as e:
            logger.error(f"Error applying filter {filter_name}: {str(e)}")

    return text

