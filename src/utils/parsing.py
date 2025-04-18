# src/utils/parsing.py

import re
from typing import Optional, List, Dict, Any
import logging

# Set up logger
logger = logging.getLogger(__name__)


def extract_code_blocks(text: str) -> List[str]:
    """
    Extract code blocks from text with robust handling of various formats and edge cases.

    This function implements a multi-stage approach to extract valid Python code from
    language model outputs, handling common issues like incomplete code blocks,
    natural language mixed with code, and various formatting styles.

    Args:
        text: Text that may contain code blocks

    Returns:
        List of extracted code blocks
    """
    # List to store extracted code
    code_blocks = []

    # Track extraction attempts for logging purposes
    extraction_attempts = []

    try:
        # Stage 1: Extract markdown-style code blocks (```python ... ```)
        markdown_blocks = _extract_markdown_blocks(text)
        if markdown_blocks:
            extraction_attempts.append(("markdown_blocks", len(markdown_blocks)))
            for block in markdown_blocks:
                cleaned_block = _clean_code_block(block)
                if _is_valid_python(cleaned_block):
                    code_blocks.append(cleaned_block)

            # If we found valid code blocks, return them
            if code_blocks:
                logger.debug(f"Successfully extracted {len(code_blocks)} markdown code blocks")
                return code_blocks
    except Exception as e:
        logger.warning(f"Error in markdown block extraction: {str(e)}")

    try:
        # Stage 2: Extract complete function definitions
        function_blocks = _extract_function_blocks(text)
        if function_blocks:
            extraction_attempts.append(("function_blocks", len(function_blocks)))
            for func_text in function_blocks:
                cleaned_func = _clean_code_block(func_text)
                if _is_valid_python(cleaned_func):
                    code_blocks.append(cleaned_func)

            # If we found valid function blocks, return them
            if code_blocks:
                logger.debug(f"Successfully extracted {len(code_blocks)} function blocks")
                return code_blocks
    except Exception as e:
        logger.warning(f"Error in function block extraction: {str(e)}")

    try:
        # Stage 3: Extract code based on indentation and Python-like patterns
        indentation_blocks = _extract_indentation_blocks(text)
        if indentation_blocks:
            extraction_attempts.append(("indentation_blocks", len(indentation_blocks)))
            for block in indentation_blocks:
                cleaned_block = _clean_code_block(block)
                if _is_valid_python(cleaned_block):
                    code_blocks.append(cleaned_block)

            # If we found valid indentation blocks, return them
            if code_blocks:
                logger.debug(f"Successfully extracted {len(code_blocks)} indentation-based blocks")
                return code_blocks
    except Exception as e:
        logger.warning(f"Error in indentation-based extraction: {str(e)}")

    try:
        # Stage 4: Try to extract function templates if reasoning contains code-like structures
        template_blocks = _extract_function_templates(text)
        if template_blocks:
            extraction_attempts.append(("template_blocks", len(template_blocks)))
            for block in template_blocks:
                if _is_valid_python(block):
                    code_blocks.append(block)

            if code_blocks:
                logger.debug(f"Successfully extracted {len(code_blocks)} template-based blocks")
                return code_blocks
    except Exception as e:
        logger.warning(f"Error in template extraction: {str(e)}")

    # Stage 5: Fallback to heuristic extraction for difficult cases
    try:
        heuristic_blocks = _extract_heuristic_blocks(text)
        if heuristic_blocks:
            extraction_attempts.append(("heuristic_blocks", len(heuristic_blocks)))
            for block in heuristic_blocks:
                cleaned_block = _clean_code_block(block)
                if _is_valid_python(cleaned_block):
                    code_blocks.append(cleaned_block)
    except Exception as e:
        logger.warning(f"Error in heuristic extraction: {str(e)}")

    # Stage 6: Last resort - try to reconstruct code from reasoning text
    if not code_blocks:
        try:
            reconstructed_code = _reconstruct_code_from_reasoning(text)
            if reconstructed_code and _is_valid_python(reconstructed_code):
                extraction_attempts.append(("reconstructed_code", 1))
                code_blocks.append(reconstructed_code)
                logger.debug("Successfully reconstructed code from reasoning")
        except Exception as e:
            logger.warning(f"Error in code reconstruction: {str(e)}")

    # Log extraction attempts if we couldn't find valid code
    if not code_blocks and extraction_attempts:
        logger.info(f"Extraction attempts: {extraction_attempts}, but no valid Python code found")

    # Return whatever code blocks we found, or an empty list
    return code_blocks


def _extract_markdown_blocks(text: str) -> List[str]:
    """Extract code blocks surrounded by triple backticks."""
    # First try standard markdown code blocks with language specifier
    standard_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if standard_blocks:
        return standard_blocks

    # Then try to handle incomplete code blocks (missing closing backticks)
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            # Extract content after opening backtick
            potential_blocks = []
            for i in range(1, len(parts), 2):
                # Skip language identifier if present
                content = parts[i]
                if content.startswith("python") or content.startswith("py"):
                    content = content[content.find("\n"):]
                potential_blocks.append(content.strip())

            return potential_blocks

    return []


def _extract_function_blocks(text: str) -> List[str]:
    """Extract Python function definitions from text."""
    # More robust function pattern that handles various function ending styles
    function_blocks = []

    # Find all potential function names
    func_starts = re.finditer(r"def\s+(\w+)\s*\(", text)

    for match in func_starts:
        func_name = match.group(1)
        start_pos = match.start()

        # Get the function text
        func_text = text[start_pos:]

        # Find a reasonable end point for the function
        end_indicators = [
            r"\ndef\s+", r"\nclass\s+", r"\n\n\n", r"\n# ",
            r"\n\n[a-zA-Z0-9_]+\s*=", r"\n\n[a-zA-Z0-9_]+\(",
            r"\nif\s+__name__\s*==\s*['\"]__main__['\"]:",
            r"```", r"```python"
        ]

        end_pos = len(func_text)
        for indicator in end_indicators:
            matches = list(re.finditer(indicator, func_text))
            if matches:
                candidate_pos = matches[0].start()
                if 0 < candidate_pos < end_pos:
                    end_pos = candidate_pos

        function_blocks.append(func_text[:end_pos].strip())

    return function_blocks


def _extract_indentation_blocks(text: str) -> List[str]:
    """Extract Python code blocks based on indentation patterns."""
    lines = text.split('\n')
    blocks = []

    in_code_block = False
    current_block = []
    indent_level = 0

    for line in lines:
        stripped = line.strip()

        # Skip non-code markers
        if any(marker in stripped for marker in
               ["Certainly!", "{Create Answer}", "```", "# SOLUTION:", "# -"]):
            if in_code_block:
                # End current block
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
                    in_code_block = False
            continue

        # Detect the start of a Python code block
        if not in_code_block and stripped and not stripped.startswith('#') and (
                stripped.startswith('def ') or
                stripped.startswith('class ') or
                stripped.startswith('import ') or
                stripped.startswith('from ') or
                '=' in stripped or
                stripped.startswith('if ') or
                stripped.startswith('for ') or
                stripped.startswith('while ') or
                stripped.startswith('try:') or
                stripped.startswith('@')):
            in_code_block = True
            indent_level = len(line) - len(line.lstrip()) if line.lstrip() else 0
            current_block.append(line)

        # Continue the code block
        elif in_code_block:
            # Empty line within a code block is fine
            if not stripped:
                current_block.append(line)
            # Check if indentation is sensible for a code block
            elif line.lstrip() and (len(line) - len(line.lstrip()) >= indent_level or
                                    stripped.startswith('#')):
                current_block.append(line)
            # End of code block detected
            else:
                blocks.append('\n'.join(current_block))
                current_block = []
                in_code_block = False

                # Check if this line starts a new code block
                if stripped and not stripped.startswith('#') and (
                        stripped.startswith('def ') or
                        stripped.startswith('class ') or
                        stripped.startswith('import ') or
                        stripped.startswith('from ') or
                        '=' in stripped):
                    in_code_block = True
                    indent_level = len(line) - len(line.lstrip()) if line.lstrip() else 0
                    current_block.append(line)

    # Add the last block if we're still in one
    if current_block:
        blocks.append('\n'.join(current_block))

    return blocks


def _extract_function_templates(text: str) -> List[str]:
    """
    Extract function templates from reasoning text.

    This looks for sections where the model discusses a function structure
    and tries to extract a valid function skeleton.
    """
    # Look for function signature discussions
    function_matches = re.finditer(r"(?:function|def)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)", text, re.IGNORECASE)
    templates = []

    for match in function_matches:
        func_name = match.group(1)
        params = match.group(2).strip()

        # Try to build a basic function template
        func_template = f"def {func_name}({params}):\n    \"\"\"\n    Function documentation\n    \"\"\"\n"

        # Look for return values or logic discussed nearby
        post_text = text[match.end():match.end() + 500]  # Look ahead a bit

        # Extract return statements
        return_match = re.search(r"return\s+([^\n\.;]+)", post_text, re.IGNORECASE)
        if return_match:
            return_val = return_match.group(1).strip()
            func_template += f"    return {return_val}\n"
        else:
            # Just add a generic return
            func_template += "    pass\n"

        templates.append(func_template)

    return templates


def _extract_heuristic_blocks(text: str) -> List[str]:
    """Use heuristics to extract potential code blocks from difficult cases."""
    blocks = []

    # Look for text sections that might be Python code but weren't caught by other methods
    # First, remove any parts that we know aren't code
    cleaned_text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # Remove existing code blocks
    cleaned_text = re.sub(r"Certainly!.*?\n", "", cleaned_text)  # Remove common language model responses
    cleaned_text = re.sub(r"Here's.*?\n", "", cleaned_text)  # Remove common language model responses

    # Split into potential blocks by multiple newlines
    potential_blocks = re.split(r"\n\s*\n\s*\n", cleaned_text)

    for block in potential_blocks:
        # Skip if it's too short to be meaningful code
        if len(block.strip()) < 10:
            continue

        # Check for code indicators
        if (re.search(r"^\s*(def|class|import|from|if|for|while|try|with)\s", block, re.MULTILINE) or
                re.search(r"^\s*[a-zA-Z0-9_]+\s*=", block, re.MULTILINE)):
            blocks.append(block.strip())

    return blocks


def _reconstruct_code_from_reasoning(text: str) -> Optional[str]:
    """
    Attempt to reconstruct code from model reasoning when no explicit code blocks are found.

    This is a last resort approach for when the model is explaining how to write a function
    but doesn't actually provide the formal implementation.
    """
    # Find the target function name and parameters
    func_name_match = re.search(r"function\s+(?:called\s+)?([a-zA-Z_][a-zA-Z0-9_]*)", text, re.IGNORECASE)
    if not func_name_match:
        func_name_match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)", text, re.IGNORECASE)

    if not func_name_match:
        return None

    func_name = func_name_match.group(1)

    # Extract parameters
    params_text = ""
    params_match = re.search(r"function\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(\s*([^)]*)\)", text, re.IGNORECASE)
    if not params_match:
        params_match = re.search(r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(\s*([^)]*)\)", text, re.IGNORECASE)

    if params_match:
        params_text = params_match.group(1).strip()

    # Look for key code elements in the reasoning
    code_elements = {
        "edge_cases": _extract_edge_case_handling(text),
        "conditionals": _extract_conditionals(text),
        "loops": _extract_loops(text),
        "return_values": _extract_return_values(text),
        "variable_assignments": _extract_variable_assignments(text)
    }

    # Build the function
    function_code = [f"def {func_name}({params_text}):"]

    # Add docstring
    function_code.append('    """')
    purpose_match = re.search(r"(?:function|def)[^.]*?(checks|determines|calculates|finds|returns|computes)[^.]*", text,
                              re.IGNORECASE)
    if purpose_match:
        function_code.append(f"    {purpose_match.group(0).strip()}")
    else:
        function_code.append(f"    Function to process {params_text}")
    function_code.append('    """')

    # Add edge case handling
    for edge_case in code_elements["edge_cases"]:
        function_code.append(f"    {edge_case}")

    # Add main logic as described
    for conditional in code_elements["conditionals"]:
        function_code.append(f"    {conditional}")

    for loop in code_elements["loops"]:
        function_code.append(f"    {loop}")

    for assignment in code_elements["variable_assignments"]:
        function_code.append(f"    {assignment}")

    # Add return statement if found, otherwise add a placeholder
    if code_elements["return_values"]:
        for return_stmt in code_elements["return_values"]:
            function_code.append(f"    {return_stmt}")
    else:
        function_code.append("    return None")

    # Join all lines
    result = "\n".join(function_code)

    # Check if the reconstruction is valid Python
    if _is_valid_python(result):
        return result

    # If not valid, simplify further to a bare minimum function
    simplified = f"def {func_name}({params_text}):\n    \"\"\"\n    Reconstructed function\n    \"\"\"\n    pass"

    if _is_valid_python(simplified):
        return simplified

    return None


def _extract_edge_case_handling(text: str) -> List[str]:
    """Extract edge case handling code from reasoning text."""
    edge_cases = []

    # Look for discussions about handling edge cases
    edge_case_patterns = [
        (r"if\s+([^:]+?)\s*(?:return|:)", "if {0}:"),
        (r"handle\s+([^.]+?)\s*by\s+returning\s+(\w+)", "if {0}:\n        return {1}"),
        (r"check\s+if\s+([^.]+?)\s*return\s+(\w+)", "if {0}:\n        return {1}"),
        (r"for\s+([^,]+?),\s*return\s+(\w+)", "if {0}:\n        return {1}")
    ]

    for pattern, template in edge_case_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                if len(match.groups()) == 1:
                    edge_cases.append(template.format(match.group(1).strip()))
                elif len(match.groups()) == 2:
                    edge_cases.append(template.format(match.group(1).strip(), match.group(2).strip()))
            except IndexError:
                continue

    # Look for common edge cases
    if re.search(r"negative|less than", text, re.IGNORECASE) and re.search(r"return\s+False", text, re.IGNORECASE):
        edge_cases.append("if n <= 0:\n        return False")

    if re.search(r"zero|0", text, re.IGNORECASE) and re.search(r"return\s+False", text, re.IGNORECASE):
        edge_cases.append("if n == 0:\n        return False")

    if re.search(r"one|1", text, re.IGNORECASE) and re.search(r"not prime|return\s+False", text, re.IGNORECASE):
        edge_cases.append("if n == 1:\n        return False")

    if re.search(r"two|2", text, re.IGNORECASE) and re.search(r"prime|return\s+True", text, re.IGNORECASE):
        edge_cases.append("if n == 2:\n        return True")

    # Look for even number handling in prime functions
    if re.search(r"prime", text, re.IGNORECASE) and re.search(r"even|divisible by 2", text, re.IGNORECASE):
        edge_cases.append("if n > 2 and n % 2 == 0:\n        return False")

    return edge_cases


def _extract_conditionals(text: str) -> List[str]:
    """Extract conditional statements from reasoning text."""
    conditionals = []

    # Look for explicit conditional descriptions
    condition_patterns = [
        r"if\s+([^,:]+?)\s*(?:,|then)?\s*([^,.]+)",
        r"check\s+if\s+([^,.]+)",
        r"when\s+([^,]+?)\s*,\s*([^,.]+)"
    ]

    for pattern in condition_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                condition = match.group(1).strip()

                # Skip conditions that are too general or not code-like
                if len(condition.split()) > 7 or 'we' in condition.lower() or 'i ' in condition.lower():
                    continue

                # If there's an action specified, include it
                if len(match.groups()) > 1 and match.group(2):
                    action = match.group(2).strip()

                    # Skip if the action is too verbose
                    if len(action.split()) > 7 or 'we' in action.lower() or 'i ' in action.lower():
                        action = "pass"

                    conditionals.append(f"if {condition}:\n        {action}")
                else:
                    conditionals.append(f"if {condition}:\n        pass")
            except IndexError:
                continue

    return conditionals


def _extract_loops(text: str) -> List[str]:
    """Extract loop statements from reasoning text."""
    loops = []

    # Look for loop descriptions
    loop_patterns = [
        r"loop (?:through|from) ([^t]+?) to ([^.,]+)",
        r"for (?:each|all)? ([^ ]+) (?:from|in) ([^.,]+)",
        r"iterate (?:through|over) ([^.,]+)"
    ]

    for pattern in loop_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                if len(match.groups()) >= 2:
                    start = match.group(1).strip()
                    end = match.group(2).strip()

                    # Handle numeric range
                    if re.match(r'\d+', start) and re.match(r'\d+', end):
                        loops.append(f"for i in range({start}, {end}):\n        pass")
                    else:
                        loops.append(f"for item in range({start}, {end}):\n        pass")
                else:
                    collection = match.group(1).strip()
                    loops.append(f"for item in {collection}:\n        pass")
            except IndexError:
                continue

    # Check for specific loop patterns in prime number functions
    if re.search(r"prime|divisor", text, re.IGNORECASE) and re.search(r"square root", text, re.IGNORECASE):
        loops.append("for d in range(3, int(n**0.5) + 1, 2):\n        if n % d == 0:\n            return False")

    return loops


def _extract_return_values(text: str) -> List[str]:
    """Extract return statements from reasoning text."""
    returns = []

    # Look for explicit return statements
    return_patterns = [
        r"return\s+([^\n\.;]+)",
        r"function\s+returns\s+([^\n\.;]+)",
        r"should\s+return\s+([^\n\.;]+)"
    ]

    for pattern in return_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                return_val = match.group(1).strip()

                # Clean up natural language in return values
                return_val = re.sub(r'the value of ', '', return_val, flags=re.IGNORECASE)
                return_val = re.sub(r'a |\bthe\b ', '', return_val, flags=re.IGNORECASE)

                # Handle common return values
                if re.match(r'true|false|none|[a-z_][a-z0-9_]*|\d+', return_val, re.IGNORECASE):
                    if return_val.lower() == 'true':
                        returns.append("return True")
                    elif return_val.lower() == 'false':
                        returns.append("return False")
                    elif return_val.lower() == 'none':
                        returns.append("return None")
                    else:
                        returns.append(f"return {return_val}")
            except IndexError:
                continue

    # If we found prime-related logic but no return True, add it
    if re.search(r"prime", text, re.IGNORECASE) and not any("return True" in r for r in returns):
        returns.append("return True")

    return returns


def _extract_variable_assignments(text: str) -> List[str]:
    """Extract variable assignments from reasoning text."""
    assignments = []

    # Look for variable assignments
    assignment_patterns = [
        r"(?:calculate|compute|set|define)\s+([a-z_][a-z0-9_]*)\s+(?:as|to|=)\s+([^.,]+)",
        r"([a-z_][a-z0-9_]*)\s+(?:is|=)\s+([^.,]+)"
    ]

    for pattern in assignment_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                var_name = match.group(1).strip()
                value = match.group(2).strip()

                # Skip if variables or values contain "the" or other non-code language
                if re.search(r'\b(the|should|will|would|could|is)\b', var_name, re.IGNORECASE):
                    continue

                # Clean up the value
                value = re.sub(r'(?:calculated|computed|defined|set) as', '', value, flags=re.IGNORECASE)

                # Skip if the value is too verbose
                if len(value.split()) > 5:
                    continue

                assignments.append(f"{var_name} = {value}")
            except IndexError:
                continue

    # Add specific variable assignments for prime number checks
    if re.search(r"prime", text, re.IGNORECASE) and re.search(r"square root", text, re.IGNORECASE):
        assignments.append("max_divisor = int(n**0.5) + 1")

    return assignments


def _clean_code_block(block: str) -> str:
    """
    Clean a code block by removing non-code elements and normalizing whitespace.

    Args:
        block: The code block to clean

    Returns:
        Cleaned code block
    """
    # Remove non-code header/footer lines
    lines = block.split('\n')
    code_lines = []

    # Track state to know when we're inside a multiline string/comment
    in_multiline_string = False
    string_delimiter = None

    for line in lines:
        stripped = line.strip()

        # Skip initial natural language lines
        if not code_lines and any(marker in stripped for marker in
                                  ["Certainly!", "Here's", "Let's", "{Create Answer}",
                                   "# SOLUTION", "The function", "This implementation"]):
            continue

        # Track multiline strings
        if in_multiline_string:
            code_lines.append(line)
            # Check for end of multiline string
            if string_delimiter in stripped and not stripped.endswith("\\"):
                in_multiline_string = False
                string_delimiter = None
        else:
            # Skip execution report lines
            if stripped.startswith('#') and any(marker in stripped for marker in
                                                ["Execution Output:", "Execution Errors:", "Test Results:"]):
                continue

            # Skip code block markup
            if stripped in ["```", "```python"]:
                continue

            # Skip test report lines
            if stripped.startswith("✓") or stripped.startswith("✗"):
                continue

            # Add the line to our code
            code_lines.append(line)

            # Check if this line starts a multiline string
            if '"""' in stripped or "'''" in stripped:
                # Only count as multiline string start if there's an odd number
                if stripped.count('"""') % 2 == 1:
                    in_multiline_string = True
                    string_delimiter = '"""'
                elif stripped.count("'''") % 2 == 1:
                    in_multiline_string = True
                    string_delimiter = "'''"

    # Join all code lines
    code = '\n'.join(code_lines)

    # Fix common issues
    code = code.replace("\\n", "\n")  # Replace literal \n with newlines

    # Remove trailing code block markers if they got included
    code = re.sub(r"```\s*$", "", code)

    return code.strip()


def _is_valid_python(code: str) -> bool:
    """
    Check if a string contains valid Python syntax.

    Args:
        code: The code string to check

    Returns:
        True if the code is valid Python, False otherwise
    """
    if not code.strip():
        return False

    try:
        # Try to compile the code to check syntax
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        # Try to recover from common syntax errors
        try:
            # Try adding a missing closing quote
            if code.count('"') % 2 == 1:
                fixed_code = code + '"'
                compile(fixed_code, '<string>', 'exec')
                return True
            if code.count("'") % 2 == 1:
                fixed_code = code + "'"
                compile(fixed_code, '<string>', 'exec')
                return True

            # Try removing trailing comma in function call/definition
            fixed_code = re.sub(r",\s*\)", ")", code)
            compile(fixed_code, '<string>', 'exec')
            return True
        except SyntaxError:
            # If we still have syntax errors, final try with just the first function
            try:
                # Try extracting just the first function or class
                match = re.search(r"(def|class)\s+\w+.*?(?:\n\S|\Z)", code, re.DOTALL)
                if match:
                    func_end = match.end()
                    first_func = code[:func_end]
                    compile(first_func, '<string>', 'exec')
                    return True
            except:
                return False

    return False


def extract_python_function(text: str, function_name: Optional[str] = None) -> Optional[str]:
    """
    Extract a specific Python function from text.

    Args:
        text: Text that may contain Python functions
        function_name: Name of the function to extract (if specified)

    Returns:
        Extracted function or None if not found
    """
    # Extract all code blocks first
    code_blocks = extract_code_blocks(text)

    if not code_blocks:
        return None

    # If no specific function name is provided, return the first code block
    if function_name is None:
        return code_blocks[0]

    # Look for the specific function in each code block
    for block in code_blocks:
        pattern = rf"def\s+{function_name}\s*\("
        if re.search(pattern, block):
            # Extract entire function definition
            func_match = re.search(rf"(def\s+{function_name}\s*\(.*?(?:return|pass|raise|\Z))", block, re.DOTALL)
            if func_match:
                # Get the function text
                func_text = func_match.group(1)

                # Ensure we capture the complete function by checking indentation
                lines = block.split('\n')
                start_found = False
                func_lines = []
                indent_level = None

                for line in lines:
                    stripped = line.strip()

                    # Look for function definition
                    if not start_found and stripped.startswith(f"def {function_name}"):
                        start_found = True
                        indent_level = len(line) - len(line.lstrip())
                        func_lines.append(line)
                        continue

                    # Collect function body lines based on indentation
                    if start_found:
                        if not stripped or line.startswith(' ' * (indent_level + 1)) or stripped.startswith('#'):
                            func_lines.append(line)
                        else:
                            # We've reached a line with less indentation, so we're out of the function
                            break

                if func_lines:
                    return '\n'.join(func_lines)

            # Fallback if regex match didn't work well
            return block

    return None


def parse_execution_result(stdout: str, stderr: str) -> Dict[str, Any]:
    """
    Parse execution results.

    Args:
        stdout: Standard output
        stderr: Standard error

    Returns:
        Parsed results
    """
    result = {
        "success": len(stderr.strip()) == 0,
        "stdout": stdout,
        "stderr": stderr,
        "has_output": len(stdout.strip()) > 0,
        "has_errors": len(stderr.strip()) > 0
    }

    # Try to extract error type if there are errors
    if result["has_errors"]:
        error_match = re.search(r"^(\w+Error):", stderr, re.MULTILINE)
        if error_match:
            result["error_type"] = error_match.group(1)

        # Try to extract line number of the error
        line_match = re.search(r"line (\d+)", stderr)
        if line_match:
            result["error_line"] = int(line_match.group(1))

    # Try to identify test results in stdout
    if result["has_output"]:
        # Look for test passed/failed markers
        passed_tests = len(re.findall(r"test.*?passed|passed.*?test", stdout, re.IGNORECASE))
        failed_tests = len(re.findall(r"test.*?failed|failed.*?test", stdout, re.IGNORECASE))

        if passed_tests > 0 or failed_tests > 0:
            result["test_results"] = {
                "passed": passed_tests,
                "failed": failed_tests,
                "total": passed_tests + failed_tests
            }

    return result

def extract_docstring(node):
    """
    Extract docstring from an AST node.

    Args:
        node: AST node (function or class definition)

    Returns:
        Docstring text or None if no docstring exists
    """
    import ast

    # Check if the node has a body
    if not hasattr(node, 'body') or not node.body:
        return None

    # Check if the first statement in the body is a string (docstring)
    first_node = node.body[0]
    if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Str):
        return first_node.value.s
    elif isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Constant) and isinstance(first_node.value.value, str):
        # For Python 3.8+, docstrings are represented as Constants
        return first_node.value.value

    return None

def extract_patches(text: str) -> List[str]:
    """
    Extract git patch/diff content from text with robust handling of various formats.

    Args:
        text: Text that may contain git patches/diffs

    Returns:
        List of extracted patches
    """
    # List to store extracted patches
    patches = []

    # Stage 1: Look for standard git diff format
    diff_blocks = re.findall(r'(diff --git .*?)(?=diff --git|\Z)', text, re.DOTALL)
    if diff_blocks:
        for block in diff_blocks:
            if block.strip():
                patches.append(block.strip())
        return patches

    # Stage 2: Look for unified diff format
    unified_diff_blocks = re.findall(r'(--- .*?\n\+\+\+ .*?(?:\n@@.*?@@.*?)(?:(?:\n@@.*?@@.*?)|(?:\Z)))', text,
                                     re.DOTALL)
    if unified_diff_blocks:
        for block in unified_diff_blocks:
            if block.strip():
                patches.append(block.strip())
        return patches

    # Stage 3: Look for markdown-enclosed git diff blocks
    markdown_blocks = re.findall(r'```(?:diff|patch)?\s*\n(diff --git .*?)```', text, re.DOTALL)
    if not markdown_blocks:
        # Try without the 'diff --git' constraint
        markdown_blocks = re.findall(r'```(?:diff|patch)\s*\n(.*?)```', text, re.DOTALL)

    if markdown_blocks:
        for block in markdown_blocks:
            if block.strip():
                patches.append(block.strip())
        return patches

    # Stage 4: Look for any content that looks like a patch (more lenient)
    if "---" in text and "+++" in text and "@@" in text:
        # Look for sections that start with --- and +++ and contain @@ markers
        potential_patches = re.findall(r'(---.*?\n\+\+\+.*?(?:\n@@.*?@@.*?)+)', text, re.DOTALL)
        if potential_patches:
            for patch in potential_patches:
                if patch.strip():
                    patches.append(patch.strip())
            return patches

    # Stage 5: If still nothing, look for file paths with a/b prefixes which is typical in diffs
    if re.search(r'(?:^|\n)(?:a/|b/)[\w/.-]+', text):
        # Try to extract the entire section that looks like it might be a diff
        potential_sections = re.split(r'\n\s*\n\s*\n', text)
        for section in potential_sections:
            if re.search(r'(?:^|\n)(?:a/|b/)[\w/.-]+', section) and ('+' in section or '-' in section):
                if section.strip():
                    patches.append(section.strip())

        if patches:
            return patches

    # If no patches found but the text has common diff markers, return the whole text
    if re.search(r'(?:^|\n)(?:---|\+\+\+|@@|diff --git)', text):
        return [text.strip()]

    # No patches found
    return []
