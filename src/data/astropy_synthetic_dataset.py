# src/data/astropy_synthetic_dataset.py

"""
Implementation Bug Generator

This script creates synthetic test cases by introducing bugs into Astropy's implementation code
rather than the tests themselves. It identifies the implementation code being tested by each test,
introduces bugs there, creates Git branches with the introduced bugs, and generates a comprehensive
CSV dataset.
"""
import argparse
import csv
import subprocess
import os
import re
import random
import difflib
import datetime
import uuid
import ast
import importlib
import inspect

# Define paths
REPO_PATH = '/storage/homefs/jp22b083/SSI/S-R-1/data/repositories/astropy/astropy/astropy/'
TESTS_DIR = os.path.join(REPO_PATH, 'wcs/wcsapi/tests')
IMPL_DIR = os.path.join(REPO_PATH, 'wcs/wcsapi')
ENV_PATH = '/storage/homefs/jp22b083/.conda/envs/astropy-dev/bin/python'
OUTPUT_CSV = 'astropy_implementation_bugs_dataset.csv'

# Dataset fields
FIELDS = [
    'Path_repo',
    'path_env',
    'problem_statement',
    'FAIL_TO_PASS',  # Now contains buggy implementation code
    'PASS_TO_PASS',  # Now contains correct implementation code
    'hint_text',
    'GT_test_patch',  # Diff showing the fix
    'complexity',
    'branch_name',  # Branch name with the bug
    'test_file_path',  # Path to test file
    'test_function_name',  # Name of test function
    'impl_file_path',  # Path to implementation file
    'impl_function_name'  # Name of implementation function
]

# Number of examples to generate
NUM_EXAMPLES = 5  # Default value, will be used if no argument is provided

# Bug types to introduce in implementation code
BUG_TYPES = {
    'easy': [
        'incorrect_return_value',  # Change a return value
        'wrong_operator',  # Change == to != or vice versa
        'off_by_one_error',  # Change a numeric value slightly
    ],
    'moderate': [
        'type_error',  # Change parameter types
        'wrong_parameter_value',  # Use incorrect parameter value
        'wrong_variable_name',  # Change a variable name
        'missing_calculation',  # Comment out a calculation line
    ],
    'complicated': [
        'multiple_bugs',  # Introduce multiple issues
        'complex_condition',  # Change complex conditionals
        'missing_function_call',  # Comment out a function call
        'inverted_logic'  # Invert logic conditions
    ]
}


def run_test(test_path, test_function, env_path=ENV_PATH):
    """
    Run a specific test function and check if it passes or fails

    Parameters:
    -----------
    test_path : str
        Path to the test file
    test_function : str
        Name of the test function to run
    env_path : str
        Path to the Python environment

    Returns:
    --------
    bool
        True if the test passes, False if it fails
    """
    cmd = f"{env_path} -m pytest {test_path}::{test_function} -v"
    result = run_command(cmd)

    # Check if the test passed or failed
    if result and "PASSED" in result:
        return True
    else:
        return False


def validate_bug_causes_failure(branch_name, test_file_path, test_function, repo_path):
    """
    Validate that the introduced bug actually causes the test to fail

    Parameters:
    -----------
    branch_name : str
        Name of the branch with the bug
    test_file_path : str
        Path to the test file
    test_function : str
        Name of the test function
    repo_path : str
        Path to the repository

    Returns:
    --------
    bool
        True if validation is successful (test fails on buggy branch and passes on main),
        False otherwise
    """
    # Get current directory to return to it later
    current_dir = os.getcwd()

    try:
        # Navigate to repository
        root_repo_path = repo_path.rstrip('/')
        # Go up until we find the .git directory
        while not os.path.exists(os.path.join(root_repo_path, '.git')) and root_repo_path:
            root_repo_path = os.path.dirname(root_repo_path)

        if not root_repo_path:
            print(f"Could not find Git repository for {repo_path}")
            return False

        os.chdir(root_repo_path)

        # First check if the test passes on main/master branch
        run_command("git checkout main || git checkout master")
        passes_on_main = run_test(test_file_path, test_function)

        if not passes_on_main:
            print(f"Test {test_function} already fails on main branch!")
            return False

        # Now check if it fails on the buggy branch
        run_command(f"git checkout {branch_name}")
        fails_on_branch = not run_test(test_file_path, test_function)

        # Switch back to main branch
        run_command("git checkout main || git checkout master")

        if fails_on_branch:
            print(f"Validation successful: Test {test_function} passes on main and fails on {branch_name}")
            return True
        else:
            print(f"Validation failed: Test {test_function} does not fail on {branch_name}")
            return False

    except Exception as e:
        print(f"Error during validation: {e}")
        return False

    finally:
        # Navigate back to original directory
        os.chdir(current_dir)


def run_command(cmd, cwd=None):
    """Run a shell command and return its output"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
            capture_output=True,
            cwd=cwd
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        return None


def find_test_files(directory):
    """Find all test files in a directory"""
    cmd = f"find {directory} -name 'test_*.py'"
    output = run_command(cmd)
    if output:
        return output.strip().split('\n')
    return []


def extract_test_functions(file_path):
    """Extract test function names from a file"""
    cmd = f"grep -E '^def test_' {file_path} | sed 's/def \\(test_[^(]*\\).*/\\1/'"
    output = run_command(cmd)
    if output:
        return output.strip().split('\n')
    return []


def read_file_content(file_path):
    """Read the content of a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def find_imports_in_test(file_content):
    """Find import statements in the test file"""
    try:
        tree = ast.parse(file_content)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    imports.append(f"{module}.{name.name}")

        return imports
    except SyntaxError:
        print("Syntax error parsing the file")
        return []


def replace_function_in_file(file_content: str, start_line: int, end_line: int, new_function_code: str) -> str:
    """
    Replaces a block of lines in the file with new content (based on line numbers).
    Lines are 1-indexed.
    """
    file_lines = file_content.splitlines()
    new_lines = new_function_code.splitlines()
    return '\n'.join(file_lines[:start_line - 1] + new_lines + file_lines[end_line:])


def find_imported_modules_in_test(test_file_path):
    """Find astropy modules imported in the test file"""
    content = read_file_content(test_file_path)
    if not content:
        return []

    imports = find_imports_in_test(content)
    # Filter imports to only include astropy modules, focusing on modeling but including others
    astropy_imports = [imp for imp in imports if 'astropy.' in imp]
    # Sort with modeling modules first (they're more likely relevant)
    astropy_imports.sort(key=lambda x: 0 if 'astropy.modeling' in x else 1)
    return astropy_imports


def extract_function_calls_in_test(test_file_path, function_name):
    """
    Extract function calls in a test function to identify implementation code
    Uses more sophisticated AST parsing to identify all types of function calls
    """
    content = read_file_content(test_file_path)
    if not content:
        return []

    try:
        tree = ast.parse(content)
        test_function = None

        # Find the test function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                test_function = node
                break

        if not test_function:
            return []

        # Find function calls in the test function
        function_calls = []
        class_names = []
        variable_types = {}  # Track variable types for better analysis

        for node in ast.walk(test_function):
            # Direct function calls
            if isinstance(node, ast.Call):
                if hasattr(node.func, 'id'):
                    function_calls.append(node.func.id)
                elif hasattr(node.func, 'attr'):
                    function_calls.append(node.func.attr)

                    # Also record class names from instantiations
                    if hasattr(node.func, 'id') and node.func.id not in function_calls and node.func.id not in [
                        'assert', 'print']:
                        class_names.append(node.func.id)

            # Class instantiations (for object-oriented code)
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call) and hasattr(node.value.func, 'id'):
                    class_name = node.value.func.id
                    class_names.append(class_name)

                    # Track variable type if possible
                    if len(node.targets) > 0 and isinstance(node.targets[0], ast.Name):
                        var_name = node.targets[0].id
                        variable_types[var_name] = class_name

            # Method calls on objects
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
                function_calls.append(method_name)

                # If we know the class type, record Class.method for better matching
                if isinstance(node.func.value, ast.Name):
                    var_name = node.func.value.id
                    if var_name in variable_types:
                        class_method = f"{variable_types[var_name]}.{method_name}"
                        function_calls.append(class_method)

        # Filter out common assertion and testing methods
        excluded = ['assert', 'assertEqual', 'assertTrue', 'assertFalse', 'assertRaises',
                    'assertAlmostEqual', 'assertIsInstance', 'assertEquals', 'print', 'range']

        function_calls = [call for call in function_calls if call not in excluded]
        class_names = [name for name in class_names if name not in excluded]

        # Add class names to the list of potential functions being tested
        function_calls.extend(class_names)

        # Remove duplicates
        function_calls = list(set(function_calls))

        return function_calls
    except SyntaxError:
        print(f"Syntax error parsing the test file: {test_file_path}")
        return []
    except Exception as e:
        print(f"Error extracting function calls from {test_file_path}: {e}")
        return []


def find_implementation_file(imported_module, base_path=REPO_PATH):
    """Find the implementation file corresponding to an imported module"""
    # Convert dotted module path to file path
    module_path = imported_module.replace('.', '/')

    # Look for the module file
    potential_paths = [
        f"{base_path}/{module_path}.py",  # Direct module
        f"{base_path}/{module_path}/__init__.py",  # Package
    ]

    # If the module includes a specific part (e.g., astropy.modeling.functional_models)
    # also check for subdirectories with related files
    module_parts = imported_module.split('.')
    if len(module_parts) > 2:
        # Get the parent directory (e.g., astropy/modeling)
        parent_path = '/'.join(module_parts[:-1]).replace('.', '/')
        parent_dir = f"{base_path}/{parent_path}"

        # If it exists, add all Python files in that directory as potential matches
        if os.path.exists(parent_dir) and os.path.isdir(parent_dir):
            for file in os.listdir(parent_dir):
                if file.endswith('.py') and not file.startswith('__'):
                    potential_paths.append(f"{parent_dir}/{file}")

    for path in potential_paths:
        if os.path.exists(path):
            return path

    return None


def extract_implementation_functions(file_path):
    """
    Extract implementation functions and classes from a file
    Includes class methods for better test-implementation matching
    """
    content = read_file_content(file_path)
    if not content:
        return []

    try:
        tree = ast.parse(content)
        functions = []

        # Track which class we're in
        current_class = None

        for node in ast.walk(tree):
            # Record class definitions
            if isinstance(node, ast.ClassDef):
                current_class = node.name
                functions.append(current_class)  # Add the class itself as a potential match

                # Extract methods from the class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        # Skip private methods (but include special methods)
                        if not item.name.startswith('_') or (item.name.startswith('__') and item.name.endswith('__')):
                            # Add both method name and Class.method format
                            functions.append(item.name)
                            functions.append(f"{current_class}.{item.name}")

            # Top-level functions
            elif isinstance(node, ast.FunctionDef) and not hasattr(node, 'parent_class'):
                # Skip private functions
                if not node.name.startswith('_') or (node.name.startswith('__') and node.name.endswith('__')):
                    functions.append(node.name)

        # Look also for any objects that might be created in the file
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith('_'):
                        functions.append(target.id)

        # Remove duplicates and sort
        functions = sorted(list(set(functions)))
        return functions
    except SyntaxError:
        print(f"Syntax error parsing the implementation file: {file_path}")
        return []
    except Exception as e:
        print(f"Error extracting implementation functions from {file_path}: {e}")
        return []


def match_test_to_implementation(test_file_path, test_function_name):
    """
    Match a test function to its corresponding implementation function(s)
    Returns a list of (implementation_file_path, implementation_function_name) tuples
    Uses multiple strategies to find the most likely implementation function
    """
    matches = []

    # Extract test content
    test_content = read_file_content(test_file_path)
    if not test_content:
        return matches

    # Strategy 1: Parse imports and find directly called functions
    imported_modules = find_imported_modules_in_test(test_file_path)
    function_calls = extract_function_calls_in_test(test_file_path, test_function_name)

    # Print for debugging
    print(f"  Imported modules: {imported_modules}")
    print(f"  Function calls: {function_calls}")

    # For each imported module, check if it contains functions that are called in the test
    for module in imported_modules:
        impl_file = find_implementation_file(module)
        if impl_file:
            print(f"  Found implementation file: {impl_file}")
            impl_functions = extract_implementation_functions(impl_file)
            print(f"  Implementation functions: {impl_functions}")
            for func_name in impl_functions:
                if func_name in function_calls:
                    matches.append((impl_file, func_name))

    # Strategy 2: Try to find class instances being created and methods called on them
    try:
        tree = ast.parse(test_content)
        test_func_node = None

        # Find the test function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == test_function_name:
                test_func_node = node
                break

        if test_func_node:
            # Look for class instantiations and method calls
            class_instances = {}  # Map variable names to class names

            for node in ast.walk(test_func_node):
                # Find variable assignments with class instantiations
                if isinstance(node, ast.Assign):
                    if isinstance(node.value, ast.Call) and hasattr(node.value.func, 'id'):
                        class_name = node.value.func.id
                        # Assuming the first target is the variable name
                        if len(node.targets) > 0 and isinstance(node.targets[0], ast.Name):
                            var_name = node.targets[0].id
                            class_instances[var_name] = class_name

                # Find method calls on objects
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        var_name = node.func.value.id
                        method_name = node.func.attr
                        if var_name in class_instances:
                            class_name = class_instances[var_name]
                            # Look for this class in implementation files
                            for module in imported_modules:
                                impl_file = find_implementation_file(module)
                                if impl_file:
                                    # Look for the class and method implementation
                                    content = read_file_content(impl_file)
                                    if content and f"class {class_name}" in content and f"def {method_name}" in content:
                                        matches.append((impl_file, method_name))
    except Exception as e:
        print(f"  Error in AST analysis: {e}")

    # Strategy 3: Name-based matching (if no matches found)
    if not matches:
        # Remove "test_" prefix from test function name
        if test_function_name.startswith('test_'):
            potential_impl_name = test_function_name[5:]
        else:
            potential_impl_name = test_function_name

        print(f"  Using name-based matching with: {potential_impl_name}")

        # Try to find potential files first
        potential_files = []

        # Check for specific naming patterns in file paths
        test_dir_parts = os.path.dirname(test_file_path).split(os.sep)
        if 'tests' in test_dir_parts:
            tests_index = test_dir_parts.index('tests')
            if tests_index > 0:
                # Get the parent module directory
                parent_module = test_dir_parts[tests_index - 1]
                # Look in the parent module directory
                parent_dir = os.path.join(REPO_PATH, parent_module)
                if os.path.exists(parent_dir):
                    for root, dirs, files in os.walk(parent_dir):
                        if 'tests' in root.split(os.sep):
                            continue  # Skip test directories
                        for file in files:
                            if file.endswith('.py') and not file.startswith('test_'):
                                potential_files.append(os.path.join(root, file))

        # Also consider imported modules
        for module in imported_modules:
            impl_file = find_implementation_file(module)
            if impl_file and impl_file not in potential_files:
                potential_files.append(impl_file)

        print(f"  Potential implementation files: {potential_files}")

        # Search through potential files for matching function/class names
        for impl_file in potential_files:
            impl_content = read_file_content(impl_file)
            if not impl_content:
                continue

            # Extract all function and class names
            impl_functions = extract_implementation_functions(impl_file)

            # Look for functions with similar names
            for func_name in impl_functions:
                # Check if implementation function name is similar to test function name
                similarity_score = name_similarity(potential_impl_name, func_name)
                if similarity_score > 0.6:  # Threshold for similarity
                    print(f"  Name match: {func_name} (score: {similarity_score})")
                    matches.append((impl_file, func_name))
                # Also check if the test function name contains the implementation function name
                elif func_name in potential_impl_name:
                    print(f"  Substring match: {func_name} in {potential_impl_name}")
                    matches.append((impl_file, func_name))

    # If still no matches, try to infer from popular model classes in test content
    if not matches:
        # Look for common model classes that might be tested
        model_classes = ['Gaussian1D', 'Polynomial1D', 'Gaussian2D', 'Rotation2D', 'Shift', 'Scale',
                         'Polynomial2D', 'Chebyshev1D', 'Legendre1D', 'Linear1D', 'Sine1D', 'Cosine1D']

        for model_class in model_classes:
            if model_class in test_content:
                # Found a model class being tested, look for its implementation
                for module in imported_modules:
                    if 'model' in module or 'func' in module:  # Focus on model-related modules
                        impl_file = find_implementation_file(module)
                        if impl_file:
                            impl_content = read_file_content(impl_file)
                            if impl_content and f"class {model_class}" in impl_content:
                                # Extract the class methods that might be tested
                                try:
                                    tree = ast.parse(impl_content)
                                    for node in ast.walk(tree):
                                        if isinstance(node, ast.ClassDef) and node.name == model_class:
                                            for sub_node in ast.walk(node):
                                                if isinstance(sub_node,
                                                              ast.FunctionDef) and not sub_node.name.startswith('_'):
                                                    matches.append((impl_file, sub_node.name))
                                except Exception as e:
                                    print(f"  Error in model class analysis: {e}")

    return matches


def name_similarity(name1, name2):
    """Calculate similarity between two names using simple heuristics"""
    if name1 == name2:
        return 1.0

    # Convert to lowercase for comparison
    name1 = name1.lower()
    name2 = name2.lower()

    # Remove common prefixes and suffixes
    prefixes = ['test_', 'get_', 'set_', 'calc_', 'compute_', 'is_']
    for prefix in prefixes:
        if name1.startswith(prefix):
            name1 = name1[len(prefix):]
        if name2.startswith(prefix):
            name2 = name2[len(prefix):]

    # If one is a substring of the other, it's a strong match
    if name1 in name2 or name2 in name1:
        return 0.8

    # Compute Levenshtein distance
    distance = len(set(name1) ^ set(name2))
    max_len = max(len(name1), len(name2))
    if max_len == 0:
        return 0

    similarity = 1 - (distance / max_len)
    return similarity


def extract_function(file_content, function_name):
    """
    Extract a function, class, or class method from file content with precise line numbers
    """
    if not file_content or not function_name:
        return None, None, None

    # Handle class.method notation
    if '.' in function_name:
        class_name, method_name = function_name.split('.', 1)
        # Find the class first
        class_content, class_start, class_end = extract_function(file_content, class_name)
        if class_content:
            # Then find the method within the class
            method_content, method_start_relative, method_end_relative = extract_function(class_content, method_name)
            if method_content and method_start_relative and method_end_relative:
                # Adjust line numbers to be relative to the file
                method_start_absolute = class_start + method_start_relative - 1
                method_end_absolute = class_start + method_end_relative - 1
                return method_content, method_start_absolute, method_end_absolute
        return None, None, None

    # Use AST for precise parsing
    try:
        tree = ast.parse(file_content)
        lines = file_content.splitlines()

        for node in ast.walk(tree):
            # Match function definitions
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                start_line = node.lineno

                # Find the end line by examining indentation
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    end_line = node.end_lineno
                else:
                    # For older Python versions without end_lineno
                    # Find where indentation returns to the same level or less
                    if start_line > len(lines):
                        continue  # Skip if line number is out of bounds

                    function_indent = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
                    end_line = start_line

                    for i in range(start_line, len(lines)):
                        if not lines[i].strip():  # Skip empty lines
                            end_line = i
                            continue

                        current_indent = len(lines[i]) - len(lines[i].lstrip())
                        if current_indent <= function_indent and lines[i].strip():
                            end_line = i - 1
                            break
                        end_line = i

                # Extract function source
                function_source = '\n'.join(lines[start_line - 1:end_line + 1])
                return function_source, start_line, end_line + 1

            # Match class definitions
            elif isinstance(node, ast.ClassDef) and node.name == function_name:
                start_line = node.lineno

                # Find the end line by examining indentation
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    end_line = node.end_lineno
                else:
                    # For older Python versions without end_lineno
                    if start_line > len(lines):
                        continue  # Skip if line number is out of bounds

                    class_indent = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
                    end_line = start_line

                    for i in range(start_line, len(lines)):
                        if i >= len(lines):
                            break  # Avoid index out of range

                        if not lines[i].strip():  # Skip empty lines
                            end_line = i
                            continue

                        current_indent = len(lines[i]) - len(lines[i].lstrip())
                        if current_indent <= class_indent and lines[i].strip():
                            end_line = i - 1
                            break
                        end_line = i

                # Extract class source
                class_source = '\n'.join(lines[start_line - 1:end_line + 1])
                return class_source, start_line, end_line + 1

    except SyntaxError as e:
        print(f"Syntax error parsing file: {e}")
        # Fall back to regex if AST parsing fails

    # Regex fallback
    try:
        lines = file_content.splitlines()

        # For functions
        pattern = rf"def\s+{re.escape(function_name)}\s*\([^)]*\):"
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                start_line = i + 1
                if start_line > len(lines):
                    continue  # Skip if line number is out of bounds

                function_indent = len(line) - len(line.lstrip())
                end_line = start_line

                for j in range(start_line, len(lines)):
                    if j >= len(lines):
                        break  # Avoid index out of range

                    if not lines[j].strip():  # Skip empty lines
                        end_line = j + 1
                        continue

                    current_indent = len(lines[j]) - len(lines[j].lstrip())
                    if current_indent <= function_indent and lines[j].strip():
                        end_line = j
                        break
                    end_line = j + 1

                function_source = '\n'.join(lines[start_line - 1:end_line])
                return function_source, start_line, end_line

        # For classes
        pattern = rf"class\s+{re.escape(function_name)}\s*(?:\([^)]*\))?:"
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                start_line = i + 1
                if start_line > len(lines):
                    continue  # Skip if line number is out of bounds

                class_indent = len(line) - len(line.lstrip())
                end_line = start_line

                for j in range(start_line, len(lines)):
                    if j >= len(lines):
                        break  # Avoid index out of range

                    if not lines[j].strip():  # Skip empty lines
                        end_line = j + 1
                        continue

                    current_indent = len(lines[j]) - len(lines[j].lstrip())
                    if current_indent <= class_indent and lines[j].strip():
                        end_line = j
                        break
                    end_line = j + 1

                class_source = '\n'.join(lines[start_line - 1:end_line])
                return class_source, start_line, end_line

    except Exception as e:
        print(f"Error in regex extraction for {function_name}: {e}")

    return None, None, None


def assess_function_complexity(function_content):
    """Assess the complexity of a function based on its content"""
    complexity_score = 0

    # Check for complex features
    if 'for' in function_content:
        complexity_score += 2
    if 'while' in function_content:
        complexity_score += 3
    if function_content.count('if') > 2:
        complexity_score += 3
    if function_content.count('return') > 2:
        complexity_score += 2
    if len(function_content.split('\n')) > 15:
        complexity_score += 2
    if 'try' in function_content and 'except' in function_content:
        complexity_score += 3
    if 'with' in function_content:
        complexity_score += 1
    if function_content.count('def') > 1:  # Nested functions
        complexity_score += 4
    if function_content.count('class') > 0:  # Class definitions
        complexity_score += 3

    # Determine complexity level
    if complexity_score <= 3:
        return 'easy'
    elif complexity_score <= 7:
        return 'moderate'
    else:
        return 'complicated'


def validate_python_syntax(code_string):
    """Check if a string is valid Python code with proper indentation"""
    try:
        ast.parse(code_string)
        return True
    except SyntaxError as e:
        print(f"Syntax error in generated code: {e}")
        return False


def introduce_bug(function_content, bug_type, complexity):
    """Introduce a specific type of bug into the implementation function while preserving indentation"""
    lines = function_content.splitlines()
    modified_lines = lines.copy()
    bug_introduced = False

    # Easy bugs
    if bug_type == 'incorrect_return_value':
        # Change a return value
        for i, line in enumerate(lines):
            if 'return' in line:
                # Preserve leading whitespace
                leading_whitespace = re.match(r'^\s*', line).group(0)
                content_part = line.lstrip()

                if any(c.isdigit() for c in content_part):
                    # Replace a digit in the return
                    modified_content = ""
                    for char in content_part:
                        if char.isdigit():
                            modified_content += str((int(char) + 1) % 10)  # Increment the digit
                        else:
                            modified_content += char
                    modified_lines[i] = leading_whitespace + modified_content
                    bug_introduced = True
                    break
                elif 'True' in content_part:
                    modified_lines[i] = leading_whitespace + content_part.replace('True', 'False')
                    bug_introduced = True
                    break
                elif 'False' in content_part:
                    modified_lines[i] = leading_whitespace + content_part.replace('False', 'True')
                    bug_introduced = True
                    break

    elif bug_type == 'wrong_operator':
        # Change a comparison operator
        operator_pairs = [('==', '!='), ('!=', '=='), ('>', '<'), ('<', '>'),
                          ('>=', '<='), ('<=', '>='), ('is', 'is not'), ('is not', 'is')]

        for i, line in enumerate(lines):
            # Preserve leading whitespace
            leading_whitespace = re.match(r'^\s*', line).group(0)
            content_part = line.lstrip()

            if 'if' in content_part or 'while' in content_part or 'return' in content_part:
                for old_op, new_op in operator_pairs:
                    if f" {old_op} " in content_part:
                        modified_content = content_part.replace(f" {old_op} ", f" {new_op} ")
                        modified_lines[i] = leading_whitespace + modified_content
                        bug_introduced = True
                        break
                if bug_introduced:
                    break

    elif bug_type == 'off_by_one_error':
        # Introduce an off-by-one error in a calculation or array access
        for i, line in enumerate(lines):
            # Preserve leading whitespace
            leading_whitespace = re.match(r'^\s*', line).group(0)
            content_part = line.lstrip()

            # Check for array access [n]
            match = re.search(r'\[\s*(\d+)\s*\]', content_part)
            if match:
                index = int(match.group(1))
                new_index = index + 1  # Increase index by 1
                modified_content = content_part.replace(f"[{index}]", f"[{new_index}]")
                modified_lines[i] = leading_whitespace + modified_content
                bug_introduced = True
                break

            # Check for range() or similar
            match = re.search(r'range\(\s*(\d+)\s*\)', content_part)
            if match:
                end = int(match.group(1))
                new_end = end - 1  # Decrease range end by 1
                modified_content = content_part.replace(f"range({end})", f"range({new_end})")
                modified_lines[i] = leading_whitespace + modified_content
                bug_introduced = True
                break

            # Check for any number in calculation
            if '+' in content_part or '-' in content_part or '*' in content_part or '/' in content_part:
                match = re.search(r'(\d+)', content_part)
                if match:
                    num = match.group(1)
                    new_num = str(int(num) + 1)
                    modified_content = content_part.replace(num, new_num)
                    modified_lines[i] = leading_whitespace + modified_content
                    bug_introduced = True
                    break

    # Moderate bugs
    elif bug_type == 'type_error':
        # Change parameter types
        for i, line in enumerate(lines):
            # Preserve leading whitespace
            leading_whitespace = re.match(r'^\s*', line).group(0)
            content_part = line.lstrip()

            if '=' in content_part and not content_part.strip().startswith('#'):
                # Check for type conversions
                if 'int(' in content_part:
                    modified_content = content_part.replace('int(', 'str(')
                    modified_lines[i] = leading_whitespace + modified_content
                    bug_introduced = True
                    break
                elif 'float(' in content_part:
                    modified_content = content_part.replace('float(', 'str(')
                    modified_lines[i] = leading_whitespace + modified_content
                    bug_introduced = True
                    break
                # Remove a type conversion
                elif 'int(' in content_part:
                    pattern = r'int\((.*?)\)'
                    modified_content = re.sub(pattern, r'\1', content_part)
                    modified_lines[i] = leading_whitespace + modified_content
                    bug_introduced = True
                    break
                elif 'float(' in content_part:
                    pattern = r'float\((.*?)\)'
                    modified_content = re.sub(pattern, r'\1', content_part)
                    modified_lines[i] = leading_whitespace + modified_content
                    bug_introduced = True
                    break

    elif bug_type == 'wrong_parameter_value':
        # Change a parameter value to an invalid one
        for i, line in enumerate(lines):
            # Preserve leading whitespace
            leading_whitespace = re.match(r'^\s*', line).group(0)
            content_part = line.lstrip()

            if '=' in content_part and any(c.isdigit() for c in content_part) and not content_part.strip().startswith(
                    '#'):
                # Replace positive number with negative or zero
                if re.search(r'=\s*\d+', content_part):
                    modified_content = re.sub(r'=\s*(\d+)', r'= -\1', content_part)
                    modified_lines[i] = leading_whitespace + modified_content
                    bug_introduced = True
                    break
                # Or change a number in any position
                elif re.search(r'\d+', content_part):
                    modified_content = re.sub(r'(\d+)', lambda m: str(int(m.group(1)) + 1), content_part)
                    modified_lines[i] = leading_whitespace + modified_content
                    bug_introduced = True
                    break

    elif bug_type == 'wrong_variable_name':
        # Change a variable name while preserving indentation
        var_names = set()
        for line in lines:
            content_part = line.lstrip()
            if '=' in content_part and not content_part.strip().startswith('#'):
                parts = content_part.split('=', 1)
                var_name = parts[0].strip()
                if var_name and var_name.isidentifier():
                    var_names.add(var_name)

        if var_names:
            # Change a variable name throughout the function
            chosen_var = random.choice(list(var_names))
            new_var = f"{chosen_var}_wrong"

            # Only replace the chosen variable names
            for i, line in enumerate(lines):
                leading_whitespace = re.match(r'^\s*', line).group(0)
                content_part = line.lstrip()

                if chosen_var in content_part:
                    # Make sure we're replacing a variable and not part of a string
                    parts = []
                    in_string = False
                    string_char = None
                    current_part = ""

                    for char in content_part:
                        if in_string:
                            current_part += char
                            if char == string_char:
                                in_string = False
                                parts.append(current_part)
                                current_part = ""
                        else:
                            if char in ['"', "'"]:
                                in_string = True
                                string_char = char
                                if current_part:
                                    parts.append(current_part)
                                current_part = char
                            else:
                                current_part += char

                    if current_part:
                        parts.append(current_part)

                    # Replace variable names in non-string parts
                    new_content = ""
                    for part in parts:
                        if part and part[0] not in ['"', "'"]:
                            # Replace whole words only, not parts of other words
                            pattern = r'\b' + re.escape(chosen_var) + r'\b'
                            part = re.sub(pattern, new_var, part)
                        new_content += part

                    modified_lines[i] = leading_whitespace + new_content
                    bug_introduced = True

    elif bug_type == 'missing_calculation':
        # Comment out a calculation line while preserving indentation
        for i, line in enumerate(lines):
            leading_whitespace = re.match(r'^\s*', line).group(0)
            content_part = line.lstrip()

            if ('=' in content_part and
                    ('+' in content_part or '-' in content_part or '*' in content_part or '/' in content_part) and
                    not content_part.strip().startswith('#') and
                    not 'if' in content_part and
                    not 'for' in content_part and
                    not 'while' in content_part):
                modified_lines[i] = leading_whitespace + "# " + content_part + "  # Bug: commented out calculation"
                bug_introduced = True
                break

    # Complicated bugs
    elif bug_type == 'multiple_bugs':
        # Introduce multiple issues while preserving indentation
        bugs_added = 0

        # First bug: change a return value
        for i, line in enumerate(lines):
            leading_whitespace = re.match(r'^\s*', line).group(0)
            content_part = line.lstrip()

            if 'return' in content_part and not content_part.strip().startswith('#'):
                if any(c.isdigit() for c in content_part):
                    modified_content = re.sub(r'(\d+)', lambda m: str(int(m.group(1)) + 1), content_part)
                    modified_lines[i] = leading_whitespace + modified_content
                    bugs_added += 1
                    break
                elif 'True' in content_part:
                    modified_content = content_part.replace('True', 'False')
                    modified_lines[i] = leading_whitespace + modified_content
                    bugs_added += 1
                    break

        # Second bug: change a calculation
        if bugs_added > 0:
            for i, line in enumerate(lines):
                leading_whitespace = re.match(r'^\s*', line).group(0)
                content_part = line.lstrip()

                if ('=' in content_part and
                        ('+' in content_part or '-' in content_part or '*' in content_part or '/' in content_part) and
                        not content_part.strip().startswith('#') and
                        'return' not in content_part):

                    # Change a + to - or vice versa
                    if '+' in content_part:
                        modified_content = content_part.replace('+', '-')
                    elif '-' in content_part:
                        modified_content = content_part.replace('-', '+')
                    elif '*' in content_part:
                        modified_content = content_part.replace('*', '/')
                    elif '/' in content_part:
                        modified_content = content_part.replace('/', '*')

                    modified_lines[i] = leading_whitespace + modified_content
                    bugs_added += 1
                    break

        bug_introduced = bugs_added > 0

    elif bug_type == 'complex_condition':
        # Change a complex conditional while preserving indentation
        for i, line in enumerate(lines):
            leading_whitespace = re.match(r'^\s*', line).group(0)
            content_part = line.lstrip()

            if 'if' in content_part and ('and' in content_part or 'or' in content_part):
                if 'and' in content_part:
                    modified_content = content_part.replace('and', 'or')
                else:
                    modified_content = content_part.replace('or', 'and')
                modified_lines[i] = leading_whitespace + modified_content
                bug_introduced = True
                break
            elif 'if' in content_part and (
                    '==' in content_part or '!=' in content_part or '>' in content_part or '<' in content_part):
                # Negate the condition
                modified_content = content_part.replace('if ', 'if not ')
                modified_lines[i] = leading_whitespace + modified_content
                bug_introduced = True
                break

    elif bug_type == 'missing_function_call':
        # Comment out an important function call while preserving indentation
        for i, line in enumerate(lines):
            leading_whitespace = re.match(r'^\s*', line).group(0)
            content_part = line.lstrip()

            # Look for function calls (not if/for/while)
            line_stripped = content_part.strip()
            if ('(' in content_part and ')' in content_part and
                    not line_stripped.startswith('if') and
                    not line_stripped.startswith('for') and
                    not line_stripped.startswith('while') and
                    not line_stripped.startswith('def') and
                    not line_stripped.startswith('#')):
                modified_lines[i] = leading_whitespace + "# " + content_part + "  # Bug: commented out function call"
                bug_introduced = True
                break

    elif bug_type == 'inverted_logic':
        # Invert logic in conditionals while preserving indentation
        for i, line in enumerate(lines):
            leading_whitespace = re.match(r'^\s*', line).group(0)
            content_part = line.lstrip()

            if 'if' in content_part:
                # Negate the whole condition
                condition_start = content_part.find('if') + 2
                condition_end = content_part.find(':')
                if condition_end > condition_start:
                    condition = content_part[condition_start:condition_end].strip()
                    new_content = f"{content_part[:condition_start]} not ({condition}){content_part[condition_end:]}"
                    modified_lines[i] = leading_whitespace + new_content
                    bug_introduced = True
                    break

    # If none of the specific strategies worked, fall back to a simple bug
    if not bug_introduced:
        # Comment out the first non-empty line that's not a function definition
        for i, line in enumerate(lines):
            leading_whitespace = re.match(r'^\s*', line).group(0)
            content_part = line.lstrip()

            if content_part.strip() and not content_part.strip().startswith(
                    'def') and not content_part.strip().startswith('#'):
                modified_lines[i] = leading_whitespace + "# " + content_part + "  # Bug: commented out important line"
                bug_introduced = True
                break

    result = '\n'.join(modified_lines)

    # Validate that the result has valid Python syntax
    if not validate_python_syntax(result):
        print("Introduced bug resulted in invalid Python syntax! Trying a simpler approach...")

        # Fall back to a very simple bug - comment out a line
        modified_lines = lines.copy()
        for i, line in enumerate(lines):
            leading_whitespace = re.match(r'^\s*', line).group(0)
            content_part = line.lstrip()

            if (content_part.strip() and
                    not content_part.strip().startswith('def') and
                    not content_part.strip().startswith('class') and
                    not content_part.strip().startswith('#')):
                modified_lines[i] = leading_whitespace + "# " + content_part + "  # Bug: safer fallback"
                break

        result = '\n'.join(modified_lines)

        # Validate again
        if not validate_python_syntax(result):
            print("Even the simplest bug introduction failed! Returning original code...")
            return function_content

    return result


def create_diff(original, modified):
    """Create a diff between two code versions"""
    # Use difflib for more accurate diffs
    diff = difflib.unified_diff(
        original.splitlines(),
        modified.splitlines(),
        fromfile='original',
        tofile='modified',
        lineterm=''
    )

    diff_text = '\n'.join(list(diff))
    return f"```diff\n{diff_text}\n```"


def create_diff_with_validation(
        original: str,
        modified: str,
        file_path: str,
        context_lines: int = 3,
) -> tuple[str, bool]:
    """
    Return a fenced unified-diff between *original* and *modified*, plus a flag
    that tells whether any added line’s indentation looks suspicious.

    Parameters
    ----------
    original, modified
        Full text of the pre- and post-patch file (or snippet).
    file_path
        Path to display in the diff header.  If ``None`` (default) the headers
        are simply ``original`` and ``modified`` so that legacy two-parameter
        call-sites continue to work.
    context_lines
        Number of context lines to keep around the changes (``3`` mimics the
        traditional GNU diff default; set to ``0`` for a “minimal” patch).
    """
    # Choose headers
    if file_path is None:
        from_hdr, to_hdr = "original", "modified"
    else:
        from_hdr, to_hdr = f"a/{file_path}", f"b/{file_path}"

    diff_iter = difflib.unified_diff(
        original.splitlines(),
        modified.splitlines(),
        fromfile=from_hdr,
        tofile=to_hdr,
        lineterm="",
        n=context_lines,
    )
    diff_text = "\n".join(diff_iter)

    # ---------- quick indentation sanity-check ------------------------------
    indentation_problems = False
    body_lines = [
        l for l in original.splitlines()
        if l.strip() and not l.lstrip().startswith("#")
    ]
    if body_lines:
        avg_indent = sum(len(l) - len(l.lstrip()) for l in body_lines) / len(body_lines)
        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                this_indent = len(line) - len(line.lstrip())
                if abs(this_indent - avg_indent) > 8:  # heuristic
                    indentation_problems = True
                    break
    # -----------------------------------------------------------------------

    fenced = f"```diff\n{diff_text}\n```"
    return fenced, indentation_problems


def create_problem_statement(test_function_name, impl_function_name, bug_type, complexity):
    """Create a descriptive problem statement based on bug type and complexity without revealing the implementation"""
    if complexity == 'easy':
        if bug_type == 'incorrect_return_value':
            return f"The test {test_function_name} is failing because a function returns an incorrect value."
        elif bug_type == 'wrong_operator':
            return f"The test {test_function_name} is failing because of an incorrect comparison operator."
        elif bug_type == 'off_by_one_error':
            return f"The test {test_function_name} is failing due to an off-by-one error."

    elif complexity == 'moderate':
        if bug_type == 'type_error':
            return f"The test {test_function_name} is failing because of a type error in a calculation."
        elif bug_type == 'wrong_parameter_value':
            return f"The test {test_function_name} is failing due to an invalid parameter value."
        elif bug_type == 'wrong_variable_name':
            return f"The test {test_function_name} is failing because of a typo in a variable name."
        elif bug_type == 'missing_calculation':
            return f"The test {test_function_name} is failing because a critical calculation is missing."

    elif complexity == 'complicated':
        if bug_type == 'multiple_bugs':
            return f"The test {test_function_name} is failing due to multiple issues, including an incorrect return value and a calculation error."
        elif bug_type == 'complex_condition':
            return f"The test {test_function_name} is failing because of a logic error in a conditional statement."
        elif bug_type == 'missing_function_call':
            return f"The test {test_function_name} is failing because a critical function call is missing."
        elif bug_type == 'inverted_logic':
            return f"The test {test_function_name} is failing due to inverted logic in a conditional statement."

    return f"The test {test_function_name} is failing due to a bug in the code."


def create_hint(bug_type, complexity):
    """Create a hint based on bug type and complexity without revealing the implementation location"""
    if complexity == 'easy':
        if bug_type == 'incorrect_return_value':
            return "Check return values in the code. Something might be returning an incorrect value or calculation."
        elif bug_type == 'wrong_operator':
            return "Look for comparison operators (==, !=, >, <, etc.) that might be incorrect."
        elif bug_type == 'off_by_one_error':
            return "Check for off-by-one errors in calculations, array indices, or loop ranges."

    elif complexity == 'moderate':
        if bug_type == 'type_error':
            return "Examine the types of variables used in calculations. There might be a type mismatch or conversion error."
        elif bug_type == 'wrong_parameter_value':
            return "Look for parameter values that might be invalid or incompatible."
        elif bug_type == 'wrong_variable_name':
            return "Check variable names for typos or inconsistencies."
        elif bug_type == 'missing_calculation':
            return "A critical operation might be commented out or missing."

    elif complexity == 'complicated':
        if bug_type == 'multiple_bugs':
            return "There are multiple issues. Start by checking return values and then look for calculation errors."
        elif bug_type == 'complex_condition':
            return "Examine conditional statements. Logic operations (and/or) might be incorrect, or conditions might be improperly negated."
        elif bug_type == 'missing_function_call':
            return "A critical function call might be commented out or missing."
        elif bug_type == 'inverted_logic':
            return "Check conditional logic that might be inverted or negated incorrectly."

    return "Examine the code carefully and look for syntax or logical errors."


def create_branch_with_bug(repo_path, impl_file_path, impl_function, modified_content, original_content):
    """
    Create a new branch with the modified implementation file (containing the bug)

    Returns the branch name
    """
    # Create a safer function name for branch (remove special characters)
    safe_function_name = re.sub(r'[^a-zA-Z0-9_-]', '_', impl_function)
    if len(safe_function_name) > 30:  # Avoid overly long branch names
        safe_function_name = safe_function_name[:30]

    # Create a unique branch name
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    short_uid = str(uuid.uuid4())[:8]
    branch_name = f"bug-{safe_function_name}-{timestamp}-{short_uid}"

    # Get current directory to return to it later
    current_dir = os.getcwd()

    try:
        # Navigate to repository
        root_repo_path = repo_path.rstrip('/')
        # Go up until we find the .git directory
        while not os.path.exists(os.path.join(root_repo_path, '.git')) and root_repo_path:
            root_repo_path = os.path.dirname(root_repo_path)

        if not root_repo_path:
            print(f"Could not find Git repository for {repo_path}")
            return None

        os.chdir(root_repo_path)

        # Make sure we're on the main branch and up to date
        run_command("git checkout main || git checkout master")
        run_command("git pull")

        # Create new branch
        run_command(f"git checkout -b {branch_name}")

        # Extract file path relative to repository root
        relative_file_path = impl_file_path
        if impl_file_path.startswith(root_repo_path):
            relative_file_path = impl_file_path[len(root_repo_path):].lstrip('/')

        # Read the original file
        file_content = read_file_content(relative_file_path)
        if not file_content:
            print(f"Could not read file {relative_file_path}")
            return None

        # Handle different input types for impl_function (function, class, method)
        if '.' in impl_function:
            # This is likely a class.method notation
            class_name, method_name = impl_function.split('.', 1)
            try:
                # Find the class first
                class_content, class_start, class_end = extract_function(file_content, class_name)
                if not class_content:
                    print(f"Could not find class {class_name} in {relative_file_path}")
                    return None

                # Find the method within the class
                method_content, method_start_relative, method_end_relative = extract_function(class_content,
                                                                                              method_name)
                if not method_content:
                    print(f"Could not find method {method_name} in class {class_name}")
                    return None

                # Adjust start and end lines to be relative to the file
                start_line = class_start + method_start_relative - 1
                end_line = start_line + (method_end_relative - method_start_relative)

                # Make sure we have valid line numbers
                if start_line <= 0 or end_line <= 0 or start_line > len(file_content.splitlines()) or end_line > len(
                        file_content.splitlines()):
                    print(f"Invalid line numbers calculated: {start_line}-{end_line}")
                    return None
            except Exception as e:
                print(f"Error extracting class.method: {e}")
                return None
        else:
            # Try to extract the function or class directly
            function_content, start_line, end_line = extract_function(file_content, impl_function)
            if not function_content or not start_line or not end_line:
                print(f"Could not extract {impl_function} from {relative_file_path}")
                return None

        print(f"Found {impl_function} at lines {start_line}-{end_line}")

        # Make sure we have the content split into lines properly
        original_lines = file_content.splitlines()
        modified_lines = original_lines.copy()

        # Make sure the start and end lines are valid
        if start_line <= 0 or end_line > len(original_lines):
            print(f"Invalid line range: {start_line}-{end_line}, file has {len(original_lines)} lines")
            return None

        # Replace the implementation with the modified version
        try:
            modified_content_lines = modified_content.splitlines()

            # Make sure indentation is preserved when replacing the function
            if len(modified_content_lines) > 0 and len(original_lines) > start_line - 1:
                first_orig_line = original_lines[start_line - 1]
                first_mod_line = modified_content_lines[0]

                # Get original indentation
                orig_indent = re.match(r'^\s*', first_orig_line).group(0)
                mod_indent = re.match(r'^\s*', first_mod_line).group(0)

                # Adjust indentation if needed
                if orig_indent != mod_indent:
                    # Recalculate indentation for all lines in modified content
                    adjusted_lines = []
                    for line in modified_content_lines:
                        if line.strip():  # For non-empty lines
                            # Remove existing indentation and add original indentation level
                            line_content = line.lstrip()
                            line_indent_level = len(line) - len(line_content)

                            if len(mod_indent) > 0:
                                # Calculate relative indentation level
                                indent_ratio = line_indent_level / len(mod_indent)
                                # Apply to original indent
                                new_indent = orig_indent * int(indent_ratio) if indent_ratio > 0 else orig_indent
                                adjusted_lines.append(new_indent + line_content)
                            else:
                                adjusted_lines.append(orig_indent + line_content)
                        else:
                            adjusted_lines.append(line)  # Empty line

                    modified_content_lines = adjusted_lines

            modified_lines[start_line - 1:end_line] = modified_content_lines
            modified_file_content = '\n'.join(modified_lines)

            # Validate the modified file content to ensure it has valid Python syntax
            if not validate_python_syntax(modified_file_content):
                print("Modified file has syntax errors! Using a safer approach...")
                # Fall back to only commenting out a single line
                modified_lines = original_lines.copy()
                for i in range(start_line - 1, end_line):
                    if original_lines[i].strip() and not original_lines[i].strip().startswith('#'):
                        leading_whitespace = re.match(r'^\s*', original_lines[i]).group(0)
                        content_part = original_lines[i].lstrip()
                        modified_lines[i] = leading_whitespace + "# " + content_part + "  # Bug: safer fallback"
                        break

                modified_file_content = '\n'.join(modified_lines)

        except Exception as e:
            print(f"Error replacing content: {e}")
            return None

        # Write the modified content back to the file
        try:
            with open(relative_file_path, 'w', encoding='utf-8') as f:
                f.write(modified_file_content)
        except Exception as e:
            print(f"Error writing modified file: {e}")
            return None

        # Commit the changes
        commit_result = run_command(f"git add {relative_file_path}")
        if commit_result is None:
            print("Error adding file to git")
            return None

        commit_msg_result = run_command(f'git commit -m "Introduce bug in {impl_function} for testing"')
        if commit_msg_result is None:
            print("Error committing changes")
            return None

        # Switch back to main branch
        checkout_result = run_command("git checkout main || git checkout master")
        if checkout_result is None:
            print("Error checking out main branch")
            # Not returning None here since we've already created the branch

        print(f"Created local branch {branch_name} with bug in {impl_function}")
        return branch_name

    except Exception as e:
        print(f"Error creating branch: {e}")
        return None

    finally:
        # Navigate back to original directory
        os.chdir(current_dir)


def main(num_examples=None):
    """
    Main execution function

    Parameters:
    -----------
    num_examples : int, optional
        The maximum number of examples to generate. If not provided,
        uses the global NUM_EXAMPLES value.
    """
    global NUM_EXAMPLES
    if num_examples is not None:
        NUM_EXAMPLES = num_examples

    print(f"Starting implementation bug generator (max examples: {NUM_EXAMPLES})")

    # Find all test files in the modeling/tests directory
    test_files = find_test_files(TESTS_DIR)
    print(f"Found {len(test_files)} test files in the modeling/tests directory")

    # If no test files found, exit
    if not test_files:
        print(f"No test files found in {TESTS_DIR}")
        return

    # First, scan and index all implementation files for better matching
    print("Scanning implementation files...")
    implementation_files = []
    for root, dirs, files in os.walk(IMPL_DIR):
        if 'tests' in root.split(os.sep):
            continue  # Skip test directories
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                implementation_files.append(os.path.join(root, file))

    print(f"Found {len(implementation_files)} implementation files")

    # Build a map of class names and functions to their file locations
    implementation_index = {}
    for impl_file in implementation_files:
        try:
            functions = extract_implementation_functions(impl_file)
            for func in functions:
                if func not in implementation_index:
                    implementation_index[func] = []
                implementation_index[func].append(impl_file)
        except Exception as e:
            print(f"Error indexing file {impl_file}: {e}")

    print(f"Indexed {len(implementation_index)} unique implementation functions/classes")

    # Collect all test functions and their corresponding implementation functions
    test_to_impl_matches = []

    # Process a subset of test files if needed for testing
    # test_files = test_files[:5]  # Uncomment to test with just 5 files

    for test_file_path in test_files:
        test_functions = extract_test_functions(test_file_path)
        # Optional: subset of functions per file for testing
        # test_functions = test_functions[:2]

        for test_function in test_functions:
            print(f"Analyzing test function: {test_function} in {test_file_path}")

            # Match test to implementation
            matches = match_test_to_implementation(test_file_path, test_function)

            # If no matches found through regular means, try using the implementation index
            if not matches:
                print("  No direct matches found, trying implementation index...")
                test_content = read_file_content(test_file_path)

                # Extract potential class/function names from the test
                potential_names = []

                # First, try to extract from the test function name
                if test_function.startswith('test_'):
                    base_name = test_function[5:]  # Remove 'test_' prefix
                    potential_names.append(base_name)

                    # Try different variations
                    parts = base_name.split('_')
                    if len(parts) > 1:
                        # Try camelCase and PascalCase variations
                        camel_case = parts[0] + ''.join(p.capitalize() for p in parts[1:])
                        pascal_case = ''.join(p.capitalize() for p in parts)
                        potential_names.append(camel_case)
                        potential_names.append(pascal_case)

                # Second, look for classes and functions in the test
                for impl_func in implementation_index.keys():
                    # Check if the implementation function/class appears in the test content
                    if impl_func in test_content and len(impl_func) > 3:  # Avoid short names that might be common
                        potential_names.append(impl_func)

                # Try to find matches using the index
                for name in potential_names:
                    if name in implementation_index:
                        impl_files = implementation_index[name]
                        for impl_file in impl_files:
                            matches.append((impl_file, name))
                            print(f"  Found match through index: {name} in {impl_file}")

            if matches:
                for impl_file_path, impl_function in matches:
                    print(f"  Matched to implementation: {impl_function} in {impl_file_path}")

                    # Get implementation function content
                    impl_content = read_file_content(impl_file_path)
                    if impl_content:
                        # Try to extract the function or class definition
                        impl_function_content, start_line, end_line = extract_function(impl_content, impl_function)

                        # If it's a class method (Class.method), try to extract just the method
                        if not impl_function_content and '.' in impl_function:
                            class_name, method_name = impl_function.split('.')
                            # First look for the class
                            class_content, class_start, class_end = extract_function(impl_content, class_name)
                            if class_content:
                                # Then extract the method from within the class
                                method_content, method_start, method_end = extract_function(class_content, method_name)
                                # Adjust the line numbers relative to the file
                                if method_start and method_end:
                                    start_line = class_start + method_start - 1
                                    end_line = start_line + (method_end - method_start)

                        # If extraction failed, try to find the function/class in the file
                        if not impl_function_content:
                            print(f"  Could not extract {impl_function}, trying regex...")
                            # Try to find it using regex
                            if '.' in impl_function:
                                # It's a method, extract just the method name
                                method_name = impl_function.split('.')[-1]
                                pattern = rf"def\s+{re.escape(method_name)}\s*\([^)]*\):.*?(?=\n(?:\s*def|\s*class|\Z))"
                            else:
                                # It's a function or class
                                pattern = rf"(?:def|class)\s+{re.escape(impl_function)}\s*(?:\(.*?\)|\(.*?\):.*?)(?=\n(?:\s*def|\s*class|\Z))"

                            match = re.search(pattern, impl_content, re.DOTALL)
                            if match:
                                impl_function_content = match.group(0)
                                # Estimate line numbers
                                lines_before = impl_content[:match.start()].count('\n') + 1
                                lines_in_function = impl_function_content.count('\n') + 1
                                start_line = lines_before
                                end_line = start_line + lines_in_function - 1

                        if impl_function_content:
                            complexity = assess_function_complexity(impl_function_content)
                            test_to_impl_matches.append({
                                'test_file_path': test_file_path,
                                'test_function': test_function,
                                'impl_file_path': impl_file_path,
                                'impl_function': impl_function,
                                'impl_content': impl_function_content,
                                'start_line': start_line,
                                'end_line': end_line,
                                'complexity': complexity
                            })
                        else:
                            print(f"  Could not extract content for {impl_function}")
                    else:
                        print(f"  Could not read implementation file: {impl_file_path}")

    print(f"Found {len(test_to_impl_matches)} test-to-implementation matches")

    # Group matches by complexity
    matches_by_complexity = {
        'easy': [m for m in test_to_impl_matches if m['complexity'] == 'easy'],
        'moderate': [m for m in test_to_impl_matches if m['complexity'] == 'moderate'],
        'complicated': [m for m in test_to_impl_matches if m['complexity'] == 'complicated']
    }

    print(f"Match distribution by complexity:")
    for complexity, matches in matches_by_complexity.items():
        print(f"  {complexity}: {len(matches)} matches")

    # Generate dataset with balanced complexity
    dataset = []
    examples_per_complexity = {
        'easy': NUM_EXAMPLES // 3 + (1 if NUM_EXAMPLES % 3 > 0 else 0),
        'moderate': NUM_EXAMPLES // 3 + (1 if NUM_EXAMPLES % 3 > 1 else 0),
        'complicated': NUM_EXAMPLES // 3
    }

    # Process matches for each complexity level
    for complexity, target_count in examples_per_complexity.items():
        print(f"\nGenerating {target_count} {complexity} examples")

        # Shuffle matches for randomness
        if matches_by_complexity[complexity]:
            random.shuffle(matches_by_complexity[complexity])

            count = 0
            for match in matches_by_complexity[complexity]:
                if count >= target_count:
                    break

                test_file_path = match['test_file_path']
                test_function = match['test_function']
                impl_file_path = match['impl_file_path']
                impl_function = match['impl_function']
                impl_content = match['impl_content']

                print(f"Processing {complexity} match: {test_function} -> {impl_function}")

                # Choose a bug type for this complexity
                bug_type = random.choice(BUG_TYPES[complexity])
                print(f"  Introducing bug type: {bug_type}")

                # Create buggy version
                buggy_version = introduce_bug(impl_content, bug_type, complexity)

                # Skip if no difference was made
                if buggy_version == impl_content:
                    print(f"  Failed to introduce a bug, skipping")
                    continue

                # Replace function in the full file and compute the correct diff
                full_file_content = read_file_content(impl_file_path)
                diff_text, indentation_problems = create_diff_with_validation(
                    impl_content,
                    buggy_version,
                    os.path.relpath(impl_file_path, REPO_PATH),  # keep path clean and relative
                    context_lines=0
                )

                # Check that the diff is meaningful and doesn't have indentation problems
                if diff_text.count(
                        '\n') <= 5 or indentation_problems:  # Just the diff header lines or indentation issues
                    print(f"  Generated diff is too small or has indentation problems, skipping")
                    continue

                # Validate the buggy code syntax
                if not validate_python_syntax(buggy_version):
                    print(f"  Generated buggy code has syntax errors, skipping")
                    continue

                # Create a branch with the buggy version
                branch_name = create_branch_with_bug(
                    os.path.dirname(REPO_PATH),  # Go up one level to get the repo root
                    impl_file_path,
                    impl_function,
                    buggy_version,
                    impl_content
                )

                # Add validation step
                validation_successful = validate_bug_causes_failure(
                    branch_name,
                    test_file_path,
                    test_function,
                    os.path.dirname(REPO_PATH)
                )

                if not validation_successful:
                    print(f"  Bug does not cause test to fail, skipping")
                    # Remove the branch since it doesn't meet our criteria
                    run_command(f"git branch -D {branch_name}")
                    continue

                if not branch_name:
                    print(f"  Failed to create branch, skipping")
                    continue

                other_passing_tests = []
                for match in test_to_impl_matches:
                    # If it's a different test but uses the same implementation file/function
                    if (match['test_file_path'] != test_file_path or match['test_function'] != test_function) and \
                            (match['impl_file_path'] == impl_file_path and match['impl_function'] == impl_function):
                        other_passing_tests.append((match['test_file_path'], match['test_function']))

                # Choose a different test that passes (if available)
                if other_passing_tests:
                    pass_test_file, pass_test_func = random.choice(other_passing_tests)
                    pass_to_pass = f"{pass_test_file}::{pass_test_func}"
                else:
                    # If no other tests found, use the same test (less ideal but fallback option)
                    pass_to_pass = f"{test_file_path}::{test_function}"

                # Create dataset entry with a more generic problem statement that doesn't reveal the implementation
                entry = {
                    'Path_repo': REPO_PATH,
                    'path_env': ENV_PATH,
                    'problem_statement': create_problem_statement(test_function, impl_function, bug_type, complexity),
                    'FAIL_TO_PASS': f"{test_file_path}::{test_function}",  # Buggy implementation
                    'PASS_TO_PASS': pass_to_pass,  # Correct implementation
                    'hint_text': create_hint(bug_type, complexity),
                    'GT_test_patch': diff_text,
                    'complexity': complexity,
                    'branch_name': branch_name,
                    'test_file_path': test_file_path,
                    'test_function_name': test_function,
                    'impl_file_path': impl_file_path,
                    'impl_function_name': impl_function
                }

                dataset.append(entry)
                count += 1
                print(f"  Added {complexity} example ({count}/{target_count})")
        else:
            print(f"No {complexity} matches found, skipping this complexity level")

    # Save dataset
    if dataset:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(dataset)

        print(f"\nDataset with {len(dataset)} entries saved to {OUTPUT_CSV}")
        complexity_counts = {c: sum(1 for entry in dataset if entry['complexity'] == c)
                             for c in ['easy', 'moderate', 'complicated']}
        print(f"Complexity distribution: {complexity_counts}")
    else:
        print("\nNo dataset entries were created")


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Generate synthetic test cases with implementation bugs')
    parser.add_argument('--num-examples', type=int, default=NUM_EXAMPLES,
                        help=f'Number of examples to generate (default: {NUM_EXAMPLES})')
    parser.add_argument('--repo-path', type=str, default=REPO_PATH,
                        help='Path to the repository')
    parser.add_argument('--tests-dir', type=str, default=TESTS_DIR,
                        help='Path to the tests directory')
    parser.add_argument('--env-path', type=str, default=ENV_PATH,
                        help='Path to the Python environment')
    parser.add_argument('--output', type=str, default=OUTPUT_CSV,
                        help='Output CSV file path')

    args = parser.parse_args()

    # Update global variables based on command-line arguments
    global_num_examples = args.num_examples
    REPO_PATH = args.repo_path
    TESTS_DIR = args.tests_dir
    ENV_PATH = args.env_path
    OUTPUT_CSV = args.output

    print(f"Generating up to {global_num_examples} examples")
    main(global_num_examples)
