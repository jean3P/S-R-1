# src/context/progressive_disclosure.py

import re
import ast
import math
from typing import Dict, List, Any, Optional, Tuple, Set
from src.utils.logging import get_logger
from src.utils.tokenization import count_tokens


class ProgressiveDisclosure:
    """
    Implements a progressive disclosure pattern for code context:
    - Starts with high-level summaries
    - Provides detail on demand
    - Expands context in areas of focus
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(self.__class__.__name__)
        self.config = config or {}
        self.code_cache = {}  # Cache for file content to avoid repeated disk reads
        self.max_token_per_implementation = self.config.get("max_token_per_implementation", 1000)
        self.max_components_per_query = self.config.get("max_components_per_query", 3)
        self.model_name = self.config.get("model_name", "default")

    def create_initial_context(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create initial context with high-level summaries.

        Args:
            task: Task details

        Returns:
            Dictionary with initial context
        """
        file_index = task.get("file_index", {})
        file_summaries = task.get("file_summaries", {})

        # Start with the project structure
        initial_context = {
            "project_structure": {
                "files": list(file_index.get("files", {}).keys()),
                "main_components": self._extract_main_components(file_index)
            },
            "problem_statement": task.get("description", ""),
            "available_details": {
                "files": True,
                "classes": True,
                "functions": True,
                "implementations": False  # Full implementations not included initially
            }
        }

        # Add high-level file summaries
        file_overview = {}
        for file_path, summary in file_summaries.items():
            file_overview[file_path] = {
                "name": summary.get("file_name", ""),
                "classes": [cls.get("name") for cls in summary.get("classes", [])],
                "functions": [func.get("name") for func in summary.get("functions", [])]
            }

        initial_context["file_overview"] = file_overview

        # Add the most relevant signatures
        problem_statement = task.get("description", "")
        if problem_statement and file_summaries:
            key_components = self._find_key_components_from_description(problem_statement, file_summaries)
            if key_components:
                initial_context["key_components"] = key_components

        return initial_context

    def _extract_main_components(self, file_index: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract main components from file index.

        Args:
            file_index: File index dictionary

        Returns:
            Dictionary with main components
        """
        # Find most important classes and functions based on references
        classes = list(file_index.get("classes", {}).keys())
        functions = list(file_index.get("functions", {}).keys())

        # Score classes by number of methods they contain
        class_scores = {}
        for class_name, class_info in file_index.get("classes", {}).items():
            methods = class_info.get("methods", [])
            class_scores[class_name] = len(methods)

        # Sort classes by score (number of methods)
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        top_classes = [cls[0] for cls in sorted_classes[:5]]

        # Filter out common utility functions (is_*, get_*, etc)
        important_functions = []
        for func_name in functions:
            # Skip getter/setter/utility functions unless there are very few functions
            if len(functions) > 10 and (func_name.startswith('get_') or
                                        func_name.startswith('set_') or
                                        func_name.startswith('is_')):
                continue
            important_functions.append(func_name)
            if len(important_functions) >= 5:
                break

        return {
            "classes": top_classes if top_classes else classes[:5],
            "functions": important_functions if important_functions else functions[:5]
        }

    def _find_key_components_from_description(self, description: str, file_summaries: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find key components that are likely relevant to the task description.

        Args:
            description: Task description
            file_summaries: Dictionary of file summaries

        Returns:
            Dictionary with key components (functions, classes)
        """
        # Extract potential terms from the description
        terms = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', description)
        # Filter out common words and Python keywords
        common_words = {
            'the', 'and', 'or', 'if', 'else', 'elif', 'for', 'while', 'in', 'is', 'not',
            'def', 'class', 'return', 'with', 'as', 'import', 'from', 'print', 'function',
            'method', 'implement', 'fix', 'debug', 'error', 'issue', 'bug', 'problem'
        }
        terms = [term for term in terms if term.lower() not in common_words and len(term) > 2]

        # Score functions and classes by their relevance to the terms
        functions = {}
        classes = {}

        for file_path, summary in file_summaries.items():
            # Check classes
            for cls in summary.get("classes", []):
                cls_name = cls.get("name", "")
                score = self._calculate_relevance_to_terms(cls_name, cls.get("docstring", ""), terms)

                if score > 0:
                    # Include the class's file path and signature
                    signature = f"class {cls_name}"
                    if cls.get("bases"):
                        signature += "(" + ", ".join(cls.get("bases")) + ")"
                    else:
                        signature += "()"

                    classes[cls_name] = {
                        "file_path": file_path,
                        "signature": signature,
                        "methods": [m.get("name") for m in cls.get("methods", [])],
                        "relevance": score
                    }

            # Check functions
            for func in summary.get("functions", []):
                func_name = func.get("name", "")
                score = self._calculate_relevance_to_terms(func_name, func.get("docstring", ""), terms)

                if score > 0:
                    # Create function signature
                    params = ", ".join(func.get("args", []))
                    returns = f" -> {func.get('returns')}" if func.get("returns") else ""
                    signature = f"def {func_name}({params}){returns}"

                    functions[func_name] = {
                        "file_path": file_path,
                        "signature": signature,
                        "relevance": score
                    }

        # Sort by relevance and take top 3 each
        sorted_functions = sorted(functions.items(), key=lambda x: x[1]["relevance"], reverse=True)
        sorted_classes = sorted(classes.items(), key=lambda x: x[1]["relevance"], reverse=True)

        return {
            "functions": {name: info for name, info in sorted_functions[:3]},
            "classes": {name: info for name, info in sorted_classes[:3]}
        }

    def _calculate_relevance_to_terms(self, name: str, docstring: str, terms: List[str]) -> float:
        """
        Calculate how relevant a component is to the search terms.

        Args:
            name: Component name
            docstring: Component docstring
            terms: Search terms

        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = 0.0

        # Check name matches (exact match is highly relevant)
        for term in terms:
            if term.lower() == name.lower():
                score += 1.0
            elif term.lower() in name.lower():
                score += 0.5

        # Check docstring for term mentions
        if docstring:
            docstring_lower = docstring.lower()
            for term in terms:
                term_lower = term.lower()
                matches = docstring_lower.count(term_lower)
                if matches > 0:
                    # Logarithmic scaling to prevent high counts from dominating
                    score += 0.2 * math.log(1 + matches)

        return score

    def respond_to_query(self,
                         query: str,
                         context: Dict[str, Any],
                         file_summaries: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Respond to a query for more information.

        Args:
            query: Query string
            context: Current context
            file_summaries: Dictionary of file summaries

        Returns:
            Tuple of (response, new_context)
        """
        # Parse the query to understand what information is being requested
        query_type = self._categorize_query(query)
        query_targets = self._extract_query_targets(query)

        # Limit the number of targets to handle
        if len(query_targets) > self.max_components_per_query:
            query_targets = query_targets[:self.max_components_per_query]

        self.logger.info(f"Query type: {query_type}, targets: {query_targets}")

        response = ""
        new_context = context.copy()

        if query_type == "implementation":
            # Model is asking for implementation details
            for target in query_targets:
                implementation = self._find_implementation(target, file_summaries)
                if implementation:
                    response += f"\nImplementation of {target}:\n```python\n{implementation}\n```\n"
                    # Add to context
                    if "implementations" not in new_context:
                        new_context["implementations"] = {}
                    new_context["implementations"][target] = implementation
                else:
                    response += f"\nCould not find implementation for '{target}'.\n"

        elif query_type == "signature":
            # Model is asking for function/method signatures
            for target in query_targets:
                signature = self._find_signature(target, file_summaries)
                if signature:
                    response += f"\nSignature of {target}: `{signature}`\n"
                    # Add to context
                    if "signatures" not in new_context:
                        new_context["signatures"] = {}
                    new_context["signatures"][target] = signature
                else:
                    response += f"\nCould not find signature for '{target}'.\n"

        elif query_type == "relationship":
            # Model is asking about relationships between components
            for target in query_targets:
                relationships = self._find_relationships(target, file_summaries)
                if relationships:
                    response += f"\nRelationships for {target}:\n{relationships}\n"
                    # Add to context
                    if "relationships" not in new_context:
                        new_context["relationships"] = {}
                    new_context["relationships"][target] = relationships
                else:
                    response += f"\nCould not find relationship information for '{target}'.\n"

        else:
            # General request
            # Try to infer what they might be looking for
            for target in query_targets:
                # First check for implementations
                implementation = self._find_implementation(target, file_summaries)
                if implementation:
                    response += f"\nHere's the implementation of {target}:\n```python\n{implementation}\n```\n"
                    if "implementations" not in new_context:
                        new_context["implementations"] = {}
                    new_context["implementations"][target] = implementation
                    continue

                # Then check for signatures
                signature = self._find_signature(target, file_summaries)
                if signature:
                    response += f"\nSignature of {target}: `{signature}`\n"
                    if "signatures" not in new_context:
                        new_context["signatures"] = {}
                    new_context["signatures"][target] = signature
                    continue

                # Finally check for relationships
                relationships = self._find_relationships(target, file_summaries)
                if relationships:
                    response += f"\nRelationships for {target}:\n{relationships}\n"
                    if "relationships" not in new_context:
                        new_context["relationships"] = {}
                    new_context["relationships"][target] = relationships
                    continue

                response += f"\nCould not find information about '{target}'.\n"

            if not query_targets:
                response = "I need more specific information about what code details you need. You can ask for implementations, signatures, or relationships between components."

        return response, new_context

    def _categorize_query(self, query: str) -> str:
        """
        Categorize the type of query.

        Args:
            query: Query string

        Returns:
            Query type
        """
        query = query.lower()

        # Check for implementation-related terms
        if any(term in query for term in [
            "implementation", "code", "how is", "source", "definition",
            "show me the code", "full code", "implemented", "function body",
            "method body", "code for", "implementation of"
        ]):
            return "implementation"

        # Check for signature-related terms
        elif any(term in query for term in [
            "signature", "parameters", "arguments", "return", "inputs",
            "what parameters", "function header", "method signature",
            "api", "interface", "takes", "returns", "declaration"
        ]):
            return "signature"

        # Check for relationship-related terms
        elif any(term in query for term in [
            "relationship", "connected", "calls", "uses", "depends",
            "imported", "references", "inheritance", "inherits",
            "subclass", "parent", "child", "derived", "base", "extends"
        ]):
            return "relationship"

        # Default to general query
        else:
            return "general"

    def _extract_query_targets(self, query: str) -> List[str]:
        """
        Extract the targets (functions, classes, etc.) from the query.

        Args:
            query: Query string

        Returns:
            List of target names
        """
        # Look for quoted names (highest priority)
        quoted = re.findall(r'[\'"`](.*?)[\'"`]', query)

        # Look for Python-style identifiers with namespace notation (like module.Class.method)
        namespaced = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\b', query)

        # Look for function or method calls
        calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', query)

        # Look for potential code identifiers (class names tend to be CamelCase)
        class_names = re.findall(r'\b([A-Z][a-zA-Z0-9_]*)\b', query)

        # Look for other potential identifiers
        identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', query)

        # Combine and filter common words
        common_words = {
            'implementation', 'code', 'function', 'class', 'method',
            'of', 'the', 'a', 'an', 'in', 'for', 'show', 'get', 'find',
            'what', 'how', 'why', 'where', 'is', 'are', 'details', 'about',
            'me', 'signature', 'relationship', 'source', 'body', 'definition',
            'tell', 'explain', 'provide', 'give', 'need', 'please', 'would',
            'could', 'can', 'want', 'like', 'more', 'information'
        }

        # Prepare all potential targets with order of precedence
        all_targets = []

        # Quoted targets have highest priority
        all_targets.extend(quoted)

        # Namespaced identifiers have next priority
        all_targets.extend(namespaced)

        # Function/method calls have next priority
        all_targets.extend(calls)

        # Class names have next priority
        all_targets.extend(class_names)

        # Other identifiers have lowest priority
        all_targets.extend(identifiers)

        # Filter out common words and ensure minimum length
        targets = [t for t in all_targets if t.lower() not in common_words and len(t) > 2]

        # Remove duplicates while preserving order
        unique_targets = []
        seen = set()
        for target in targets:
            if target not in seen:
                seen.add(target)
                unique_targets.append(target)

        return unique_targets

    def _read_file_content(self, file_path: str) -> Optional[str]:
        """
        Read file content, using cache if available.

        Args:
            file_path: Path to the file

        Returns:
            File content or None if file cannot be read
        """
        # Check cache first
        if file_path in self.code_cache:
            return self.code_cache[file_path]

        # Attempt to read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Cache the content for future use
                self.code_cache[file_path] = content
                return content
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return None

    def _find_implementation_in_file(self, target_name: str, file_path: str) -> Optional[str]:
        """
        Find implementation of target in a file.

        Args:
            target_name: Name of the target to find
            file_path: Path to the file

        Returns:
            Implementation or None if not found
        """
        content = self._read_file_content(file_path)
        if not content:
            return None

        try:
            # Parse the file
            tree = ast.parse(content)

            # Handle module.Class.method or module.function notation
            target_parts = target_name.split('.')
            if len(target_parts) > 1:
                # This is a namespaced reference
                if len(target_parts) == 2:
                    # Could be module.function or module.Class
                    module_name, target = target_parts
                    # First look for Class
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name == target:
                            code_lines = content.splitlines()[node.lineno - 1:node.end_lineno]
                            return '\n'.join(code_lines)

                    # Then look for function
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name == target:
                            code_lines = content.splitlines()[node.lineno - 1:node.end_lineno]
                            return '\n'.join(code_lines)

                elif len(target_parts) == 3:
                    # This is module.Class.method
                    module_name, class_name, method_name = target_parts

                    # Find the class
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name == class_name:
                            # Find the method in the class
                            for child in node.body:
                                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                                    code_lines = content.splitlines()[child.lineno - 1:child.end_lineno]
                                    return '\n'.join(code_lines)

            # Direct class or function reference
            for node in ast.walk(tree):
                if ((isinstance(node, ast.ClassDef) or isinstance(node, ast.FunctionDef))
                        and node.name == target_name):
                    code_lines = content.splitlines()[node.lineno - 1:node.end_lineno]
                    return '\n'.join(code_lines)

            return None

        except Exception as e:
            self.logger.error(f"Error parsing file {file_path}: {e}")
            return None

    def _find_implementation(self, target: str, file_summaries: Dict[str, Any]) -> Optional[str]:
        """
        Find implementation of a component.

        Args:
            target: Target name to find
            file_summaries: Dictionary of file summaries

        Returns:
            Implementation code or None if not found
        """
        # If target includes a namespace (module.function or class.method)
        if '.' in target:
            parts = target.split('.')

            # Handle class.method pattern
            if len(parts) == 2:
                class_name, method_name = parts

                # Look for the class in all files
                for file_path, summary in file_summaries.items():
                    for cls in summary.get("classes", []):
                        if cls.get("name") == class_name:
                            # Check if the method exists in this class
                            for method in cls.get("methods", []):
                                if method.get("name") == method_name:
                                    # Found the method, get its implementation
                                    impl = self._find_implementation_in_file(
                                        f"{class_name}.{method_name}", file_path)
                                    if impl:
                                        # Check token length and truncate if needed
                                        tokens = count_tokens(impl, self.model_name)
                                        if tokens > self.max_token_per_implementation:
                                            # Truncate implementation
                                            lines = impl.split('\n')
                                            middle_idx = len(lines) // 2
                                            top_part = '\n'.join(lines[:middle_idx // 2])
                                            bottom_part = '\n'.join(lines[-(middle_idx // 2):])
                                            return f"{top_part}\n\n# ... [implementation truncated] ...\n\n{bottom_part}"
                                        return impl

        # Look for direct match in classes and functions
        for file_path, summary in file_summaries.items():
            # Check for class with this name
            for cls in summary.get("classes", []):
                if cls.get("name") == target:
                    impl = self._find_implementation_in_file(target, file_path)
                    if impl:
                        # Check token length and truncate if needed
                        tokens = count_tokens(impl, self.model_name)
                        if tokens > self.max_token_per_implementation:
                            # Truncate implementation
                            lines = impl.split('\n')
                            middle_idx = len(lines) // 2
                            top_part = '\n'.join(lines[:middle_idx // 2])
                            bottom_part = '\n'.join(lines[-(middle_idx // 2):])
                            return f"{top_part}\n\n# ... [implementation truncated] ...\n\n{bottom_part}"
                        return impl

            # Check for function with this name
            for func in summary.get("functions", []):
                if func.get("name") == target:
                    impl = self._find_implementation_in_file(target, file_path)
                    if impl:
                        # Check token length and truncate if needed
                        tokens = count_tokens(impl, self.model_name)
                        if tokens > self.max_token_per_implementation:
                            # Truncate implementation
                            lines = impl.split('\n')
                            middle_idx = len(lines) // 2
                            top_part = '\n'.join(lines[:middle_idx // 2])
                            bottom_part = '\n'.join(lines[-(middle_idx // 2):])
                            return f"{top_part}\n\n# ... [implementation truncated] ...\n\n{bottom_part}"
                        return impl

        return None

    def _find_signature(self, target: str, file_summaries: Dict[str, Any]) -> Optional[str]:
        """
        Find signature of a component.

        Args:
            target: Target name to find
            file_summaries: Dictionary of file summaries

        Returns:
            Signature string or None if not found
        """
        # Handle namespaced targets
        if '.' in target:
            parts = target.split('.')

            # Handle class.method pattern
            if len(parts) == 2:
                class_name, method_name = parts

                # Look for the class in all files
                for file_path, summary in file_summaries.items():
                    for cls in summary.get("classes", []):
                        if cls.get("name") == class_name:
                            # Check if the method exists in this class
                            for method in cls.get("methods", []):
                                if method.get("name") == method_name:
                                    # Construct method signature
                                    args_str = ", ".join(method.get("args", []))
                                    returns = f" -> {method.get('returns')}" if method.get("returns") else ""
                                    return f"def {method_name}({args_str}){returns}"

        # Look for direct match in classes and functions
        for file_path, summary in file_summaries.items():
            # Check for class with this name
            for cls in summary.get("classes", []):
                if cls.get("name") == target:
                    # Construct class signature
                    bases = ", ".join(cls.get("bases", []))
                    return f"class {target}({bases})"

            # Check for function with this name
            for func in summary.get("functions", []):
                if func.get("name") == target:
                    # Construct function signature
                    args_str = ", ".join(func.get("args", []))
                    returns = f" -> {func.get('returns')}" if func.get("returns") else ""
                    return f"def {target}({args_str}){returns}"

        return None

    def _find_imports_in_file(self, file_path: str) -> List[str]:
        """
        Find import statements in a file.

        Args:
            file_path: Path to the file

        Returns:
            List of import statements
        """
        content = self._read_file_content(file_path)
        if not content:
            return []

        try:
            # Parse the file
            tree = ast.parse(content)

            imports = []
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(f"import {name.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for name in node.names:
                        imports.append(f"from {module} import {name.name}")

            return imports
        except Exception as e:
            self.logger.error(f"Error parsing file {file_path}: {e}")
            return []

    def _find_function_calls_in_impl(self, implementation: str) -> Set[str]:
        """
        Find function calls in an implementation.

        Args:
            implementation: Implementation code

        Returns:
            Set of function names that are called
        """
        if not implementation:
            return set()

        try:
            # Parse the code
            tree = ast.parse(implementation)

            function_calls = set()
            # Extract function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        function_calls.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        # Handle method calls (e.g., obj.method())
                        function_calls.add(node.func.attr)

            return function_calls
        except Exception as e:
            # Fallback to regex for partial code that might not parse
            calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', implementation)
            return set(calls)

    def _find_relationships(self, target: str, file_summaries: Dict[str, Any]) -> Optional[str]:
        """
        Find relationships for a component.

        Args:
            target: Target name to find
            file_summaries: Dictionary of file summaries

        Returns:
            Relationship description or None if not found
        """
        result = []

        # Get information about where this component is defined
        definition_info = None
        component_type = None
        file_path = None

        # Find where the target is defined
        for path, summary in file_summaries.items():
            # Check classes
            for cls in summary.get("classes", []):
                if cls.get("name") == target:
                    component_type = "class"
                    file_path = path
                    definition_info = cls
                    break

                # Check methods in this class
                for method in cls.get("methods", []):
                    if method.get("name") == target:
                        component_type = "method"
                        file_path = path
                        definition_info = method
                        parent_class = cls.get("name")
                        result.append(f"Method '{target}' is defined in class '{parent_class}'")
                        break

            # Check functions
            if not definition_info:
                for func in summary.get("functions", []):
                    if func.get("name") == target:
                        component_type = "function"
                        file_path = path
                        definition_info = func
                        break

            if definition_info:
                break

        if not definition_info:
            return None

        # Add file information
        if file_path:
            result.append(f"Defined in file: {file_path}")

            # Add import information
            imports = self._find_imports_in_file(file_path)
            if imports:
                result.append("Imports:")
                for imp in imports[:5]:  # Limit to 5 imports
                    result.append(f"  - {imp}")
                if len(imports) > 5:
                    result.append(f"  - ... {len(imports) - 5} more imports")

        # Handle class relationships
        if component_type == "class":
            # List base classes
            if definition_info.get("bases"):
                result.append("Inherits from:")
                for base in definition_info.get("bases", []):
                    result.append(f"  - {base}")

            # List methods
            if definition_info.get("methods"):
                methods = definition_info.get("methods", [])
                result.append(f"Contains {len(methods)} methods:")
                for method in methods[:5]:  # Limit to 5 methods
                    result.append(f"  - {method.get('name')}")
                if len(methods) > 5:
                    result.append(f"  - ... {len(methods) - 5} more methods")

            # Find subclasses (classes that inherit from this one)
            subclasses = []
            for path, summary in file_summaries.items():
                for cls in summary.get("classes", []):
                    if target in cls.get("bases", []):
                        subclasses.append(cls.get("name"))

            if subclasses:
                result.append("Subclasses:")
                for subclass in subclasses:
                    result.append(f"  - {subclass}")

        # Handle function/method relationships
        elif component_type in ["function", "method"]:
            # Get function implementation
            implementation = self._find_implementation(target, file_summaries)

            if implementation:
                # Find what this function calls
                function_calls = self._find_function_calls_in_impl(implementation)
                if function_calls:
                    result.append("Calls:")
                    for call in list(function_calls)[:5]:  # Limit to 5 calls
                        result.append(f"  - {call}()")
                    if len(function_calls) > 5:
                        result.append(f"  - ... {len(function_calls) - 5} more function calls")

                # Find what other functions call this function
                callers = []
                for path, summary in file_summaries.items():
                    for cls in summary.get("classes", []):
                        for method in cls.get("methods", []):
                            method_impl = self._find_implementation(f"{cls.get('name')}.{method.get('name')}",
                                                                    file_summaries)
                            if method_impl and target in self._find_function_calls_in_impl(method_impl):
                                callers.append(f"{cls.get('name')}.{method.get('name')}")

                    for func in summary.get("functions", []):
                        func_impl = self._find_implementation(func.get("name"), file_summaries)
                        if func_impl and target in self._find_function_calls_in_impl(func_impl):
                            callers.append(func.get("name"))

                if callers:
                    result.append("Called by:")
                    for caller in callers[:5]:  # Limit to 5 callers
                        result.append(f"  - {caller}")
                    if len(callers) > 5:
                        result.append(f"  - ... {len(callers) - 5} more callers")

        return "\n".join(result) if result else None

    def expand_context(self, focus_areas: List[str]) -> Dict[str, Any]:
        """
        Expand context with additional details for the specified focus areas.

        Args:
            focus_areas: List of focus areas to expand context for

        Returns:
            Dictionary with additional context
        """
        self.logger.info(f"Expanding context for focus areas: {focus_areas}")

        additional_context = {}

        for area in focus_areas:
            # First, determine what kind of area this is (class, function, concept, etc.)
            area_type = self._infer_area_type(area)

            if area_type == "class" or area_type == "function":
                # For classes and functions, provide signature information
                signature = self._find_signature_for_focus(area)
                if signature:
                    if "signatures" not in additional_context:
                        additional_context["signatures"] = {}
                    additional_context["signatures"][area] = signature

            elif area_type == "implementation":
                # For implementation requests, provide code snippets if available
                implementation = self._find_implementation_for_focus(area)
                if implementation:
                    if "implementations" not in additional_context:
                        additional_context["implementations"] = {}
                    additional_context["implementations"][area] = implementation

            elif area_type == "concept":
                # For conceptual areas, provide explanatory context
                explanation = self._find_explanation_for_concept(area)
                if explanation:
                    if "explanations" not in additional_context:
                        additional_context["explanations"] = {}
                    additional_context["explanations"][area] = explanation

        return additional_context

    def _infer_area_type(self, area: str) -> str:
        """
        Infer the type of a focus area.

        Args:
            area: Focus area to infer type for

        Returns:
            Type of the focus area ('class', 'function', 'implementation', or 'concept')
        """
        # If area contains terms related to implementation
        if any(term in area.lower() for term in ["implementation", "code", "how to"]):
            return "implementation"

        # If area has CamelCase, likely a class
        if area and area[0].isupper() and not area.isupper() and '_' not in area:
            return "class"

        # If area ends with (), likely a function
        if area.endswith("()") or "function" in area.lower() or "method" in area.lower():
            area_clean = area.replace("()", "")
            if area_clean and area_clean[0].islower():
                return "function"

        # Default to concept
        return "concept"

    def _find_signature_for_focus(self, area: str) -> Optional[str]:
        """
        Find signature information for a focus area.

        Args:
            area: Focus area to find signature for

        Returns:
            Signature information or None if not found
        """
        # This could delegate to the existing _find_signature method if applicable
        # For now, return a placeholder
        return f"Signature information for {area}"

    def _find_implementation_for_focus(self, area: str) -> Optional[str]:
        """
        Find implementation details for a focus area.

        Args:
            area: Focus area to find implementation for

        Returns:
            Implementation details or None if not found
        """
        # This could delegate to the existing _find_implementation method if applicable
        # For now, return a placeholder
        return f"Implementation details for {area}"

    def _find_explanation_for_concept(self, concept: str) -> Optional[str]:
        """
        Find explanation for a conceptual focus area.

        Args:
            concept: Concept to find explanation for

        Returns:
            Explanation or None if not found
        """
        # This would need to extract relevant information from available context
        # For now, return a placeholder
        return f"Explanation for concept: {concept}"