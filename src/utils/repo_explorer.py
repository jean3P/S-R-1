import os
import re
from typing import Dict, Any, List, Optional

from src.utils.logging import get_logger

logger = get_logger("repo_explorer")


class RepoExplorer:
    """
    Utility class for exploring repository structure and finding relevant files.
    """

    def __init__(self, repo_path: str):
        """
        Initialize the repository explorer.

        Args:
            repo_path: Path to the repository
        """
        print(f"[DEBUG] Initializing RepoExplorer with repo_path='{repo_path}'")
        self.repo_path = repo_path
        self.file_cache = {}

    def find_file(self, file_name: str) -> List[str]:
        """
        Find all instances of a file in the repository.

        Args:
            file_name: Name of the file to find

        Returns:
            List of relative paths to matching files
        """
        print(f"[DEBUG] find_file called with file_name='{file_name}'")
        found_files = []
        for root, dirs, files in os.walk(self.repo_path):
            if file_name in files:
                rel_path = os.path.relpath(os.path.join(root, file_name), self.repo_path)
                found_files.append(rel_path)
                print(f"[DEBUG] Found '{file_name}' at '{rel_path}'")

        print(f"[DEBUG] Total matches found for '{file_name}': {len(found_files)}")
        return found_files

    def find_rule_definition(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """
        Find the definition of a linting rule in the repository.

        Args:
            rule_id: ID of the rule (e.g., L031)

        Returns:
            Dictionary with rule information or None if not found
        """
        print(f"[DEBUG] find_rule_definition called with rule_id='{rule_id}'")
        # Look for rule files with the rule ID
        rule_files = self.find_file(f"{rule_id}.py")
        print(f"[DEBUG] Initial rule_files search result: {rule_files}")

        if not rule_files:
            print("[DEBUG] No direct match found; trying general search for class definitions.")
            # Try a more general search
            rule_files = []
            for root, dirs, files in os.walk(self.repo_path):
                for file in files:
                    if file.endswith(".py"):
                        full_path = os.path.join(root, file)
                        try:
                            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                                content = f.read()
                            # Check if class definition matches either Rule_X or X
                            if f"class Rule_{rule_id}" in content or f"class {rule_id}" in content:
                                rel_path = os.path.relpath(full_path, self.repo_path)
                                rule_files.append(rel_path)
                                print(f"[DEBUG] Found rule class in '{rel_path}'")
                        except Exception as e:
                            logger.error(f"Error reading file {full_path}: {str(e)}")

        if not rule_files:
            logger.warning(f"No rule file found for {rule_id}")
            print(f"[DEBUG] No rule file found for '{rule_id}' after general search.")
            return None

        # Use the first match (most likely the correct one)
        rule_file = rule_files[0]
        full_path = os.path.join(self.repo_path, rule_file)
        print(f"[DEBUG] Using rule file '{rule_file}' for final analysis.")

        # Read the file content
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            # Find the rule class definition
            rule_class_match = re.search(
                rf'class (?:Rule_)?{rule_id}\(.*?\):(.*?)(?:\n\w+|$)',
                content, re.DOTALL
            )

            if rule_class_match:
                # Find the class content
                class_content = rule_class_match.group(1)

                # Find the line number
                line_number = content[:rule_class_match.start()].count('\n') + 1

                # Extract the class docstring if available
                docstring_match = re.search(r'"""(.*?)"""', class_content, re.DOTALL)
                docstring = docstring_match.group(1).strip() if docstring_match else ""

                # Find methods in the class
                method_matches = re.finditer(r'def\s+(\w+)\s*\(', class_content)
                methods = [match.group(1) for match in method_matches]

                # Extract a snippet of the class definition and first method
                snippet_start = max(0, rule_class_match.start() - 100)
                snippet_end = min(len(content), rule_class_match.end() + 500)
                code_snippet = content[snippet_start:snippet_end]

                result = {
                    "file_path": full_path,
                    "context": {
                        "line_number": line_number,
                        "methods": methods,
                        "docstring": docstring,
                        "code_snippet": code_snippet
                    }
                }
                print(f"[DEBUG] Successfully extracted rule definition for '{rule_id}'.")
                return result
            else:
                print("[DEBUG] Could not find matching class definition in file content.")

        except Exception as e:
            logger.error(f"Error reading rule file {rule_file}: {str(e)}")
            print(f"[DEBUG] Exception while reading or parsing the rule file: {e}")

        return None

    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Get the content of a file.

        Args:
            file_path: Path to the file (relative to repo root)

        Returns:
            File content or None if not found
        """
        print(f"[DEBUG] get_file_content called with file_path='{file_path}'")
        # Check cache first
        if file_path in self.file_cache:
            print("[DEBUG] Found file content in cache.")
            return self.file_cache[file_path]

        full_path = os.path.join(self.repo_path, file_path)
        if not os.path.exists(full_path):
            print(f"[DEBUG] File does not exist: '{full_path}'")
            return None

        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            # Cache the content
            self.file_cache[file_path] = content
            print("[DEBUG] File content successfully read and cached.")
            return content

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            print(f"[DEBUG] Exception in get_file_content: {e}")
            return None

    def get_function_definition(self, file_path: str, function_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the definition of a function.

        Args:
            file_path: Path to the file (relative to repo root)
            function_name: Name of the function

        Returns:
            Dictionary with function information or None if not found
        """
        print(f"[DEBUG] get_function_definition called with file_path='{file_path}', function_name='{function_name}'")
        content = self.get_file_content(file_path)
        if not content:
            print("[DEBUG] No file content available, returning None.")
            return None

        pattern = re.compile(
            rf'def\s+{re.escape(function_name)}\s*\(.*?\):(.*?)(?:(?:\n\s*def|\n\s*class)|\Z)',
            re.DOTALL
        )
        match = pattern.search(content)

        if not match:
            print("[DEBUG] Function definition not found in the file content.")
            return None

        # Get the function body
        body = match.group(1).strip()

        # Get the line number
        line_number = content[:match.start()].count('\n') + 1

        # Extract the function signature
        signature_match = re.search(
            rf'def\s+{re.escape(function_name)}\s*\((.*?)\):',
            content
        )
        signature = signature_match.group(1) if signature_match else ""

        # Extract parameters
        parameters = [param.strip().split(':')[0].strip() for param in signature.split(',') if param.strip()]

        # Extract docstring if available
        docstring_match = re.search(r'"""(.*?)"""', body, re.DOTALL)
        docstring = docstring_match.group(1).strip() if docstring_match else ""

        result = {
            "name": function_name,
            "file_path": file_path,
            "line_number": line_number,
            "signature": signature,
            "parameters": parameters,
            "docstring": docstring,
            "body": body
        }

        print("[DEBUG] Successfully extracted function definition.")
        return result

    def get_class_definition(self, file_path: str, class_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the definition of a class.

        Args:
            file_path: Path to the file (relative to repo root)
            class_name: Name of the class

        Returns:
            Dictionary with class information or None if not found
        """
        print(f"[DEBUG] get_class_definition called with file_path='{file_path}', class_name='{class_name}'")
        content = self.get_file_content(file_path)
        if not content:
            print("[DEBUG] No file content available, returning None.")
            return None

        pattern = re.compile(
            rf'class\s+{re.escape(class_name)}\s*(?:\(.*?\))?:(.*?)(?:(?:\n\s*class|\n\s*def\s+[^_])|\Z)',
            re.DOTALL
        )
        match = pattern.search(content)

        if not match:
            print("[DEBUG] Class definition not found in the file content.")
            return None

        # Get the class body
        body = match.group(1).strip()

        # Get the line number
        line_number = content[:match.start()].count('\n') + 1

        # Extract the class inheritance
        inheritance_match = re.search(
            rf'class\s+{re.escape(class_name)}\s*\((.*?)\):',
            content
        )
        inheritance = inheritance_match.group(1) if inheritance_match else ""

        # Extract docstring if available
        docstring_match = re.search(r'"""(.*?)"""', body, re.DOTALL)
        docstring = docstring_match.group(1).strip() if docstring_match else ""

        # Extract methods
        method_matches = re.finditer(r'def\s+(\w+)\s*\(', body)
        methods = [match.group(1) for match in method_matches]

        result = {
            "name": class_name,
            "file_path": file_path,
            "line_number": line_number,
            "inheritance": inheritance,
            "docstring": docstring,
            "methods": methods,
            "body": body
        }

        print("[DEBUG] Successfully extracted class definition.")
        return result

    def find_imports(self, file_path: str) -> List[str]:
        """
        Find import statements in a file.

        Args:
            file_path: Path to the file (relative to repo root)

        Returns:
            List of import statements
        """
        print(f"[DEBUG] find_imports called with file_path='{file_path}'")
        content = self.get_file_content(file_path)
        if not content:
            print("[DEBUG] No content, returning empty list.")
            return []

        import_pattern = re.compile(r'^(?:from\s+[\w\.]+\s+import|import\s+[\w\.]+).*$', re.MULTILINE)
        matches = import_pattern.findall(content)
        print(f"[DEBUG] Found {len(matches)} import statements in '{file_path}'.")
        return matches

    def find_references(self, name: str, max_files: int = 10) -> List[Dict[str, Any]]:
        """
        Find references to a name across the repository.

        Args:
            name: Name to search for
            max_files: Maximum number of files to search

        Returns:
            List of references
        """
        print(f"[DEBUG] find_references called with name='{name}', max_files={max_files}")
        references = []
        count = 0

        for root, dirs, files in os.walk(self.repo_path):
            for file in files:
                if not file.endswith(".py"):
                    continue

                if count >= max_files:
                    print(f"[DEBUG] Reached max_files={max_files} limit.")
                    break

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.repo_path)

                try:
                    with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()

                    if name in content:
                        lines = content.split('\n')
                        references_in_file = []
                        for i, line_content in enumerate(lines):
                            if name in line_content:
                                # Ensure it's an actual reference, not just part of a word
                                if re.search(rf'\b{re.escape(name)}\b', line_content):
                                    references_in_file.append({
                                        "line_number": i + 1,
                                        "line_content": line_content.strip()
                                    })

                        if references_in_file:
                            references.append({
                                "file_path": rel_path,
                                "references": references_in_file
                            })
                            count += 1
                            print(f"[DEBUG] Found {len(references_in_file)} references to '{name}' in '{rel_path}'")

                except Exception as e:
                    logger.error(f"Error reading file {rel_path}: {str(e)}")
                    print(f"[DEBUG] Exception while searching references: {e}")

        print(f"[DEBUG] Returning {len(references)} total references.")
        return references
