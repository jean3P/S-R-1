# src/datasets/coding_problems.py

from typing import Dict, Any, Optional, List

from src.datasets.json_dataset import JSONDataset


class CodingProblemsDataset(JSONDataset):
    """Dataset implementation for coding problems."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the coding problems dataset.

        Args:
            config: Dataset configuration
        """
        super().__init__(config)

        # Extract additional configuration
        self.difficulty_levels = config.get("difficulty_levels", ["easy", "medium", "hard"])
        self.languages = config.get("languages", ["python", "javascript", "java", "cpp"])
        self.problem_categories = config.get("problem_categories", [
            "algorithms", "data_structures", "math", "string", "array",
            "dynamic_programming", "recursion", "sorting", "searching"
        ])

        # Define validation schema
        self.schema = {
            "id": {"type": str, "required": True},
            "title": {"type": str, "required": True},
            "description": {"type": str, "required": True},
            "difficulty": {"type": str, "required": True, "values": self.difficulty_levels},
            "category": {"type": str, "required": True, "values": self.problem_categories},
            "constraints": {"type": list, "required": False},
            "examples": {"type": list, "required": True},
            "solutions": {"type": dict, "required": False},
            "test_cases": {"type": list, "required": False}
        }

    def load(self) -> None:
        """
        Load the dataset and validate.

        Raises:
            FileNotFoundError: If the JSON file does not exist
            ValueError: If the JSON format is invalid or validation fails
        """
        # Load the dataset using parent method
        super().load()

        # Validate the loaded data
        invalid_examples = []
        for i, example in enumerate(self.data):
            validation_errors = self._validate_example(example)
            if validation_errors:
                invalid_examples.append((i, validation_errors))

        if invalid_examples:
            for i, errors in invalid_examples:
                self.logger.warning(f"Example {i} has validation errors: {errors}")
            self.logger.warning(f"Found {len(invalid_examples)} invalid examples out of {len(self.data)}")

    def _validate_example(self, example: Dict[str, Any]) -> List[str]:
        """
        Validate an example against the schema.

        Args:
            example: Example to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields and types
        for field, spec in self.schema.items():
            if spec.get("required", False) and field not in example:
                errors.append(f"Missing required field: {field}")
                continue

            if field in example:
                if not isinstance(example[field], spec["type"]):
                    errors.append(
                        f"Field {field} has wrong type: expected {spec['type'].__name__}, got {type(example[field]).__name__}")

                # Check allowed values if specified
                if "values" in spec and example[field] not in spec["values"]:
                    errors.append(
                        f"Field {field} has invalid value: {example[field]}, allowed values: {spec['values']}")

        # Validate examples format
        if "examples" in example and isinstance(example["examples"], list):
            for i, ex in enumerate(example["examples"]):
                if not isinstance(ex, dict):
                    errors.append(f"Example {i} is not a dictionary")
                    continue

                if "input" not in ex:
                    errors.append(f"Example {i} missing input")
                if "output" not in ex:
                    errors.append(f"Example {i} missing output")

        # Validate test cases format
        if "test_cases" in example and isinstance(example["test_cases"], list):
            for i, test in enumerate(example["test_cases"]):
                if not isinstance(test, dict):
                    errors.append(f"Test case {i} is not a dictionary")
                    continue

                if "input" not in test:
                    errors.append(f"Test case {i} missing input")
                if "expected" not in test:
                    errors.append(f"Test case {i} missing expected output")

        # Validate solutions format
        if "solutions" in example and isinstance(example["solutions"], dict):
            for lang, solution in example["solutions"].items():
                if lang not in self.languages:
                    errors.append(f"Solution language {lang} not in allowed languages: {self.languages}")

                if not isinstance(solution, str):
                    errors.append(f"Solution for language {lang} is not a string")

        return errors

    def get_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """
        Get problems by difficulty level.

        Args:
            difficulty: Difficulty level

        Returns:
            List of matching problems
        """
        if difficulty not in self.difficulty_levels:
            raise ValueError(f"Invalid difficulty level: {difficulty}, allowed values: {self.difficulty_levels}")

        return self.filter(lambda example: example.get("difficulty") == difficulty)

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get problems by category.

        Args:
            category: Problem category

        Returns:
            List of matching problems
        """
        if category not in self.problem_categories:
            raise ValueError(f"Invalid category: {category}, allowed values: {self.problem_categories}")

        return self.filter(lambda example: example.get("category") == category)

    def get_problem_by_id(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a problem by ID.

        Args:
            problem_id: Problem ID

        Returns:
            Problem or None if not found
        """
        matches = self.filter(lambda example: example.get("id") == problem_id)
        return matches[0] if matches else None

    def add_problem(self, problem: Dict[str, Any]) -> None:
        """
        Add a new problem to the dataset.

        Args:
            problem: Problem to add

        Raises:
            ValueError: If the problem is invalid
        """
        # Validate the problem
        validation_errors = self._validate_example(problem)
        if validation_errors:
            error_msg = "; ".join(validation_errors)
            raise ValueError(f"Invalid problem: {error_msg}")

        # Check if problem with same ID already exists
        if "id" in problem:
            existing = self.get_problem_by_id(problem["id"])
            if existing:
                self.logger.warning(f"Overwriting existing problem with ID: {problem['id']}")

        # Add the problem
        self.add_example(problem)

    def add_solution(self, problem_id: str, language: str, solution: str) -> bool:
        """
        Add a solution to a problem.

        Args:
            problem_id: Problem ID
            language: Programming language
            solution: Solution code

        Returns:
            True if the solution was added, False if the problem was not found

        Raises:
            ValueError: If the language is not supported
        """
        if language not in self.languages:
            raise ValueError(f"Unsupported language: {language}, allowed values: {self.languages}")

        # Find the problem
        problem = self.get_problem_by_id(problem_id)
        if not problem:
            return False

        # Initialize solutions dictionary if it doesn't exist
        if "solutions" not in problem:
            problem["solutions"] = {}

        # Add the solution
        problem["solutions"][language] = solution
        return True
