# src/reasoning/tree_of_thought.py
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class TreeOfThought:
    """
    Implementation of Tree of Thought reasoning for solving GitHub issues.
    """

    def __init__(self, config, model):
        """
        Initialize Tree of Thought reasoning.

        Args:
            config: Configuration object.
            model: Language model instance.
        """
        self.config = config
        self.model = model

        # Load ToT-specific configurations
        model_configs = config.get_model_config(model.model_name)
        self.tot_config = config.model_configs.get("tree_of_thought", {})
        self.prompt_template = self.tot_config.get("prompt_template", "")

        # ToT parameters
        self.tot_breadth = config["reasoning"].get("tot_breadth", 3)
        self.tot_depth = config["reasoning"].get("tot_depth", 3)

    def solve(self, issue_description: str, codebase_context: str) -> Dict[str, Any]:
        """
        Solve a GitHub issue using Tree of Thought reasoning.

        Args:
            issue_description: Description of the GitHub issue.
            codebase_context: Context from the codebase related to the issue.

        Returns:
            Dictionary containing the branches, solutions, and final implementation.
        """
        logger.info(f"Solving issue using Tree of Thought reasoning with {self.model.model_name}")

        # Create branches for problem understanding
        branches = self._explore_branches(issue_description, codebase_context)

        # Evaluate branches and select the most promising one
        best_branch_idx = self._evaluate_branches(branches)
        best_branch = branches[best_branch_idx]

        # Generate solution candidates from the best branch
        solutions = self._generate_solutions(best_branch, issue_description, codebase_context)

        # Evaluate solution candidates and select the best one
        best_solution_idx = self._evaluate_solutions(solutions)
        best_solution = solutions[best_solution_idx]

        # Create implementation plan for the best solution
        implementation = self._create_implementation(best_solution, issue_description, codebase_context)

        return {
            "branches": branches,
            "best_branch_idx": best_branch_idx,
            "solutions": solutions,
            "best_solution_idx": best_solution_idx,
            "implementation": implementation
        }

    def _explore_branches(self, issue_description: str, codebase_context: str) -> List[str]:
        """Explore different branches of problem understanding."""
        branches = []

        branch_prompt = "Analyze the issue from this perspective and identify key aspects."

        # Format the base prompt
        prompt = self.prompt_template.format(
            issue_description=f"{issue_description}\n\n{codebase_context}",
            branch_prompt=branch_prompt,
            solution_prompt="",
            implementation_prompt=""
        )

        # Generate branches
        response = self.model.generate(prompt)

        # Parse branches from response
        sections = response.split("BRANCH")
        for i in range(1, min(self.tot_breadth + 1, len(sections))):
            if i < len(sections):
                branches.append(sections[i].strip())

        return branches

    def _evaluate_branches(self, branches: List[str]) -> int:
        """Evaluate branches and return the index of the best one."""
        if not branches:
            return 0

        # For now, simple evaluation strategy - choose the first branch
        # This can be enhanced with a more sophisticated evaluation
        return 0

    def _generate_solutions(self, branch: str, issue_description: str, codebase_context: str) -> List[str]:
        """Generate solution candidates from the selected branch."""
        solutions = []

        solution_prompt = "Propose a detailed solution approach."

        # Format the prompt with the selected branch
        prompt = f"""
            Based on this understanding of the issue:
            
            {branch}
            
            And the original issue description:
            {issue_description}
            
            And the codebase context:
            {codebase_context}
            
            Generate {self.tot_breadth} different solutions to address this issue.
            For each solution, provide:
            1. A high-level approach
            2. Key implementation details
            3. Potential advantages and disadvantages
            """

        # Generate solutions
        response = self.model.generate(prompt)

        # Parse solutions from response
        sections = response.split("SOLUTION")
        for i in range(1, min(self.tot_breadth + 1, len(sections))):
            if i < len(sections):
                solutions.append(sections[i].strip())

        return solutions

    def _evaluate_solutions(self, solutions: List[str]) -> int:
        """Evaluate solutions and return the index of the best one."""
        if not solutions:
            return 0

        # For now, simple evaluation strategy - choose the first solution
        # This can be enhanced with a more sophisticated evaluation
        return 0

    def _create_implementation(self, solution: str, issue_description: str, codebase_context: str) -> str:
        """Create a detailed implementation plan for the selected solution."""
        implementation_prompt = "Provide a detailed implementation plan."

        # Format the prompt with the selected solution
        prompt = f"""
            Based on this solution approach:
            
            {solution}
            
            And the original issue description:
            {issue_description}
            
            And the codebase context:
            {codebase_context}
            
            Create a detailed implementation plan that includes:
            1. Specific files that need to be modified
            2. Exact code changes to be made
            3. How to test the implementation
            """

        # Generate implementation plan
        implementation = self.model.generate(prompt)

        return implementation
