import logging
import re
from typing import Dict, List, Any, Optional

from ..utils.patch_validator import PatchValidator

logger = logging.getLogger(__name__)


class EnhancedTreeOfThought:
    """
    Enhanced Tree of Thought reasoning with integrated validation.
    Implements the second phase of the optimized bug fixing pipeline.
    """

    def __init__(self, config, model):
        """
        Initialize Tree of Thought reasoning with validation.

        Args:
            config: Configuration object.
            model: Language model instance.
        """
        self.config = config
        self.model = model
        self.max_branches = config["reasoning"].get("tot_breadth", 3)
        self.patch_validator = PatchValidator(config)

        # Metrics tracking
        self.metrics = {
            "logical_consistency": 0.0,
            "code_pattern_match": 0.0,
            "edge_case_coverage": 0.0,
            "overall_confidence": 0.0,
        }

    def explore_with_validation(
            self,
            issue_id: str,
            issue_description: str,
            codebase_context: str,
            bug_location: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explore different branches with integrated patch validation.

        Args:
            issue_id: Issue ID for validation.
            issue_description: Description of the issue.
            codebase_context: Context from the codebase.
            bug_location: Location of the bug from previous phase.

        Returns:
            Dictionary containing branches, evaluations, and validated solutions.
        """
        logger.info("Starting Enhanced Tree of Thought exploration with integrated validation")

        # Default result structure
        default_result = {
            "branches": [],
            "depth": 0.0,
            "early_stopped": False,
            "valid_solution": None,
            "best_branch": None
        }

        # Format the prompt with bug location information
        prompt = self._create_exploration_prompt(issue_description, codebase_context, bug_location)

        # Generate the initial exploration
        try:
            logger.info("Generating initial branches for ToT exploration")
            response = self.model.generate(prompt)
            logger.debug(f"ToT response: {response[:500]}...")
        except Exception as e:
            logger.error(f"Error generating ToT exploration: {e}")
            return default_result  # Return default instead of {"error": str(e)}

        # Parse branches from the response
        branches = self._parse_branches(response)
        logger.info(f"Parsed {len(branches)} branches from response")

        # If no branches were found, return default result
        if not branches:
            logger.warning("No branches were parsed from the ToT response")
            return default_result

        # Evaluate and validate each branch
        branch_results = []
        valid_patch_found = False
        valid_solution = None

        for i, branch in enumerate(branches[:self.max_branches]):
            branch_num = i + 1
            logger.info(f"Evaluating branch {branch_num}/{len(branches)}")

            # Evaluate branch confidence
            confidence = self._evaluate_branch_confidence(branch, bug_location)

            # Create result for this branch
            branch_result = {
                "branch_number": branch_num,
                "content": branch,
                "confidence": confidence,
                "evaluation": {
                    "logical_consistency": self.metrics["logical_consistency"],
                    "code_pattern_match": self.metrics["code_pattern_match"],
                    "edge_case_coverage": self.metrics["edge_case_coverage"],
                }
            }

            # Skip patch generation if confidence is too low
            if confidence < 0.6:
                logger.info(f"Branch {branch_num} confidence too low ({confidence:.2f}), skipping patch generation")
                branch_results.append(branch_result)
                continue

            # Try to generate a patch from this branch
            logger.info(f"Generating patch for branch {branch_num}")
            patch = self._generate_patch_from_branch(branch, bug_location, issue_description, codebase_context)

            if patch:
                # Save generated patch
                branch_result["patch_generated"] = True
                branch_result["patch"] = patch

                # Validate the patch
                try:
                    validation_result = self.patch_validator.validate_patch(patch, issue_id)
                    branch_result["validation"] = validation_result

                    # If valid patch, mark for early stopping
                    if validation_result.get("success", False):
                        valid_patch_found = True
                        valid_solution = {
                            "branch": branch_num,
                            "patch": patch,
                            "validation": validation_result,
                        }
                        logger.info(f"Found valid patch in branch {branch_num}")
                except Exception as e:
                    logger.error(f"Error validating patch for branch {branch_num}: {e}")
                    branch_result["validation_error"] = str(e)
            else:
                branch_result["patch_generated"] = False

            branch_results.append(branch_result)

            # Early stopping if valid patch found
            if valid_patch_found:
                logger.info("Valid patch found, stopping branch exploration early")
                break

        # Identify the best branch
        best_branch = None
        best_confidence = 0.0

        for branch in branch_results:
            if branch.get("confidence", 0) > best_confidence:
                best_confidence = branch.get("confidence", 0)
                best_branch = branch

        # Compile results
        result = {
            "branches": branch_results,
            "valid_patch_found": valid_patch_found,
            "valid_solution": valid_solution,
            "best_branch": best_branch,
            "depth": best_confidence if best_branch else 0.0,
            "early_stopped": valid_patch_found
        }

        return result

    def _create_exploration_prompt(self, issue_description: str, codebase_context: str,
                                   bug_location: Dict[str, Any]) -> str:
        """
        Create the exploration prompt with bug location information.

        Args:
            issue_description: Issue description text.
            codebase_context: Codebase context information.
            bug_location: Bug location dictionary.

        Returns:
            Formatted prompt string.
        """
        prompt = f"""
        You are an expert software engineer using a Tree of Thought methodology to explore the root cause of a bug.

        ISSUE DESCRIPTION:
        {issue_description}

        BUG LOCATION:
        File: {bug_location.get('file', 'Unknown')}
        Function: {bug_location.get('function', 'Unknown')}
        Line Numbers: {bug_location.get('line_numbers', 'Unknown')}
        Issue: {bug_location.get('issue', 'Unknown')}

        CODEBASE CONTEXT:
        {codebase_context}

        Your task is to explore multiple hypotheses for the root cause of this bug. For each hypothesis (branch):
        1. Clearly state the hypothesis about what causes the bug
        2. Analyze the code patterns and logic that support this hypothesis
        3. Explain how this hypothesis explains the observed symptoms
        4. Discuss potential edge cases or conditions where the bug manifests

        Generate {self.max_branches} distinct hypotheses, each labeled as BRANCH 1, BRANCH 2, etc.
        For each branch, provide deep technical analysis that demonstrates your understanding of the code.
        """

        return prompt

    def _parse_branches(self, response: str) -> List[str]:
        """
        Parse branches from the model response.

        Args:
            response: Model response text.

        Returns:
            List of branch contents.
        """
        branches = []

        # Look for BRANCH x: headings
        branch_pattern = r'BRANCH\s+(\d+):(.*?)(?=BRANCH\s+\d+:|$)'
        branch_matches = re.finditer(branch_pattern, response, re.DOTALL)

        for match in branch_matches:
            branch_content = match.group(2).strip()
            if branch_content:
                branches.append(branch_content)

        # If no BRANCH headings found, try alternative formats
        if not branches:
            # Try numbered hypotheses
            hypothesis_pattern = r'(?:Hypothesis|HYPOTHESIS)\s+(\d+):(.*?)(?=(?:Hypothesis|HYPOTHESIS)\s+\d+:|$)'
            hypothesis_matches = re.finditer(hypothesis_pattern, response, re.DOTALL)

            for match in hypothesis_matches:
                branch_content = match.group(2).strip()
                if branch_content:
                    branches.append(branch_content)

        # If still no branches, split by double newlines and take sections
        if not branches and response.strip():
            sections = re.split(r'\n\s*\n', response)
            for section in sections:
                if len(section.strip()) > 100:  # Only substantial sections
                    branches.append(section.strip())

        return branches

    def _generate_more_branches(
            self,
            issue_description: str,
            codebase_context: str,
            bug_location: Dict[str, Any],
            existing_branch: str
    ) -> List[str]:
        """
        Generate additional branches when initial exploration produces too few.

        Args:
            issue_description: Issue description text.
            codebase_context: Codebase context information.
            bug_location: Bug location dictionary.
            existing_branch: Content of the existing branch.

        Returns:
            List of additional branch contents.
        """
        prompt = f"""
        You are an expert software engineer exploring alternative root causes for a bug.

        ISSUE DESCRIPTION:
        {issue_description}

        BUG LOCATION:
        File: {bug_location.get('file', 'Unknown')}
        Function: {bug_location.get('function', 'Unknown')}
        Line Numbers: {bug_location.get('line_numbers', 'Unknown')}
        Issue: {bug_location.get('issue', 'Unknown')}

        EXISTING HYPOTHESIS:
        {existing_branch}

        Your task is to generate two ALTERNATIVE hypotheses that explore different possible root causes than the existing one.
        Create hypotheses that are substantively different from the existing one by:
        1. Considering different code patterns or logic
        2. Exploring different edge cases or conditions
        3. Focusing on different parts of the code or different types of issues

        For each alternative hypothesis:
        1. Clearly state the hypothesis about what causes the bug
        2. Analyze the code patterns and logic that support this hypothesis
        3. Explain how this hypothesis explains the observed symptoms
        4. Discuss potential edge cases or conditions where the bug manifests

        Label them as ALTERNATIVE 1 and ALTERNATIVE 2.
        """

        try:
            # Generate alternatives
            response = self.model.generate(prompt)

            # Parse alternatives
            alternatives = []
            alternative_pattern = r'ALTERNATIVE\s+(\d+):(.*?)(?=ALTERNATIVE\s+\d+:|$)'
            alternative_matches = re.finditer(alternative_pattern, response, re.DOTALL)

            for match in alternative_matches:
                alternative_content = match.group(2).strip()
                if alternative_content:
                    alternatives.append(alternative_content)

            # If no structured alternatives found, split by double newlines
            if not alternatives and response.strip():
                sections = re.split(r'\n\s*\n', response)
                for section in sections:
                    if len(section.strip()) > 100:  # Only substantial sections
                        alternatives.append(section.strip())

            return alternatives

        except Exception as e:
            logger.error(f"Error generating additional branches: {e}")
            return []

    def _generate_patch_from_branch(
            self,
            branch: str,
            bug_location: Dict[str, Any],
            issue_description: str,
            codebase_context: str
    ) -> Optional[str]:
        """
        Generate a patch based on a branch hypothesis.

        Args:
            branch: Branch content text.
            bug_location: Bug location information.
            issue_description: Issue description text.
            codebase_context: Codebase context information.

        Returns:
            Generated patch string or None.
        """
        prompt = f"""
        You are an expert software engineer fixing a bug based on a specific hypothesis.

        ISSUE DESCRIPTION:
        {issue_description}

        BUG LOCATION:
        File: {bug_location.get('file', 'Unknown')}
        Function: {bug_location.get('function', 'Unknown')}
        Line Numbers: {bug_location.get('line_numbers', 'Unknown')}

        ROOT CAUSE HYPOTHESIS:
        {branch}

        Your task is to create a precise Git patch that fixes this issue based on the hypothesis.
        The patch must:
        1. Be in proper Git patch format starting with "diff --git"
        2. Include correct file paths based on the bug location
        3. Use correct line numbers in the hunk headers
        4. Include context lines around the changes
        5. Make minimal changes to fix the specific issue identified in the hypothesis

        IMPORTANT: Your response must be ONLY the Git patch.
        """

        try:
            # Generate patch
            response = self.model.generate(prompt)

            # Extract patch from response
            return self._extract_patch(response)
        except Exception as e:
            logger.error(f"Error generating patch from branch: {e}")
            return None

    def _extract_patch(self, response: str) -> Optional[str]:
        """
        Extract a Git patch from the model response.

        Args:
            response: Model response text.

        Returns:
            Extracted patch string or None.
        """
        # Look for diff --git pattern
        patch_pattern = r'(diff --git.*?)(?=^```|\Z)'
        patch_match = re.search(patch_pattern, response, re.MULTILINE | re.DOTALL)

        if patch_match:
            return patch_match.group(1).strip()

        # Look for code blocks
        code_block_pattern = r'```(?:diff|patch|git)?\n(.*?)```'
        code_match = re.search(code_block_pattern, response, re.MULTILINE | re.DOTALL)

        if code_match:
            content = code_match.group(1).strip()
            if content.startswith('diff --git') or ('---' in content and '+++' in content):
                return content

        # Check if the response itself is a patch
        if response.strip().startswith('diff --git') or ('---' in response and '+++' in response):
            return response.strip()

        return None

    def solve(self, issue_description, codebase_context):
        """
        Solve using Tree of Thought reasoning.
        """
        logger.info(f"Solving issue using Tree of Thought reasoning with {self.model.model_name}")

        try:
            # Generate initial thoughts
            prompt = self._format_prompt(issue_description, codebase_context)
            response = self.model.generate(prompt)

            # Parse branches from response
            branches = self._parse_branches(response)
            logger.info(f"Generated {len(branches)} branches through ToT reasoning")

            # Initialize empty results
            confidence_scores = []
            branch_details = []

            # Safety check for empty branches
            if not branches:
                logger.warning("No branches could be parsed from the response")
                return {
                    "branches": [],
                    "depth": 0.0,
                    "best_branch_idx": -1
                }

            # Evaluate each branch
            for i, branch in enumerate(branches):
                confidence = self._evaluate_branch_confidence(branch)
                confidence_scores.append(confidence)
                branch_details.append({
                    "content": branch,
                    "confidence": confidence,
                    "evaluation": self.metrics.copy()
                })

            # Find best branch safely
            if confidence_scores:
                best_branch_idx = confidence_scores.index(max(confidence_scores))
                best_confidence = confidence_scores[best_branch_idx]
                # Only access branches if we have valid index
                if 0 <= best_branch_idx < len(branches):
                    best_branch = branches[best_branch_idx]
                else:
                    best_branch = ""
                    best_branch_idx = -1
            else:
                # Default values if no branches with confidence scores
                best_branch_idx = -1
                best_confidence = 0.0
                best_branch = ""

            return {
                "branches": branch_details,
                "depth": best_confidence,
                "best_branch_idx": best_branch_idx
            }
        except Exception as e:
            logger.error(f"Error in Tree of Thought reasoning: {e}", exc_info=True)
            return {
                "branches": [],
                "depth": 0.0,
                "best_branch_idx": -1
            }

    def _format_prompt(self, issue_description: str, codebase_context: str) -> str:
        """
        Format the prompt for Tree of Thought exploration.

        Args:
            issue_description: Issue description text.
            codebase_context: Codebase context information.

        Returns:
            Formatted prompt string.
        """
        # This function should create a prompt similar to _create_exploration_prompt
        # but without requiring bug_location
        prompt = f"""
        You are an expert software engineer using a Tree of Thought methodology to explore the root cause of a bug.

        ISSUE DESCRIPTION:
        {issue_description}

        CODEBASE CONTEXT:
        {codebase_context}

        Your task is to explore multiple hypotheses for the root cause of this bug. For each hypothesis (branch):
        1. Clearly state the hypothesis about what causes the bug
        2. Analyze the code patterns and logic that support this hypothesis
        3. Explain how this hypothesis explains the observed symptoms
        4. Discuss potential edge cases or conditions where the bug manifests

        Generate {self.max_branches} distinct hypotheses, each labeled as BRANCH 1, BRANCH 2, etc.
        For each branch, provide deep technical analysis that demonstrates your understanding of the code.
        """
        return prompt

    def _evaluate_branch_confidence(self, branch: str, bug_location: Dict[str, Any] = None) -> float:
        """
        Evaluate the confidence score for a branch.

        Args:
            branch: Branch content text.
            bug_location: Optional bug location information (used for additional scoring).

        Returns:
            Confidence score between 0 and 1.
        """
        # Reset metrics
        self.metrics = {
            "logical_consistency": 0.0,
            "code_pattern_match": 0.0,
            "edge_case_coverage": 0.0,
            "overall_confidence": 0.0,
        }

        score = 0.4  # Base score for a plausible explanation

        # Check logical consistency
        logical_indicators = [
            'causes', 'leads to', 'results in', 'triggers', 'because',
            'when', 'if', 'condition', 'scenario', 'case'
        ]
        logical_count = sum(1 for term in logical_indicators if term in branch.lower())
        logical_score = min(0.3, logical_count * 0.05)
        score += logical_score
        self.metrics["logical_consistency"] = logical_score

        # Check code pattern analysis
        code_indicators = [
            'function', 'method', 'variable', 'parameter', 'return',
            'loop', 'condition', 'check', 'value', 'type'
        ]
        code_count = sum(1 for term in code_indicators if term in branch.lower())

        # Extra points for specific references to the identified bug location if provided
        if bug_location:
            file_name = bug_location.get('file', '')
            if file_name and file_name.lower() in branch.lower():
                code_count += 2

            function_name = bug_location.get('function', '')
            if function_name and function_name.lower() in branch.lower():
                code_count += 2

        code_score = min(0.3, code_count * 0.03)
        score += code_score
        self.metrics["code_pattern_match"] = code_score

        # Check for edge case identification
        edge_indicators = [
            'edge case', 'special case', 'boundary', 'limit', 'exception',
            'corner case', 'specific scenario', 'unexpected', 'rare'
        ]
        edge_count = sum(1 for term in edge_indicators if term in branch.lower())
        edge_score = min(0.2, edge_count * 0.05)
        score += edge_score
        self.metrics["edge_case_coverage"] = edge_score

        # Check for code examples or specific references
        if '```' in branch:
            score += 0.1  # Contains code examples

        # Check for specific line number references
        if re.search(r'line\s+\d+', branch.lower()):
            score += 0.05

        self.metrics["overall_confidence"] = min(1.0, score)
        return min(1.0, score)
