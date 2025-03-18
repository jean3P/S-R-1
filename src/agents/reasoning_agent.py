# src/agents/reasoning_agent.py
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.utils.file_utils import save_json
from src.utils.tokenization import count_tokens


class ReasoningAgent(BaseAgent):
    """
    Agent that performs multi-step reasoning to solve complex problems.
    This agent breaks down a problem, thinks step by step, and progressively
    refines its understanding to reach a solution.
    """

    def reflect(self, initial_prompt: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the reasoning process with self-reflection.

        Args:
            initial_prompt: Initial problem description or question
            task: Task details and constraints

        Returns:
            Dict containing the detailed reasoning process and final solution
        """
        self.logger.info(f"Starting reasoning process for task: {task.get('name', 'unnamed')}")

        # Initialize metrics
        self._start_metrics()

        # Extract task-specific parameters
        max_reasoning_steps = task.get("max_reasoning_steps", self.config.get("max_reasoning_steps", 5))
        reasoning_strategy = task.get("reasoning_strategy", self.config.get("reasoning_strategy", "cot"))

        # Store steps in the reasoning process
        reasoning_steps = []
        current_prompt = initial_prompt

        # Prepare the initial reasoning state
        reasoning_state = {
            "problem": initial_prompt,
            "current_understanding": "",
            "sub_problems": [],
            "facts": [],
            "hypotheses": [],
            "solved_sub_problems": [],
            "solution_progress": 0.0,  # 0.0 to 1.0
            "confidence": 0.0  # 0.0 to 1.0
        }

        # Main reasoning loop
        for step in range(1, max_reasoning_steps + 1):
            self.logger.info(f"Reasoning step {step}/{max_reasoning_steps}")

            # Step 1: Generate the next reasoning step
            generation_input = self._format_reasoning_prompt(
                current_prompt=current_prompt,
                reasoning_state=reasoning_state,
                reasoning_strategy=reasoning_strategy,
                step=step,
                task=task
            )

            tokens_used = count_tokens(generation_input, self.model.config.get("model_name", "default"))
            reasoning_output, generation_time = self._measure_execution(self.model.generate, generation_input)

            # Record metrics
            self._record_generation(tokens_used, generation_time, True)

            # Step 2: Parse and structure the reasoning output
            parsed_output = self._parse_reasoning_output(reasoning_output, reasoning_strategy)

            # Step 3: Update reasoning state based on new insights
            reasoning_state = self._update_reasoning_state(reasoning_state, parsed_output)

            # Step 4: Log this reasoning step
            step_result = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "reasoning_prompt": generation_input,
                "reasoning_output": reasoning_output,
                "parsed_output": parsed_output,
                "reasoning_state": reasoning_state.copy()
            }
            reasoning_steps.append(step_result)

            # Update prompt for next iteration - includes the thinking so far
            current_prompt = self._generate_next_prompt(
                initial_prompt=initial_prompt,
                reasoning_steps=reasoning_steps,
                reasoning_state=reasoning_state
            )

            # Early stopping conditions
            if reasoning_state["solution_progress"] >= 0.99 and reasoning_state["confidence"] >= 0.95:
                self.logger.info("Early stopping: Solution found with high confidence")
                break

        # Final reasoning step - generate the solution
        solution_prompt = self._format_solution_prompt(
            initial_prompt=initial_prompt,
            reasoning_steps=reasoning_steps,
            reasoning_state=reasoning_state,
            task=task
        )

        tokens_used = count_tokens(solution_prompt, self.model.config.get("model_name", "default"))
        final_solution, solution_time = self._measure_execution(self.model.generate, solution_prompt)

        # Record metrics
        self._record_generation(tokens_used, solution_time, True)

        # Finalize metrics
        self._end_metrics()

        # Prepare results
        results = {
            "task": task,
            "initial_prompt": initial_prompt,
            "reasoning_steps": reasoning_steps,
            "final_reasoning_state": reasoning_state,
            "final_solution": final_solution,
            "metrics": self.metrics
        }

        # Save results if configured
        if self.config.get("save_results", True):
            output_dir = self.config.get("output_dir", "data/results")
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"{task.get('name', 'reasoning')}_{timestamp}.json")
            save_json(results, output_file)
            self.logger.info(f"Results saved to {output_file}")

        self.logger.info(f"Completed reasoning process with {len(reasoning_steps)} steps")
        return results

    def _format_reasoning_prompt(self, current_prompt: str, reasoning_state: Dict[str, Any],
                                 reasoning_strategy: str, step: int, task: Dict[str, Any]) -> str:
        """Format the prompt for the next reasoning step."""
        return self.prompt.format_reasoning(
            current_prompt=current_prompt,
            reasoning_state=reasoning_state,
            reasoning_strategy=reasoning_strategy,
            step=step,
            task=task
        )

    def _format_solution_prompt(self, initial_prompt: str, reasoning_steps: List[Dict[str, Any]],
                                reasoning_state: Dict[str, Any], task: Dict[str, Any]) -> str:
        """Format the prompt for generating the final solution."""
        return self.prompt.format_solution(
            initial_prompt=initial_prompt,
            reasoning_steps=reasoning_steps,
            reasoning_state=reasoning_state,
            task=task
        )

    def _parse_reasoning_output(self, output: str, strategy: str) -> Dict[str, Any]:
        """
        Parse the reasoning output from the model.
        Different strategies may have different parsing logic.
        """
        parsed = {
            "thoughts": output,
            "new_facts": [],
            "new_hypotheses": [],
            "solved_sub_problems": [],
            "solution_progress_update": None,
            "confidence_update": None
        }

        # Strategy-specific parsing
        if strategy == "cot":  # Chain of Thought
            # Simple extraction of key components using markers
            if "FACT:" in output:
                facts_section = output.split("FACT:")[1].split("\n")[0].strip()
                parsed["new_facts"] = [facts_section]

            if "HYPOTHESIS:" in output:
                hypothesis_section = output.split("HYPOTHESIS:")[1].split("\n")[0].strip()
                parsed["new_hypotheses"] = [hypothesis_section]

            if "PROGRESS:" in output:
                try:
                    progress = float(output.split("PROGRESS:")[1].split("\n")[0].strip().rstrip("%")) / 100.0
                    parsed["solution_progress_update"] = progress
                except (ValueError, IndexError):
                    pass

            if "CONFIDENCE:" in output:
                try:
                    confidence = float(output.split("CONFIDENCE:")[1].split("\n")[0].strip().rstrip("%")) / 100.0
                    parsed["confidence_update"] = confidence
                except (ValueError, IndexError):
                    pass

        elif strategy == "tot":  # Tree of Thought
            # TODO: Implement Tree of Thought specific parsing
            pass

        elif strategy == "got":  # Graph of Thought
            # TODO: Implement Graph of Thought specific parsing
            pass

        # Return the parsed output
        return parsed

    def _update_reasoning_state(self, current_state: Dict[str, Any], parsed_output: Dict[str, Any]) -> Dict[str, Any]:
        """Update the reasoning state based on new parsed output."""
        new_state = current_state.copy()

        # Update current understanding with the latest thoughts
        new_state["current_understanding"] = parsed_output["thoughts"]

        # Add new facts
        if parsed_output["new_facts"]:
            new_state["facts"].extend(parsed_output["new_facts"])

        # Add new hypotheses
        if parsed_output["new_hypotheses"]:
            new_state["hypotheses"].extend(parsed_output["new_hypotheses"])

        # Add solved sub-problems
        if parsed_output["solved_sub_problems"]:
            new_state["solved_sub_problems"].extend(parsed_output["solved_sub_problems"])

        # Update solution progress if provided
        if parsed_output["solution_progress_update"] is not None:
            new_state["solution_progress"] = parsed_output["solution_progress_update"]

        # Update confidence if provided
        if parsed_output["confidence_update"] is not None:
            new_state["confidence"] = parsed_output["confidence_update"]

        return new_state

    def _generate_next_prompt(self, initial_prompt: str, reasoning_steps: List[Dict[str, Any]],
                              reasoning_state: Dict[str, Any]) -> str:
        """Generate the prompt for the next iteration, incorporating previous reasoning."""
        # Start with the initial problem
        next_prompt = f"Problem: {initial_prompt}\n\n"

        # Add a summary of the reasoning so far
        next_prompt += "Reasoning so far:\n"

        # Include the last few reasoning steps (to avoid exceeding token limits)
        max_steps_to_include = min(3, len(reasoning_steps))
        for step in reasoning_steps[-max_steps_to_include:]:
            next_prompt += f"Step {step['step']}: {step['parsed_output']['thoughts']}\n\n"

        # Add the current state
        next_prompt += "Current understanding: " + reasoning_state["current_understanding"] + "\n\n"

        # Add discovered facts
        if reasoning_state["facts"]:
            next_prompt += "Facts discovered:\n"
            for fact in reasoning_state["facts"][-5:]:  # Only include the most recent facts
                next_prompt += f"- {fact}\n"
            next_prompt += "\n"

        # Add current hypotheses
        if reasoning_state["hypotheses"]:
            next_prompt += "Current hypotheses:\n"
            for hypothesis in reasoning_state["hypotheses"][-3:]:  # Only include the most recent hypotheses
                next_prompt += f"- {hypothesis}\n"
            next_prompt += "\n"

        # Add progress indicators
        next_prompt += f"Solution progress: {reasoning_state['solution_progress'] * 100:.1f}%\n"
        next_prompt += f"Confidence: {reasoning_state['confidence'] * 100:.1f}%\n\n"

        # Add instruction for the next step
        next_prompt += "Continue the reasoning process. What's the next step in solving this problem? Consider new insights, identify any errors in previous reasoning, and make progress toward a solution."

        return next_prompt
    