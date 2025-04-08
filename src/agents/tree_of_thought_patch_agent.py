# src/agents/tree_of_thought_patch_agent.py

import os
import re
from typing import Dict, Any, List, Optional
from src.agents.base_agent import BaseAgent
from src.utils.parsing import extract_patches
from src.utils.repo_knowledge_graph import RepoKnowledgeGraph


class TreeOfThoughtPatchAgent(BaseAgent):
    """
    Agent that applies Tree of Thought reasoning to generate and refine GitHub patches.
    Explores multiple reasoning paths concurrently and selects the most promising solution.
    """

    def __init__(self, model_id: str, prompt_id: str, evaluator_id: str, config: Dict[str, Any]):
        super().__init__(model_id, prompt_id, evaluator_id, config)

        # ToT-specific parameters
        self.max_branches = config.get("max_branches", 3)  # Maximum branches to explore at each step
        self.max_depth = config.get("max_depth", 3)  # Maximum reasoning depth
        self.selection_strategy = config.get("selection_strategy", "best_first")  # Strategy for selecting branches
        self.temperature_schedule = config.get("temperature_schedule", [0.7, 0.5, 0.3])  # Decreasing temperature
        
        # Knowledge graph parameters
        self.use_knowledge_graph = config.get("use_knowledge_graph", True)
        self.knowledge_graph = None
        self.kg_cache_dir = config.get("kg_cache_dir", "data/knowledge_graphs")
        self.kg_max_entities = config.get("kg_max_entities", 5)

    def reflect(self, initial_prompt: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Tree of Thought reasoning process for patch generation.

        Args:
            initial_prompt: Initial problem statement
            task: Task details including repository information

        Returns:
            Dictionary containing the reasoning process and results
        """
        self._start_metrics()
        self.task = task
        self.logger.info(f"===== STARTING TREE OF THOUGHT PATCH GENERATION =====")
        self.logger.info(f"Task: {task.get('name')}")
        self.logger.info(f"Task structure: name={task.get('name')}, keys={list(task.keys())}")
        self.logger.info(f"repo_info keys: {list(task.get('repo_info', {}).keys())}")
        self.logger.info(f"Initial prompt length: {len(initial_prompt)} characters")
        
        # Initialize knowledge graph if enabled
        if self.use_knowledge_graph:
            self.logger.info(f"Initializing knowledge graph for repository analysis")
            self._initialize_knowledge_graph(task)

        # Initialize the reasoning tree with the root node
        reasoning_tree = {
            "root": {
                "id": "root",
                "prompt": initial_prompt,
                "children": [],
                "depth": 0,
                "solution": None,
                "evaluation": None,
                "status": "pending"
            }
        }
        self.logger.info(f"Initialized reasoning tree with root node")

        # Track active branches for exploration
        active_branches = ["root"]

        # Store the best solution found
        best_solution = {
            "patch": None,
            "score": float('-inf'),
            "node_id": None
        }
        self.logger.info(f"Maximum exploration depth: {self.max_depth}, branches per node: {self.max_branches}")

        # Execute ToT reasoning process
        for depth in range(1, self.max_depth + 1):
            self.logger.info(f"===== EXPLORING REASONING DEPTH {depth}/{self.max_depth} =====")

            # Set temperature for this depth
            temperature = self.temperature_schedule[min(depth - 1, len(self.temperature_schedule) - 1)]
            self.logger.info(f"Using temperature {temperature} for depth {depth}")

            # Generate and evaluate new branches
            next_active_branches = []
            self.logger.info(f"Processing {len(active_branches)} active branches at depth {depth}")

            for parent_id in active_branches:
                parent_node = reasoning_tree[parent_id]
                self.logger.info(f"Generating branches from parent node {parent_id}")

                # Generate branches (different reasoning paths)
                branches = self._generate_branches(parent_node, depth, temperature, self.max_branches)
                self.logger.info(f"Generated {len(branches)} branches from parent {parent_id}")

                for branch_id, branch in branches.items():
                    reasoning_tree[branch_id] = branch
                    parent_node["children"].append(branch_id)

                    # Evaluate the solution if one was generated
                    if branch.get("solution"):
                        self.logger.info(f"Evaluating solution from branch {branch_id} (length: {len(branch['solution'])})")
                        evaluation = self._evaluate_solution(branch["solution"])
                        branch["evaluation"] = evaluation

                        # Update best solution if better
                        score = self._calculate_solution_score(evaluation)
                        if score > best_solution["score"]:
                            best_solution["patch"] = branch["solution"]
                            best_solution["score"] = score
                            best_solution["node_id"] = branch_id
                            self.logger.info(f"✅ Found better solution with score {score:.2f} in branch {branch_id}")
                        else:
                            self.logger.info(f"Solution score {score:.2f} not better than current best {best_solution['score']:.2f}")

                    # Add promising branches to next iteration
                    if branch["status"] == "active":
                        next_active_branches.append(branch_id)

            # Update active branches for next iteration
            active_branches = self._select_branches(next_active_branches, reasoning_tree)
            self.logger.info(f"Selected {len(active_branches)} branches for depth {depth+1}")

            # Early stopping if we've found a good solution
            if best_solution["score"] >= self.config.get("early_stop_threshold", 0.8):
                self.logger.info(f"🎯 Early stopping at depth {depth}: found good solution with score {best_solution['score']:.2f}")
                break

            # Also stop if we have no active branches
            if not active_branches:
                self.logger.info(f"⛔ Stopping at depth {depth}: no active branches")
                break

        # If we didn't find a solution but generated patches, use the last one
        if best_solution["patch"] is None:
            self.logger.info(f"No solution evaluated as successful, searching for any valid patch...")
            # Find any branch with a solution
            for branch_id, branch in reasoning_tree.items():
                if branch.get("solution"):
                    best_solution["patch"] = branch["solution"]
                    best_solution["node_id"] = branch_id
                    best_solution["score"] = 0.5  # Assign a moderate score
                    self.logger.info(f"Using solution from {branch_id} as fallback (score: 0.5)")
                    break
            
            if best_solution["patch"] is None:
                self.logger.info(f"❌ No valid patch found in any branch")
            else:
                self.logger.info(f"Found fallback solution of length {len(best_solution['patch'])}")

        # Finalize metrics
        self._end_metrics()
        self.logger.info(f"Tree of Thought exploration completed in {self.metrics.get('average_generation_time', 0):.2f}s average per branch")

        # Extract reasoning path if we have a solution
        reasoning_path = []
        if best_solution["node_id"]:
            self.logger.info(f"Extracting reasoning path for node {best_solution['node_id']}")
            # Simple path extraction - just the node itself
            node = reasoning_tree.get(best_solution["node_id"])
            if node:
                reasoning_path = [{
                    "depth": node["depth"],
                    "reasoning": node.get("reasoning", ""),
                    "solution": node.get("solution", "")
                }]
                self.logger.info(f"Extracted reasoning path at depth {node['depth']}")

        # Prepare results
        results = {
            "task": task,
            "reasoning_tree": reasoning_tree,
            "best_solution": best_solution["patch"],
            "best_node_id": best_solution["node_id"],
            "reasoning_path": reasoning_path,
            "metrics": self.metrics,
            "success": best_solution["patch"] is not None
        }
        
        # Save results to JSON file if configured
        self._save_results(results, task)
        
        self.logger.info(f"===== TREE OF THOUGHT PATCH GENERATION COMPLETED =====")
        self.logger.info(f"Final solution found: {'Yes' if best_solution['patch'] else 'No'}")
        self.logger.info(f"Best solution score: {best_solution['score']:.2f}")
        self.logger.info(f"Solution length: {len(best_solution['patch']) if best_solution['patch'] else 0} characters")
        
        return results
    
    def _initialize_knowledge_graph(self, task: Dict[str, Any]) -> None:
        """
        Initialize the knowledge graph for the repository.
        
        Args:
            task: Task details including repository information
        """
        import os
        import time
        
        # Get repository information
        repo_info = task.get("repo_info", {})
        repo_name = repo_info.get("repo", "")
        
        if not repo_name:
            self.logger.warning("⚠️ Repository name not found in task, knowledge graph disabled")
            self.use_knowledge_graph = False
            return
        
        # Form repository path
        repos_dir = self.config.get("repos_dir", "data/repositories")
        repo_path = os.path.join(repos_dir, repo_name.replace("/", "_"))
        self.logger.info(f"Repository path: {repo_path}")
        
        if not os.path.exists(repo_path):
            self.logger.warning(f"⚠️ Repository path not found: {repo_path}, knowledge graph disabled")
            self.use_knowledge_graph = False
            return
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.kg_cache_dir, exist_ok=True)
        
        # Check if we have a cached knowledge graph
        cache_file = os.path.join(self.kg_cache_dir, f"{repo_name.replace('/', '_')}.json")
        self.logger.info(f"Knowledge graph cache file: {cache_file}")
        
        # Initialize knowledge graph
        kg_config = self.config.get("knowledge_graph_config", {})
        self.knowledge_graph = RepoKnowledgeGraph(repo_path, kg_config)
        
        if os.path.exists(cache_file):
            # Load from cache
            self.logger.info(f"📂 Loading knowledge graph from cache: {cache_file}")
            try:
                self.knowledge_graph.load_graph(cache_file)
                self.logger.info("✅ Knowledge graph loaded successfully from cache")
                return
            except Exception as e:
                self.logger.error(f"❌ Error loading knowledge graph from cache: {e}")
                self.logger.info("Will build knowledge graph from scratch")
                # Fall through to build the graph
        else:
            self.logger.info("No cached knowledge graph found, building from scratch")
        
        # Build the knowledge graph
        self.logger.info(f"🔄 Building knowledge graph for repository: {repo_name}")
        start_time = time.time()
        
        try:
            # Limit the number of files to process for large repositories
            max_files = self.config.get("kg_max_files", 300)
            
            # Get Python files in the repository
            python_files = []
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
                        if len(python_files) >= max_files:
                            break
                if len(python_files) >= max_files:
                    break
            
            self.logger.info(f"Building knowledge graph from {len(python_files)} files (limited to {max_files})")
            
            # Build the graph with the limited set of files
            self.knowledge_graph.build_graph(python_files)
            
            build_time = time.time() - start_time
            self.logger.info(f"✅ Knowledge graph built in {build_time:.2f} seconds")
            
            # Save to cache
            self.knowledge_graph.save_graph(cache_file)
            self.logger.info(f"💾 Knowledge graph saved to cache: {cache_file}")
        except Exception as e:
            self.logger.error(f"❌ Error building knowledge graph: {e}")
            self.logger.warning("Knowledge graph functionality will be disabled")
            self.use_knowledge_graph = False
    
    def _save_results(self, results: Dict[str, Any], task: Dict[str, Any]) -> None:
        """
        Save results to a JSON file.
        
        Args:
            results: Results to save
            task: Task details
        """
        import json
        import os
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        output_dir = self.config.get("output_dir", "results/swe_bench")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a filename based on task name and timestamp
        task_name = task.get("name", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_name}_{timestamp}.json"
        
        # Full path to output file
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Save results to file
            with open(output_path, 'w', encoding='utf-8') as f:
                # Use a smaller subset of results to avoid huge files
                compact_results = {
                    "task_name": task.get("name", "unknown"),
                    "timestamp": timestamp,
                    "success": results.get("success", False),
                    "best_solution": results.get("best_solution", ""),
                    "metrics": results.get("metrics", {})
                }
                
                # Add reasoning path if available
                if results.get("reasoning_path"):
                    compact_results["reasoning_path"] = results["reasoning_path"]
                
                json.dump(compact_results, f, indent=2)
                
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results to {output_path}: {e}")

    def _generate_branches(self, parent_node: Dict[str, Any], depth: int,
                           temperature: float, max_branches: int) -> Dict[str, Dict[str, Any]]:
        """Generate new reasoning branches from a parent node."""
        branches = {}
        self.logger.info(f"Generating {max_branches} branches at depth {depth} from parent {parent_node['id']}")

        # Create distinct reasoning prompts for different branches
        for i in range(max_branches):
            try:
                # Generate prompt that includes specific reasoning strategy or focus
                self.logger.info(f"Creating prompt for branch {i+1}/{max_branches}")
                branch_prompt = self._create_branch_prompt(parent_node, depth, i)
                prompt_length = len(branch_prompt)
                self.logger.info(f"Branch {i+1} prompt created (length: {prompt_length} chars)")

                # Generate a solution for this branch
                self.logger.info(f"🔄 Generating branch {i+1}/{max_branches} at depth {depth} with temperature {temperature}")
                solution, generation_time = self._measure_execution(
                    self.model.generate,
                    branch_prompt
                )
                self.logger.info(f"✅ Generation completed in {generation_time:.2f}s (response length: {len(solution)} chars)")

                # Extract patch if present
                patches = self._extract_patches_improved(solution)
                patch = patches[0] if patches else None
                
                if patch:
                    self.logger.info(f"📄 Extracted patch of length {len(patch)}")
                else:
                    self.logger.info("⚠️ No patch extracted from response, trying alternative extraction")
                    # Try to extract patch from code blocks if no patch was found
                    import re
                    code_blocks = re.findall(r'```(?:diff)?(.*?)```', solution, re.DOTALL)
                    self.logger.info(f"Found {len(code_blocks)} code blocks in response")
                    for block in code_blocks:
                        if 'diff --git' in block or '+++' in block or '---' in block:
                            patch = block.strip()
                            self.logger.info(f"📄 Extracted patch from code block, length: {len(patch)}")
                            break

                # Create branch node
                branch_id = f"{parent_node['id']}-{depth}-{i}"
                branches[branch_id] = {
                    "id": branch_id,
                    "parent": parent_node["id"],
                    "depth": depth,
                    "branch_index": i,
                    "prompt": branch_prompt,
                    "reasoning": solution,
                    "solution": patch,
                    "evaluation": None,
                    "children": [],
                    "status": "active" if patch else "terminated"
                }
                
                if patch:
                    self.logger.info(f"Branch {branch_id} created with valid patch")
                else:
                    self.logger.info(f"Branch {branch_id} created but no valid patch found (status: terminated)")
                
            except Exception as e:
                self.logger.error(f"❌ Error generating branch {i} at depth {depth}: {str(e)}")
                # Create a fallback branch node
                branch_id = f"{parent_node['id']}-{depth}-{i}"
                branches[branch_id] = {
                    "id": branch_id,
                    "parent": parent_node["id"],
                    "depth": depth,
                    "branch_index": i,
                    "prompt": "Error generating prompt",
                    "reasoning": f"Error occurred: {str(e)}",
                    "solution": None,
                    "evaluation": None,
                    "children": [],
                    "status": "terminated"
                }
                self.logger.info(f"Created fallback branch {branch_id} due to error")

        self.logger.info(f"Generated {len(branches)} branches, {sum(1 for b in branches.values() if b['status'] == 'active')} active")
        return branches
        
    def _extract_patches_improved(self, text: str) -> List[str]:
        """
        Extract patches from text with improved heuristics.
        
        This method uses multiple strategies to find git patches in the text:
        1. Look for standard diff format
        2. Look for code blocks that might contain patches
        3. Look for sections that have patch-like formatting
        
        Args:
            text: Text to extract patches from
            
        Returns:
            List of extracted patches
        """
        
        # First try the standard extract_patches function
        patches = extract_patches(text)
        if patches:
            return patches
            
        # Try to find diff blocks
        diff_blocks = []
        
        # Look for diff --git style patches
        diff_pattern = re.compile(r'diff --git.*?(?=diff --git|\Z)', re.DOTALL)
        matches = diff_pattern.findall(text)
        diff_blocks.extend(matches)
        
        # Look for +++ and --- style patches
        plusminus_pattern = re.compile(r'(?:---.*?\n\+\+\+.*?\n)(?:@@.*?@@.*?)(?=\n---|\Z)', re.DOTALL)
        matches = plusminus_pattern.findall(text)
        diff_blocks.extend(matches)
        
        # Look for code blocks that might contain patches
        code_block_pattern = re.compile(r'```(?:diff)?(.*?)```', re.DOTALL)
        code_blocks = code_block_pattern.findall(text)
        
        for block in code_blocks:
            if 'diff --git' in block or '+++' in block or '---' in block or '@@ ' in block:
                diff_blocks.append(block.strip())
                
        # Clean up and return unique patches
        cleaned_patches = []
        for patch in diff_blocks:
            # Clean up the patch
            patch = patch.strip()
            if patch and len(patch) > 10:  # Minimum size for a valid patch
                cleaned_patches.append(patch)
                
        return list(set(cleaned_patches))

    def _create_branch_prompt(self, parent_node: Dict[str, Any], depth: int, branch_index: int) -> str:
        """Create a prompt for a specific reasoning branch."""
        # Base prompt from parent
        base_prompt = parent_node["prompt"]

        # Different reasoning strategies based on branch_index
        strategies = [
            "Focus on identifying the minimal required changes",
            "Focus on ensuring tests will pass",
            "Focus on maintaining code style consistency"
        ]

        # Add specific guidance from the parent reasoning if available
        parent_reasoning = parent_node.get("reasoning", "")

        # Select a strategy based on branch index (cycle through if more branches than strategies)
        strategy = strategies[branch_index % len(strategies)]
        
        # Prepare code context for this branch
        code_context = self._prepare_code_context(parent_node, depth, branch_index)

        # Create comprehensive prompt template
        prompt_template = self.prompt.format_tot_reasoning(
            original_prompt=base_prompt,
            parent_reasoning=parent_reasoning,
            depth=depth,
            strategy=strategy,
            task=self.task,
            context=code_context
        )

        return prompt_template
        
    def _prepare_code_context(self, parent_node: Dict[str, Any], depth: int, branch_index: int) -> Dict[str, Any]:
        """
        Prepare code context for a specific branch.
        
        Args:
            parent_node: Parent node in the reasoning tree
            depth: Current reasoning depth
            branch_index: Branch index
            
        Returns:
            Dictionary with code context
        """
        # Initialize code context
        code_context = {}
        
        # If we have task information with repo_info and test_info
        if self.task and "repo_info" in self.task:
            repo_info = self.task.get("repo_info", {})
            repo_name = repo_info.get("repo", "")
            
            # Extract file paths from the problem statement or parent reasoning
            problem_statement = self.task.get("initial_prompt", "")
            parent_reasoning_text = parent_node.get("reasoning", "")
            
            # Combine texts to extract file paths
            combined_text = problem_statement + "\n" + parent_reasoning_text
            
            # Use knowledge graph if available
            if self.use_knowledge_graph and self.knowledge_graph:
                # Get knowledge graph context
                kg_context = self._get_knowledge_graph_context(combined_text)
                if kg_context:
                    code_context["knowledge_graph"] = kg_context
            
            # Extract file paths and get their content
            file_paths = self._extract_file_paths_from_text(combined_text)
            
            # If we have file paths, try to get their content
            if file_paths:
                relevant_files = {}
                for file_path in file_paths[:3]:  # Limit to 3 files to avoid context overflow
                    file_content = self._get_file_content(repo_name, file_path)
                    if file_content:
                        relevant_files[file_path] = file_content
                
                if relevant_files:
                    code_context["relevant_files"] = relevant_files
            
            # Extract error information if available
            error_info = self._extract_error_info(combined_text)
            if error_info:
                code_context["code_context"] = {
                    "error_location": error_info
                }
                
            # If we have a previous solution, include it
            if parent_node.get("solution"):
                code_context["code_context"] = code_context.get("code_context", {})
                code_context["code_context"]["previous_solution"] = parent_node["solution"]
        
        return code_context
    
    def _get_knowledge_graph_context(self, query_text: str) -> Dict[str, Any]:
        """
        Get context from the knowledge graph based on the query text.
        
        Args:
            query_text: Query text to search for relevant entities
            
        Returns:
            Dictionary with knowledge graph context
        """
        if not self.knowledge_graph:
            return {}
        
        try:
            # Get context for the query
            context = self.knowledge_graph.get_context_for_query(
                query_text, 
                max_entities=self.kg_max_entities
            )
            
            # Format the context for inclusion in the prompt
            formatted_context = {
                "relevant_entities": [],
                "entity_relationships": {}
            }
            
            # Format entities
            for entity in context.get("entities", []):
                formatted_entity = {
                    "type": entity.get("type"),
                    "name": entity.get("name"),
                    "description": entity.get("description"),
                    "file": entity.get("file"),
                    "code_snippet": entity.get("code")
                }
                formatted_context["relevant_entities"].append(formatted_entity)
            
            # Format relationships
            for entity_id, relationships in context.get("relationships", {}).items():
                entity_name = None
                for entity in context.get("entities", []):
                    if entity.get("id") == entity_id:
                        entity_name = entity.get("name")
                        break
                
                if entity_name:
                    formatted_relationships = {}
                    for rel_type, related_entities in relationships.items():
                        formatted_relationships[rel_type] = [
                            {"type": e.get("type"), "name": e.get("name")}
                            for e in related_entities
                        ]
                    
                    formatted_context["entity_relationships"][entity_name] = formatted_relationships
            
            return formatted_context
        
        except Exception as e:
            self.logger.error(f"Error getting knowledge graph context: {e}")
            return {}
        
    def _extract_file_paths_from_text(self, text: str) -> List[str]:
        """
        Extract file paths from text.
        
        Args:
            text: Text to extract file paths from
            
        Returns:
            List of file paths
        """
        
        # Look for file paths with extensions
        file_pattern = re.compile(r'\b([a-zA-Z0-9_/.-]+\.(py|sql|js|html|css|java|cpp|h|md|json|yaml|yml))\b')
        matches = file_pattern.findall(text)
        file_paths = [m[0] for m in matches]  # Return just the full file paths
        
        # Also look for paths mentioned in diff format
        diff_pattern = re.compile(r'(?:---|\+\+\+) [ab]/(.+?)(?:\s|$)')
        diff_matches = diff_pattern.findall(text)
        file_paths.extend(diff_matches)
        
        # Look for file paths in error messages
        error_pattern = re.compile(r'(?:Error|Exception|Traceback).*?[\'"]([a-zA-Z0-9_/.-]+\.[a-zA-Z0-9]+)[\'"]', re.IGNORECASE)
        error_matches = error_pattern.findall(text)
        file_paths.extend(error_matches)
        
        # Remove duplicates while preserving order
        unique_paths = []
        seen = set()
        for path in file_paths:
            # Clean path by removing a/ or b/ prefixes
            if path.startswith('a/') or path.startswith('b/'):
                path = path[2:]
            
            # Remove .orig suffix if present
            if path.endswith('.orig'):
                path = path[:-5]
                
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
                
        return unique_paths
        
    def _extract_error_info(self, text: str) -> Dict[str, Any]:
        """
        Extract error information from text.
        
        Args:
            text: Text to extract error information from
            
        Returns:
            Dictionary with error information
        """
        import re
        
        error_info = {}
        
        # Look for file and line information in error messages
        file_line_pattern = re.compile(r'File "([^"]+)", line (\d+)')
        matches = file_line_pattern.findall(text)
        
        if matches:
            file_path, line_number = matches[0]
            error_info["file"] = file_path
            error_info["line"] = line_number
            
            # Try to extract function/class context
            function_pattern = re.compile(r'in (\w+)')
            function_matches = function_pattern.findall(text)
            if function_matches:
                error_info["function"] = function_matches[0]
        
        return error_info
        
    def _get_file_content(self, repo_name: str, file_path: str) -> Optional[str]:
        """
        Get content of a file from the repository.
        
        Args:
            repo_name: Repository name
            file_path: Path to the file
            
        Returns:
            File content or None if file cannot be read
        """
        
        # Form repository path
        repo_path = os.path.join(self.config.get("repos_dir", "data/repositories"), 
                                repo_name.replace("/", "_"))
        
        # Form full file path - remove any 'a/' or 'b/' prefixes from git diff format
        clean_file_path = file_path
        if clean_file_path.startswith('a/') or clean_file_path.startswith('b/'):
            clean_file_path = clean_file_path[2:]
        
        # Remove .orig suffix if present
        if clean_file_path.endswith('.orig'):
            clean_file_path = clean_file_path[:-5]
            
        full_path = os.path.join(repo_path, clean_file_path)
        
        # Try to read the file
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            # Try alternative paths if the direct path fails
            try:
                # Try without any directory structure
                base_filename = os.path.basename(clean_file_path)
                alt_path = os.path.join(repo_path, base_filename)
                with open(alt_path, 'r', encoding='utf-8') as f:
                    self.logger.info(f"Found file at alternative path: {alt_path}")
                    return f.read()
            except Exception:
                # Try searching for the file in the repository
                try:
                    import glob
                    possible_files = glob.glob(f"{repo_path}/**/{os.path.basename(clean_file_path)}", recursive=True)
                    if possible_files:
                        with open(possible_files[0], 'r', encoding='utf-8') as f:
                            self.logger.info(f"Found file by searching: {possible_files[0]}")
                            return f.read()
                except Exception:
                    pass
                    
            self.logger.error(f"Error reading file {full_path}: {e}")
            return None

    def _evaluate_solution(self, patch: str) -> Dict[str, Any]:
        """Evaluate a solution patch."""
        if not patch:
            return {"success": False, "error": "No patch generated"}

        # Use the evaluator to test the patch
        try:
            self.logger.info("🧪 Evaluating patch with evaluator")

            # Ensure task has required fields for SWE-bench evaluation
            if self.task is None:
                self.logger.error("❌ Task is None, cannot evaluate")
                return {"success": False, "error": "Task information is missing"}

            # Ensure task has the required structure before passing to evaluator
            if not isinstance(self.task, dict):
                self.logger.error(f"❌ Task is not a dictionary: {type(self.task)}")
                return {"success": False, "error": "Task is not in the correct format"}
                
            # Ensure task has the minimum required fields
            required_fields = ['name', 'repo_info', 'test_info']
            missing_fields = [field for field in required_fields if field not in self.task]
            if missing_fields:
                self.logger.error(f"❌ Task is missing required fields: {missing_fields}")
                return {"success": False, "error": f"Task is missing required fields: {missing_fields}"}
                
            # Log task structure for debugging
            self.logger.info(f"Task keys: {list(self.task.keys())}")
            self.logger.info(f"repo_info keys: {list(self.task.get('repo_info', {}).keys())}")
            self.logger.info(f"test_info keys: {list(self.task.get('test_info', {}).keys())}")

            # Call evaluate with the task as a keyword argument
            self.logger.info(f"Calling evaluator with patch of length {len(patch)}")
            output, errors = self.evaluator.evaluate(patch, task=self.task)

            if errors:
                self.logger.info(f"❌ Evaluation failed with errors: {errors[:100]}...")
            else:
                self.logger.info(f"✅ Evaluation succeeded")
                
            self.logger.info(
                f"Evaluation result: output={output[:100] if output else ''}, errors={errors[:100] if errors else 'None'}")

            # Parse evaluation results
            return {
                "success": not errors,
                "output": output,
                "errors": errors
            }
        except Exception as eval_error:
            self.logger.error(f"❌ Evaluator error: {str(eval_error)}")
            return {
                "success": False,
                "output": "",
                "errors": f"Evaluation error: {str(eval_error)}"
            }

    def _calculate_solution_score(self, evaluation: Dict[str, Any]) -> float:
        """Calculate a score for a solution based on its evaluation."""
        if not evaluation:
            return float('-inf')

        # Start with base score
        score = 0.0

        # Success is the most important factor
        if evaluation.get("success", False):
            score += 0.8

        # Consider other factors like test results
        output = evaluation.get("output", "")
        if "passing" in output.lower() or "passed" in output.lower():
            score += 0.2

        # Penalize for errors
        if evaluation.get("errors"):
            score -= 0.3

        return score

    def _select_branches(self, branches: List[str], reasoning_tree: Dict[str, Dict[str, Any]]) -> List[str]:
        """Select which branches to continue exploring."""
        if not branches:
            self.logger.info("No branches to select from")
            return []

        self.logger.info(f"Selecting branches using strategy: {self.selection_strategy}")
        self.logger.info(f"Selecting from {len(branches)} candidate branches")

        if self.selection_strategy == "breadth_first":
            # Explore all branches up to max_branches
            selected = branches[:self.max_branches]
            self.logger.info(f"Breadth-first selection: {len(selected)}/{len(branches)} branches")
            return selected

        elif self.selection_strategy == "best_first":
            # Sort branches by evaluation score and take the best ones
            scored_branches = []
            for branch_id in branches:
                branch = reasoning_tree[branch_id]
                
                # Default score based on whether a solution exists
                if branch.get("solution"):
                    # Give a higher base score to branches that have a solution
                    score = 0.5
                else:
                    score = 0.0

                # Calculate branch score from evaluation if available
                if branch.get("evaluation"):
                    score = self._calculate_solution_score(branch["evaluation"])

                scored_branches.append((branch_id, score))
                self.logger.info(f"Branch {branch_id} scored {score:.2f}")

            # Sort by score (descending) and take top branches
            scored_branches.sort(key=lambda x: x[1], reverse=True)
            selected = [branch_id for branch_id, _ in scored_branches[:self.max_branches]]
            
            # Log selection details
            self.logger.info(f"Best-first selection: {len(selected)}/{len(branches)} branches")
            if selected:
                top_scores = [f"{branch_id}:{score:.2f}" for branch_id, score in scored_branches[:self.max_branches]]
                self.logger.info(f"Top branches: {', '.join(top_scores)}")
            
            return selected

        else:  # Default to all branches
            self.logger.info(f"Using default selection strategy (all branches)")
            return branches
