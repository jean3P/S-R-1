"""
Repository Knowledge Graph.

This module provides functionality to extract entities and relationships from Python files
in a repository, creating a knowledge graph that can be queried for relevant information.
"""

import os
import ast
import re
import json
import logging
import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
from collections import defaultdict

from src.utils.logging import get_logger

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False


class RepoKnowledgeGraph:
    """
    Knowledge graph for repository code analysis.
    
    Extracts entities (classes, functions, methods) and relationships from Python files
    and builds a searchable graph that can be queried for relevant information.
    """
    
    def __init__(self, repo_path: str, config: Dict[str, Any] = None):
        """
        Initialize the repository knowledge graph.
        
        Args:
            repo_path: Path to the repository
            config: Configuration dictionary
        """
        self.logger = get_logger(self.__class__.__name__)
        self.repo_path = repo_path
        self.config = config or {}
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Initialize embeddings model if available
        self.embedding_model = None
        self.embeddings = {}
        self.entity_descriptions = {}
        
        if HAVE_SENTENCE_TRANSFORMERS and self.config.get("use_embeddings", True):
            model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
            try:
                self.logger.info(f"Loading embedding model: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
            except Exception as e:
                self.logger.error(f"Error loading embedding model: {e}")
        
        # Cache for file content
        self.file_cache = {}
        
    def build_graph(self, file_paths: List[str] = None) -> None:
        """
        Build the knowledge graph from repository files.
        
        Args:
            file_paths: List of file paths to process (if None, all Python files in repo are processed)
        """
        if file_paths is None:
            # Find all Python files in the repository
            file_paths = []
            for root, _, files in os.walk(self.repo_path):
                for file in files:
                    if file.endswith('.py'):
                        file_paths.append(os.path.join(root, file))
        
        self.logger.info(f"Building knowledge graph from {len(file_paths)} files")
        
        # Process each file
        for file_path in file_paths:
            try:
                self._process_file(file_path)
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
        
        # Build cross-file relationships
        self._build_cross_file_relationships()
        
        # Generate embeddings for all entities
        if self.embedding_model:
            self._generate_embeddings()
            
        self.logger.info(f"Knowledge graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
    
    def _process_file(self, file_path: str) -> None:
        """
        Process a single Python file and extract entities and relationships.
        
        Args:
            file_path: Path to the Python file
        """
        rel_path = os.path.relpath(file_path, self.repo_path)
        self.logger.debug(f"Processing file: {rel_path}")
        
        # Read file content
        content = self._read_file(file_path)
        if not content:
            return
        
        # Parse the file
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {rel_path}: {e}")
            return
        
        # Add file node
        file_node_id = f"file:{rel_path}"
        self.graph.add_node(file_node_id, 
                           type="file", 
                           name=os.path.basename(file_path),
                           path=rel_path)
        
        # Extract imports
        imports = self._extract_imports(tree)
        
        # Add import relationships
        for imp in imports:
            import_node_id = f"import:{imp['module']}"
            self.graph.add_node(import_node_id, 
                               type="import", 
                               name=imp["module"])
            
            self.graph.add_edge(file_node_id, import_node_id, 
                               type="imports",
                               alias=imp.get("alias"))
        
        # Extract classes and functions
        self._extract_classes_and_functions(tree, file_node_id, rel_path, content)
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, str]]:
        """
        Extract import statements from an AST.
        
        Args:
            tree: AST of a Python file
            
        Returns:
            List of import dictionaries
        """
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({
                        "module": name.name,
                        "alias": name.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    if module:
                        full_name = f"{module}.{name.name}"
                    else:
                        full_name = name.name
                    
                    imports.append({
                        "module": full_name,
                        "alias": name.asname,
                        "from_import": True
                    })
        
        return imports
    
    def _extract_classes_and_functions(self, tree: ast.AST, file_node_id: str, 
                                      file_path: str, content: str) -> None:
        """
        Extract classes and functions from an AST.
        
        Args:
            tree: AST of a Python file
            file_node_id: ID of the file node
            file_path: Path to the file
            content: Content of the file
        """
        # Get line mapping for the file
        lines = content.splitlines()
        
        # Process top-level nodes
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                self._process_class(node, file_node_id, file_path, lines)
            elif isinstance(node, ast.FunctionDef):
                self._process_function(node, file_node_id, None, file_path, lines)
    
    def _process_class(self, node: ast.ClassDef, file_node_id: str, 
                      file_path: str, lines: List[str]) -> str:
        """
        Process a class definition.
        
        Args:
            node: Class definition node
            file_node_id: ID of the file node
            file_path: Path to the file
            lines: Lines of the file
            
        Returns:
            ID of the class node
        """
        # Create class node
        class_name = node.name
        class_node_id = f"class:{file_path}:{class_name}"
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{base.value.id}.{base.attr}")
        
        # Get class source code
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
        while end_line < len(lines) and not lines[end_line].strip().startswith('class '):
            end_line += 1
        class_code = '\n'.join(lines[start_line:end_line])
        
        # Add class node
        self.graph.add_node(class_node_id,
                           type="class",
                           name=class_name,
                           docstring=docstring,
                           bases=bases,
                           code=class_code,
                           file=file_path,
                           lineno=node.lineno)
        
        # Add relationship to file
        self.graph.add_edge(file_node_id, class_node_id, type="contains")
        
        # Add base class relationships
        for base in bases:
            base_node_id = f"class:{base}"
            self.graph.add_edge(class_node_id, base_node_id, type="inherits_from")
        
        # Process methods
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                method_node_id = self._process_function(child, file_node_id, class_node_id, file_path, lines)
                self.graph.add_edge(class_node_id, method_node_id, type="has_method")
        
        return class_node_id
    
    def _process_function(self, node: ast.FunctionDef, file_node_id: str, 
                         class_node_id: Optional[str], file_path: str, 
                         lines: List[str]) -> str:
        """
        Process a function definition.
        
        Args:
            node: Function definition node
            file_node_id: ID of the file node
            class_node_id: ID of the class node (if method)
            file_path: Path to the file
            lines: Lines of the file
            
        Returns:
            ID of the function node
        """
        # Create function node
        func_name = node.name
        
        if class_node_id:
            # This is a method
            class_name = class_node_id.split(':')[-1]
            func_node_id = f"method:{file_path}:{class_name}.{func_name}"
            node_type = "method"
        else:
            # This is a function
            func_node_id = f"function:{file_path}:{func_name}"
            node_type = "function"
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract parameters
        params = []
        for arg in node.args.args:
            params.append(arg.arg)
        
        # Extract return annotation if available
        returns = None
        if node.returns:
            if isinstance(node.returns, ast.Name):
                returns = node.returns.id
            elif isinstance(node.returns, ast.Subscript):
                returns = ast.unparse(node.returns)
        
        # Get function source code
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
        while end_line < len(lines) and not (lines[end_line].strip().startswith('def ') or 
                                           lines[end_line].strip().startswith('class ')):
            end_line += 1
        func_code = '\n'.join(lines[start_line:end_line])
        
        # Add function node
        self.graph.add_node(func_node_id,
                           type=node_type,
                           name=func_name,
                           docstring=docstring,
                           params=params,
                           returns=returns,
                           code=func_code,
                           file=file_path,
                           lineno=node.lineno,
                           class_name=class_node_id.split(':')[-1] if class_node_id else None)
        
        # Add relationship to file
        self.graph.add_edge(file_node_id, func_node_id, type="contains")
        
        # Extract function calls
        calls = self._extract_function_calls(node)
        
        # Add call relationships
        for call in calls:
            call_node_id = f"function:{call}"
            self.graph.add_edge(func_node_id, call_node_id, type="calls")
        
        return func_node_id
    
    def _extract_function_calls(self, node: ast.AST) -> Set[str]:
        """
        Extract function calls from an AST node.
        
        Args:
            node: AST node
            
        Returns:
            Set of function names that are called
        """
        calls = set()
        
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call):
                if isinstance(subnode.func, ast.Name):
                    calls.add(subnode.func.id)
                elif isinstance(subnode.func, ast.Attribute):
                    if isinstance(subnode.func.value, ast.Name):
                        calls.add(f"{subnode.func.value.id}.{subnode.func.attr}")
        
        return calls
    
    def _build_cross_file_relationships(self) -> None:
        """Build relationships between entities across different files."""
        # Find all class inheritance relationships
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('type') == 'class':
                for base in node_data.get('bases', []):
                    # Look for the base class in the graph
                    for other_id, other_data in self.graph.nodes(data=True):
                        if (other_data.get('type') == 'class' and 
                            other_data.get('name') == base.split('.')[-1]):
                            self.graph.add_edge(node_id, other_id, type="inherits_from")
        
        # Find all function call relationships
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('type') in ('function', 'method'):
                # Get outgoing call edges
                for _, target_id, edge_data in self.graph.out_edges(node_id, data=True):
                    if edge_data.get('type') == 'calls':
                        # Find the actual function node
                        target_name = target_id.split(':')[-1]
                        for other_id, other_data in self.graph.nodes(data=True):
                            if (other_data.get('type') in ('function', 'method') and 
                                other_data.get('name') == target_name):
                                self.graph.add_edge(node_id, other_id, type="calls")
    
    def _generate_embeddings(self) -> None:
        """Generate embeddings for all entities in the graph."""
        if not self.embedding_model:
            return
        
        self.logger.info("Generating embeddings for knowledge graph entities")
        
        # Prepare texts and their corresponding node IDs
        texts = []
        node_ids = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('type')
            
            if node_type in ('class', 'function', 'method'):
                # Create a descriptive text for the entity
                description = self._create_entity_description(node_id, node_data)
                self.entity_descriptions[node_id] = description
                
                texts.append(description)
                node_ids.append(node_id)
        
        # Generate embeddings in batches
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_ids = node_ids[i:i+batch_size]
            
            try:
                batch_embeddings = self.embedding_model.encode(batch_texts)
                
                # Store embeddings
                for j, node_id in enumerate(batch_ids):
                    self.embeddings[node_id] = batch_embeddings[j]
            except Exception as e:
                self.logger.error(f"Error generating embeddings for batch {i//batch_size}: {e}")
        
        self.logger.info(f"Generated embeddings for {len(self.embeddings)} entities")
    
    def _create_entity_description(self, node_id: str, node_data: Dict[str, Any]) -> str:
        """
        Create a descriptive text for an entity.
        
        Args:
            node_id: ID of the node
            node_data: Node data
            
        Returns:
            Descriptive text
        """
        node_type = node_data.get('type')
        name = node_data.get('name', '')
        docstring = node_data.get('docstring', '')
        
        description = f"{node_type}: {name}\n"
        
        if docstring:
            description += f"Documentation: {docstring}\n"
        
        if node_type == 'class':
            bases = node_data.get('bases', [])
            if bases:
                description += f"Inherits from: {', '.join(bases)}\n"
            
            # Add methods
            methods = []
            for _, method_id in self.graph.out_edges(node_id):
                method_data = self.graph.nodes.get(method_id, {})
                if method_data.get('type') == 'method':
                    methods.append(method_data.get('name', ''))
            
            if methods:
                description += f"Methods: {', '.join(methods)}\n"
        
        elif node_type in ('function', 'method'):
            params = node_data.get('params', [])
            returns = node_data.get('returns')
            
            if params:
                description += f"Parameters: {', '.join(params)}\n"
            
            if returns:
                description += f"Returns: {returns}\n"
            
            # Add function calls
            calls = []
            for _, call_id, edge_data in self.graph.out_edges(node_id, data=True):
                if edge_data.get('type') == 'calls':
                    call_name = call_id.split(':')[-1]
                    calls.append(call_name)
            
            if calls:
                description += f"Calls: {', '.join(calls)}\n"
        
        return description
    
    def _read_file(self, file_path: str) -> Optional[str]:
        """
        Read a file, using cache if available.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content or None if file cannot be read
        """
        if file_path in self.file_cache:
            return self.file_cache[file_path]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.file_cache[file_path] = content
                return content
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def find_relevant_entities(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find entities relevant to a query using embedding similarity.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of relevant entities with their data
        """
        if not self.embedding_model or not self.embeddings:
            self.logger.warning("Embeddings not available, falling back to keyword search")
            return self.find_entities_by_keywords(query, top_k)
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Calculate similarities
            similarities = {}
            for node_id, embedding in self.embeddings.items():
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                similarities[node_id] = similarity
            
            # Sort by similarity
            sorted_entities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Get top-k results
            results = []
            for node_id, similarity in sorted_entities[:top_k]:
                node_data = self.graph.nodes[node_id]
                results.append({
                    "id": node_id,
                    "type": node_data.get('type'),
                    "name": node_data.get('name'),
                    "similarity": similarity,
                    "description": self.entity_descriptions.get(node_id, ""),
                    "code": node_data.get('code', ""),
                    "file": node_data.get('file'),
                    "lineno": node_data.get('lineno')
                })
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error finding relevant entities: {e}")
            return self.find_entities_by_keywords(query, top_k)
    
    def find_entities_by_keywords(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find entities by keyword matching (fallback method).
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of relevant entities with their data
        """
        # Extract keywords from query
        keywords = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query.lower())
        
        # Score entities by keyword matches
        scores = defaultdict(float)
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('type') not in ('class', 'function', 'method'):
                continue
            
            name = node_data.get('name', '').lower()
            docstring = node_data.get('docstring', '').lower()
            
            # Score by name match
            for keyword in keywords:
                if keyword == name:
                    scores[node_id] += 3.0
                elif keyword in name:
                    scores[node_id] += 1.0
                
                # Score by docstring match
                if keyword in docstring:
                    scores[node_id] += 0.5
        
        # Sort by score
        sorted_entities = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        results = []
        for node_id, score in sorted_entities[:top_k]:
            node_data = self.graph.nodes[node_id]
            results.append({
                "id": node_id,
                "type": node_data.get('type'),
                "name": node_data.get('name'),
                "similarity": score,
                "description": self._create_entity_description(node_id, node_data),
                "code": node_data.get('code', ""),
                "file": node_data.get('file'),
                "lineno": node_data.get('lineno')
            })
        
        return results
    
    def get_entity_relationships(self, entity_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get relationships for an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Dictionary of relationship types to lists of related entities
        """
        if entity_id not in self.graph:
            return {}
        
        relationships = defaultdict(list)
        
        # Get outgoing relationships
        for _, target_id, edge_data in self.graph.out_edges(entity_id, data=True):
            rel_type = edge_data.get('type', 'related_to')
            target_data = self.graph.nodes[target_id]
            
            relationships[rel_type].append({
                "id": target_id,
                "type": target_data.get('type'),
                "name": target_data.get('name')
            })
        
        # Get incoming relationships
        for source_id, _, edge_data in self.graph.in_edges(entity_id, data=True):
            rel_type = f"is_{edge_data.get('type', 'related_to')}_by"
            source_data = self.graph.nodes[source_id]
            
            relationships[rel_type].append({
                "id": source_id,
                "type": source_data.get('type'),
                "name": source_data.get('name')
            })
        
        return dict(relationships)
    
    def get_context_for_query(self, query: str, max_entities: int = 5) -> Dict[str, Any]:
        """
        Get context information for a query.
        
        Args:
            query: Query string
            max_entities: Maximum number of entities to include
            
        Returns:
            Dictionary with context information
        """
        # Find relevant entities
        entities = self.find_relevant_entities(query, top_k=max_entities)
        
        context = {
            "entities": [],
            "relationships": {}
        }
        
        # Add entities to context
        for entity in entities:
            context["entities"].append({
                "id": entity["id"],
                "type": entity["type"],
                "name": entity["name"],
                "description": entity["description"],
                "code": entity["code"],
                "file": entity["file"],
                "lineno": entity["lineno"]
            })
            
            # Add relationships for this entity
            entity_id = entity["id"]
            relationships = self.get_entity_relationships(entity_id)
            if relationships:
                context["relationships"][entity_id] = relationships
        
        return context
    
    def save_graph(self, output_path: str) -> None:
        """
        Save the knowledge graph to a file.
        
        Args:
            output_path: Path to save the graph
        """
        # Convert graph to dictionary
        data = {
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        for node_id, node_data in self.graph.nodes(data=True):
            node_dict = {"id": node_id}
            node_dict.update(node_data)
            data["nodes"].append(node_dict)
        
        # Add edges
        for source, target, edge_data in self.graph.edges(data=True):
            edge_dict = {
                "source": source,
                "target": target
            }
            edge_dict.update(edge_data)
            data["edges"].append(edge_dict)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Knowledge graph saved to {output_path}")
    
    def load_graph(self, input_path: str) -> None:
        """
        Load the knowledge graph from a file.
        
        Args:
            input_path: Path to load the graph from
        """
        # Load from file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create new graph
        self.graph = nx.DiGraph()
        
        # Add nodes
        for node_dict in data["nodes"]:
            node_id = node_dict.pop("id")
            self.graph.add_node(node_id, **node_dict)
        
        # Add edges
        for edge_dict in data["edges"]:
            source = edge_dict.pop("source")
            target = edge_dict.pop("target")
            self.graph.add_edge(source, target, **edge_dict)
        
        self.logger.info(f"Knowledge graph loaded from {input_path} with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        
        # Regenerate embeddings if needed
        if self.embedding_model:
            self._generate_embeddings()
