# S-R-1: Self-Reasoning LLM Framework

S-R-1 is a framework for LLM-based self-reasoning and code refinement, allowing language models to iteratively improve their solutions through execution feedback and self-reflection.

## Overview

This framework enables large language models to:
- Generate initial solutions to programming problems
- Execute the code and collect feedback
- Refine solutions through multiple iterations of self-reasoning
- Evaluate code quality, correctness, and efficiency

## System Architecture

The system follows a modular design with the following components:

- **Agents**: Implement self-reflection strategies (CodeRefinementAgent, ReasoningAgent)
- **Models**: Interface with language models (HuggingFace, OpenAI, Anthropic)
- **Prompts**: Manage templates for model interactions
- **Evaluators**: Execute and analyze generated code
- **Datasets**: Store and manage problem data
- **Utils**: Provide helper functions for logging, parsing, tokenization, etc.

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU 
- 90+ GB RAM recommended for 32B+ parameter models

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/S-R-1.git
   cd S-R-1
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -e .
   ```

## Running Experiments

### Using Slurm (HPC Cluster)

The repository includes a Slurm job script for running experiments on HPC clusters with GPU resources:

```bash
sbatch run_gpu_job.sh
```

### Locally (Manual Execution)

If you don't have access to a Slurm cluster, you can run the experiment locally:

1. Create the necessary directories:
   ```bash
   mkdir -p configs/{models,agents,prompts,evaluators,experiments} results problems
   ```

2. Create configuration files as shown in the `run_gpu_job.sh` script

3. Run the experiment:
   ```bash
   # Set environment variables
   export PYTHONPATH=$(pwd)
   export TOKENIZERS_PARALLELISM=false
   
   # Create experiment configuration
   python -m src.main create \
     --name "experiment_name" \
     --agent code_refinement \
     --model your_model_id \
     --prompt code_gen \
     --evaluator python_exec \
     --task "./problems/your_problem.txt"
   
   # Run the experiment
   python -m src.main run \
     --config "configs/experiments/experiment_name.yaml" \
     --output-dir "results/experiment_name_$(date +%Y%m%d_%H%M%S)" \
     --log-level INFO
   ```

## Configuration

### Model Configuration

Create a model configuration in `configs/models/your_model.yaml`:

```yaml
id: "your_model_id"
type: "huggingface"  # or "openai", "anthropic"
config:
  model_name: "ModelProvider/ModelName"
  device_map: "auto"
  use_fp16: true
  # Additional model-specific parameters
```

### Problem Definition

Create problem files in `problems/` directory:

```
Write a Python function that...
- Requirement 1
- Requirement 2
...

Examples:
- Example 1 input/output
- Example 2 input/output
```

### Supported Models

- HuggingFace models (local execution)
- OpenAI models (API-based)
- Anthropic Claude models (API-based)

## Available Commands

The system supports several commands:

```bash
# List available components
python -m src.main list --type models
python -m src.main list --type agents
python -m src.main list --type prompts
python -m src.main list --type evaluators

# Create experiment configuration
python -m src.main create --name "experiment_name" --agent agent_id --model model_id --prompt prompt_id --evaluator evaluator_id --task "path/to/problem.txt"

# Run experiment
python -m src.main run --config "configs/experiments/experiment_name.yaml" --output-dir "results/output_dir"

# Generate a solution without creating an experiment
python -m src.main generate --model model_id --prompt prompt_id --evaluator evaluator_id --problem "path/to/problem.txt" --iterations 3
```

## Important Notes

### Using with Qwen Models

When using Qwen models, the system automatically registers tokenizers. To avoid warnings, create a `src/utils/tokenizer_config.py` file with the following content:

```python
from transformers import AutoTokenizer
from src.utils.tokenization import register_tokenizer
from src.utils.logging import get_logger

logger = get_logger("tokenizer_config")

def register_qwen_tokenizer():
    """Register tokenizer for Qwen2.5 models."""
    try:
        qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
        
        def qwen_tokenize_fn(text):
            """Tokenize text using Qwen tokenizer."""
            return qwen_tokenizer.encode(text)
        
        register_tokenizer("Qwen/Qwen2.5-Coder-32B-Instruct", qwen_tokenize_fn)
        register_tokenizer("qwen", qwen_tokenize_fn)
        
        logger.info("Successfully registered Qwen tokenizer")
        return True
    except Exception as e:
        logger.warning(f"Failed to register Qwen tokenizer: {e}")
        return False

register_qwen_tokenizer()
```

### Early Stopping Behavior

By default, the CodeRefinementAgent uses an early stopping strategy, ending iterations when a successful solution is found. To disable this and force the system to perform all iterations regardless of success:

1. Edit the agent configuration:
   ```yaml
   # configs/agents/code_refinement.yaml
   id: "code_refinement"
   type: "code_refinement"
   config:
     max_iterations: 3
     early_stop_on_success: false  # Set to false to disable early stopping
     save_results: true
     output_dir: "results"
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).