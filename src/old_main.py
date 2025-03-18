# src/main.py

import os
import json
import logging
import subprocess
import tempfile
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("../experiment.log", mode="a")
        ]
    )
    return logging.getLogger(__name__)


def load_model(model_name):
    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    logger.info("Model loaded successfully.")
    # Return both generator and tokenizer for later use
    return generator, tokenizer


def generate_code(generator, tokenizer, prompt, additional_tokens=100):
    # Tokenize the prompt to get its length
    prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1]
    # Set max_length dynamically
    max_length = prompt_tokens + additional_tokens
    logger.info("Generating code for prompt (prompt tokens: %d, max_length: %d):\n%s", prompt_tokens, max_length, prompt)
    outputs = generator(prompt, max_length=max_length, do_sample=True, top_p=0.95, num_return_sequences=1)
    generated_text = outputs[0]['generated_text']
    logger.info("Generated code:\n%s", generated_text)
    return generated_text


def evaluate_code(code):
    # Save the code to a temporary file and try to execute it
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(code)
        tmp_filename = tmp_file.name
    logger.info("Evaluating code from temporary file: %s", tmp_filename)

    try:
        result = subprocess.run(
            ["python", tmp_filename],
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout
        errors = result.stderr
    except Exception as e:
        output = ""
        errors = str(e)
    # Clean up the temporary file
    os.remove(tmp_filename)
    logger.info("Execution output:\n%s", output)
    logger.info("Execution errors:\n%s", errors)
    return output, errors


def main():
    global logger
    logger = setup_logging()

    # Setup model and parameters
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
    generator, tokenizer = load_model(model_name)

    # Number of iterations for self-reasoning loop
    K = 3
    initial_prompt = ("# TASK: Write a Python function that checks if a number is prime.\n"
                      "# Please only focus in to provide the solution, then the solution is:")

    # List to store experiment details for each iteration
    experiment_results = []

    # Start with the initial prompt
    prompt = initial_prompt
    for i in range(1, K + 1):
        logger.info("========== Iteration %d ==========", i)
        # Generation: Create a solution based on the current prompt
        solution = generate_code(generator, tokenizer, prompt)

        # Execution: Evaluate the generated solution by running it as a script
        output, errors = evaluate_code(solution)

        # Reprompting: Concatenate the solution and its execution result, and ask for improvements
        reprompt = (
            solution +
            "\n\n# Execution Output:\n" + output +
            "\n# Execution Errors:\n" + errors +
            "\n\n# Based on the above, please refine the solution if needed."
        )
        logger.info("Reprompting with the following prompt:\n%s", reprompt)

        # Generate an improved solution from the reprompt
        refined_solution = generate_code(generator, tokenizer, reprompt)

        # Log the iteration results in a dictionary
        iteration_result = {
            "iteration": i,
            "timestamp": datetime.now().isoformat(),
            "initial_prompt": prompt,
            "solution": solution,
            "execution_output": output,
            "execution_errors": errors,
            "reprompt": reprompt,
            "refined_solution": refined_solution
        }
        experiment_results.append(iteration_result)

        # Use the refined solution as the new prompt for the next iteration
        prompt = refined_solution

    # Save all results to a JSON file
    output_filename = "experiment_results.json"
    with open(output_filename, "w") as json_file:
        json.dump(experiment_results, json_file, indent=4)
    logger.info("Experiment completed. Results saved to %s", output_filename)


if __name__ == "__main__":
    main()
