import argparse
import time 

from typing import Literal

from dotenv import load_dotenv
from inspect_ai import eval_set, eval

from mountain.tasks import agentharm, agentharm_benign
import random
import numpy as np
import yaml

load_dotenv()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def run_eval_set_with_different_context_length(
        model_id: str,
        task_type: Literal["harmful", "benign"],
        context_lengths: list[int],
        padding_type: Literal["random"]| None,
        log_subdir_name: str,
):
    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_"
        if task_type == "harmful":
            tasks.append(agentharm(padding_type=padding_type, padding_token_length=context_length, name_prefix=name_prefix))
        elif task_type == "benign":
            tasks.append(agentharm_benign(padding_type=padding_type, padding_token_length=context_length, name_prefix=name_prefix))

    logs = eval_set(tasks=tasks, log_dir= f"logs/{log_subdir_name}" , model=model_id)
    return logs



def eval_gpt41_nano_random_all_contexts_val_subset_benign():
    model_id="openai/gpt-4.1-nano-2025-04-14"
    context_lengths=[0, 1000, 10_000, 50_000, 100_000, 250_000, 500_000, 990_000]
    padding_type="random" 

    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_val_subset_"
        task = agentharm_benign(
            padding_type=padding_type,padding_token_length=context_length, name_prefix=name_prefix,
            hint_included=False, split="val", detailed_behaviors=False
        )
        tasks.append(task)


    eval_set(tasks=tasks, log_dir= "logs/nano_all_contexts_benign" , model=model_id, retry_on_error=3)

def eval_gpt_non_relevant_padding():
    model_id = "openai/gpt-4.1-nano-2025-04-14"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"
    
    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_test_subset_"
        task = agentharm(
            padding_type=padding_type,padding_token_length=context_length, name_prefix=name_prefix,
            hint_included=False, split="test_public", detailed_behaviors=False
        )
        tasks.append(task)


    eval_set(tasks=tasks, log_dir= "logs/testSplit/gpt4nano/nano_all_contexts_irrelevant" , model=model_id, retry_on_error=3)

def eval_gpt_non_relevant_padding_benign():
    model_id = "openai/gpt-4.1-nano"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"
    
    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_test_subset_"
        task = agentharm_benign(
            padding_type=padding_type,padding_token_length=context_length, name_prefix=name_prefix,
            hint_included=False, split="test_public", detailed_behaviors=False
        )
        tasks.append(task)


    eval_set(tasks=tasks, log_dir= "logs/testSplit/gpt4nano/nano_all_contexts_irrelevant_benign" , model=model_id, retry_on_error=3)

def eval_gpt4mini_non_relevant_padding():
    model_id = "openai/gpt-4o-mini-2024-07-18"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"
    
    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_val_subset_"
        task = agentharm(
            padding_type=padding_type,padding_token_length=context_length, name_prefix=name_prefix,
            hint_included=False, split="val", detailed_behaviors=False
        )
        tasks.append(task)


    eval_set(tasks=tasks, log_dir= "logs/afterContext/4omini_all_contexts_irrelevant" , model=model_id, retry_on_error=3)

def eval_gpt5nano_non_relevant_padding():
    model_id = "openai/gpt-5-nano-2025-08-07"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000, 200_000]
    padding_type = "non_relevant"
    
    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_val_subset_"
        task = agentharm(
            padding_type=padding_type,padding_token_length=context_length, name_prefix=name_prefix,
            hint_included=False, split="test_public", detailed_behaviors=False
        )
        tasks.append(task)


    eval_set(tasks=tasks, log_dir= "logs/testSplit/gpt5/7gpt5nano_all_contexts_irrelevant" , model=model_id, retry_on_error=3)


def eval_gpt5nano_non_relevant_padding_benign():
    model_id = "openai/gpt-5-nano-2025-08-07"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000, 200_000]
    padding_type = "non_relevant"
    
    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_val_subset_"
        task = agentharm_benign(
            padding_type=padding_type,padding_token_length=context_length, name_prefix=name_prefix,
            hint_included=False, split="test_public", detailed_behaviors=False
        )
        tasks.append(task)


    eval_set(tasks=tasks, log_dir= "logs/testSplit/gpt5/gpt5nano_all_contexts_irrelevant_benign" , model=model_id, retry_on_error=3)

def eval_claude_haiku3_non_relevant_padding():
    model_id = "anthropic/claude-3-haiku-20240307"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"
    
    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_val_subset_"
        task = agentharm(
            padding_type=padding_type,padding_token_length=context_length, name_prefix=name_prefix,
            hint_included=False, split="val", detailed_behaviors=False
        )
        tasks.append(task)


    eval_set(tasks=tasks, log_dir= "logs/haiku3_all_contexts_irrelevant" , model=model_id, retry_on_error=3)

def eval_test_claude_haiku3_non_relevant_padding():
    model_id = "openrouter/anthropic/claude-3-haiku"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"
    
    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_test_subset_"
        task = agentharm(
            padding_type=padding_type,padding_token_length=context_length, name_prefix=name_prefix,
            hint_included=False, split="test_public", detailed_behaviors=False
        )
        tasks.append(task)


    eval_set(tasks=tasks, log_dir= "logs/testSplit/7haiku3_all_contexts_irrelevant" , model=model_id, retry_on_error=3)

def eval_test_claude_haiku3_non_relevant_padding_benign():
    model_id = "openrouter/anthropic/claude-3-haiku"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"
    
    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_test_subset_"
        task = agentharm_benign(
            padding_type=padding_type,padding_token_length=context_length, name_prefix=name_prefix,
            hint_included=False, split="test_public", detailed_behaviors=False
        )
        tasks.append(task)


    eval_set(tasks=tasks, log_dir= "logs/testSplit/3haiku3_all_contexts_irrelevant_benign" , model=model_id, retry_on_error=3)

def eval_claude_sonnet4_non_relevant_padding():
    model_id = "openrouter/anthropic/claude-sonnet-4"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"
    
    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_val_subset_"
        task = agentharm(
            padding_type=padding_type,padding_token_length=context_length, name_prefix=name_prefix,
            hint_included=False, split="test_public", detailed_behaviors=False
        )
        tasks.append(task)


    eval_set(tasks=tasks, log_dir= "logs/testSplit/sonnet4/2sonnet4_all_contexts_irrelevant" , model=model_id, retry_on_error=3)

def eval_claude_sonnet4_non_relevant_padding_benign():
    model_id = "openrouter/anthropic/claude-sonnet-4"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"
    
    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_val_subset_"
        task = agentharm_benign(
            padding_type=padding_type,padding_token_length=context_length, name_prefix=name_prefix,
            hint_included=False, split="test_public", detailed_behaviors=False
        )
        tasks.append(task)


    eval_set(tasks=tasks, log_dir= "logs/testSplit/sonnet4/sonnet4_all_contexts_irrelevant_benign" , model=model_id, retry_on_error=3)

def run_eval_set_with_config(config: dict):
    """
    Run evaluation based on YAML configuration
    
    Args:
        config: Dictionary containing evaluation configuration
    """
    # Extract configuration with defaults
    model_id = config["model_id"]
    task_type = config.get("task_type", "harmful")
    context_lengths = config["context_lengths"]
    padding_type = config["padding_type"]
    log_dir = config["log_dir"]
    retry_on_error = config.get("retry_on_error", 3)
    split = config.get("split", "val")
    hint_included = config.get("hint_included", False)
    detailed_behaviors = config.get("detailed_behaviors", False)
    seed = config.get("seed", 42)  # Default seed if not specified
    
    log_dir = log_dir + "/" + str(seed)
    # Set seed for reproducibility
    set_seed(seed)
    
    tasks = []
    for context_length in context_lengths:
        # Extract model name prefix (handling various model ID formats)
        model_parts = model_id.split("/")
        if len(model_parts) > 1:
            model_name = model_parts[1]
            # Remove date suffix if present (e.g., "-2025-04-14")
            if "-" in model_name and len(model_name.split("-")) > 1:
                model_id_prefix = model_name.rsplit("-", 1)[0]
            else:
                model_id_prefix = model_name
        else:
            model_id_prefix = model_id
        
        name_prefix = f"{model_id_prefix}_{context_length}_{padding_type}_{split}_"
        
        # Create appropriate task based on type
        if task_type == "harmful":
            task = agentharm(
                padding_type=padding_type,
                padding_token_length=context_length, 
                name_prefix=name_prefix,
                hint_included=hint_included, 
                split=split, 
                detailed_behaviors=detailed_behaviors
            )
        elif task_type == "benign":
            task = agentharm_benign(
                padding_type=padding_type,
                padding_token_length=context_length, 
                name_prefix=name_prefix,
                hint_included=hint_included, 
                split=split, 
                detailed_behaviors=detailed_behaviors
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        tasks.append(task)
    
    # Run evaluation
    logs = eval_set(
        tasks=tasks, 
        log_dir=log_dir, 
        model=model_id, 
        retry_on_error=retry_on_error
    )
    return logs

def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    set_seed(42)
    available_tasks = {
        name: obj
        for name, obj in globals().items()
        if callable(obj) and name.startswith("eval_")
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task_name", choices=sorted(available_tasks.keys()),
        help="The name of the evaluation function to run. Function should start with 'eval_'.",
        required=False
    )
    parser.add_argument(
        "--config_file", 
        type=str,
        help="Path to YAML configuration file",
        required=False
    )
    parser.add_argument("--seed", type=int, help="Override config file seed", required=False)
    parser.add_argument("--long_context", type=bool, default=False, required=False)

    args = parser.parse_args()

    # Load configuration
    config = load_config_from_yaml(args.config_file)
    # To set the seed via command line or shell script. Optional.
    if args.seed is not None:
        config["seed"] = args.seed
    
    # If no config file or eval function provided, terminate
    if args.config_file is None and args.task_name is None:
        print("No config or evaluation function provided. Exiting...")
        exit()

    # Set the seed
    set_seed(config["seed"])

    # Run evaluation
    if args.config_file is not None:
        run_eval_set_with_config(config)

    if args.task_name is not None:
        task_function = available_tasks[args.task_name]
        task_function()

if __name__ == "__main__":
    main()