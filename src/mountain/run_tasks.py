import argparse
import time 

from typing import Literal

from dotenv import load_dotenv
from inspect_ai import eval_set, eval

from mountain.tasks import agentharm, agentharm_benign
import yaml

load_dotenv()


def eval_gpt41_nano_single_rand() -> None:
    for pad_len in [800_000]:
        model_id="openai/gpt-4.1-nano-2025-04-14"
        random_seeds = [3] #[14,15,92,65] 
        padding_length = pad_len
        padding_type = "random"
        padding_kwargs={"padding_position": "after"}
        tasks = []
        for random_seed in random_seeds:
            model_id_prefix = model_id.split("/")[1][:-11]
            name_prefix = f"{model_id_prefix}_rs_{random_seed}_hint_detprompt_test"
            task = agentharm(
                split="test_public",
                padding_type=padding_type, padding_length=padding_length, padding_kwargs=padding_kwargs,
                hint_included=True, detailed_behaviors=True,
                random_seed=random_seed,
                name_prefix=name_prefix,
            )
            tasks.append(task)
        eval_set(
            tasks=tasks, 
            log_dir= f"logs/harm/random_padding/{pad_len}_after/gpt_4.1_nano/test_hint_det_prompt",
            model=model_id,
            retry_on_error=3,
            timeout=500,
            max_connections=1
        )
        time.sleep(25)

def run_eval_set_with_config(config: dict):
    """
    Run evaluation based on YAML configuration
    
    Args:
        config: Dictionary containing evaluation configuration
    """
    # Extract configuration with defaults
    model_id = config["model_id"]
    task_type = config.get("task_type", "harm")
    context_lengths = config["context_lengths"]
    padding_type = config["padding_type"]
    log_dir = config["log_dir"] # log directory root
    retry_on_error = config.get("retry_on_error", 3)
    split = config.get("split", "val")
    hint_included = config.get("hint_included", False)
    detailed_behaviors = config.get("detailed_behaviors", False)
    padding_position = config.get("padding_position", "before")
    random_seeds = config.get("random_seeds")  # Default seed if not specified
    padding_kwargs={"padding_position": padding_position}

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

        tasks = []
        for random_seed in random_seeds:
            # Create appropriate task based on type
            if task_type == "harm":
                task = agentharm(
                    padding_type=padding_type,
                    padding_length=context_length, 
                    name_prefix=name_prefix,
                    hint_included=hint_included, 
                    split=split, 
                    detailed_behaviors=detailed_behaviors,
                    random_seed=random_seed,
                    padding_kwargs=padding_kwargs
                )
            elif task_type == "benign":
                task = agentharm_benign(
                    padding_type=padding_type,
                    padding_length=context_length, 
                    name_prefix=name_prefix,
                    hint_included=hint_included, 
                    split=split, 
                    detailed_behaviors=detailed_behaviors,
                    random_seed=random_seed,
                    padding_kwargs=padding_kwargs
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            tasks.append(task)

        hint_flag = "hint" if hint_included else "no_hint"
        detailed_behavior_flag = "det" if detailed_behaviors else "no_det"

        _log_dir = f"{log_dir}/{task_type}/{padding_type}/{context_length}_{padding_position}/{model_id_prefix}/test_{hint_flag}_{detailed_behavior_flag}_prompt"

        # Run evaluation
        eval_set(
            tasks=tasks,
            log_dir=_log_dir,
            model=model_id,
            retry_on_error=retry_on_error
            )
        time.sleep(25)

def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    available_tasks = {
        name: obj
        for name, obj in globals().items()
        if callable(obj) and name.startswith("eval_")
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name", choices=sorted(available_tasks.keys()),
        help="The name of the evaluation function to run. Function should start with 'eval_'.",
        required=False
    )
    parser.add_argument(
        "--config_file", 
        type=str,
        help="Path to YAML configuration file",
        required=False
    )
    parser.add_argument("--long_context", type=bool, default=False, required=False)

    args = parser.parse_args()

    # Load configuration
    # And run evaluation
    if args.config_file is not None:
        config = load_config_from_yaml(args.config_file)
        run_eval_set_with_config(config)


    if args.task_name is not None:
        task_function = available_tasks[args.task_name]
        task_function()

if __name__ == "__main__":
    main()