import argparse
import yaml
import random
import numpy as np
from dotenv import load_dotenv
from inspect_ai import eval_set

from mountain.tasks import agentharm, agentharm_benign

load_dotenv()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


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
                detailed_behaviors=detailed_behaviors,
            )
        elif task_type == "benign":
            task = agentharm_benign(
                padding_type=padding_type,
                padding_token_length=context_length,
                name_prefix=name_prefix,
                hint_included=hint_included,
                split=split,
                detailed_behaviors=detailed_behaviors,
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        tasks.append(task)

    # Run evaluation
    logs = eval_set(
        tasks=tasks, log_dir=log_dir, model=model_id, retry_on_error=retry_on_error
    )
    return logs


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation with YAML configuration"
    )
    parser.add_argument(
        "--config_file", type=str, help="Path to YAML configuration file"
    )
    parser.add_argument("--seed", type=int, help="Override config file seed")

    args = parser.parse_args()

    # Load configuration
    config = load_config_from_yaml(args.config_file)
    if args.seed is not None:
        config["seed"] = args.seed

    # Run evaluation
    run_eval_set_with_config(config)


if __name__ == "__main__":
    main()
