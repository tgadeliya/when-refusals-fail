import argparse
import time

import yaml
from dotenv import load_dotenv
from inspect_ai import eval_set

from mountain.tasks import agentharm, agentharm_benign

load_dotenv()

def eval_gpt41_nano() -> None:
    for task_type in ["harm", "benign"]:
        for pad_len in [5,10,30,50]:
            model_id = "openai/gpt-4.1-nano-2025-04-14"
            random_seeds = [3,14]  # [14,15,92,65]
            padding_length = pad_len
            padding_type = "multi_task"
            padding_kwargs = {"task_type": "benign", "padding_position": "after_with_task_instructed"}
            tasks = []
            for random_seed in random_seeds:
                model_id_prefix = model_id.split("/")[1][:-11]
                name_prefix = f"{model_id_prefix}_rs_{random_seed}_hint_detprompt_test"
                if task_type == "harm":
                    task = agentharm(
                        split="test_public",
                        padding_type=padding_type,
                        padding_length=padding_length,
                        padding_kwargs=padding_kwargs,
                        hint_included=True,
                        detailed_behaviors=True,
                        random_seed=random_seed,
                        name_prefix=name_prefix,
                    )
                elif task_type == "benign":
                    task = agentharm_benign(
                        split="test_public",
                        padding_type=padding_type,
                        padding_length=padding_length,
                        padding_kwargs=padding_kwargs,
                        hint_included=True,
                        detailed_behaviors=True,
                        random_seed=random_seed,
                        name_prefix=name_prefix,
                    )
                tasks.append(task)
            eval_set(
                tasks=tasks,
                log_dir=f"logs/{task_type}/multi_task_padding/benign_new/{pad_len}_after_with_task_instructed/gpt_4.1_nano/test_hint_det_prompt",
                model=model_id,
                retry_on_error=3,
                timeout=500,
                max_connections=1,
            )
            time.sleep(25)

def eval_gpt5_rand_pad() -> None:
    for pad_pos in ["after"]:
        for task_type in ["harm"]:
            for pad_len in [100000, 200000]:
                model_id = "openrouter/openai/gpt-5"
                random_seeds = [3] # [14,15,92,65]
                padding_length = pad_len
                padding_type = "random"
                padding_kwargs = {"padding_position": pad_pos}
                tasks = []
                for random_seed in random_seeds:
                    model_id_prefix = model_id.split("/")[2]
                    name_prefix = f"{model_id_prefix}_rs_{random_seed}_hint_detprompt_test"
                    if task_type == "harm":
                        task = agentharm(
                            split="test_public",
                            padding_type=padding_type,
                            padding_length=padding_length,
                            padding_kwargs=padding_kwargs,
                            hint_included=True,
                            detailed_behaviors=True,
                            random_seed=random_seed,
                            name_prefix=name_prefix,
                            refusal_judge="openai/gpt-4o-2024-08-06",
                        )
                    elif task_type == "benign":
                        task = agentharm_benign(
                            split="test_public",
                            padding_type=padding_type,
                            padding_length=padding_length,
                            padding_kwargs=padding_kwargs,
                            hint_included=True,
                            detailed_behaviors=True,
                            random_seed=random_seed,
                            name_prefix=name_prefix,
                        )
                    tasks.append(task)
                eval_set(
                    tasks=tasks,
                    log_dir=f"logs/{task_type}/gpt-5_random_padding/{pad_len}_{pad_pos}/test_hint_det_prompt",
                    model=model_id,
                    retry_on_error=3,
                    timeout=500,
                    batch=True,
                    reasoning_effort="minimal"
                )
                time.sleep(25)

def eval_grok_rand_pad() -> None:
    for pad_pos in ["after"]:
        for task_type in ["harm","benign"]:
            for pad_len in [0, 1_000, 5_000, 10_000, 50_000, 100_000, 200_000]:
                model_id = "openrouter/x-ai/grok-4-fast:free"
                random_seeds = [3, 14]  # [14,15,92,65]
                padding_length = pad_len
                padding_type = "random"
                padding_kwargs = {"padding_position": pad_pos}
                tasks = []
                for random_seed in random_seeds:
                    model_id_prefix = model_id.split("/")[1][:-11]
                    name_prefix = f"{model_id_prefix}_rs_{random_seed}_no_hint_no_detprompt_test"
                    if task_type == "harm":
                        task = agentharm(
                            split="test_public",
                            padding_type=padding_type,
                            padding_length=padding_length,
                            padding_kwargs=padding_kwargs,
                            hint_included=False,
                            detailed_behaviors=False,
                            random_seed=random_seed,
                            name_prefix=name_prefix,
                        )
                    elif task_type == "benign":
                        task = agentharm_benign(
                            split="test_public",
                            padding_type=padding_type,
                            padding_length=padding_length,
                            padding_kwargs=padding_kwargs,
                            hint_included=False,
                            detailed_behaviors=False,
                            random_seed=random_seed,
                            name_prefix=name_prefix,
                        )
                    tasks.append(task)
                eval_set(
                    tasks=tasks,
                    log_dir=f"logs/{task_type}/grok-4-fast_random_padding/{pad_len}_{pad_pos}/test_no_hint_no_det_prompt",
                    model=model_id,
                    retry_on_error=3,
                    timeout=500,
                    max_connections=1,
                )
                time.sleep(25)

def eval_models_ds() -> None:
    for task_type in ["harm", "benign"]:
        for pad_len in [72_000]:
            for model in ["deepseek/deepseek-chat-v3.1"]:    
                hint_included = True 
                detailed_behaviors = True 
                model_id = f"openrouter/{model}"
                random_seeds = [3,14] #,15] # 92,65]
                padding_length = pad_len
                padding_type = "random"
                padding_kwargs = {"padding_position": "after"}
                tasks = []
                model_id_prefix = model_id.split("/")[2] 

                for random_seed in random_seeds:
                    name_prefix = f"{model_id_prefix}_rs_{random_seed}_hint_test"
                    if task_type == "harm":
                        task = agentharm(
                            split="test_public",
                            padding_type=padding_type,
                            padding_length=padding_length,
                            padding_kwargs=padding_kwargs,
                            hint_included=hint_included,
                            detailed_behaviors=detailed_behaviors,
                            random_seed=random_seed,
                            name_prefix=name_prefix,
                        )
                    elif task_type == "benign":
                        task = agentharm_benign(
                            split="test_public",
                            padding_type=padding_type,
                            padding_length=padding_length,
                            padding_kwargs=padding_kwargs,
                            hint_included=hint_included,
                            detailed_behaviors=detailed_behaviors,
                            random_seed=random_seed,
                            name_prefix=name_prefix,
                        )
                    else:
                        raise ValueError(f"Wrong {task_type=}")
                    tasks.append(task)
                
                log_dir_name_low = "hint_det_prompt" if (hint_included and detailed_behaviors) else "no_hint_no_det_prompt" 
                eval_set(
                    tasks=tasks,
                    log_dir=f"logs/{task_type}/{model_id_prefix}_random_padding/{pad_len}_after/test_{log_dir_name_low}",
                    model=model_id,
                    retry_on_error=3,
                    max_tasks=5,
                    timeout=500,
                    max_connections=20,
                )
                time.sleep(25)

def eval_gemini25_flash_lite() -> None:
    model_id="google/gemini-2.5-flash-lite"
    random_seeds = [3] #,14,15,92,65]
    padding_length = 0
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
        log_dir= f"logs/benign/random_padding/0/gemini_2.5_flash_lite/test_hint_det_prompt",
        log_level="trace",
        model=model_id,
        retry_on_error=3,
        timeout=500,
        max_connections=1,
        max_subprocesses=1,
        limit=20,
        debug_errors=True,
        max_samples=1,
        max_tasks=1,
        max_tool_output=1
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
    log_dir = config["log_dir"]  # log directory root
    retry_on_error = config.get("retry_on_error", 3)
    split = config.get("split", "val")
    hint_included = config.get("hint_included", None)
    detailed_behaviors = config.get("detailed_behaviors", None)
    padding_position = config.get("padding_position", "before")
    random_seeds = config.get("random_seeds")  # Default seed if not specified
    behavior_ids = config.get("behavior_ids", None)
    semantic_judge = config.get("semantic_judge", "openai/gpt-4.1-nano-2025-04-14")
    refusal_judge = config.get("refusal_judge", "openai/gpt-4.1-nano-2025-04-14")
    prompt_technique = config.get("prompt_technique", "standard")
    agent_kwargs = {"prompt_technique": prompt_technique}
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
                    behavior_ids=behavior_ids,
                    padding_length=context_length,
                    name_prefix=name_prefix,
                    hint_included=hint_included, 
                    split=split, 
                    refusal_judge=refusal_judge,
                    semantic_judge=semantic_judge,
                    detailed_behaviors=detailed_behaviors,
                    random_seed=random_seed,
                    padding_kwargs=padding_kwargs,
                    agent_kwargs=agent_kwargs
                )
            elif task_type == "benign":
                task = agentharm_benign(
                    padding_type=padding_type,
                    behavior_ids=behavior_ids,
                    padding_length=context_length,
                    name_prefix=name_prefix,
                    hint_included=hint_included, 
                    split=split, 
                    refusal_judge=refusal_judge,
                    semantic_judge=semantic_judge,
                    detailed_behaviors=detailed_behaviors,
                    random_seed=random_seed,
                    padding_kwargs=padding_kwargs,
                    agent_kwargs=agent_kwargs
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            tasks.append(task)

        hint_flag = (
            "hint"
            if hint_included == True
            else "no_hint"
            if hint_included == False
            else "all"
        )
        detailed_behavior_flag = (
            "det"
            if detailed_behaviors == True
            else "no_det"
            if detailed_behaviors == False
            else "all"
        )

        _log_dir = f"{log_dir}/{task_type}/{padding_type}/{context_length}_{padding_position}/{model_id_prefix}/test_{hint_flag}_{detailed_behavior_flag}_prompt"

        # Run evaluation
        eval_set(
            tasks=tasks, log_dir=_log_dir, model=model_id, retry_on_error=retry_on_error
        )
        time.sleep(25)

def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
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
        "--task_name",
        choices=sorted(available_tasks.keys()),
        help="The name of the evaluation function to run. Function should start with 'eval_'.",
        required=False,
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to YAML configuration file",
        required=False,
    )

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
