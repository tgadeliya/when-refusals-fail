import argparse
import time 

from typing import Literal
import random
import numpy as np
from dotenv import load_dotenv
from inspect_ai import eval_set, eval

from mountain.tasks import agentharm, agentharm_benign

load_dotenv()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


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


if __name__ == "__main__":
def run_eval_set_with_different_context_length(
    model_id: str,
    task_type: Literal["harmful", "benign"],
    context_lengths: list[int],
    padding_type: Literal["random"] | None,
    log_subdir_name: str,
):
    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_"
        if task_type == "harmful":
            tasks.append(
                agentharm(
                    padding_type=padding_type,
                    padding_token_length=context_length,
                    name_prefix=name_prefix,
                )
            )
        elif task_type == "benign":
            tasks.append(
                agentharm_benign(
                    padding_type=padding_type,
                    padding_token_length=context_length,
                    name_prefix=name_prefix,
                )
            )

    logs = eval_set(tasks=tasks, log_dir=f"logs/{log_subdir_name}", model=model_id)
    return logs


def eval_gpt41_nano_random_all_contexts_val_subset_benign():
    model_id = "openai/gpt-4.1-nano-2025-04-14"
    context_lengths = [0, 1000, 10_000, 50_000, 100_000, 250_000, 500_000, 990_000]
    padding_type = "random"

    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_val_subset_"
        task = agentharm_benign(
            padding_type=padding_type,
            padding_token_length=context_length,
            name_prefix=name_prefix,
            hint_included=False,
            split="val",
            detailed_behaviors=False,
        )
        tasks.append(task)

    eval_set(
        tasks=tasks,
        log_dir="logs/nano_all_contexts_benign",
        model=model_id,
        retry_on_error=3,
    )


def eval_gpt_non_relevant_padding():
    model_id = "openai/gpt-4.1-nano-2025-04-14"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"

    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_test_subset_"
        task = agentharm(
            padding_type=padding_type,
            padding_token_length=context_length,
            name_prefix=name_prefix,
            hint_included=False,
            split="test_public",
            detailed_behaviors=False,
        )
        tasks.append(task)

    eval_set(
        tasks=tasks,
        log_dir="logs/testSplit/gpt4nano/nano_all_contexts_irrelevant",
        model=model_id,
        retry_on_error=3,
    )


def eval_gpt_non_relevant_padding_benign():
    model_id = "openai/gpt-4.1-nano"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"

    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_test_subset_"
        task = agentharm_benign(
            padding_type=padding_type,
            padding_token_length=context_length,
            name_prefix=name_prefix,
            hint_included=False,
            split="test_public",
            detailed_behaviors=False,
        )
        tasks.append(task)

    eval_set(
        tasks=tasks,
        log_dir="logs/testSplit/gpt4nano/nano_all_contexts_irrelevant_benign",
        model=model_id,
        retry_on_error=3,
    )


def eval_gpt4mini_non_relevant_padding():
    model_id = "openai/gpt-4o-mini-2024-07-18"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"

    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_val_subset_"
        task = agentharm(
            padding_type=padding_type,
            padding_token_length=context_length,
            name_prefix=name_prefix,
            hint_included=False,
            split="val",
            detailed_behaviors=False,
        )
        tasks.append(task)

    eval_set(
        tasks=tasks,
        log_dir="logs/afterContext/4omini_all_contexts_irrelevant",
        model=model_id,
        retry_on_error=3,
    )


def eval_gpt5nano_non_relevant_padding():
    model_id = "openai/gpt-5-nano-2025-08-07"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000, 200_000]
    padding_type = "non_relevant"

    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_val_subset_"
        task = agentharm(
            padding_type=padding_type,
            padding_token_length=context_length,
            name_prefix=name_prefix,
            hint_included=False,
            split="test_public",
            detailed_behaviors=False,
        )
        tasks.append(task)

    eval_set(
        tasks=tasks,
        log_dir="logs/testSplit/gpt5/7gpt5nano_all_contexts_irrelevant",
        model=model_id,
        retry_on_error=3,
    )


def eval_gpt5nano_non_relevant_padding_benign():
    model_id = "openai/gpt-5-nano-2025-08-07"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000, 200_000]
    padding_type = "non_relevant"

    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_val_subset_"
        task = agentharm_benign(
            padding_type=padding_type,
            padding_token_length=context_length,
            name_prefix=name_prefix,
            hint_included=False,
            split="test_public",
            detailed_behaviors=False,
        )
        tasks.append(task)

    eval_set(
        tasks=tasks,
        log_dir="logs/testSplit/gpt5/gpt5nano_all_contexts_irrelevant_benign",
        model=model_id,
        retry_on_error=3,
    )


def eval_claude_haiku3_non_relevant_padding():
    model_id = "anthropic/claude-3-haiku-20240307"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"

    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_val_subset_"
        task = agentharm(
            padding_type=padding_type,
            padding_token_length=context_length,
            name_prefix=name_prefix,
            hint_included=False,
            split="val",
            detailed_behaviors=False,
        )
        tasks.append(task)

    eval_set(
        tasks=tasks,
        log_dir="logs/haiku3_all_contexts_irrelevant",
        model=model_id,
        retry_on_error=3,
    )


def eval_test_claude_haiku3_non_relevant_padding():
    model_id = "openrouter/anthropic/claude-3-haiku"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"

    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_test_subset_"
        task = agentharm(
            padding_type=padding_type,
            padding_token_length=context_length,
            name_prefix=name_prefix,
            hint_included=False,
            split="test_public",
            detailed_behaviors=False,
        )
        tasks.append(task)

    eval_set(
        tasks=tasks,
        log_dir="logs/testSplit/7haiku3_all_contexts_irrelevant",
        model=model_id,
        retry_on_error=3,
    )


def eval_test_claude_haiku3_non_relevant_padding_benign():
    model_id = "openrouter/anthropic/claude-3-haiku"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"

    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_test_subset_"
        task = agentharm_benign(
            padding_type=padding_type,
            padding_token_length=context_length,
            name_prefix=name_prefix,
            hint_included=False,
            split="test_public",
            detailed_behaviors=False,
        )
        tasks.append(task)

    eval_set(
        tasks=tasks,
        log_dir="logs/testSplit/3haiku3_all_contexts_irrelevant_benign",
        model=model_id,
        retry_on_error=3,
    )


def eval_claude_sonnet4_non_relevant_padding():
    model_id = "openrouter/anthropic/claude-sonnet-4"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"

    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_val_subset_"
        task = agentharm(
            padding_type=padding_type,
            padding_token_length=context_length,
            name_prefix=name_prefix,
            hint_included=False,
            split="test_public",
            detailed_behaviors=False,
        )
        tasks.append(task)

    eval_set(
        tasks=tasks,
        log_dir="logs/testSplit/sonnet4/2sonnet4_all_contexts_irrelevant",
        model=model_id,
        retry_on_error=3,
    )


def eval_claude_sonnet4_non_relevant_padding_benign():
    model_id = "openrouter/anthropic/claude-sonnet-4"
    context_lengths = [0, 100, 1000, 10_000, 50_000, 100_000]
    padding_type = "non_relevant"

    tasks = []
    for context_length in context_lengths:
        model_id_prefix = model_id.split("/")[1][:-11]
        name_prefix = f"{model_id_prefix}_{padding_type}_{context_length}_val_subset_"
        task = agentharm_benign(
            padding_type=padding_type,
            padding_token_length=context_length,
            name_prefix=name_prefix,
            hint_included=False,
            split="test_public",
            detailed_behaviors=False,
        )
        tasks.append(task)

    eval_set(
        tasks=tasks,
        log_dir="logs/testSplit/sonnet4/sonnet4_all_contexts_irrelevant_benign",
        model=model_id,
        retry_on_error=3,
    )


def main():
    set_seed(42)  # 2, 7, 42
    available_tasks = {
        name: obj
        for name, obj in globals().items()
        if callable(obj) and name.startswith("eval_")
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task_name",
        choices=sorted(available_tasks.keys()),
        help="The name of the evaluation function to run. Function should start with 'eval_'.",
    )
    args = parser.parse_args()
    task_function = available_tasks[args.task_name]
    task_function()


if __name__ == "__main__":
    main()
