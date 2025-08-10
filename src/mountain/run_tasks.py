import argparse
from typing import Literal

from dotenv import load_dotenv
from inspect_ai import eval_set

from mountain.tasks import agentharm, agentharm_benign

load_dotenv()


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


def main():
    available_tasks = {
        name: obj
        for name, obj in globals().items()
        if callable(obj) and name.startswith("eval_")
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task_name", choices=sorted(available_tasks.keys()),
        help="The name of the evaluation function to run. Function should start with 'eval_'."
    )
    args = parser.parse_args()
    task_function = available_tasks[args.task_name]
    task_function()

if __name__ == "__main__":
    main()