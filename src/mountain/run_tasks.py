import argparse
import time 

from dotenv import load_dotenv
from inspect_ai import eval_set, eval

from mountain.tasks import agentharm, agentharm_benign

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


if __name__ == "__main__":
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