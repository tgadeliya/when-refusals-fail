from inspect_ai import Task, task, eval
from mountain.agentharm.agentharm import run_agent_benchmark

from dotenv import load_dotenv
load_dotenv()



@task
def run_debug_agentharm() -> Task:
    return run_agent_benchmark(
        agent="default",
        behavior_ids=[],
        task_name="harmful",
        split="val",
        hint_included=False,
        refusal_judge="openai/gpt-4o-mini",
        semantic_judge="openai/gpt-4o-mini",
        task_max_messages=4
    )
 