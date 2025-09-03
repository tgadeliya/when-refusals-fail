from typing import Any, Dict, Literal, Optional

from inspect_ai import Task, task
from dotenv import load_dotenv
from inspect_ai import eval_set, eval

from mountain.agentharm.agentharm import run_agent_benchmark

load_dotenv()


@task
def run_debug_agentharm() -> Task:
    return run_agent_benchmark(
        agent="default",
        behavior_ids=[],
        task_name="benign",
        split="val",
        hint_included=False,
        detailed_behaviors=False,
        refusal_judge="openai/gpt-4.1-nano-2025-04-14",
        semantic_judge="openai/gpt-4.1-nano-2025-04-14",
        message_limit=20, # no output tokens
        name_prefix="debug_",
        padding_type=None,
        padding_token_length=None
    )

@task
def agentharm_benign(
    agent: str = "default",
    behavior_ids: list[str] = [],
    split: Literal["val", "test_private", "test_public"] = "test_public",
    detailed_behaviors: Optional[bool] = None,
    hint_included: Optional[bool] = None,
    n_irrelevant_tools: int = 0,
    refusal_judge: str = "openai/gpt-4.1-nano-2025-04-14",
    semantic_judge: str = "openai/gpt-4.1-nano-2025-04-14",
    agent_kwargs: Optional[Dict[str, Any]] = None,
    name_prefix: str = "",
    random_seed: int = 0,
    padding_type: Literal["random"]| None = None,
    padding_token_length: int| None = None,
) -> Task:
    return run_agent_benchmark(
        agent=agent,
        behavior_ids=behavior_ids if behavior_ids else [],
        task_name="benign",
        split=split,
        detailed_behaviors=detailed_behaviors,
        hint_included=hint_included,
        n_irrelevant_tools=n_irrelevant_tools,
        refusal_judge=refusal_judge,
        semantic_judge=semantic_judge,
        agent_kwargs=agent_kwargs,
        message_limit=20,
        name_prefix=name_prefix,
        random_seed=random_seed,
        padding_type=padding_type,
        padding_token_length=padding_token_length
    )

@task
def agentharm(
    agent: str = "default",
    behavior_ids: list[str] = [],
    split: Literal["val", "test_private", "test_public"] = "test_public",
    detailed_behaviors: Optional[bool] = None,
    hint_included: Optional[bool] = None,
    n_irrelevant_tools: int = 0,
    refusal_judge: str = "openai/gpt-4.1-nano-2025-04-14",
    semantic_judge: str = "openai/gpt-4.1-nano-2025-04-14",
    agent_kwargs: Optional[Dict[str, Any]] = None,
    chat_dataset: Optional[bool] = False,
    name_prefix: str = "",
    random_seed: int = 0,
    padding_type: Literal["random"]| None = None,
    padding_token_length: int| None = None,
) -> Task:
    return run_agent_benchmark(
        agent=agent,
        behavior_ids=behavior_ids if behavior_ids else [],
        task_name="harmful" if not chat_dataset else "chat",
        split=split,
        detailed_behaviors=detailed_behaviors,
        hint_included=hint_included,
        n_irrelevant_tools=n_irrelevant_tools,
        refusal_judge=refusal_judge,
        semantic_judge=semantic_judge,
        agent_kwargs=agent_kwargs,
        message_limit=20,
        name_prefix=name_prefix,
        random_seed=random_seed,
        padding_type=padding_type,
        padding_token_length=padding_token_length
    )

if __name__ == "__main__":
    model_id = "openai/gpt-4o-mini-2024-07-18"
    tasks = [run_debug_agentharm()]
    logs = eval(tasks=tasks, log_dir="./logs" , model=model_id)
