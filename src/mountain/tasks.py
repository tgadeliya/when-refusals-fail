from os import name
from typing import Any, Dict, Literal, Optional

from inspect_ai import Task, eval_set, task
from dotenv import load_dotenv

from mountain.agentharm.agentharm import run_agent_benchmark

load_dotenv()



@task()
def run_debug_agentharm(padding_type: Literal["random"]|None = None, padding_token_length: int|None = None,) -> Task:
    return run_agent_benchmark(
        agent="default",
        behavior_ids=[],
        task_name="harmful",
        split="test_public",
        hint_included=None,
        detailed_behaviors=None,
        refusal_judge="openai/gpt-4.1-nano-2025-04-14",
        semantic_judge="openai/gpt-4.1-nano-2025-04-14",
        message_limit=1, # no output tokens
        name_prefix="debug_",
        padding_type=None,
        padding_token_length=None
    )

@task()
def agentharm_benign(
    agent: str = "default",
    behavior_ids: list[str] = [],
    split: Literal["val", "test_private", "test_public"] = "test_public",
    detailed_behaviors: Optional[bool] = None,
    hint_included: Optional[bool] = None,
    n_irrelevant_tools: int = 0,
    refusal_judge: str = "openai/gpt-4o-2024-08-06",
    semantic_judge: str = "openai/gpt-4o-2024-08-06",
    agent_kwargs: Optional[Dict[str, Any]] = None,
    name_prefix: str = "",
    padding_type: Literal["random"]| None = None,
    padding_token_length: int| None = None,
) -> Task:
    """
    AgentHarm benchmark task.

    Args:
        agent (str): The name of the solver to use for the agent. We provide a default agent if unspecified.
        behavior_ids (list[str]): Behavior IDs to evaluate on (all by default).
        split (Literal['val', 'test']): Split to evaluate on.
        detailed_behaviors (Optional[bool]): If true, evaluate only on detailed behaviors. If false, evaluate only on open-ended behaviors.
        hint_included (Optional[bool]): If true, evaluate only on hint-included behaviors. If false, evaluate only on hint-not-included behaviors.
        n_irrelevant_tools (int): Number of irrelevant tools.
        refusal_judge (str): Refusal judge model.
        semantic_judge (str): Semantic judge model.
        agent_kwargs: Additional arguments to pass to the agent. The kwargs for the default agent are:
            - tool_choice (Literal['forced_first', 'auto', 'none']): The choice of tool to use. Default is 'auto'.
            - user_prompt_template (str): Name of user prompt in `prompts.py`. Default is 'empty'.
            - system_prompt_template (str): Name of system prompt in `prompts.py`. Default is None.
            - prompt_technique (Literal['standard', 'cot', 'react', 'refusal']): Prompting technique to apply to the system prompt. Default is 'standard'.
    """
    return run_agent_benchmark(
        agent=agent,
        behavior_ids=behavior_ids,
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
    refusal_judge: str = "openai/gpt-4o-2024-08-06",
    semantic_judge: str = "openai/gpt-4o-2024-08-06",
    agent_kwargs: Optional[Dict[str, Any]] = None,
    chat_dataset: Optional[bool] = False,
    name_prefix: str = "",
    padding_type: Literal["random"]| None = None,
    padding_token_length: int| None = None,
) -> Task:
    """
    AgentHarm benchmark task.

    Args:
        agent (str): The name of the solver to use for the agent. We provide a default agent if unspecified.
        behavior_ids (list[str]): Behavior IDs to evaluate on (all by default).
        split (Literal['val', 'test']): Split to evaluate on.
        detailed_behaviors (Optional[bool]): If true, evaluate only on detailed behaviors. If false, evaluate only on open-ended behaviors.
        hint_included (Optional[bool]): If true, evaluate only on hint-included behaviors. If false, evaluate only on hint-not-included behaviors.
        n_irrelevant_tools (int): Number of irrelevant tools.
        refusal_judge (str): Refusal judge model.
        semantic_judge (str): Semantic judge model.
        agent_kwargs: Additional arguments to pass to the agent. The kwargs fro the default agent are:
            - tool_choice (Literal['forced_first', 'auto', 'none']): The choice of tool to use. Default is 'auto'.
            - user_prompt_template (str): Name of user prompt in `prompts.py`. Default is 'empty'.
            - system_prompt_template (str): Name of system prompt in `prompts.py`. Default is None.
            - prompt_technique (Literal['standard', 'cot', 'react', 'refusal']): Prompting technique to apply to the system prompt. Default is 'standard'.
        chat_dataset (bool): If true, evaluate on the chat dataset.
    """
    return run_agent_benchmark(
        agent=agent,
        behavior_ids=behavior_ids,
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
        padding_type=padding_type,
        padding_token_length=padding_token_length
    )


def run_eval_set_with_different_context_length(
        model_id: str|list[str],
        task_type: Literal["harmful", "benign"],
        context_lengths: list[int|None],
        padding_type: Literal["random"]| None
) -> None:
    tasks = []
    for context_length in context_lengths:
        name_prefix = f"padding_{padding_type}_cont_len_{context_length}_"
        if task_type == "harmful":
            tasks.append(agentharm(padding_type=padding_type, padding_token_length=context_length, name_prefix=name_prefix))
        elif task_type == "benign":
            tasks.append(agentharm_benign(padding_type=padding_type, padding_token_length=context_length, name_prefix=name_prefix))

    logs = eval_set(tasks=tasks, log_dir="./logs_set/context_length_experiments_nano_100000", model=model_id)



if __name__ == "__main__":
    model_ids = ["openai/gpt-4.1-nano-2025-04-14"]
    context_lengths = [100_000]
    run_eval_set_with_different_context_length(
        model_id=model_ids, task_type="benign", # type: ignore
        context_lengths=context_lengths, padding_type="random"
    )