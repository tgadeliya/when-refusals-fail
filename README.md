# ⛰️ Mountain

Repo for research on safety and capabilities degradation in long-context tasks 

## Installation

This project uses `uv` for Python environment and package management.

To create a virtual environment (`.venv` by default) and install all necessary dependencies from `pyproject.toml`, run:
```bash
uv sync
```

# Run tasks

### Option 1: Inspect VSCode extension
1. Install extenstion from VScode marketplace
2. Add function with `@task` decorator in the `src/mountain/tasks.py`
3. Using extension interface run your task

### Option 2: Run task from command-line (perfect for multiple tasks setups)
1. In the file `src/mountain/run_tasks.py` add function that creates `Tasks` object and run `eval` or `eval_set`. Function should start with `prefix eval` to be discoverable  
    Do not forget to specify new subdirectory in `logs`
2. To run your evals use
```
uv run src/mountain/run_tasks.py <your_function_name>
```

# Model names for the experiments 

- openai/gpt-4o-mini-2024-07-18
- openai/gpt-4.1-nano-2025-04-14
- openai/gpt-5-nano-2025-08-07
- anthropic/claude-3-haiku-20240307
- claude-sonnet-4-20250514

# For linting before commit, do:
- uv run ruff check

# To fix issues automatically (some would require manual fixing), do:
- uv run ruff check --fix

# To format the repo, do:
- uv run ruff format