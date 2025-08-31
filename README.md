
# Parameters for experimentation subset
- split = "val"
- hint_included = False
- detailed_prompt = False



# Model names
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