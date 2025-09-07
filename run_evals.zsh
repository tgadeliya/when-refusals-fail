#!/bin/zsh

# Exit immediately if a command fails
set -e

# Optional: activate virtual environment if you have one
source /Users/jauhar/Repositories/algoverse_mountain/.venv/bin/activate

echo "Starting Python script sequence..."

uv run src/mountain/run_evals.py --config_file configs/config_gpt41_nano_harm.yaml --seed 42
echo "Finished GPT4.1-nano for harmful prompts ✅"

uv run src/mountain/run_evals.py --config_file configs/config_gpt41_nano_benign.yaml --seed 42
echo "Finished GPT4.1-nano for benign prompts ✅"

uv run src/mountain/run_evals.py --config_file configs/config_gpt4o_mini_harm.yaml --seed 42
echo "Finished GPT4o-mini for harmful prompts ✅"

uv run src/mountain/run_evals.py --config_file configs/config_gpt4o_mini_benign.yaml --seed 42
echo "Finished GPT4o-mini for benign prompts ✅"

uv run src/mountain/run_evals.py --config_file configs/config_gpt5_nano_harm.yaml --seed 42
echo "Finished GPT5-nano for hamrful prompts"

uv run src/mountain/run_evals.py --config_file configs/config_gpt5_nano_benign.yaml --seed 42
echo "Finished GPT5-nano for benign prompts"

# uv run src/mountain/run_evals.py --config_file configs/config_gemini2o_flash_benign.yaml
# echo "Finished Gemini 2 flash for benign prompts ✅"

# uv run src/mountain/run_evals.py --config_file configs/config_gemini2o_flash_harm.yaml
# echo "Finished Gemini 2 flash for harmful prompts ✅"

echo "All scripts executed successfully 🎉"
