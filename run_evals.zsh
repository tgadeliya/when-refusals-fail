#!/bin/zsh

# Exit immediately if a command fails
set -e

# activate virtual env
source /Users/jauhar/Repositories/algoverse_mountain/.venv/bin/activate

echo "Starting Python script sequence..."

uv run src/mountain/run_tasks.py --config configs/sonoma_dusk_harm.yaml
echo "Finished Sonoma Dusk for harmful prompts ✅"

# uv run src/mountain/run_tasks.py --config configs/sonoma_dusk_benign.yaml
# echo "Finished Sonoma Dusk for benign prompts ✅"

# uv run src/mountain/run_tasks.py --config configs/sonoma_sky_harm.yaml
# echo "Finished Sonoma Sky for harmful prompts ✅"

# uv run src/mountain/run_tasks.py --config configs/sonoma_sky_benign.yaml
# echo "Finished Sonoma Sky for benign prompts ✅"


echo "All scripts executed successfully 🎉"
