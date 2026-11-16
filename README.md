# When Refusals Fail: Unstable Safety Mechanisms in Long-Context LLM Agents

[![arXiv](https://img.shields.io/badge/arXiv-2512.02445-b31b1b.svg)](https://arxiv.org/abs/2512.02445)
[![AAAI 2026 TrustAgent Workshop](https://img.shields.io/badge/AAAI%202026-TrustAgent%20Workshop-blue.svg)](https://arxiv.org/abs/2512.02445)
[![Built with Inspect](https://img.shields.io/badge/built%20with-Inspect%20AI-4b8bbe.svg)](https://inspect.aisi.org.uk/)

This repository holds the code for the paper:

> **When Refusals Fail: Unstable Safety Mechanisms in Long-Context LLM Agents**
> Tsimur Hadeliya, Mohammad Ali Jauhar, Nidhi Sakpal, Diogo Cruz
> *AAAI 2026 TrustAgent Workshop* · [arXiv:2512.02445](https://arxiv.org/abs/2512.02445) · [PDF](https://arxiv.org/pdf/2512.02445) · [HTML](https://arxiv.org/html/2512.02445v1)

We study how an LLM agent's **capability** and **safety** change as its context window fills up. We take agentic tasks from [AgentHarm](https://arxiv.org/abs/2410.09024), pad them with up to hundreds of thousands of extra tokens, and measure how task performance and refusal behavior change.

> [!WARNING]
> AgentHarm contains prompts that describe harmful behavior (fraud, cybercrime, harassment and other categories), and the tools return synthetic data. This code is for safety research only.

## Main findings

<p align="center">
  <img src="assets/results_harm_score.svg" alt="AgentHarm score on harmful and benign tasks against random padding length for Grok 4 Fast, GPT-5, GPT-4.1-nano and DeepSeek-V3.1." width="49%">
  <img src="assets/results_refusal_rate.svg" alt="Refusal rate on harmful tasks against random padding length for Grok 4 Fast, GPT-5, GPT-4.1-nano and DeepSeek-V3.1." width="49%">
</p>

<p align="center"><em>Left: AgentHarm score on harmful (solid) and benign (dashed) tasks. Right: refusal rate on harmful tasks. Both are plotted against the length of random padding placed after the task (Figure 2 in the paper).</em></p>

- **Capability drops well before the advertised context limit.** Models with 1M–2M-token windows lose more than 50% of their performance at around 100K tokens of padding, on both benign and harmful tasks.
- **Refusal rates change unpredictably.** At 200K tokens, GPT-4.1-nano's refusal rate rises from about 5% to about 40%, while Grok 4 Fast's falls from about 80% to about 10%.
- **The type of padding matters.** Coherent text hurts less than random tokens, and random tokens hurt less than unrelated task descriptions (multi-task padding).
- **The position of padding matters.** Padding placed *before* the task degrades performance less than padding placed *after* it.
- **Some refusals happen late.** Agents sometimes refuse only after they have already made tool calls, so part of the harmful action may already be done.

## Evaluation setup

<p align="center">
  <img src="assets/overview.svg" alt="Overview of the evaluation: an AgentHarm task is combined with padding of one of four types, placed before or after the task, and the agent's tool-call trajectory is graded." width="85%">
</p>

<p align="center"><em>Each AgentHarm task gets padding of one of four types, placed before or after the task description. The agent then runs and its tool calls are graded (Figure 1 in the paper).</em></p>

## Relation to AgentHarm and `inspect_evals`

This codebase **extends the AgentHarm implementation from [`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/agentharm)** (UK AISI). The benchmark, tools, grading functions and judges in [`src/mountain/agentharm/`](src/mountain/agentharm/) come from that implementation. We added:

- **Context padding** ([`padder.py`](src/mountain/agentharm/padder.py)), which inserts extra tokens into each sample at a chosen length and position.
- **Padding corpora** ([`paddings/`](src/mountain/agentharm/paddings/)): fiction, category-relevant text and multi-task prompts.
- **Prompting variants** for the agent (`standard`, `cot`, `react`, `refusal`).
- **Configurable judges** and run scripts for sweeping models, context lengths and random seeds.

The dataset itself is downloaded from Hugging Face ([`ai-safety-institute/AgentHarm`](https://huggingface.co/datasets/ai-safety-institute/AgentHarm)) the first time you run an evaluation.

## Padding types and positions

| `padding_type` | What gets inserted |
|---|---|
| `random` | Random tokens sampled from the `o200k_base` tokenizer with a fixed seed |
| `non_relevant` | Fiction unrelated to the task (humor, mythology, sci-fi, crime, romance) |
| `relevant` | Wikipedia or textbook text about the task's harm category |
| `multi_task` | Other AgentHarm task descriptions, separated by an instruction prompt |

`padding_position` can be `before` (between the system prompt and the task), `after` (after the task) or `around` (both). `multi_task` padding also accepts `*_with_task_instructed` variants. `padding_length` is measured in tokens.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.13.3.

```bash
git clone https://github.com/diogo-cruz/algoverse_mountain.git
cd algoverse_mountain
uv sync
```

Create a `.env` file in the repository root with API keys for the providers you plan to use. It is loaded automatically.

```bash
OPENAI_API_KEY=...        # also needed for the default judges (gpt-4.1-nano)
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...    # Grok, DeepSeek, Qwen, etc.
GOOGLE_API_KEY=...
```

## Running evaluations

The tasks `agentharm` (harmful) and `agentharm_benign` are defined in [`src/mountain/tasks.py`](src/mountain/tasks.py) and registered with Inspect. You can run them in three ways.

### 1. Inspect CLI

```bash
uv run inspect eval mountain/agentharm \
  --model openai/gpt-4.1-nano-2025-04-14 \
  -T split=test_public -T hint_included=true -T detailed_behaviors=true \
  -T padding_type=random -T padding_length=10000 \
  -T 'padding_kwargs={"padding_position": "after"}'
```

### 2. YAML config (sweeping context lengths and seeds)

```bash
uv run src/mountain/run_tasks.py --config_file configs/sonoma_dusk_harm.yaml
```

Example config:

```yaml
model_id: "openrouter/qwen/qwen3-235b-a22b:free"
task_type: "harm"                 # "harm" or "benign"
context_lengths: [0, 1000, 5000, 10000, 50000, 100000]
padding_type: "non_relevant"      # random | non_relevant | relevant | multi_task
padding_position: "after"         # before | after | around
split: "test_public"              # val | test_public
hint_included: true
detailed_behaviors: true
random_seeds: [3, 14, 15]
semantic_judge: "openai/gpt-4.1-nano-2025-04-14"
refusal_judge: "openai/gpt-4.1-nano-2025-04-14"
prompt_technique: "cot"           # standard | cot | react | refusal
log_dir: "logs/cot/qwen3/"
retry_on_error: 3
```

More examples are in [`configs/`](configs/). [`run_evals.zsh`](run_evals.zsh) shows how to chain several configs.

### 3. Python functions

Add a function whose name starts with `eval_` to [`src/mountain/run_tasks.py`](src/mountain/run_tasks.py). It should build the tasks and call `eval` or `eval_set`, with a new subdirectory under `logs/`. Then run it by name:

```bash
uv run src/mountain/run_tasks.py --task_name eval_grok_rand_pad
```

You can also launch tasks from the [Inspect VS Code extension](https://marketplace.visualstudio.com/items?itemName=ukaisi.inspect-ai).

### Viewing results

```bash
uv run inspect view --log-dir logs/
```


## Repository structure

```
src/mountain/
├── tasks.py               # Inspect @task entry points: agentharm, agentharm_benign
├── run_tasks.py           # CLI: YAML-config runner and eval_* experiment functions
└── agentharm/             # AgentHarm, extended from inspect_evals
    ├── agentharm.py       # Task construction (dataset, padding, agent, scorer)
    ├── padder.py          # Padding logic (random / non_relevant / relevant / multi_task)
    ├── paddings/          # Padding corpora
    ├── agents/            # Default and refusal agents, prompt techniques
    ├── benchmark/         # Tools and grading functions (harmful and benign)
    ├── scorer.py          # Combined scorer, refusal and semantic judges
    └── prompts.py         # System prompts
configs/                   # YAML experiment configs
notebooks/                 # Log parsing, analysis and paper figures (plot_eval_logs/paper_plots.ipynb)
```


