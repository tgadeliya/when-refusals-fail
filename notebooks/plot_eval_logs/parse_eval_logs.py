from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
import ast

from inspect_ai import task
from inspect_ai.log import EvalSampleScore, read_eval_log 
from inspect_ai.log import EvalSample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def list_all_logs_in_dir(dir_path: Path):
    files_path = []
    for item in dir_path.iterdir():
        if item.is_file() and item.suffix == ".eval":
            files_path.append(Path(item))
    return files_path


def parse_grading_line(line: str) -> dict:
    try:
        start = line.index("{")
        end = line.rindex("}")
        dict_str = line[start:end+1]
        return ast.literal_eval(dict_str)
    except (ValueError, SyntaxError):
        return {}

@dataclass
class EvalRun:
    task_name: str
    model_name: str
    split: str
    padding_type: str | None
    padding_length: int
    random_seed: int
    metrics_overall: dict
    metrics_per_sample: list[dict]
    num_turns_per_sample: list[int]
    num_turns_per_sample_avg: dict
    grading_breakdown_per_sample: list[dict[str, Any]]
    log_path: Path
    task_args: dict[str, Any]
    _samples: list[EvalSample] | None

    @classmethod
    def from_log_file(cls, log_path: Path) -> "EvalRun":
        logs = read_eval_log(log_path)
        md = []
        num_turns_per_sample = []
        for idx, sample in enumerate(logs.samples or []):
            score: EvalSampleScore = logs.reductions[0].samples[idx]
            md.append(
                {
                    "id": sample.metadata["id"], 
                    "id_original": sample.metadata["id_original"], 
                    "hint_included": sample.metadata["hint_included"], 
                    "detailed_prompt": sample.metadata["detailed_prompt"],
                    'name': sample.metadata["name"],
                    'category': sample.metadata["category"],
                    "score_obj": score,
                    "score": score.value["score"]},
            )
            num_turns_per_sample.append(len(sample.messages))
        task_args = logs.eval.task_args_passed
        task_args["padding_token_length"] = task_args.get("padding_token_length", task_args.get("padding_length", None)) 
        import random
        task_args["random_seed"] = task_args.get("random_seed", random.randint(0, 2**32 - 1))
        return cls(
            task_name=logs.plan.steps[0].params["task_name"],
            model_name = logs.eval.model,
            split=task_args["split"],
            random_seed=task_args["random_seed"],
            padding_type=task_args["padding_type"],
            padding_length= 0 if task_args["padding_token_length"] is None else task_args["padding_token_length"],
            metrics_overall=logs.results.scores[0].metrics,
            metrics_per_sample=md,
            log_path=log_path,
            task_args=task_args,
            grading_breakdown_per_sample=[parse_grading_line(sample.explanation) for sample in logs.reductions[0].samples],
            num_turns_per_sample= num_turns_per_sample,
            num_turns_per_sample_avg={"num_turns_avg" : np.mean(num_turns_per_sample)},
            _samples = logs.samples
        )

    @property
    def metrics_per_sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics_per_sample)



@dataclass
class EvalRunRandomSeedAgg:
    task_name: str
    model_name: str
    split: str
    padding_type: str | None
    padding_length: int
    random_seeds: list[int]
    metrics_overall: list[dict]
    num_turns_per_sample: list
    num_turns_per_sample_avg: list[dict]
    metrics_per_sample: list[list[dict]]
    grading_breakdown_per_sample: list[list[dict[str, Any]]]
    log_pathes: list[Path]
    task_args: dict[str, Any]
    list_of_evals: list[EvalRun]

    @classmethod
    def from_eval_run_list(cls, evals: list[EvalRun]) -> "EvalRunRandomSeedAgg":
        return cls(
            task_name=evals[0].task_name,
            model_name =evals[0].model_name,
            split=evals[0].split,
            random_seeds=[eval.random_seed for eval in evals],
            padding_type=evals[0].padding_type,
            padding_length=evals[0].padding_length,
            metrics_overall=[eval.metrics_overall for eval in evals],
            metrics_per_sample=[eval.metrics_per_sample for eval in evals],
            log_pathes=[eval.log_path for eval in evals],
            task_args=evals[0].task_args,
            list_of_evals=evals,
            num_turns_per_sample=[eval.num_turns_per_sample for eval in evals],
            num_turns_per_sample_avg=[eval.num_turns_per_sample_avg for eval in evals],
            grading_breakdown_per_sample=[eval.grading_breakdown_per_sample for eval in evals],
        )

    @property
    def metrics_overall_agg(self) -> dict[str, list]:
        return self._agg_dicts(self.metrics_overall)

    def _agg_dicts(self, list_of_dicts: list[dict]) -> dict[str, Any]:
        aggregated = defaultdict(list)
        for d in list_of_dicts:
            for key, value in d.items():
                aggregated[key].append(value.value)
        return dict(aggregated)

    @property
    def metrics_num_turns_agg(self) -> list: 
        aggregated = defaultdict(list)
        for d in self.num_turns_per_sample_avg:
            for key, value in d.items():
                aggregated[key].append(float(value))
        return dict(aggregated)["num_turns_avg"]



def agg_rs_results(dir_path: Path) -> EvalRunRandomSeedAgg:
    pathes = list_all_logs_in_dir(dir_path)
    return  EvalRunRandomSeedAgg.from_eval_run_list([EvalRun.from_log_file(p) for p in pathes])


def plot_dict_with_rs(
    dict_with_rs: dict,
    dict_with_rs2: Optional[dict] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "score",
    connect_keys: bool = False,
    figsize: Tuple[float, float] | None = None,
    dict1_color: str = "blue",
    dict2_color: str | None = None,
    labels: tuple[str, str] = ("", ""),
) -> None:
    """
    Plots one or two dicts of values.
    
    dict_with_rs: {key (str or hashable): [float, ...]}
    dict_with_rs2: optional second dict (must have same keys in same order if provided).
    
    Shows raw runs (gray dots) + mean with ±SE error bars.
    """

    def compute_stats(d: dict):
        cats = list(d.keys())
        means, errs, all_x_pos, all_y = [], [], [], []
        for i, k in enumerate(cats):
            runs = np.asarray(d[k], dtype=float)
            if runs.size == 0:
                continue
            means.append(runs.mean())
            sd = runs.std(ddof=1) if runs.size > 1 else 0.0
            se = sd / np.sqrt(runs.size) if runs.size > 0 else 0.0
            errs.append(se)
            all_x_pos.extend([i] * runs.size)
            all_y.extend(runs.tolist())
        return cats, means, errs, all_x_pos, all_y

    # --- Compute stats for dict1
    cats1, means1, errs1, all_x1, all_y1 = compute_stats(dict_with_rs)

    plt.figure(figsize= figsize or (7, 4.5))

    # Scatter for dict1
    plt.scatter(all_x1, all_y1, color="gray", alpha=0.6, label=f"Runs {labels[0]}")
    fmt = "o-" if connect_keys else "o"
    plt.errorbar(
        np.arange(len(cats1)),
        means1,
        yerr=errs1,
        fmt=fmt,
        color=dict1_color,
        capsize=5,
        label=f"{labels[0]} mean ± SE",
    )

    # --- If second dict provided
    if dict_with_rs2 is not None:
        cats2, means2, errs2, all_x2, all_y2 = compute_stats(dict_with_rs2)

        if cats1 != cats2:
            raise ValueError("Both dicts must have the same keys in the same order!")

        plt.scatter(all_x2, all_y2, color="black", alpha=0.6, label=f"Runs {labels[1]}")
        plt.errorbar(
            np.arange(len(cats2)),
            means2,
            yerr=errs2,
            fmt=fmt,
            color=dict2_color if dict2_color is not None else ("red" if dict1_color != "red" else "blue"),
            capsize=5,
            label=f"{labels[1]} mean ± SE",
        )

    # Axes, ticks, labels
    plt.xticks(np.arange(len(cats1)), [str(k) for k in cats1], rotation=45)
    # change xticks size
    plt.tick_params(axis='x', labelsize=7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

def plot_dicts_with_rs(
    dicts_with_rs: List[dict],
    dict_colors: List[str],
    labels: List[str],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "score",
    connect_keys: bool = False,
    figsize: Tuple[float, float] | None = None,
) -> None:

    n_dicts = len(dicts_with_rs)
    # Check that all dicts share identical keys (and order)
    cats_ref = list(dicts_with_rs[0].keys())
    for j, d in enumerate(dicts_with_rs[1:], start=1):
        if list(d.keys()) != cats_ref:
            raise ValueError(
                f"All dicts must have the same keys in the same order. Mismatch at dict index {j}."
            )

    x_positions = np.arange(len(cats_ref))

    def compute_stats(d: dict, cats: list[str]):
        """
        Returns positions where data exists, means, SEs, and all raw points (x,y).
        Skips categories with empty lists for means/SEs, but keeps raw points empty if no runs.
        """
        means_idx, means, errs = [], [], []
        all_x, all_y = [], []

        for i, k in enumerate(cats):
            runs = np.asarray(d[k], dtype=float)
            if runs.size > 0:
                mean = runs.mean()
                sd = runs.std(ddof=1) if runs.size > 1 else 0.0
                se = sd / np.sqrt(runs.size)
                means_idx.append(i)
                means.append(mean)
                errs.append(se)
                all_x.extend([i] * runs.size)
                all_y.extend(runs.tolist())

        return np.asarray(means_idx), np.asarray(means), np.asarray(errs), np.asarray(all_x), np.asarray(all_y)

    plt.figure(figsize=figsize or (7, 4.5))

    # Plot each dict
    for j, (dct, color, label) in enumerate(zip(dicts_with_rs, dict_colors, labels)):
        means_idx, means, errs, all_x, all_y = compute_stats(dct, cats_ref)

        # Raw runs (slightly transparent). Offset x a tiny bit per series to reduce overlap.
        if all_x.size > 0:
            # offset = (j - (n_dicts - 1) / 2) * 0.08  # small deterministic offset
            plt.scatter(all_x, all_y, alpha=0.35, s=16, label=None, color=color)

        # Means + SE
        fmt = "o-" if connect_keys else "o"
        if means_idx.size > 0:
            plt.errorbar(
                means_idx,
                means,
                yerr=errs,
                fmt=fmt,
                color=color,
                capsize=5,
                label=f"{label} mean ± SE",
            )

    # Axes, ticks, labels
    plt.xticks(x_positions, [str(k) for k in cats_ref], rotation=45)
    plt.tick_params(axis="x", labelsize=7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def aggregate_histograms(
    runs: List[List[int]],
    bins: np.ndarray | None = None,
    density: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate histograms across multiple seeds (runs) of one experiment.

    runs: list of seeds; each seed is a list of integer 'turns per sample'
    bins: optional shared bin edges; if None, computed from data (integer bins)
    density: if True, return probabilities; else return counts

    Returns:
        mean_hist: mean histogram across seeds (probabilities if density=True)
        std_hist:  std across seeds (same scale as mean_hist)
        bins:      the bin edges used
    """
    if not runs:
        raise ValueError("runs is empty")

    # Build bins if not provided (integer bins covering all observed values)
    if bins is None:
        all_vals = [v for run in runs for v in run]
        if len(all_vals) == 0:
            raise ValueError("All runs are empty")
        vmin = int(np.floor(min(all_vals)))
        vmax = int(np.ceil(max(all_vals)))
        # integer bins, centered at k+0.5
        bins = np.arange(vmin - 0.5, vmax + 1.5, 1.0)

    hists = []
    for run in runs:
        hist, _ = np.histogram(run, bins=bins, density=density)
        hists.append(hist)

    hists = np.asarray(hists)
    mean_hist = hists.mean(axis=0)
    std_hist = hists.std(axis=0, ddof=1) if len(hists) > 1 else np.zeros_like(mean_hist)
    return mean_hist, std_hist, bins


def plot_mean_histograms(
    experiments: Dict[str, List[List[int]]],
    bins: np.ndarray | None = None,
    density: bool = True,
    with_std: bool = True
) -> Dict[str, dict]:
    """
    Plot mean histogram per experiment. Returns a dict with per-experiment stats.

    experiments: {experiment_name: [seed1_turns, seed2_turns, ...]}
    bins: optional shared bin edges for all experiments; if None, computed globally
    density: if True, plot probability mass; else plot mean counts
    with_std: if True, plot ±1 std as a filled band

    Returns:
        stats: {name: {"mean": mean_hist, "std": std_hist, "bins": bins}}
    """
    if not experiments:
        raise ValueError("experiments is empty")

    # If bins not given, compute a global integer binning across all experiments
    if bins is None:
        all_vals = []
        for runs in experiments.values():
            for run in runs:
                all_vals.extend(run)
        if len(all_vals) == 0:
            raise ValueError("All experiments are empty")
        vmin = int(np.floor(min(all_vals)))
        vmax = int(np.ceil(max(all_vals)))
        bins = np.arange(vmin - 0.5, vmax + 1.5, 1.0)

    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    plt.figure(figsize=(8, 5))
    stats = {}

    for name, runs in experiments.items():
        mean_hist, std_hist, _ = aggregate_histograms(runs, bins=bins, density=density)
        stats[name] = {"mean": mean_hist, "std": std_hist, "bins": bins}

        # Use a line over integer-centered bins (clean overlay for multiple experiments)
        plt.plot(bin_centers, mean_hist, marker="o", label=name)
        if with_std and np.any(std_hist > 0):
            upper = mean_hist + std_hist
            lower = mean_hist - std_hist
            lower = np.clip(lower, a_min=0, a_max=None)  # keep non-negative for probs/counts
            plt.fill_between(bin_centers, lower, upper, alpha=0.2)

    plt.xlabel("Turns per sample")
    plt.ylabel("Probability" if density else "Mean count")
    plt.title("Mean histogram per experiment (aggregated across seeds)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(title="Experiment")
    plt.tight_layout()
    plt.show()

    return stats

def plot_category_heatmaps(log_dict, figsize=(18, 16)):
    """
    Create two heatmaps stacked vertically:
      1. Harm Scores
      2. Refusal Rates
    
    Parameters:
    -----------
    log_dict : dict
        Dictionary where each key maps to another dictionary of metrics (each metric has a list with one value).
    figsize : tuple
        Size of the figure (width, height).

    Example:
    -----------
    >>> results_dict = {}

    >>> log_dirs = [
    >>>     "0_before", "100_before", "1000_before",
    >>> ]

    >>> for dir_name in log_dirs:
    >>>     key_dir_name = dir_name.replace("00000", "00k").replace("0000", "0k").replace("000", "k")
    >>>     results_dict[key_dir_name] = agg_rs_results(LOGS_DIR / f"harm/non_relevant/")

    >>> plot_split_heatmaps(results_dict)

    """
    df = pd.DataFrame({k: {m: v[0] for m, v in log_dict[k].metrics_overall_agg.items()} for k, _ in log_dict.items()}).T
    score_df = df[[col for col in df.columns if "score" in col.lower()]]
    other_df = df[[col for col in df.columns if "score" not in col.lower()]]

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    fig.patch.set_facecolor('#fafafa')
    
    score_mask = score_df.isnull()
    refusal_mask = other_df.isnull()
    
    # colors
    pastel_mint = LinearSegmentedColormap.from_list('pastel_mint', [
        '#e8f6f3',  # Very light mint
        '#d6efe9',  # Light mint
        '#c4e8e0',  # Medium mint
        '#b2e1d6',  # Mint
        '#a0dacd',  # Soft seafoam
        '#8ed3c3'   # Pastel seafoam
    ])
    
    score_heatmap = sns.heatmap(
        score_df, 
        annot=True, 
        cmap=pastel_mint, 
        fmt=".3f", 
        ax=axes[0],
        mask=score_mask,
        cbar_kws={'label': 'Score Value', 'shrink': 0.8},
        linewidths=0.5,
        linecolor='white',
        annot_kws={'size': 10, 'weight': 'bold', 'color': '#555555'}
    )

    axes[0].set_title("Harm Scores", fontsize=16, fontweight='bold', pad=20, color='#555555')
    axes[0].set_xlabel("Metrics", fontsize=12, fontweight='bold', color='#555555')
    axes[0].set_ylabel("Padding Length", fontsize=12, fontweight='bold', color='#555555')
    axes[0].tick_params(axis="x", rotation=45, labelsize=10, colors='#555555')
    axes[0].tick_params(axis="y", labelsize=10, colors='#555555')
    
    cbar = score_heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10, colors='#555555')
    cbar.set_label('Score Value', fontsize=12, fontweight='bold', color='#555555')
    
    for _, spine in axes[0].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('#dddddd')
    
    refusal_heatmap = sns.heatmap(
        other_df, 
        annot=True, 
        cmap=pastel_mint, 
        fmt=".3f", 
        ax=axes[1],
        mask=refusal_mask,
        cbar_kws={'label': 'Refusal Rate', 'shrink': 0.8},
        linewidths=0.5,
        linecolor='white',
        annot_kws={'size': 10, 'weight': 'bold', 'color': '#555555'}
    )
    axes[1].set_title("Refusal Rates", fontsize=16, fontweight='bold', pad=20, color='#555555')
    axes[1].set_xlabel("Metrics", fontsize=12, fontweight='bold', color='#555555')
    axes[1].set_ylabel("Padding Length", fontsize=12, fontweight='bold', color='#555555')
    axes[1].tick_params(axis="x", rotation=45, labelsize=10, colors='#555555')
    axes[1].tick_params(axis="y", labelsize=10, colors='#555555')
    
    cbar = refusal_heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10, colors='#555555')
    cbar.set_label('Refusal Rate', fontsize=12, fontweight='bold', color='#555555')
    
    for _, spine in axes[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('#dddddd')
    
    # Adjust layout with more padding
    plt.tight_layout(pad=3.0)