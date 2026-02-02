import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import scienceplots

plt.style.use(["science", "no-latex"])


def load_training_results(results_dir="training_results", json_files=None):
    results_path = Path(results_dir)
    experiments = []

    if json_files:
        json_paths = [results_path / f for f in json_files]
    else:
        json_paths = sorted(results_path.glob("*.json"))

    for json_file in json_paths:
        with open(json_file, "r") as f:
            data = json.load(f)
            experiments.append(data)

    return experiments


def plot_training_results(
    experiments, mode="both", save_path="training_results.png", log_scale=True
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for idx, exp in enumerate(experiments):
        exp_name = exp["experiment_name"]
        epochs = np.arange(1, len(exp["train_perplexities"]) + 1)

        if mode in ["train", "both"]:
            line = ax.plot(
                epochs,
                exp["train_perplexities"],
                label=f"{exp_name} (train)",
                linestyle="-",
                linewidth=1.5,
            )
            color = line[0].get_color()
            min_train_ppl = min(exp["train_perplexities"])
            min_train_epoch = exp["train_perplexities"].index(min_train_ppl) + 1
            ax.plot(
                min_train_epoch,
                min_train_ppl,
                marker="*",
                markersize=12,
                color=color,
                markeredgecolor="black",
                markeredgewidth=1,
                zorder=5,
            )
            ax.annotate(
                f"{min_train_ppl:.1f}",
                xy=(min_train_epoch, min_train_ppl),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )

        if mode in ["val", "both"]:
            line = ax.plot(
                epochs,
                exp["val_perplexities"],
                label=f"{exp_name} (val)",
                linestyle="--",
                linewidth=1.5,
            )
            color = line[0].get_color()
            min_val_ppl = min(exp["val_perplexities"])
            min_val_epoch = exp["val_perplexities"].index(min_val_ppl) + 1
            ax.plot(
                min_val_epoch,
                min_val_ppl,
                marker="*",
                markersize=12,
                color=color,
                markeredgecolor="black",
                markeredgewidth=1,
                zorder=5,
            )
            ax.annotate(
                f"{min_val_ppl:.1f}",
                xy=(min_val_epoch, min_val_ppl),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")

    title_map = {
        "train": "Training Perplexity",
        "val": "Validation Perplexity",
        "both": "Training and Validation Perplexity",
    }
    ax.set_title(title_map[mode])

    handles, labels = ax.get_legend_handles_labels()
    star_marker = plt.Line2D(
        [0],
        [0],
        marker="*",
        color="black",
        linestyle="None",
        markersize=12,
        markeredgecolor="black",
        markeredgewidth=1,
        label="Lowest Perplexity",
    )
    handles.append(star_marker)
    labels.append("Lowest Perplexity")
    ax.legend(handles, labels, loc="best", frameon=True)

    if log_scale:
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot training results from JSON files"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "val", "both"],
        default="both",
        help="Plot training, validation, or both (default: both)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_results.png",
        help="Output file path (default: training_results.png)",
    )
    parser.add_argument(
        "--json-files",
        type=str,
        nargs="+",
        default=None,
        help="Specific JSON files to plot (default: all JSON files in training_results/)",
    )
    parser.add_argument(
        "--no-log-scale",
        action="store_false",
        dest="log_scale",
        help="Disable log scale for y-axis (default: log scale enabled)",
    )

    args = parser.parse_args()

    experiments = load_training_results(json_files=args.json_files)

    if not experiments:
        print("No training results found in training_results/ directory")
        return

    plot_training_results(
        experiments, mode=args.mode, save_path=args.output, log_scale=args.log_scale
    )


if __name__ == "__main__":
    main()
