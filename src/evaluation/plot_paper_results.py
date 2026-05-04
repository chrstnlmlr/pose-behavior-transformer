import pandas as pd
import matplotlib.pyplot as plt


def plot_main():
    df = pd.read_csv("reports/table_mean_std.csv")

    metrics = ["f1_flap", "recall_flap", "f1_jump", "recall_jump"]
    labels = ["F1\n(Flapping)", "Recall\n(Flapping)", "F1\n(Jumping)", "Recall\n(Jumping)"]

    lstm = df[df["model"] == "LSTM"].set_index("measure")
    transformer = df[df["model"] == "Transformer"].set_index("measure")

    x = range(len(metrics))
    width = 0.28

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    ax.bar(
        [i - width / 2 for i in x],
        [lstm.loc[m, "mean"] for m in metrics],
        width,
        yerr=[lstm.loc[m, "std"] for m in metrics],
        capsize=4,
        label="LSTM",
        color="#4C72B0",
    )

    ax.bar(
        [i + width / 2 for i in x],
        [transformer.loc[m, "mean"] for m in metrics],
        width,
        yerr=[transformer.loc[m, "std"] for m in metrics],
        capsize=4,
        label="Transformer",
        color="#DD8452",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    plt.savefig("reports/figure_main_results.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_main()