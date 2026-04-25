import pandas as pd
import matplotlib.pyplot as plt


def plot_main():
    df = pd.read_csv("reports/table_mean_std.csv")

    metrics = ["f1_flap", "f1_jump", "recall_flap", "recall_jump"]

    lstm = df[df["model"] == "LSTM"].set_index("measure")
    transformer = df[df["model"] == "Transformer"].set_index("measure")

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        [i - width/2 for i in x],
        [lstm.loc[m, "mean"] for m in metrics],
        width,
        label="LSTM"
    )

    ax.bar(
        [i + width/2 for i in x],
        [transformer.loc[m, "mean"] for m in metrics],
        width,
        label="Transformer"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (mean over folds)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("reports/figure_main_results.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_main()