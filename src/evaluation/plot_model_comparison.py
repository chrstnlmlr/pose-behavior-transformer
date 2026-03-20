from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_mean_metrics(csv_path: str | Path) -> dict:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    return dict(zip(df["measure"], df["value"]))


def build_comparison_df(lstm_metrics: dict, transformer_metrics: dict) -> pd.DataFrame:
    selected_metrics = [
        "f1_flap",
        "recall_flap",
        "precision_flap",
        "specificity_flap",
        "f1_jump",
        "recall_jump",
        "precision_jump",
        "specificity_jump",
    ]

    rows = []
    for metric in selected_metrics:
        rows.append(
            {
                "metric": metric,
                "LSTM": lstm_metrics.get(metric),
                "Transformer": transformer_metrics.get(metric),
            }
        )

    return pd.DataFrame(rows)


def plot_comparison(comparison_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)

    x = range(len(comparison_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        [i - width / 2 for i in x],
        comparison_df["LSTM"],
        width=width,
        label="LSTM",
    )
    ax.bar(
        [i + width / 2 for i in x],
        comparison_df["Transformer"],
        width=width,
        label="Transformer",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison_df["metric"], rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("LSTM vs Transformer: Mean CV Metrics")
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    lstm_csv = Path("outputs/lstm/lstm_mean_metrics.csv")
    transformer_csv = Path("outputs/transformer/transformer_mean_metrics.csv")
    output_png = Path("outputs/model_comparison.png")

    if not lstm_csv.exists():
        raise FileNotFoundError(f"Missing file: {lstm_csv}")
    if not transformer_csv.exists():
        raise FileNotFoundError(f"Missing file: {transformer_csv}")

    lstm_metrics = load_mean_metrics(lstm_csv)
    transformer_metrics = load_mean_metrics(transformer_csv)

    comparison_df = build_comparison_df(lstm_metrics, transformer_metrics)

    print("\nComparison table:")
    print(comparison_df)

    plot_comparison(comparison_df, output_png)
    print(f"\nSaved plot to: {output_png}")


if __name__ == "__main__":
    main()