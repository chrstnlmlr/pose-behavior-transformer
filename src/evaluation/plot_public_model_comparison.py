from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


def load_mean_metrics(csv_path: str | Path) -> dict:
    df = pd.read_csv(csv_path)
    return dict(zip(df["measure"], df["value"]))


def load_label_names(label_mapping_path: str | Path) -> list[str]:
    with open(label_mapping_path, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)

    # json stores class_name -> index
    idx_to_name = {idx: name for name, idx in label_mapping.items()}
    return [idx_to_name[i] for i in sorted(idx_to_name.keys())]


def build_public_comparison_df(
    lstm_metrics: dict,
    transformer_metrics: dict,
    label_names: list[str],
) -> pd.DataFrame:
    rows = []
    for idx, label_name in enumerate(label_names):
        rows.append(
            {
                "label": label_name,
                "LSTM_F1": lstm_metrics.get(f"f1_class_{idx}"),
                "Transformer_F1": transformer_metrics.get(f"f1_class_{idx}"),
                "LSTM_Recall": lstm_metrics.get(f"recall_class_{idx}"),
                "Transformer_Recall": transformer_metrics.get(f"recall_class_{idx}"),
            }
        )
    return pd.DataFrame(rows)


def plot_f1_comparison(comparison_df: pd.DataFrame, output_path: str | Path) -> None:
    x = range(len(comparison_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        [i - width / 2 for i in x],
        comparison_df["LSTM_F1"],
        width=width,
        label="LSTM",
    )
    ax.bar(
        [i + width / 2 for i in x],
        comparison_df["Transformer_F1"],
        width=width,
        label="Transformer",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison_df["label"], rotation=30, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_title("Public benchmark: LSTM vs Transformer (F1 by class)")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_recall_comparison(comparison_df: pd.DataFrame, output_path: str | Path) -> None:
    x = range(len(comparison_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        [i - width / 2 for i in x],
        comparison_df["LSTM_Recall"],
        width=width,
        label="LSTM",
    )
    ax.bar(
        [i + width / 2 for i in x],
        comparison_df["Transformer_Recall"],
        width=width,
        label="Transformer",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison_df["label"], rotation=30, ha="right")
    ax.set_ylabel("Recall")
    ax.set_title("Public benchmark: LSTM vs Transformer (Recall by class)")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    lstm_csv = Path("outputs/public_lstm/public_lstm_mean_metrics.csv")
    transformer_csv = Path("outputs/public_transformer/public_transformer_mean_metrics.csv")
    label_mapping_json = Path("data/public_label_mapping.json")
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    lstm_metrics = load_mean_metrics(lstm_csv)
    transformer_metrics = load_mean_metrics(transformer_csv)
    label_names = load_label_names(label_mapping_json)

    comparison_df = build_public_comparison_df(
        lstm_metrics=lstm_metrics,
        transformer_metrics=transformer_metrics,
        label_names=label_names,
    )

    print("\nPublic comparison table:")
    print(comparison_df)

    comparison_df.to_csv(figures_dir / "public_model_comparison.csv", index=False)
    plot_f1_comparison(comparison_df, figures_dir / "public_model_comparison_f1.png")
    plot_recall_comparison(comparison_df, figures_dir / "public_model_comparison_recall.png")

    print(f"\nSaved CSV to: {figures_dir / 'public_model_comparison.csv'}")
    print(f"Saved F1 plot to: {figures_dir / 'public_model_comparison_f1.png'}")
    print(f"Saved recall plot to: {figures_dir / 'public_model_comparison_recall.png'}")


if __name__ == "__main__":
    main()