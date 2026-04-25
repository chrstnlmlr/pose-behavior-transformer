from pathlib import Path
import pandas as pd


def load_all_fold_metrics(folder):
    files = sorted(Path(folder).glob("*_metrics_fold_*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def compute_mean_std(df):
    grouped = df.groupby("measure")["value"]
    mean = grouped.mean()
    std = grouped.std()

    result = pd.DataFrame({
        "measure": mean.index,
        "mean": mean.values,
        "std": std.values
    })

    return result


def main():
    lstm_path = "outputs/lstm"
    transformer_path = "outputs/transformer"

    lstm_df = load_all_fold_metrics(lstm_path)
    transformer_df = load_all_fold_metrics(transformer_path)

    lstm_stats = compute_mean_std(lstm_df)
    transformer_stats = compute_mean_std(transformer_df)

    lstm_stats["model"] = "LSTM"
    transformer_stats["model"] = "Transformer"

    final = pd.concat([lstm_stats, transformer_stats])
    final.to_csv("reports/table_mean_std.csv", index=False)

    print(final)


if __name__ == "__main__":
    main()