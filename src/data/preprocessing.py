from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_SEQUENCE_LENGTH = 15


def print_label_distribution(y_data, groups_data, sequence_length, fold_type):
    no_man = np.sum(np.all(y_data == [0, 0], axis=1))
    jump = np.sum(np.all(y_data == [0, 1], axis=1))
    flap = np.sum(np.all(y_data == [1, 0], axis=1))
    flap_jump = np.sum(np.all(y_data == [1, 1], axis=1))
    group_count = len(np.unique(groups_data))

    print(f"\nLabel distribution in {fold_type}:")
    print(
        f"  no_man={no_man/sequence_length}, "
        f"jump={jump/sequence_length}, "
        f"flap={flap/sequence_length}, "
        f"flap_jump={flap_jump/sequence_length}"
    )
    print(f"  total sequences={len(y_data)/sequence_length}, unique groups={group_count}")


def select_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep:
    - rows with Flattern == 1 and Fraglich == 0
    - rows with Hüpfen == 1 and Fraglich == 0
    - rows with Manierismus == 0
    """
    mask = (
        ((df["Flattern"] == 1) & (df["Fraglich"] == 0))
        | ((df["Hüpfen"] == 1) & (df["Fraglich"] == 0))
        | (df["Manierismus"] == 0)
    )
    return df.loc[mask].copy()


def get_group_array(df: pd.DataFrame, group_column: str) -> np.ndarray:
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in dataframe.")
    return df[group_column].values


def extract_arrays(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_columns: list[str],
    group_column: str,
):
    X = df[feature_columns].values
    y = df[label_columns].values
    groups = get_group_array(df, group_column)

    print(f"\nExtracted arrays:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  groups shape: {groups.shape}")

    return X, y, groups


def standardize_features(X: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def undersample_sequences(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    random_seed: int = 0,
):
    """
    Undersample majority class [0, 0] at the sequence level.
    """
    rng = np.random.default_rng(random_seed)

    n_sequences = X.shape[0] // sequence_length
    X_reshaped = X.reshape(n_sequences, sequence_length, -1)
    y_reshaped = y.reshape(n_sequences, sequence_length, -1)
    groups_reshaped = groups[::sequence_length]

    no_man_indices = np.where(np.all(y_reshaped[:, 0] == [0, 0], axis=1))[0]
    jump_indices = np.where(np.all(y_reshaped[:, 0] == [0, 1], axis=1))[0]
    flap_indices = np.where(np.all(y_reshaped[:, 0] == [1, 0], axis=1))[0]
    flap_jump_indices = np.where(np.all(y_reshaped[:, 0] == [1, 1], axis=1))[0]

    max_no_man = len(jump_indices) + len(flap_indices) + len(flap_jump_indices)

    if len(no_man_indices) > max_no_man:
        undersampled_no_man_indices = rng.choice(
            no_man_indices,
            size=max_no_man,
            replace=False,
        )
    else:
        undersampled_no_man_indices = no_man_indices

    undersampled_indices = np.concatenate(
        [
            undersampled_no_man_indices,
            jump_indices,
            flap_indices,
            flap_jump_indices,
        ]
    )

    X_undersampled = X_reshaped[undersampled_indices]
    y_undersampled = y_reshaped[undersampled_indices]
    groups_undersampled = groups_reshaped[undersampled_indices]

    groups_undersampled_rows = np.repeat(groups_undersampled, sequence_length)

    return (
        X_undersampled.reshape(-1, X.shape[-1]),
        y_undersampled.reshape(-1, y.shape[-1]),
        groups_undersampled_rows,
    )


def make_sequence_arrays(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
):
    """
    Convert row-level arrays into sequence-level arrays.
    """
    num_sequences = X.shape[0] // sequence_length

    X_sequences = X.reshape((num_sequences, sequence_length, -1))
    y_sequences = y.reshape((num_sequences, sequence_length, -1))[:, 0, :]
    groups_sequences = groups[::sequence_length]

    print(f"\nSequence arrays created:")
    print(f"  X_sequences shape: {X_sequences.shape}")
    print(f"  y_sequences shape: {y_sequences.shape}")
    print(f"  groups_sequences shape: {groups_sequences.shape}")

    return X_sequences, y_sequences, groups_sequences


def save_sequence_arrays(
    X_sequences: np.ndarray,
    y_sequences: np.ndarray,
    groups_sequences: np.ndarray,
    output_dir: str | Path = "data",
    compressed: bool = False,
):
    """
    Save sequence arrays locally. Do not commit these to GitHub.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if compressed:
        np.savez_compressed(output_dir / "sequence_data.npz",
                            X_sequences=X_sequences,
                            y_sequences=y_sequences,
                            groups_sequences=groups_sequences)
        print(f"Saved compressed sequence data to: {output_dir / 'sequence_data.npz'}")
    else:
        np.save(output_dir / "X_sequences.npy", X_sequences)
        np.save(output_dir / "y_sequences.npy", y_sequences)
        np.save(output_dir / "groups_sequences.npy", groups_sequences)
        print(f"Saved X_sequences.npy, y_sequences.npy, groups_sequences.npy to: {output_dir}")


def prepare_sequence_data(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_columns: list[str],
    group_column: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    random_seed: int = 0,
    apply_subset: bool = True,
    apply_undersampling: bool = True,
):
    """
    Full preprocessing pipeline from dataframe to sequence arrays.
    """
    if apply_subset:
        df = select_subset(df)
        print(f"\nSubset selected: {df.shape}")

    X, y, groups = extract_arrays(
        df=df,
        feature_columns=feature_columns,
        label_columns=label_columns,
        group_column=group_column,
    )
    print_label_distribution(y, groups, sequence_length, "row-level data")

    X_scaled, scaler = standardize_features(X)

    if apply_undersampling:
        X_processed, y_processed, groups_processed = undersample_sequences(
            X=X_scaled,
            y=y,
            groups=groups,
            sequence_length=sequence_length,
            random_seed=random_seed,
        )
        print(f"\nAfter undersampling:")
        print(f"  X shape: {X_processed.shape}")
        print(f"  y shape: {y_processed.shape}")
        print(f"  groups shape: {groups_processed.shape}")
        print_label_distribution(
            y_processed,
            groups_processed,
            sequence_length,
            "undersampled row-level data",
        )
    else:
        X_processed, y_processed, groups_processed = X_scaled, y, groups

    X_sequences, y_sequences, groups_sequences = make_sequence_arrays(
        X=X_processed,
        y=y_processed,
        groups=groups_processed,
        sequence_length=sequence_length,
    )

    return X_sequences, y_sequences, groups_sequences, scaler