from pathlib import Path
import gc

from src.data.loader import load_data
from src.data.preprocessing import prepare_sequence_data, save_sequence_arrays


SEQUENCE_LENGTH = 15
GROUP_COLUMN = "Proband"

LABEL_COLUMNS = ["Flattern", "Hüpfen"]

FEATURE_COLUMNS = [
    'Shoulder_x_0', 'Shoulder_y_0', 'Elbow_x_0', 'Elbow_y_0',
    'RWrist_x_0', 'RWrist_y_0', 'LShoulder_x_0', 'LShoulder_y_0',
    'LElbow_x_0', 'LElbow_y_0', 'LWrist_x_0', 'LWrist_y_0',
    'Shoulder_x_1', 'Shoulder_y_1', 'Elbow_x_1', 'Elbow_y_1',
    'RWrist_x_1', 'RWrist_y_1', 'LShoulder_x_1', 'LShoulder_y_1',
    'LElbow_x_1', 'LElbow_y_1', 'LWrist_x_1', 'LWrist_y_1',
    'Shoulder_x_2', 'Shoulder_y_2', 'Elbow_x_2', 'Elbow_y_2',
    'RWrist_x_2', 'RWrist_y_2', 'LShoulder_x_2', 'LShoulder_y_2',
    'LElbow_x_2', 'LElbow_y_2', 'LWrist_x_2', 'LWrist_y_2',
    'Shoulder_x_3', 'Shoulder_y_3', 'Elbow_x_3', 'Elbow_y_3',
    'RWrist_x_3', 'RWrist_y_3', 'LShoulder_x_3', 'LShoulder_y_3',
    'LElbow_x_3', 'LElbow_y_3', 'LWrist_x_3', 'LWrist_y_3',
    'Shoulder_x_4', 'Shoulder_y_4', 'Elbow_x_4', 'Elbow_y_4',
    'RWrist_x_4', 'RWrist_y_4', 'LShoulder_x_4', 'LShoulder_y_4',
    'LElbow_x_4', 'LElbow_y_4', 'LWrist_x_4', 'LWrist_y_4'
]


def build_dtype_map():
    dtype_map = {}

    for col in FEATURE_COLUMNS:
        dtype_map[col] = "float32"

    dtype_map["Flattern"] = "int8"
    dtype_map["Hüpfen"] = "int8"
    dtype_map["Fraglich"] = "int8"
    dtype_map["Manierismus"] = "int8"

    # Falls Proband numerisch ist:
    # dtype_map["Proband"] = "int32"

    return dtype_map


def main():
    raw_csv_path = Path("local_data/df_cleaned.csv")
    output_dir = Path("data")

    required_columns = FEATURE_COLUMNS + LABEL_COLUMNS + ["Fraglich", "Manierismus", GROUP_COLUMN]
    dtype_map = build_dtype_map()

    df = load_data(
        raw_csv_path,
        usecols=required_columns,
        dtype=dtype_map,
    )

    X_sequences, y_sequences, groups_sequences, _ = prepare_sequence_data(
        df=df,
        feature_columns=FEATURE_COLUMNS,
        label_columns=LABEL_COLUMNS,
        group_column=GROUP_COLUMN,
        sequence_length=SEQUENCE_LENGTH,
        random_seed=0,
        apply_subset=True,
        apply_undersampling=True,
    )

    del df
    gc.collect()

    save_sequence_arrays(
        X_sequences=X_sequences,
        y_sequences=y_sequences,
        groups_sequences=groups_sequences,
        output_dir=output_dir,
        compressed=True,
    )


if __name__ == "__main__":
    main()