# Separate features and labels in full dataset
X_data = data[column_names_features].values
y_data = data[column_names_labels].values
if group_column == 'Proband':
  groups_data = data['Proband'].values
else:
  groups_data = data['video_id'].values
print(f"\nX_data Shape: {X_data.shape}")
print(f"y_data Shape: {y_data.shape}")
print(f"groups_data Shape: {groups_data.shape}")
print_label_distribution(y_data, groups_data, sequence_length, 'y_data')

# Select subset
def select_subset(df):
    df_subset = df[(df["Flattern"] == 1) & (df["Fraglich"] == 0) |
            (df["Hüpfen"] == 1) & (df["Fraglich"] == 0) |
            (df["Manierismus"] == 0)]
    return df_subset
data_subset = select_subset(data)

# Separate features and labels in subset
X = data_subset[column_names_features].values
y = data_subset[column_names_labels].values
if group_column == 'Proband':
  groups = data_subset['Proband'].values
else:
  groups = data_subset['video_id'].values
print(f"\nX_data Shape: {X.shape}")
print(f"y Shape: {y.shape}")
print(f"groups Shape: {groups.shape}")
print_label_distribution(y, groups, sequence_length, 'y')

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Undersampling
def undersample(X, y, groups, sequence_length=15):
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
        undersampled_no_man_indices = np.random.choice(no_man_indices, size=max_no_man, replace=False)
    else:
        undersampled_no_man_indices = no_man_indices
    undersampled_indices = np.concatenate([undersampled_no_man_indices, jump_indices, flap_indices, flap_jump_indices])
    X_undersampled = X_reshaped[undersampled_indices]
    y_undersampled = y_reshaped[undersampled_indices]
    groups_undersampled = groups_reshaped[undersampled_indices]
    groups_undersampled_rows = np.repeat(groups_undersampled, sequence_length)
    return X_undersampled.reshape(-1, X.shape[-1]), y_undersampled.reshape(-1, y.shape[-1]), groups_undersampled_rows
X_undersampled, y_undersampled, groups_undersampled = undersample(X, y, groups, sequence_length)
print(f"\nX_undersampled Shape: {X_undersampled.shape}")
print(f"y_undersampled Shape: {y_undersampled.shape}")
print(f"groups_undersampled Shape: {groups_undersampled.shape}")
print_label_distribution(y_undersampled, groups_undersampled, sequence_length, 'y_undersampled')

# Check group overlap
def check_group_overlap(groups_train, groups_test):
    common_groups = np.intersect1d(groups_train, groups_test)
    if len(common_groups) > 0:
        print(f"Overlap found in groups: {common_groups}")
    else:
        print("No overlap in groups between training and test sets.")

# Reshape the data
num_sequences = X_undersampled.shape[0] // sequence_length
X_sequences = X_undersampled.reshape((num_sequences, sequence_length, -1))
y_sequences = y_undersampled.reshape((num_sequences, sequence_length, -1))[:, 0, :]
groups_sequences = groups_undersampled[::sequence_length]
