# Function to load data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, sep=';')
        return df
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None

# File path and data loading
file_name = 'df_cleaned.csv'
data = load_data(file_name)

if data is not None:
    print("Data loaded successfully.")
    print(f"\nData Shape: {data.shape}")
else:
    print("Data loading failed.")