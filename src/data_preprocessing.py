import pandas as pd
import os

def load_data(path):
    return pd.read_csv(path, low_memory=False)

def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates()
    # Handle missing values (simple fill strategy)
    df = df.fillna(df.median(numeric_only=True))
    return df

if __name__ == "__main__":
    # Define input and output paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "emi_prediction_dataset.csv")
    output_path = os.path.join(base_dir, "data", "processed_data.csv")

    # Ensure data directory exists
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)

    # Load and clean
    print("📂 Loading dataset...")
    df = load_data(input_path)
    print(f"✅ Loaded {len(df)} records")

    print("🧹 Cleaning dataset...")
    df_clean = clean_data(df)

    # Save cleaned file
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved successfully at: {output_path}")
