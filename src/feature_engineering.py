import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        # Convert all values to string to avoid mixed-type errors
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    return df

if __name__ == "__main__":
    print("⚙️ Loading processed data...")
    df = pd.read_csv("/Users/sands/Desktop/EMIPredictAI/data/processed_data.csv", low_memory=False)

    print("🔢 Encoding categorical features...")
    df_encoded = encode_features(df)

    print("💾 Saving encoded data...")
    df_encoded.to_csv("/Users/sands/Desktop/EMIPredictAI/data/encoded_data.csv", index=False)

    print("✅ Feature engineering completed successfully! File saved at: data/encoded_data.csv")
