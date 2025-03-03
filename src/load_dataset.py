import pandas as pd
import os

# Define the dataset path
DATASET_PATH = "data\spotify_millsongdata.csv"

def load_and_clean_data():
    """Load the Spotify dataset and preprocess it for embedding."""
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Display first few rows to inspect
    print("Original Dataset Sample:\n", df.head())

    # Drop unnecessary columns (keep only relevant ones)
    df = df[["artist", "song", "text"]]

    # Remove rows with missing lyrics
    df = df.dropna(subset=["text"])

    # Convert 'artists' column to a clean format
    df["artist"] = df["artist"].apply(lambda x: x.strip("[]").replace("'", ""))

    # Show dataset size after cleaning
    print(f"Dataset size after filtering: {len(df)} songs")

    return df

# Run the script if executed directly
if __name__ == "__main__":
    df = load_and_clean_data()