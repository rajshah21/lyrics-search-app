import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

# Load API Key (if still using OpenAI for other tasks)
load_dotenv()

# Define paths
DATASET_PATH = "data/spotify_millsongdata.csv"
EMBEDDINGS_PATH = "data/lyrics_chunk_embeddings.parquet"

# Load Sentence-Transformers Model (supports GPU acceleration)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer("all-MiniLM-L6-v2").to(device)  # Efficient model with GPU support

CHUNK_SIZE = 2  # Number of lines per chunk
BATCH_SIZE = 64  # Larger batch size for fast GPU processing

def chunk_lyrics(df):
    """Splits all lyrics in the dataset into smaller chunks."""
    chunks = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking lyrics"):
        song, artist, lyrics = row["song"], row["artist"], row["text"]
        if not isinstance(lyrics, str) or lyrics.strip() == "":
            continue  # Skip if lyrics are empty

        lines = lyrics.split("\n")
        lyric_chunks = [" ".join(lines[i:i+CHUNK_SIZE]) for i in range(0, len(lines), CHUNK_SIZE)]

        for chunk in lyric_chunks:
            if chunk.strip():  # Ensure chunk is not empty
                chunks.append({"Song": song, "Artist": artist, "Chunk": chunk})
    
    return pd.DataFrame(chunks)  # Return as a new DataFrame

def process_dataset():
    """Loads dataset, pre-chunks lyrics, generates embeddings, and saves."""
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    df = df[["song", "artist", "text"]].dropna()

    # Pre-chunk all lyrics using Pandas
    chunk_df = chunk_lyrics(df)

    # Prepare storage for embeddings
    embeddings = []
    print(f"Generating embeddings for {len(chunk_df)} chunks...")

    # Process embeddings in GPU batches
    for i in tqdm(range(0, len(chunk_df), BATCH_SIZE), desc="Generating embeddings"):
        batch_texts = chunk_df["Chunk"][i:i+BATCH_SIZE].tolist()

        try:
            batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, device=device)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"❌ Error in batch {i // BATCH_SIZE + 1}: {e}")
            embeddings.extend([None] * len(batch_texts))  # Placeholder for failed batch
    
    # Store embeddings
    chunk_df["Embedding"] = embeddings
    chunk_df = chunk_df.dropna(subset=["Embedding"])  # Remove failed embeddings
    chunk_df.to_parquet(EMBEDDINGS_PATH, index=False)
    print(f"✅ Embeddings saved to {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    process_dataset()
