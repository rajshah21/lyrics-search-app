import os
import pandas as pd
import pinecone
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from connect_pinecone import get_pinecone_index

# Load environment variables
load_dotenv()

# Connect to Pinecone index
index = get_pinecone_index()

# Define path to the new chunk-based embeddings
EMBEDDINGS_PATH = "data/lyrics_chunk_embeddings.parquet"
BATCH_SIZE = 50  # Pinecone recommends batch upserts for performance

def store_embeddings():
    """Uploads new chunk-based embeddings to Pinecone."""
    
    df = pd.read_parquet(EMBEDDINGS_PATH)
    
    # Convert embeddings from list format to NumPy arrays
    df["Embedding"] = df["Embedding"].apply(lambda x: np.array(x, dtype=np.float32))

    print(f"🚀 Uploading {len(df)} chunk embeddings to Pinecone...")

    # Step 3: Upload embeddings in batches
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Uploading to Pinecone", unit="batch"):
        batch = df.iloc[i:i+BATCH_SIZE]

        pinecone_batch = []
        for _, row in batch.iterrows():
            unique_id = f"{row['Song']} - {row['Artist']} - {hash(row['Chunk'])}"
            embedding_vector = row["Embedding"].tolist()

            if len(embedding_vector) > 0:
                pinecone_batch.append((unique_id, embedding_vector, {
                    "song": row["Song"],
                    "artist": row["Artist"],
                    "chunk": row["Chunk"]
                }))

        if pinecone_batch:
            index.upsert(vectors=pinecone_batch)

    print("✅ All chunk embeddings have been stored in Pinecone.")

if __name__ == "__main__":
    store_embeddings()
