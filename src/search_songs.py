import os
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from connect_pinecone import get_pinecone_index

# Load environment variables
load_dotenv()

# Load Sentence-Transformer Model with GPU support
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device: {device}")

model = SentenceTransformer("all-MiniLM-L6-v2").to(device)  # Ensure this matches your stored embeddings model

# Connect to Pinecone index
index = get_pinecone_index()

def get_query_embedding(query):
    """Generate sentence-transformer embedding for the user query (lyrics snippet)."""
    query_embedding = model.encode(query, convert_to_numpy=True, device=device)
    return query_embedding

def search_songs(query, top_k=10):
    """Find the most similar lyrics chunks and return the original song."""
    query_embedding = get_query_embedding(query)

    search_results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    # Extract and group results by song
    song_matches = {}
    for match in search_results["matches"]:
        metadata = match["metadata"]
        song_title = metadata["song"]
        artist = metadata["artist"]
        chunk = metadata["chunk"]
        score = round(match["score"], 4)

        if song_title not in song_matches:
            song_matches[song_title] = {"artist": artist, "score": score, "chunks": []}

        song_matches[song_title]["chunks"].append(chunk)

    return song_matches

if __name__ == "__main__":
    query = input(" Enter a lyrics snippet: ")
    results = search_songs(query, top_k=10)

    print("\n Search Results:")
    for song, data in results.items():
        print(f"\n {song} - {data['artist']} (Score: {data['score']})")
        print("Matched Lyrics Snippet(s):")
        for chunk in data["chunks"]:
            print(f" - {chunk}")
