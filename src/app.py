import gradio as gr
import torch
from sentence_transformers import SentenceTransformer
from connect_pinecone import get_pinecone_index
import pandas as pd

# Load Sentence-Transformer Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Using device: {device}")

model = SentenceTransformer("all-MiniLM-L6-v2").to(device)  # Ensure it matches stored embeddings

# Connect to Pinecone
index = get_pinecone_index()

def get_query_embedding(query):
    """Generate sentence-transformer embedding for the user query."""
    query_embedding = model.encode(query, convert_to_numpy=True, device=device)
    return query_embedding

def search_songs(query, top_k=5):
    """Find the most similar lyrics chunks and return results in table format."""
    query_embedding = get_query_embedding(query)

    search_results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    # Extract and group results
    song_data = []
    for match in search_results["matches"]:
        metadata = match["metadata"]
        song_title = metadata["song"]
        artist = metadata["artist"]
        score = round(match["score"], 4)

        song_data.append([song_title, artist, score])

    # Convert to DataFrame for Gradio
    df = pd.DataFrame(song_data, columns=["Song Name", "Artist", "Confidence Score"])
    return df if not df.empty else pd.DataFrame([["No results found", "", ""]], columns=["Song Name", "Artist", "Score"])

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 🎶 Lyrics Search Engine")
    gr.Markdown("🔍 Enter a lyrics snippet to find matching songs!")

    with gr.Row():
        query_input = gr.Textbox(label="Enter Lyrics Snippet", placeholder="Type part of a song's lyrics here...")
        search_button = gr.Button("🔍 Search")

    results_output = gr.Dataframe(headers=["Song Name", "Artist", "Confidence Score"], interactive=False)

    search_button.click(search_songs, inputs=query_input, outputs=results_output)

# Run Gradio app
if __name__ == "__main__":
    app.launch()

