import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys and environment details
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the existing index
index = pc.Index(PINECONE_INDEX_NAME)

# Verify connection by checking index details
print("Connected to Pinecone index:", index.describe_index_stats())

# Function to return the index
def get_pinecone_index():
    return index