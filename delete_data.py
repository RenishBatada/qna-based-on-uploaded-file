import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Get the index name
index_name = os.getenv("INDEX_NAME")
index = pc.Index(index_name)

# Step 1: Fetch all IDs related to your file
query_response = index.query(
    vector=[0] * 3072,  # ✅ Updated to match Ollama's vector dimension
    top_k=1000,  # Fetch maximum possible matches
    filter={
        # "source": "D:\\third roack techno\\LangChain\\practices\\third\\gls_governor_body.txt"
    },
    include_metadata=True,
)

# Extract all vector IDs
vector_ids_to_delete = [match["id"] for match in query_response["matches"]]

# Step 2: Delete vectors by IDs
if vector_ids_to_delete:
    index.delete(ids=vector_ids_to_delete)
    print(f"✅ Successfully deleted {len(vector_ids_to_delete)} vectors.")
else:
    print("⚠ No vectors found for deletion.")
