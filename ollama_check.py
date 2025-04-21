from langchain_ollama import OllamaEmbeddings

# Initialize Ollama Embedding
embedding_model = OllamaEmbeddings(
        model="llama3.2:latest",
        base_url="http://127.0.0.1:11434",
    )  # Replace with your actual model name

# Sample text for embedding
sample_text = "This is a test to check embedding dimensions."

# Get embedding
embedding_vector = embedding_model.embed_query(sample_text)

# Print dimension
print("Embedding dimension:", len(embedding_vector))
