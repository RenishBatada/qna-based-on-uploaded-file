import os
from dotenv import load_dotenv
load_dotenv()

from typing import Dict, Any, List, Optional
import pandas as pd
import tempfile

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain_core.documents import Document
from backend.ingestion import ingest_text_chunks, ingest_tabular_chunks

Index_name = os.getenv("INDEX_NAME")

def ingest_file(file_content, file_name: str, file_type: str, page_texts=None, page_numbers=None):
    """
    Ingest a file into the Pinecone vector database.
    
    Args:
        file_content: The content of the file (DataFrame for tabular, text for others)
        file_name: Name of the file
        file_type: Type of the file (csv, txt, pdf, etc.)
        page_texts: For multi-page documents, list of text content by page
        page_numbers: For multi-page documents, list of page numbers
    """
    print(f"Ingesting file to Pinecone: {file_name}")
    
    # Check if file_content is a DataFrame - handle CSV, XLS, XLSX files
    if isinstance(file_content, pd.DataFrame):
        print(f"Processing tabular data ({file_type}): {file_name}")
        # Convert DataFrame to string format for Pinecone storage
        csv_content = file_content.to_csv(index=False)
        
        # Store in chunks to handle size limitations
        chunk_size = 1000  # Number of rows per chunk
        total_rows = len(file_content)
        
        for i in range(0, total_rows, chunk_size):
            chunk = file_content.iloc[i:min(i+chunk_size, total_rows)]
            chunk_text = chunk.to_csv(index=False)
            chunk_metadata = {
                "source": file_name,
                "file_type": file_type,
                "row_start": i,
                "row_end": min(i+chunk_size, total_rows)
            }
            # Use a single page number since this is tabular data
            ingest_text_chunks([chunk_text], file_name, file_type, [1], chunk_metadata)
            
    elif page_texts and page_numbers:
        ingest_text_chunks(page_texts, file_name, file_type, page_numbers)
    else:
        # Fallback for simple text files
        ingest_text_chunks([file_content], file_name, file_type, [1])


def get_vector_store():
    """Get the Pinecone vector store with the configured embeddings."""
    embeddings = OllamaEmbeddings(
        model="llama3.2:latest",
        base_url="http://127.0.0.1:11434",
    )
    
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


    return PineconeVectorStore(index_name=Index_name, embedding=embeddings)


def extract_clean_answer(error_msg: str) -> str:
    """Extract and clean answer from error message."""
    # Try to extract content between backticks
    if '`' in error_msg:
        start_quote = error_msg.find('`') + 1
        end_quote = error_msg.rfind('`')
        if start_quote > 0 and end_quote > start_quote:
            return error_msg[start_quote:end_quote]
    
    # If no backticks, try to extract after the colon
    if 'output:' in error_msg.lower():
        return error_msg.split('output:', 1)[1].strip()
    
    return error_msg

def clean_agent_output(output: str) -> str:
    """Clean up the agent output to remove warnings and format nicely."""
    if not output:
        return ""
        
    # Remove handle_parsing_errors message
    if "handle_parsing_errors=" in output:
        parts = output.split("to the AgentExecutor.")
        if len(parts) > 1:
            output = parts[1].strip()
    
    # Remove error prefix
    if "This is the error:" in output:
        parts = output.split("This is the error:")
        if len(parts) > 1:
            output = parts[1].strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "Final Answer: ",
        "Answer: ",
        "Here's the answer: ",
        "Could not parse LLM output:"
    ]
    
    result = output
    for prefix in prefixes_to_remove:
        if result.lower().startswith(prefix.lower()):
            result = result[len(prefix):].strip()
    
    # Clean up any markdown formatting
    result = result.replace("```", "").replace("python", "")
    
    # Remove any URLs or error messages
    if "https://" in result:
        result = result[:result.find("https://")]
    
    # Remove any "Thought:" or "Action:" lines
    lines = result.split("\n")
    cleaned_lines = []
    skip_line = False
    for line in lines:
        line = line.strip()
        if any(x in line.lower() for x in ["thought:", "action:", "input:", "assistant:", "human:", "error:", "warning:"]):
            skip_line = True
            continue
        if line and not skip_line:
            cleaned_lines.append(line)
        skip_line = False
    
    # Join and clean up extra whitespace
    result = "\n".join(cleaned_lines).strip()
    result = "\n".join(line for line in result.split("\n") if line.strip())
    
    return result

def process_tabular_query(query: str, file_name: str, file_type: str, use_file_directly=False) -> dict:
    """
    Process queries for tabular data (CSV, XLS, XLSX) using a CSV agent.
    """
    print(f"Processing tabular query for {file_name} using CSV agent")
    
    try:
        # Initialize the LLM
        # llm = ChatOllama(temperature=0, model="llama3.2")
        llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")
        
        # Always use the file from the data directory
        file_path = os.path.join("data", file_name)
        print(f"Looking for file at: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find file: {file_path}")

        # Create and use CSV agent
        agent = create_csv_agent(
            llm,
            file_path,
            verbose=True,
            handle_parsing_errors=True,
            allow_dangerous_code=True
        )

        # Add helpful context to the query
        enhanced_query = f"""Based on the CSV file data, Give the ans of user query: {query}
        Important: Format your response as a clear, readable answer without any technical details or debugging information.
        Do not include any warnings, error messages, or URLs in your response."""

        # Execute query
        try:
            result = agent.invoke({"input": enhanced_query})
            raw_answer = result.get("output", "")
            
            # Clean up the answer
            cleaned_answer = clean_agent_output(raw_answer)
            
            if not cleaned_answer:
                cleaned_answer = "Could not generate a proper response. Please try rephrasing your question."
            
            return {
                "query": query,
                "answer": cleaned_answer,
                "source": [file_name]
            }
            
        except Exception as parsing_error:
            error_msg = str(parsing_error)
            answer = clean_agent_output(extract_clean_answer(error_msg))
        
        if not answer:
            answer = "Could not generate a proper response. Please try rephrasing your question."
        
        return {
            "query": query,
            "answer": answer,
            "source": [file_name]
        }
        
    except Exception as e:
        print(f"Error processing tabular query: {str(e)}")
        error_msg = str(e)
        
        # Try to extract useful content from error message
        if "Could not parse LLM output:" in error_msg:
            answer = clean_agent_output(extract_clean_answer(error_msg))
            if answer:
                return {
                    "query": query,
                    "answer": answer,
                    "source": [file_name]
                }
        
        return {
            "query": query,
            "answer": "Sorry, I encountered an error processing your query. Please try rephrasing your question.",
            "source": []
        }


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> dict:
    """
    Process a user query against the document database.
    
    Args:
        query: The user query
        chat_history: Previous chat history
        
    Returns:
        Dictionary with query, answer, and source information
    """
    # Extract file metadata if available
    file_meta = {}
    for item in reversed(chat_history):
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "human":
            continue
        if isinstance(item, dict) and "meta" in item:
            file_meta = item["meta"]
            break
    
    file_type = file_meta.get("file_type", "")
    file_name = file_meta.get("file_name", "")
    
    # Handle tabular data differently - but this is mostly handled in main.py now
    # This is a fallback in case we need to retrieve from Pinecone
    if file_type in ["csv", "xls", "xlsx"] and file_name:
        return process_tabular_query(query, file_name, file_type, use_file_directly=False)
    
    # Process different document types - could be expanded for specific file types
    return process_text_documents(query, chat_history)


def process_text_documents(query: str, chat_history: List[Dict[str, Any]] = []) -> dict:
    """
    Process queries for text-based documents (PDF, DOCX, TXT, etc.)
    
    Args:
        query: The user query
        chat_history: Previous chat history
        
    Returns:
        Dictionary with query, answer, and source information
    """
    # For text-based documents, use retrieval QA chain
    

    embeddings = OllamaEmbeddings(
        model="llama3.2:latest",
        base_url="http://127.0.0.1:11434",
    )
    
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    
    docsearch = PineconeVectorStore(index_name=Index_name, embedding=embeddings)
    # chat = ChatOllama(temperature=0, model="llama3.2")
    chat = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")
    
    # Define a strict custom prompt that only uses provided context
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # Custom strict retrieval prompt
    strict_retrieval_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant that ONLY answers based on the provided context. 
        
Your answers must NEVER include information not provided in the context documents.
If the answer cannot be found in the context, you MUST say "I don't have enough information to answer this question."
Never invent sources. Only cite sources that are explicitly mentioned in the provided context.
Do not reference websites, books, or other materials not provided in the context.
        
Context information is below:
{context}
        
Given this context, help the human."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # Custom strict rephrase prompt
    strict_rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given the conversation history and the latest user question, rephrase the question to be a standalone question 
that strictly focuses on retrieving information from the document context.
DO NOT add any additional information or context beyond what's explicitly mentioned in the chat history.
If there is not enough information to understand the intent, use only the original question.
        
Chat History:
{chat_history}
        
User Question: {input}"""),
    ])
    
    # Create the document chain with our custom prompt
    stuff_document_chain = create_stuff_documents_chain(chat, strict_retrieval_prompt)
    
    # Create history-aware retrieval with our custom rephrase prompt
    history_aware_retrieval = create_history_aware_retriever(
        llm=chat,
        retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        prompt=strict_rephrase_prompt,
    )
    
    qa = create_retrieval_chain(
        history_aware_retrieval, combine_docs_chain=stuff_document_chain
    )
    
    # Format chat history for the chain
    formatted_history = []
    for item in chat_history:
        if isinstance(item, tuple) and len(item) == 2:
            role, message = item
            if role == "human":
                formatted_history.append({"type": "human", "content": message})
            elif role == "ai":
                formatted_history.append({"type": "ai", "content": message})
    
    result = qa.invoke(input={"input": query, "chat_history": formatted_history})
    
    # Debug the structure of the context/source information
    print("DEBUG - Result keys:", result.keys())
    if "context" in result and result["context"]:
        print("DEBUG - Context type:", type(result["context"]))
        if isinstance(result["context"], list):
            print(f"DEBUG - Context is a list with {len(result['context'])} items")
            for i, doc in enumerate(result["context"]):
                print(f"DEBUG - Doc {i} type: {type(doc)}")
                if hasattr(doc, "metadata"):
                    print(f"DEBUG - Doc {i} metadata: {doc.metadata}")
    
    # Make sure we return the full document objects with metadata intact
    return {
        "query": result["input"],
        "answer": result["answer"],
        "source": result["context"] if "context" in result else [],
    }
