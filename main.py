from typing import Set
import time
import streamlit as st
import pandas as pd
import json
import PyPDF2
import os
import tempfile
import io

from backend.core import run_llm, ingest_file, process_tabular_query
from backend.ingestion import process_file_content, clear_vector_store

# Session state
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "file_uploaded_once" not in st.session_state:
    st.session_state["file_uploaded_once"] = False
if "uploaded_text" not in st.session_state:
    st.session_state["uploaded_text"] = ""
if "file_meta" not in st.session_state:
    st.session_state["file_meta"] = {}
if "tabular_data" not in st.session_state:
    st.session_state["tabular_data"] = None
if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = ""

# Create data directory for storing tabular files
if not os.path.exists("data"):
    os.makedirs("data")

st.header("Document Processing System")
st.subheader("Upload documents and ask questions about their content")

def create_sources_string(source_urls: Set[str]) -> str:
    """Format source information for display."""
    if not source_urls:
        return ""
    formatted_sources = []
    for src in sorted(source_urls):
        if ":" in src:
            # This format assumes ingestion stored source as "filename:page"
            try:
                file_name, page = src.rsplit(":", 1)
                formatted_sources.append(f"{file_name} (Page {page})")
                print(f"DEBUG - Formatted source with page: {file_name} (Page {page})")
            except Exception as e:
                print(f"DEBUG - Error formatting source {src}: {e}")
                formatted_sources.append(src)
        else:
            formatted_sources.append(src)
    
    # Just provide the numbered list without the duplicate "Sources:" header
    result = "\n".join(f"{i+1}. {src}" for i, src in enumerate(formatted_sources))
    print(f"DEBUG - Final formatted sources: {result}")
    print(f"{formatted_sources=}")
    return result

# Always define file uploader to avoid NameError
file = None
if not st.session_state["file_uploaded_once"]:
    file = st.file_uploader(
        label="Upload a file (PDF, TXT, CSV, XLS, XLSX, DOCX, PPTX)",
        type=["csv", "txt", "pdf", "xls", "xlsx", "docx", "pptx","ppt"],
    )

if file is not None:
    with st.spinner("Processing file..."):
        file_name = file.name
        file_type = file.type or file_name.split(".")[-1].lower()
        
        # Process file content based on file type
        file_content, page_texts, page_numbers = process_file_content(file, file_name, file_type)
        
        # Create data directory if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")
            
        # Handle different file types
        if file_name.endswith((".csv", ".xls", ".xlsx")):
            # For tabular data, save locally for CSV agent and store in session state
            if isinstance(file_content, pd.DataFrame):
                # Convert Excel to CSV if needed
                if file_name.endswith((".xls", ".xlsx")):
                    file_name = os.path.splitext(file_name)[0] + ".csv"
                
                # Save to data directory for CSV agent
                csv_path = os.path.join("data", file_name)
                file_content.to_csv(csv_path, index=False)
                
                # Store metadata and ingest into Pinecone
                st.session_state["tabular_data"] = file_content
                st.session_state["file_meta"] = {
                    "file_name": file_name,
                    "file_type": "csv",
                    "local_path": csv_path
                }
                
                # Ingest into Pinecone
                ingest_file(file_content, file_name, "csv", None, None)
                
                st.success(f"Tabular file '{file_name}' processed and stored in both local directory and Pinecone!")
            else:
                st.error("Error: Could not process tabular file format")
        else:
            # For text-based documents, ingest into Pinecone
            ingest_file(file_content, file_name, file_type, page_texts, page_numbers)
            st.session_state["uploaded_text"] = file_content
            st.session_state["file_meta"] = {"file_name": file_name, "file_type": file_type}
            st.success(f"Document '{file_name}' uploaded and processed successfully!")
        
        st.session_state["file_uploaded_once"] = True
        st.session_state["file_meta"] = {"file_name": file_name, "file_type": file_type}

elif st.session_state["file_uploaded_once"]:
    st.info("📁 File already uploaded. You can ask questions about it below.")

# Optional Reset
if st.session_state["file_uploaded_once"]:
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Reset"):
            # Reset session state
            st.session_state["file_uploaded_once"] = False
            st.session_state["uploaded_text"] = ""
            st.session_state["file_meta"] = {}
            st.session_state["user_prompt_history"] = []
            st.session_state["chat_answer_history"] = []
            st.session_state["chat_history"] = []
            st.session_state["tabular_data"] = None
            st.session_state["user_prompt"] = ""
            
            st.rerun()

# File info display
if st.session_state["file_uploaded_once"] and st.session_state["file_meta"]:
    file_meta = st.session_state["file_meta"]
    with st.expander("File Information", expanded=False):
        st.write(f"**File Name:** {file_meta.get('file_name', 'Unknown')}")
        st.write(f"**File Type:** {file_meta.get('file_type', 'Unknown')}")
        
        # For tabular data, show a preview
        if file_meta.get('file_type') in ["csv", "xls", "xlsx"] and st.session_state["uploaded_text"]:
            try:
                df = pd.read_csv(io.StringIO(st.session_state["uploaded_text"]))
                st.write("**Data Preview:**")
                st.dataframe(df.head(5))
            except:
                pass

# --- User Prompt ---
st.markdown("### Ask a question about your document")
user_prompt = st.text_input("Enter your question:", key="user_prompt")

def update_session_state():
    if user_prompt:
        with st.spinner("Generating answer..."):
            # Add file metadata to chat history for context
            if st.session_state["file_meta"] and not any(isinstance(item, dict) and "meta" in item for item in st.session_state["chat_history"]):
                st.session_state["chat_history"].append({"meta": st.session_state["file_meta"]})
            
            # Get response from LLM
            file_meta = st.session_state["file_meta"]
            file_type = file_meta.get("file_type", "")
            file_name = file_meta.get("file_name", "")
            
            # Log query information
            print(f"Processing query: '{user_prompt}'")
            print(f"File metadata: {file_meta}")
            
            # Process query based on file type
            if file_type == "csv" or file_meta.get("file_name", "").endswith((".csv", ".xls", ".xlsx")):
                # Use CSV agent for tabular data
                result = process_tabular_query(
                    st.session_state["user_prompt"],
                    file_meta.get("file_name", ""),
                    file_type,
                    use_file_directly=True
                )
            else:
                # Use standard document processing for text files
                result = run_llm(
                    st.session_state["user_prompt"],
                    st.session_state["chat_history"]
                )
            
            # Format the response with source information
            sources = []
            print("DEBUG - Response type:", type(result))
            print("DEBUG - Response keys:", result.keys())
            
            if "source" in result and result["source"]:
                print("DEBUG - Source type:", type(result["source"]))
                if isinstance(result["source"], list):
                    print(f"DEBUG - Source is a list with {len(result['source'])} items")
                    for i, doc in enumerate(result["source"]):
                        print(f"DEBUG - Doc {i} type: {type(doc)}")
                        if hasattr(doc, "metadata"):
                            metadata = doc.metadata
                            print(f"DEBUG - Doc {i} metadata: {metadata}")
                            # Get the source directly from metadata
                            source = metadata.get("source", "")
                            page = metadata.get("page", "")
                            file_type = metadata.get("file_type", "")
                            
                            print(f"DEBUG - Extracted: source={source}, page={page}, file_type={file_type}")
                            
                            # For PDF/DOCX/PPTX, ensure we have the page number in the source
                            if (file_type in ["pdf", "docx", "pptx"] or 
                                "pdf" in file_type.lower() or 
                                "docx" in file_type.lower() or 
                                "pptx" in file_type.lower()) and source and ":" not in source:
                                source = f"{source}:{int(page) if isinstance(page, (int, float)) else page}"
                                print(f"DEBUG - Updated source with page: {source}")
                            
                            if source:
                                sources.append(source)
                                print(f"DEBUG - Added source: {source}")
            
            formatted_response = f"{result['answer']}"
            if sources:
                if "Sources:" not in result['answer']:
                    formatted_response += f"\n\nSources:\n{create_sources_string(set(sources))}"
                else:
                    # If the answer already has a Sources section, just skip adding another one
                    pass
            
            # Update session state
            st.session_state["user_prompt_history"].append(user_prompt)
            st.session_state["chat_answer_history"].append(formatted_response)
            st.session_state["chat_history"].append(("human", user_prompt))
            st.session_state["chat_history"].append(("ai", formatted_response))
            
            # Clear the input field by setting user_prompt to empty string
            st.session_state["user_prompt"] = ""

st.button("Submit", on_click=update_session_state)

if st.session_state["chat_answer_history"]:
    with st.expander("Chat History", expanded=True):
        for i in reversed(range(len(st.session_state["chat_answer_history"]))):
            st.chat_message("user").write(st.session_state["user_prompt_history"][i])
            st.chat_message("assistant").write(st.session_state["chat_answer_history"][i])
