# Document Processing System with LangChain and Pinecone

This project implements a document processing system using LangChain and Pinecone vector database. It allows users to upload various file types (CSV, TXT, PDF, XLS, XLSX, DOCX, PPTX), stores them in a Pinecone vector database, and provides a query-answering system based on the contents of the uploaded documents.

## Features

- **File Upload**: Support for multiple file formats (CSV, TXT, PDF, XLS, XLSX, DOCX, PPTX)
- **Vector Database Storage**: Stores document content in Pinecone for efficient retrieval
- **Intelligent Query Processing**: 
  - For PDF/DOCX/PPTX files: Provides answers with file name and page number
  - For CSV/XLS/XLSX files: Uses a CSV agent to process and join data for answering queries
- **Chat History**: Maintains conversation history for context-aware responses

## Project Structure

- **main.py**: Streamlit interface for file uploads and user queries
- **backend/core.py**: Functions for interacting with Pinecone and processing queries
- **backend/ingestion.py**: Handles document ingestion and text extraction

## Setup

1. Ensure you have Python installed (3.8+ recommended)
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up a Pinecone account and create an index
4. Create a `.env` file with your Pinecone API key and index name:
   ```
   PINECONE_API_KEY=your_api_key
   INDEX_NAME=your_index_name
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run main.py
   ```
2. Upload a document through the web interface
3. Ask questions about the document content
4. View answers with source information

## Supported File Types

- **CSV/XLS/XLSX**: Tabular data files
- **PDF**: Multi-page documents
- **TXT**: Plain text files
- **DOCX**: Microsoft Word documents
- **PPTX**: Microsoft PowerPoint presentations

## Dependencies

- LangChain for document processing and retrieval
- Pinecone for vector database storage
- Pandas for handling tabular data
- PyMuPDF for PDF text extraction
- Streamlit for the web interface
- Ollama for LLM integration

## Notes

- The system uses a local Ollama instance for embeddings and LLM functionality
- For large files, processing may take some time
- The CSV agent can handle complex queries about tabular data
