import os
import json
import asyncio
from typing import List, Dict, Any

from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

# Import the updated parser from your existing codebase
from document_parser import parse_document_to_sections

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"
CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION_NAME = "document_insights"


class IndexingPipeline:
    """
    An asynchronous pipeline that takes a PDF, processes it, generates embeddings
    using Google's async API, and indexes them in a local ChromaDB.
    """

    def __init__(self, google_api_key: str):
        """
        Initializes the pipeline and connects to the required services.
        """
        if not google_api_key:
            raise ValueError("API key for Google is required.")

        # Configure Google Generative AI
        genai.configure(api_key=google_api_key)

        # Configure ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        
        print("IndexingPipeline initialized successfully.")
        print(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' is ready.")

    def _prepare_chunk_for_embedding(self, chunk: Dict[str, Any]) -> str:
        """
        Creates a context-enriched string from a parsed data chunk.
        """
        full_path_str = " > ".join(chunk.get("full_path", []))
        return f"Section Path: {full_path_str}\nContent: {chunk.get('content', '')}"

    async def process_and_index_pdf_async(self, pdf_path: str, model_path: str, encoder_path: str):
        """
        Asynchronous method to process a single PDF and upload its content to ChromaDB.
        """
        print(f"--- Starting async processing for: {os.path.basename(pdf_path)} ---")

        # 1. Parsing remains synchronous as it's a CPU-bound task
        try:
            parsed_data = parse_document_to_sections(pdf_path, model_path, encoder_path)
            if not parsed_data:
                print(f"Warning: DocumentParser returned no data for {pdf_path}.")
                return
        except Exception as e:
            print(f"Error parsing document {pdf_path}: {e}")
            return
        print(f"Parsed {os.path.basename(pdf_path)} into {len(parsed_data)} sections.")

        # 2. Prepare chunks for embedding
        texts_to_embed = [self._prepare_chunk_for_embedding(chunk) for chunk in parsed_data]
        if not texts_to_embed:
            print(f"No text content found to embed in {os.path.basename(pdf_path)}.")
            return

        # 3. Generate embeddings asynchronously
        print(f"Generating embeddings for {len(texts_to_embed)} chunks from {os.path.basename(pdf_path)}...")
        try:
            result = await genai.embed_content_async(
                model=EMBEDDING_MODEL,
                content=texts_to_embed,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings = result['embedding']
            print(f"Embeddings generated for {os.path.basename(pdf_path)}.")
        except Exception as e:
            print(f"Error generating embeddings for {os.path.basename(pdf_path)}: {e}")
            return

        # 4. Prepare data for ChromaDB (synchronous)
        documents_to_add = texts_to_embed
        metadatas_to_add = []
        ids_to_add = []

        for i, chunk in enumerate(parsed_data):
            metadatas_to_add.append({
                "document_name": chunk.get("document_name", ""),
                "page_number": int(chunk.get("page_number", 0)),
                "section_title": chunk.get("section_title", ""),
                "full_path": " > ".join(chunk.get("full_path", [])),
                "original_content": chunk.get("content", ""),
                "bounding_box": json.dumps(chunk.get("bounding_box")) if chunk.get("bounding_box") else "{}"
            })
            ids_to_add.append(f"{os.path.basename(pdf_path)}_{i}")

        # 5. Add to ChromaDB (synchronous, but fast local operation)
        self.collection.add(
            embeddings=embeddings,
            documents=documents_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )
        
        print(f"--- Finished processing and indexing {os.path.basename(pdf_path)} ---")


async def main():
    """Main function to run the pipeline for a single file as a demonstration."""
    if GOOGLE_API_KEY:
        pipeline = IndexingPipeline(google_api_key=GOOGLE_API_KEY)
        
        sample_pdf = "pdfs/sample.pdf"
        model_file = "models/heading_classifier_model.joblib"
        encoder_file = "models/label_encoder.joblib"

        if os.path.exists(sample_pdf) and os.path.exists(model_file) and os.path.exists(encoder_file):
            # Run the single async task
            await pipeline.process_and_index_pdf_async(
                pdf_path=sample_pdf,
                model_path=model_file,
                encoder_path=encoder_file
            )
            print("\nExample PDF has been indexed asynchronously.")
        else:
            print("\nError: Make sure sample PDF and model files exist to run the example.")
    else:
        print("\n--- Please configure your Google API key! ---")
        print("Set the GOOGLE_API_KEY environment variable in a .env file.")

if __name__ == '__main__':
    # To run the async main function
    asyncio.run(main())
