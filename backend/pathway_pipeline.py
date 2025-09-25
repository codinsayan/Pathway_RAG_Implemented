import os
import tempfile
from datetime import datetime

import pathway as pw
from pathway.xpacks.llm.embedders import GoogleGeminiEmbedder
from pathway.xpacks.llm.llms import GoogleGeminiChat, prompt_chat_single_qa
from pathway.xpacks.llm.vector_stores import VectorStoreServer

# Import your custom parser
from document_parser import parse_document_to_sections

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
UPLOAD_FOLDER = "pdfs"
HOST = "0.0.0.0"
PORT = 8000
MODEL_FILE = "models/heading_classifier_model.joblib"
ENCODER_FILE = "models/label_encoder.joblib"


# --- Pathway User-Defined Function (UDF) for Custom Parsing ---
# This function will wrap your existing document parser.
def parse_pdf(data: bytes, path: str) -> list[dict]:
    """
    Parses a PDF file from raw bytes using your custom document parser.
    """
    # Create a temporary file to store the PDF content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(data)
        temp_pdf_path = temp_pdf.name

    try:
        # Call your existing parser with the path to the temporary file
        # and the required model/encoder paths.
        parsed_data = parse_document_to_sections(
            pdf_path=temp_pdf_path,
            model_path=MODEL_FILE,
            encoder_path=ENCODER_FILE,
        )
        # Add the original document name to each section for context
        for section in parsed_data:
            section["document_name"] = os.path.basename(path)
        return parsed_data
    finally:
        # Clean up the temporary file
        os.unlink(temp_pdf_path)


def run_pipeline():
    # 1. Data Source: Read documents from the local file system.
    documents = pw.io.fs.read(
        UPLOAD_FOLDER,
        mode="streaming",
        format="binary",
        with_metadata=True,
    )

    # 2. Custom Parsing: Apply your document parser to each incoming file.
    # We use a UDF to wrap the call to your parsing function.
    # The `pw.udf` decorator marks the function for use in the Pathway pipeline.
    @pw.udf
    def safe_parse_pdf(data: bytes, path: str) -> list[dict]:
        try:
            return parse_pdf(data, path)
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            return []

    # The .map() function applies the UDF to each row of the documents table.
    # The result is a table where each row contains a list of parsed sections.
    parsed_documents = documents.select(
        sections=safe_parse_pdf(pw.this.data, pw.this.path)
    )

    # 3. Flattening: Create a new table where each row is a single section.
    # This is necessary because the parser returns multiple sections per document.
    document_chunks = parsed_documents.flatten(pw.this.sections)
    document_chunks = document_chunks.select(
        chunk=pw.this.sections["content"],
        doc_id=pw.this.sections["document_name"],
    )

    # 4. Embedder and LLM: Configure the models.
    embedder = GoogleGeminiEmbedder(api_key=GOOGLE_API_KEY)
    llm = GoogleGeminiChat(api_key=GOOGLE_API_KEY)

    # 5. Vector Indexing: Create embeddings and build the real-time vector index.
    vector_store = VectorStoreServer(
        document_chunks,
        embedder=embedder,
        vector_key="chunk",
        doc_id_key="doc_id",
    )

    # 6. Query Handling: Set up the REST API endpoint for queries.
    query, response_writer = pw.io.http.rest_connector(
        host=HOST,
        port=PORT,
        schema={"query": str},
        autocommit_duration_ms=50,
    )

    # 7. RAG Logic: Generate responses using the LLM and vector store.
    responses = prompt_chat_single_qa(
        vector_store.query(query, k=3),
        query_key="query",
        llm=llm,
        prompt_template="""
        You are a helpful assistant.
        Answer the question based on the context provided.
        Context:
        {context}

        Question:
        {query}
        """,
    )

    # 8. Output: Write the responses back to the HTTP response writer.
    response_writer(responses)

    # Run the pipeline.
    print(f"Running Pathway pipeline with custom parser on http://{HOST}:{PORT}")
    pw.run()


if __name__ == "__main__":
    run_pipeline()