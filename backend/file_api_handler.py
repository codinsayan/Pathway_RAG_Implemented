# file_api_handler.py

import os
import time
import json # Import the json library
from typing import List, Dict, Any

from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# --- Configuration ---
# Ensure your GOOGLE_API_KEY is set in your .env file or environment
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
GENERATION_MODEL = "gemini-1.5-flash-latest" # As specified in the docs

# This is the powerful, all-in-one prompt for Gemini
INSIGHTS_PROMPT_TEMPLATE = """
You are an expert AI assistant for a document analysis tool. Your task is to analyze a user's selected text based ONLY on the provided context from their document library, which is referenced by the attached files.

Analyze the user's selection in light of the context and generate the following, structured as a JSON object with three top-level keys: "retrieved_sections", "generated_insights", and "podcast_script".

1.  **retrieved_sections**: An array of objects. Each object should represent a relevant text snippet and have "source_document" and "content" keys. Provide up to 5 snippets.
2.  **generated_insights**: A single string containing categorized insights. Structure this string with markdown headings: "### Contradictions (⚔️):", "### Enhancements (💡):", and "### Connections (🔗):".
3.  **podcast_script**: A single string containing a short, engaging, two-speaker conversational script (Host and Analyst) summarizing the key findings.

USER'S SELECTED TEXT:
"{user_selection}"

Analyze the user's selection now.
"""


class FileApiHandler:
    """
    Handles uploading files to the Google AI File API and generating insights using Gemini.
    """

    def __init__(self):
        # This list will now store the full File objects from the API
        self.uploaded_file_objects = []
        
        # *** CHANGE: Enable JSON Mode for reliable output ***
        self.generation_model = genai.GenerativeModel(
            GENERATION_MODEL,
            generation_config={"response_mime_type": "application/json"}
        )
        
        print("FileApiHandler initialized successfully in JSON mode.")

    def upload_pdf_to_api(self, pdf_path: str) -> Dict[str, Any]:
        """Uploads a single PDF file to the Google AI File API."""
        print(f"Uploading {os.path.basename(pdf_path)} to Google AI File API...")
        try:
            # The File API automatically handles chunking and indexing
            uploaded_file = genai.upload_file(path=pdf_path, display_name=os.path.basename(pdf_path))
            
            # We need to wait for the processing to complete
            print(f"File '{uploaded_file.display_name}' is processing...")
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                # You need to get the file again to check its updated state
                uploaded_file = genai.get_file(uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                 raise Exception(f"File processing failed for {pdf_path}. Reason: {uploaded_file.error}")

            # Store the entire File object
            self.uploaded_file_objects.append(uploaded_file)
            
            file_info = {"uri": uploaded_file.uri, "name": uploaded_file.display_name}
            print(f"Successfully uploaded and indexed {file_info['name']}. Internal Name: {uploaded_file.name}")
            return file_info
        except Exception as e:
            print(f"Error uploading file {pdf_path}: {e}")
            return None

    def get_insights_for_selection(self, user_selection: str) -> Dict[str, Any]:
        """
        Generates insights by making a single grounded generation call to Gemini,
        referencing all previously uploaded files.
        """
        if not self.uploaded_file_objects:
            return {"error": "No files have been uploaded to the library yet."}

        print(f"Generating insights for selection: '{user_selection[:50]}...'")
        
        prompt = INSIGHTS_PROMPT_TEMPLATE.format(user_selection=user_selection)

        try:
            # Pass the list of File objects directly to the model
            contents = [prompt] + self.uploaded_file_objects
            response = self.generation_model.generate_content(contents)
            
            print("Successfully received insights from Gemini.")
            
            # *** CHANGE: Parse the JSON response on the backend ***
            try:
                # The response.text will be a valid JSON string
                parsed_response = json.loads(response.text)
                return parsed_response
            except json.JSONDecodeError:
                print("Error: Failed to decode JSON from model response.")
                return {"error": "Invalid JSON format received from the model."}

        except Exception as e:
            print(f"An error occurred during insight generation: {e}")
            return {"error": "An internal error occurred while calling the Gemini API."}
