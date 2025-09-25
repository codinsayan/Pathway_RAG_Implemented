
# --- Serve PDFs statically ---
from flask import send_from_directory


import os
import asyncio
import time
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from asgiref.wsgi import WsgiToAsgi
import json

# Import our core application logic
from indexing_pipeline import IndexingPipeline
from retrieval_handler import RetrievalHandler

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
UPLOAD_FOLDER = "pdfs"
MODEL_FILE = "models/heading_classifier_model.joblib"
ENCODER_FILE = "models/label_encoder.joblib"
ALLOWED_EXTENSIONS = {'pdf'}

# --- Flask App Initialization ---
flask_app = Flask(__name__)
flask_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(flask_app.config['UPLOAD_FOLDER'], exist_ok=True)
CORS(flask_app, supports_credentials=True)

# --- Initialize Handlers ---
try:
    retrieval_handler = RetrievalHandler(google_api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"FATAL: Could not initialize RetrievalHandler: {e}")
    retrieval_handler = None

# --- Timing Decorator ---
def time_request(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = await f(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Request to '{request.path}' took {duration:.3f} seconds.")
        return result
    return decorated_function

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API Endpoints ---
@flask_app.route('/upload_batch', methods=['POST'])
@time_request
async def upload_batch():
    if 'files' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    files = request.files.getlist('files')
    saved_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(flask_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            saved_files.append(filepath)
    if not saved_files:
        return jsonify({"error": "No valid PDF files were uploaded."}), 400
    try:
        indexing_pipeline = IndexingPipeline(google_api_key=GOOGLE_API_KEY)
        tasks = [
            indexing_pipeline.process_and_index_pdf_async(
                pdf_path, MODEL_FILE, ENCODER_FILE
            ) for pdf_path in saved_files
        ]
        await asyncio.gather(*tasks)
        return jsonify({
            "message": f"Successfully indexed {len(saved_files)} files.",
            "filenames": [os.path.basename(p) for p in saved_files]
        }), 200
    except Exception as e:
        return jsonify({"error": f"Indexing error: {e}"}), 500

from generate_podcast import generate_podcast
import uuid

# --- Podcast Generation Endpoint ---
@flask_app.route('/generate_podcast', methods=['POST'])
@time_request
async def generate_podcast_endpoint():
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({"error": "Request body must be a JSON array of strings."}), 400
    # Assign voices: even indexes 'fable', odd indexes 'nova'
    conversation = []
    for idx, text in enumerate(data):
        voice = 'fable' if idx % 2 == 0 else 'nova'
        speaker = 'Speaker' if voice == 'fable' else 'Host'
        conversation.append((speaker, text, voice))
    # Generate a unique filename for each request
    output_file = f"podcast_{uuid.uuid4().hex}.mp3"
    output_path = os.path.join("pdfs", output_file)
    try:
        await generate_podcast(conversation, output_file=output_path)
        return jsonify({"audio_path": output_path}), 200
    except Exception as e:
        return jsonify({"error": f"Podcast generation failed: {e}"}), 500

@flask_app.route('/get_retrieved_sections', methods=['POST'])
@time_request
async def get_retrieved_sections():
    """
    FAST ENDPOINT: Performs a lightning-fast single query and rerank for the initial UI.
    """
    if not retrieval_handler:
        return jsonify({"error": "Backend handler not initialized."}), 500
    data = request.get_json()
    if not data or 'selection' not in data:
        return jsonify({"error": "Missing 'selection' key."}), 400
        
    user_selection = data['selection']
    print(f"\nReceived FAST retrieval request for: '{user_selection[:50]}...'")
    
    try:
        # Use the lightning-fast, single-query retrieval method
        reranked_sections = await retrieval_handler.retrieve_fast_async(user_selection)
        return jsonify({"retrieved_sections": reranked_sections})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

@flask_app.route('/get_generated_insights', methods=['POST'])
@time_request
async def get_generated_insights():
    """
    DEEP ENDPOINT: Uses the parallelized method to generate all text insights.
    """
    if not retrieval_handler:
        return jsonify({"error": "Backend handler not initialized."}), 500
    data = request.get_json()
    if not data or 'selection' not in data:
        return jsonify({"error": "Missing 'selection' key."}), 400
        
    user_selection = data['selection']
    print(f"\nReceived DEEP insight request for: '{user_selection[:50]}...'")
    
    try:
        llm_response = await retrieval_handler.generate_initial_insights_async(user_selection)
        return jsonify(llm_response)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

# *** UPDATED: This endpoint now generates all 4 persona podcasts at once ***
@flask_app.route('/get_persona_podcast', methods=['POST'])
@time_request
async def get_persona_podcast():
    """
    PODCAST ENDPOINT: Generates all 4 persona podcasts in a single parallel call.
    """
    if not retrieval_handler:
        return jsonify({"error": "Backend handler not initialized."}), 500
    data = request.get_json()
    # Now only checks for the 'selection' key
    if not data or 'selection' not in data:
        return jsonify({"error": "Missing 'selection' key."}), 400
        
    selection = data['selection']
    print(f"\nReceived parallel podcast generation request for: '{selection[:50]}...'")
    
    try:
        # The handler function now only needs the selection text
        llm_response = await retrieval_handler.generate_persona_podcast_async(selection)
        return jsonify(llm_response)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500
    

# --- List PDFs Endpoint ---
@flask_app.route('/list_pdfs', methods=['GET'])
def list_pdfs():
    try:
        pdf_dir = flask_app.config['UPLOAD_FOLDER']
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        return jsonify({"pdfs": pdf_files}), 200
    except Exception as e:
        return jsonify({"error": f"Could not list PDFs: {e}"}), 500
    


@flask_app.route('/pdfs/<path:filename>')
def serve_pdf(filename):
    pdf_dir = flask_app.config['UPLOAD_FOLDER']
    return send_from_directory(pdf_dir, filename)



# --- Delete PDF file when deleting document ---
@flask_app.route('/delete_document', methods=['POST'])
def delete_document():
    data = request.get_json()
    if not data or 'document_name' not in data:
        return jsonify({"error": "Missing 'document_name' key."}), 400
    document_name = data['document_name']
    pdf_dir = flask_app.config['UPLOAD_FOLDER']
    pdf_path = os.path.join(pdf_dir, document_name)
    try:
        # Remove PDF file if it exists
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        # Optionally: also remove from vector store here if needed
        # ...existing vector store deletion logic...
        return jsonify({"message": f"Deleted {document_name}"}), 200
    except Exception as e:
        return jsonify({"error": f"Could not delete {document_name}: {e}"}), 500

@flask_app.route('/config.js')
def runtime_config():
    """Serves runtime configuration variables to the frontend."""
    config = {
        'VITE_ADOBE_CLIENT_ID': os.environ.get('VITE_ADOBE_CLIENT_ID', '')
    }
    # Return as a JS file that creates a global object on the window
    js_payload = f"window.runtimeConfig = {json.dumps(config)};"
    print(js_payload)
    return js_payload, 200, {'Content-Type': 'application/javascript'}

# --- ASGI Wrapper ---
app = WsgiToAsgi(flask_app)