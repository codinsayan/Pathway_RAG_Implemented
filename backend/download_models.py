# download_models.py
from sentence_transformers import CrossEncoder

print("Downloading and caching reranker model...")
# This line will download the model to the default cache location
CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
print("Model downloaded successfully.")