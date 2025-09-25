# **Document Insight & Engagement System**

This project is a full-stack web application designed to help users gain deeper insights from their personal document libraries. By leveraging AI and machine learning, it transforms a static collection of PDFs into an interactive knowledge base. When a user selects text in one document, the system instantly surfaces related concepts, contradictory viewpoints, and relevant examples from all other documents in their library.

## 🎥 Demo :


https://github.com/user-attachments/assets/551ae6c6-e0e8-4b2f-8514-4a5c110a7338



## **🚀 Getting Started with Docker**

This application is containerized for easy setup and deployment. Ensure you have Docker installed and running on your machine.

### **1\. Build the Docker Image**

Navigate to the root directory of the project and run the following command to build the Docker image. This process will bundle the frontend, backend, and all necessary dependencies.

```
docker build -t my-app .
```
### **2\. Run the Docker Container**

Once the image is built, run the container using the command below. You must replace the placeholder values with your actual API keys and service endpoints.

**ADOBE\_EMBED\_API\_KEY:**  
`VITE_ADOBE_CLIENT_ID="1a27f057ace94416b2fd19b3d35ab3f4"`

The application's frontend will be accessible at http://localhost:8080, and the backend API will be available at http://localhost:8000.

```
docker run -e GOOGLE_API_KEY="Your Gemini API key" -e AZURE_TTS_KEY="Your Azure TTS Key" -e AZURE_TTS_ENDPOINT="Your Azure TTS Endpoint" -e AZURE_TTS_DEPLOYMENT="tts" -e AZURE_TTS_API_VERSION="2025-03-01-preview" -e VITE_ADOBE_CLIENT_ID="1a27f057ace94416b2fd19b3d35ab3f4" -p 8080:8080 -p 8000:8000 my-app
```

Now, you can open your web browser and navigate to http://localhost:8080 to start using the application.

## **✨ Core Features**

* **Intelligent Document Parsing**: Goes beyond simple text extraction by analyzing the structure of a PDF to understand headings, subheadings, and body content. This is powered by a custom-trained machine learning model.  
* **Context-Aware Insight Retrieval**: When you select text, the system performs a hybrid search (semantic similarity \+ metadata filtering) to find the most relevant information across your entire library.  
* **"Podcast Mode" Audio Generation**: Instantly converts the selected text and its generated insights into an audio summary, allowing for on-the-go consumption.  
* **Interactive PDF Viewing**: Integrates the Adobe PDF Embed API for a seamless and feature-rich document reading experience.  
* **Resizable Three-Panel UI**: A clean, user-friendly interface to manage documents, view content, and explore insights simultaneously.

## **🛠️ Tech Stack**

| Category | Technology |
| :---- | :---- |
| **Frontend** | [React](https://reactjs.org/), [Vite](https://vitejs.dev/), [TypeScript](https://www.typescriptlang.org/), [Tailwind CSS](https://tailwindcss.com/), [Adobe PDF Embed API](https://developer.adobe.com/document-services/docs/overview/pdf-embed-api/) |
| **Backend** | [Python 3.12+](https://www.python.org/), [Flask](https://flask.palletsprojects.com/) |
| **AI / ML** | [Sentence Transformers](https://www.sbert.net/), [Scikit-learn](https://scikit-learn.org/), [PyMuPDF](https://pymupdf.readthedocs.io/), LightGBM |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) |
| **LLM & TTS** | Google Gemini-2.5-flash, Azure OpenAI TTS |
| **Containerization** | [Docker](https://www.docker.com/), [Supervisor](http://supervisord.org/) |

## 

## **💡 Our Approach**

Our system addresses the challenge of information overload by creating a deeply contextual and intelligent experience for navigating personal document libraries. The core of our solution is a sophisticated pipeline that understands documents far beyond simple text.

1. **Hybrid-Machine-Learning-Solution-for-PDF-Structure-Extraction(Round 1A)**: When a PDF is uploaded, we don't just extract its text. We employ a custom-trained machine learning model that analyzes the visual and positional properties of every text block—like font size, weight, and location—to classify it as a `heading`, `subheading`, or `body_text`. This allows us to deconstruct the document into a hierarchical structure that retains its original context.  
2. **Document Parsing Phase(Round 1A & 1B)**: Based on the structural classification, the system deconstructs the document into "smart chunks". Unlike simple text splitting, this method groups related content under its correct heading, ensuring that the semantic and structural integrity of the information is preserved before it is indexed.  
3. **Document-Intelligence(Round 1B)**: These smart chunks, now enriched with metadata (e.g., source document, page number, structural role), are converted into vector embeddings using a Sentence Transformer model. They are then stored in a ChromaDB vector database. This method ensures that our knowledge base is not just a flat list of text fragments but a structured, context-aware network of information.  
4. **Advanced Multi-Stage Insight Retrieval**: When a user selects text, we trigger a sophisticated, multi-stage retrieval process designed for both speed and high relevance. This is the core of our insight generation engine:  
   * **Stage 1: Broad Candidate Search**: The user's selected text is first converted into a vector embedding. We then perform an initial, broad similarity search against our ChromaDB vector store to retrieve the top 100 potentially relevant document chunks. This casts a wide net to ensure we don't miss any potential insights.  
   * **Stage 2: High-Precision Re-ranking**: The initial 100 candidates are immediately passed to a Cross-Encoder model. Unlike the first search, a cross-encoder examines the user's selection and each candidate chunk *together*, providing a much more accurate relevance score. This step re-ranks the candidates based on deep contextual understanding, and we select the top 30 most relevant, de-duplicated results.  
   * **Stage 3: Parallel LLM Categorization**: To deliver insights quickly, we don't just send the re-ranked context to the LLM with a single, slow prompt. Instead, we make three **parallel, asynchronous calls** to the Gemini LLM. Each call has a specialized task: one prompt asks the LLM to find **contradictions**, another to find **enhancements** (e.g., detailed examples), and a third to find thematic **connections**. This parallel approach allows the AI to categorize all the information simultaneously, drastically reducing latency.  
5. **Seamless User Experience**: The frontend, built with React and the Adobe PDF Embed API, captures user selections in real-time. A debouncing mechanism ensures that API calls are made efficiently. The categorized insights from the retrieval pipeline are then presented in a dedicated panel, creating a fluid and interactive research environment. For an alternative consumption method, the insights can be converted into an audio "podcast" using Azure's Text-to-Speech service.

