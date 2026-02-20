# AGRID AI - LangGraph Orchestrated RAG Chatbot

An advanced Retrieval-Augmented Generation (RAG) chatbot powered by **Gemini 2.0 Flash** and orchestrated using **LangGraph**. This system processes PDFs, text files, and web URLs to provide grounded, context-aware answers with visible confidence scores and decision paths.

## 🚀 Key Features

- **Multi-Source Knowledge Base**:
  - 📄 **PDF Support**: Extract text from PDF documents.
  - 📝 **TXT Support**: Ingest simple text files.
  - 🌐 **Web Scraping**: Add URLs to scrape and index web content automatically.
- **LangGraph Orchestration**:
  - A robust 7-node workflow (Input -> Intent -> Retrieval -> Validation -> Generation -> Fallback -> Formatter).
  - **Dynamic Routing**: Intelligently switches between "Document QA" and "General Chat" based on user intent.
- **Transparency & Metrics**:
  - **Confidence Score**: Displays a percentage (0-100%) indicating how well the retrieved context matches the query (using Cosine Similarity).
  - **Node Path**: Visualizes the exact steps the AI took to generate the response (e.g., `Retrieval → Context Validation → Response Generation`).
- **Vector Search**: Uses **FAISS** with **SentenceTransformers** (`all-MiniLM-L6-v2`) for efficient similarity search.
- **Modern UI**: A clean, responsive chat interface with file management, history, and export capabilities.

## 🛠️ Prerequisites

- **Python 3.10+**
- **Gemini API Key**: You need a valid API key from [Google AI Studio](https://aistudio.google.com/).

## 📥 Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/tsilentk/ASKGALORE_TECH_TASK.git
    cd agrid-ai
    ```

2.  **Create a Virtual Environment (Recommended)**

    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a `.env` file in the `utils/` directory (or root, depending on your setup) containing your API key:
    ```env
    GEMINI_API_KEY=your_actual_api_key_here
    ```

## ▶️ Running the Application

1.  **Start the Server**

    ```bash
    python main.py
    ```

    The server will start at `http://localhost:8000`.

2.  **Access the UI**
    Open your browser and navigate to `http://localhost:8000`.

## 📖 Usage Guide

- **Upload Documents**: Click the "Upload" button or drag and drop `.pdf` or `.txt` files into the knowledge base area.
- **Add URLs**: Enter a website URL in the "Add Web Content" field and click the **+** button to index it.
- **Chat**: Type your query in the chat box.
  - The bot will automatically decide whether to search your documents or answer from general knowledge.
  - Check the **Confidence Score** and **Node Path** below each response to understand the AI's reasoning.
- **Manage Files**: View indexed files in the sidebar. Click the trash icon to remove them from the knowledge base (triggers an automatic index rebuild).

## 🧩 Project Structure

- `main.py`: FastAPI application entry point.
- `utils/langgraph_workflow.py`: The core workflow engine defining nodes and edges.
- `utils/vector_store.py`: Manages FAISS index and embeddings.
- `utils/gemini_llm.py`: Handles interaction with Google's Gemini API.
- `templates/index.html`: The frontend user interface.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for new features or improvements!

