import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from utils.pdf_loader import extract_text_from_pdf
from utils.text_loader import extract_text_from_txt
from utils.web_loader import extract_text_from_url
from utils.text_splitter import split_text
from utils.vector_store import vector_store
from utils.gemini_llm import gemini_manager

app = FastAPI(title="PDF RAG Chatbot")

@app.on_event("startup")
async def startup_event():
    if vector_store.load():
        print("Existing vector store loaded from disk.")
    else:
        print("No existing vector store found. Waiting for upload.")

# Setup templates and static (if any)
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    question: str
    history: Optional[list[dict]] = []

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return open("templates/index.html", encoding="utf-8").read()

from fastapi.responses import StreamingResponse
import json
import asyncio

@app.get("/active_files")
async def get_active_files():
    """Returns a list of currently indexed files."""
    return list(vector_store.files_metadata.keys())

@app.post("/delete")
async def delete_file(filename: str):
    """Deletes a file from the knowledge base."""
    if vector_store.remove_file(filename):
        vector_store.save()
        return {"message": f"File {filename} deleted successfully."}
    raise HTTPException(status_code=404, detail="File not found in knowledge base.")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are allowed.")
    
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    async def generate_progress():
        try:
            # Step 1: Loading
            yield f"data: {json.dumps({'step': 'loading', 'status': 'active'})}\n\n"
            await asyncio.sleep(0.5)
            yield f"data: {json.dumps({'step': 'loading', 'status': 'completed'})}\n\n"
            
            # Step 2: Analyzing Semantics
            yield f"data: {json.dumps({'step': 'extracting', 'status': 'active'})}\n\n"
            
            text = ""
            if ext == ".pdf":
                text = extract_text_from_pdf(file_path)
            elif ext == ".txt":
                text = extract_text_from_txt(file_path)
                
            if not text:
                yield f"data: {json.dumps({'error': 'Readable content not detected.'})}\n\n"
                return
            await asyncio.sleep(0.5)
            yield f"data: {json.dumps({'step': 'extracting', 'status': 'completed'})}\n\n"
            
            # Step 3: Atomic Partitioning
            yield f"data: {json.dumps({'step': 'chunking', 'status': 'active'})}\n\n"
            chunks = split_text(text)
            await asyncio.sleep(0.5)
            yield f"data: {json.dumps({'step': 'chunking', 'status': 'completed'})}\n\n"
            
            # Step 4: Vector Embedding
            yield f"data: {json.dumps({'step': 'embedding', 'status': 'active'})}\n\n"
            vector_store.add_file(file.filename, chunks)
            await asyncio.sleep(0.5)
            yield f"data: {json.dumps({'step': 'embedding', 'status': 'completed'})}\n\n"
            
            # Step 5: RAG Ready
            yield f"data: {json.dumps({'step': 'rag', 'status': 'active'})}\n\n"
            vector_store.save()
            await asyncio.sleep(0.5)
            yield f"data: {json.dumps({'step': 'rag', 'status': 'completed'})}\n\n"
            
            yield f"data: {json.dumps({'message': 'Process complete', 'filename': file.filename})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return StreamingResponse(generate_progress(), media_type="text/event-stream")

class UrlRequest(BaseModel):
    url: str

@app.post("/add_url")
async def add_url(request: UrlRequest):
    """endpoint to ingest web content"""
    text = extract_text_from_url(request.url)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from URL.")
        
    # Process the text similarly to upload (but synchronous for now or simple background)
    # For simplicity, we'll do it synchronously but split it
    chunks = split_text(text)
    filename = request.url # Use URL as filename
    vector_store.add_file(filename, chunks)
    vector_store.save()
    
    return {"message": "URL content added successfully.", "filename": filename}

from utils.langgraph_workflow import run_rag_workflow

@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        response_data = run_rag_workflow(request.question, chat_history=request.history)
        
        return {
            "answer": response_data.get("answer"),
            "sources": response_data.get("sources", []),
            "confidence": response_data.get("confidence"),
            "node_path": response_data.get("node_path")
        }
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
