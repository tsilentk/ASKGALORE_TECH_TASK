import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

class GeminiManager:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY_PAID") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash"

        # Define the response schema for structured output
        self.response_schema = {
            "type": "OBJECT",
            "properties": {
                "answer": {"type": "STRING"},
                "found": {"type": "BOOLEAN"}
            },
            "required": ["answer", "found"]
        }

    def generate_response(self, question: str, context_chunks: list[str], chat_history: list[dict] = None) -> dict:
        """
        Generates a response from Gemini. 
        Supports chat history for context and RAG for document knowledge.
        """
        history_text = ""
        if chat_history:
            history_text = "PREVIOUS CONVERSATION:\n"
            for msg in chat_history[-6:]: # Keep last 6 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"
            history_text += "\n"

        if context_chunks:
            context_text = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
            prompt = f"""
{history_text}
Answer the user's latest question based EXCLUSIVELY on the provided context. 
The context might contain information from multiple different documents/resumes. 
Look carefully for the name or specific subject the user is asking about.

If the user is asking about a specific person (e.g., Suhail) and you see their information in the context, focus on that.
If you only see information for someone else, state that clearly but do not assume the information applies to both.

CONTEXT:
{context_text}

USER LATEST QUESTION:
{question}
"""
            system_instruction = "You are an intelligent PDF analyst. You can synthesize information from multiple indexed documents and distinguish between different subjects (people/topics)."
        else:
            prompt = f"{history_text}USER LATEST QUESTION: {question}"
            system_instruction = "You are a helpful AI Assistant with memory. Reference previous messages if the user refers to them (e.g., 'it', 'him', 'that')."

        generate_content_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=self.response_schema,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=generate_content_config,
            )
            # The new SDK returns a response object; the text attribute contains the JSON string
            return json.loads(response.text)
        except Exception as e:
            if "quota" in str(e).lower() or "exhausted" in str(e).lower():
                return {
                    "answer": "I'm sorry, but the Gemini API quota has been exhausted. Please wait a minute and try again.",
                    "found": False
                }
            raise e

    def classify_intent(self, question: str) -> str:
        """Classifies the user query into 'document_qa' or 'general_chat'."""
        prompt = f"""
        Classify the following user question into one of two categories:
        1. 'document_qa': If the user is asking about specific information, documents, data, or technical details that would likely be in an uploaded knowledge base.
        2. 'general_chat': If the user is just saying hello, asking about your capabilities, or making general conversation not requiring document retrieval.

        USER QUESTION: {question}

        Respond ONLY with the category name ('document_qa' or 'general_chat').
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            intent = response.text.strip().lower()
            if 'document_qa' in intent: return 'document_qa'
            return 'general_chat'
        except:
            return 'document_qa' # Default to search if unsure

    def generate_with_prompt(self, prompt: str, system_instruction: str = None) -> dict:
        """Low-level generation with a custom prompt and schema."""
        generate_content_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=self.response_schema,
        )
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=generate_content_config,
            )
            return json.loads(response.text)
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "found": False}

gemini_manager = GeminiManager()
