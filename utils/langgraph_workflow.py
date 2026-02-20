import os
from typing import TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, START, END
from utils.vector_store import vector_store
from utils.gemini_llm import gemini_manager

# Define the state schema
class AgentState(TypedDict):
    question: str
    chat_history: List[dict]
    intent: Optional[str]
    context_chunks: List[str]
    confidence_score: float
    raw_answer: Optional[str]
    found: bool
    response: Optional[dict]
    node_path: List[str]

# --- Nodes ---

def input_processing_node(state: AgentState) -> AgentState:
    """Prepares the state for the workflow."""
    return {
        **state,
        "node_path": state.get("node_path", []) + ["Input Processing"],
        "context_chunks": [],
        "confidence_score": 0.0,
        "found": False
    }

def intent_routing_node(state: AgentState) -> AgentState:
    """Classifies the user intent."""
    intent = gemini_manager.classify_intent(state["question"])
    return {
        **state,
        "intent": intent,
        "node_path": state["node_path"] + [f"Intent Routing ({intent})"]
    }

def retrieval_node(state: AgentState) -> AgentState:
    """Retrieves context from the vector store."""
    chunks, distances = vector_store.search_with_scores(state["question"], k=10)
    
    # Scores are now Cosine Similarity (0 to 1), where 1.0 is perfect match.
    # We can use the average score directly as confidence.
    avg_score = sum(distances) / len(distances) if distances else 0.0
    confidence = max(0.0, min(1.0, avg_score)) 
    
    return {
        **state,
        "context_chunks": chunks,
        "confidence_score": confidence,
        "node_path": state["node_path"] + ["Retrieval"]
    }

def context_validation_node(state: AgentState) -> AgentState:
    """Validates if the retrieved context is sufficient."""
    # This is primarily a logic gate for the conditional edge, 
    # but we record the visit here.
    return {
        **state,
        "node_path": state["node_path"] + ["Context Validation"]
    }

def response_generation_node(state: AgentState) -> AgentState:
    """Generates a response using Gemini."""
    intent = state.get("intent", "document_qa")
    history_text = ""
    if state["chat_history"]:
        history_text = "PREVIOUS CONVERSATION:\n"
        for msg in state["chat_history"][-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "\n"

    if intent == "document_qa" and state["context_chunks"]:
        context_text = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(state["context_chunks"])])
        prompt = f"""
        {history_text}
        Answer the user's latest question based EXCLUSIVELY on the provided context.
        CONTEXT:
        {context_text}
        
        USER LATEST QUESTION:
        {state["question"]}
        """
        system_instruction = "You are an intelligent PDF analyst. Respond based on the provided context."
    else:
        prompt = f"{history_text}USER LATEST QUESTION: {state['question']}"
        system_instruction = "You are a helpful AI Assistant. Reference previous messages if needed."

    response_data = gemini_manager.generate_with_prompt(prompt, system_instruction)
    
    return {
        **state,
        "raw_answer": response_data.get("answer"),
        "found": response_data.get("found", True),
        "node_path": state["node_path"] + ["Response Generation"]
    }

def fallback_node(state: AgentState) -> AgentState:
    """Handles low-confidence or no-context scenarios."""
    if not state["context_chunks"]:
        answer = "I couldn't find any documents in the knowledge base related to your question. Would you like me to answer based on general knowledge?"
    else:
        answer = "I found some information, but I'm not very confident it directly answers your question. Here is what I found: \n\n" + state["raw_answer"] if state.get("raw_answer") else "I'm not sure I can answer that accurately from the current documents."
        
    return {
        **state,
        "raw_answer": answer,
        "found": False,
        "node_path": state["node_path"] + ["Fallback"]
    }

def response_formatter_node(state: AgentState) -> AgentState:
    """Formats the final response object."""
    # Add a suffix about the workflow to the answer if desired, 
    # or keep it separate for the UI to handle.
    return {
        **state,
        "response": {
            "answer": state["raw_answer"],
            "found": state["found"],
            "confidence": round(state["confidence_score"] * 100, 1),
            "node_path": state["node_path"],
            "sources": state["context_chunks"]
        },
        "node_path": state["node_path"] + ["Response Formatter"]
    }

# --- Conditional Edges ---

def route_by_intent(state: AgentState) -> Literal["retrieval", "generate"]:
    if state["intent"] == "document_qa":
        return "retrieval"
    return "generate"

def validate_context(state: AgentState) -> Literal["generate", "fallback"]:
    # Confidence threshold of 30% or at least some chunks
    if state["confidence_score"] > 0.3 or (state["intent"] == "document_qa" and state["context_chunks"]):
        return "generate"
    return "fallback"

# --- Graph Assembly ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("input_processing", input_processing_node)
workflow.add_node("intent_routing", intent_routing_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("context_validation", context_validation_node)
workflow.add_node("response_generation", response_generation_node)
workflow.add_node("fallback", fallback_node)
workflow.add_node("response_formatter", response_formatter_node)

# Add Edges
workflow.add_edge(START, "input_processing")
workflow.add_edge("input_processing", "intent_routing")

workflow.add_conditional_edges(
    "intent_routing",
    route_by_intent,
    {
        "retrieval": "retrieval",
        "generate": "response_generation"
    }
)

workflow.add_edge("retrieval", "context_validation")

workflow.add_conditional_edges(
    "context_validation",
    validate_context,
    {
        "generate": "response_generation",
        "fallback": "fallback"
    }
)

workflow.add_edge("response_generation", "response_formatter")
workflow.add_edge("fallback", "response_formatter")
workflow.add_edge("response_formatter", END)

# Compile
app = workflow.compile()

def run_rag_workflow(question: str, chat_history: List[dict] = None) -> dict:
    """Entry point to run the LangGraph workflow."""
    initial_state = {
        "question": question,
        "chat_history": chat_history or [],
        "node_path": []
    }
    final_state = app.invoke(initial_state)
    return final_state["response"]
