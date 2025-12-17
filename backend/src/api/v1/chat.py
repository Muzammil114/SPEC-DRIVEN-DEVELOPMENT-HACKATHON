from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from backend.src.services.rag_service import rag_service

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    confidence: float
    session_id: str

@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that uses RAG to answer questions based on textbook content.
    """
    try:
        # Get answer with retrieval from textbook content
        result = rag_service.get_answer_with_retrieval(request.message)

        # Generate a simple session ID if not provided (in a real implementation, you'd have proper session management)
        session_id = request.session_id or "session_" + str(hash(request.message))[:8]

        return ChatResponse(
            response=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@router.post("/validate-answer")
def validate_answer_endpoint(request: ChatRequest):
    """
    Endpoint to validate if an answer is grounded in textbook content.
    """
    try:
        # For now, this just returns the same result as chat, but in a real implementation
        # this would include additional validation logic
        result = rag_service.get_answer_with_retrieval(request.message)

        return {
            "is_valid": True,  # In a real implementation, this would be determined by validation logic
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating answer: {str(e)}")