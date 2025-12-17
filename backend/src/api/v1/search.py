from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

from backend.src.services.content_service import ContentService

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]

@router.post("/search")
def search_endpoint(request: SearchRequest):
    """
    Search endpoint that uses vector similarity to find relevant textbook content.
    """
    try:
        # Perform semantic search using the content service
        results = ContentService.search_content(request.query, request.limit)

        return SearchResponse(results=results)
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")

@router.get("/search/health")
def search_health():
    """
    Health check for the search functionality.
    """
    return {"status": "search service is operational"}