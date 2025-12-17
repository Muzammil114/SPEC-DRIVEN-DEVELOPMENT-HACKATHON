import asyncio
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from backend.src.config import settings

class EmbeddingService:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.collection_name = "textbook_content"

        # Initialize the collection if it doesn't exist
        self._init_collection()

    def _init_collection(self):
        """Initialize the Qdrant collection for textbook content"""
        try:
            self.qdrant_client.get_collection(self.collection_name)
        except:
            # Collection doesn't exist, create it
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            )

    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for the given text using OpenAI"""
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def store_content(self, content_id: str, text: str, metadata: Dict[str, Any] = None):
        """Store content with its embedding in Qdrant"""
        embedding = self.create_embedding(text)

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=content_id,
                    vector=embedding,
                    payload={
                        "text": text,
                        "metadata": metadata or {}
                    }
                )
            ]
        )

    def search_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for content similar to the query"""
        query_embedding = self.create_embedding(query)

        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        return [
            {
                "id": result.id,
                "text": result.payload.get("text", ""),
                "metadata": result.payload.get("metadata", {}),
                "score": result.score
            }
            for result in results
        ]

    def delete_content(self, content_id: str):
        """Delete content from the collection"""
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[content_id]
            )
        )

# Global instance
embedding_service = EmbeddingService()