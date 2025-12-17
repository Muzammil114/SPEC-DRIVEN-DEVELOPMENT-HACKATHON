from typing import List, Dict, Any
from openai import OpenAI
from backend.src.services.content_service import ContentService
from backend.src.config import settings

class RAGService:
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

    def get_answer(self, query: str, context_texts: List[str]) -> str:
        """
        Get an answer to the query based on the provided context texts.
        This ensures answers are grounded only in textbook content.
        """
        # Combine context texts into a single context
        context = "\n\n".join(context_texts)

        # Create a prompt that explicitly asks to use only the provided context
        prompt = f"""
        Answer the question based ONLY on the provided context. Do not use any external knowledge.

        Context:
        {context}

        Question: {query}

        Answer (based only on the context provided above):
        """

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI tutor for a Physical AI & Humanoid Robotics textbook. Answer questions based only on the provided context. If the context doesn't contain the information needed to answer, say 'I cannot answer this question based only on the textbook content provided.'"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )

        return response.choices[0].message.content

    def get_answer_with_retrieval(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Get an answer by first retrieving relevant content and then generating a response.
        """
        # Search for relevant content
        search_results = ContentService.search_content(query, limit=top_k)

        if not search_results:
            return {
                "answer": "I cannot answer this question based only on the textbook content provided.",
                "sources": [],
                "confidence": 0.0
            }

        # Extract text from search results
        context_texts = [result["text"] for result in search_results]

        # Generate answer based on retrieved context
        answer = self.get_answer(query, context_texts)

        # Prepare sources with metadata
        sources = [
            {
                "id": result["id"],
                "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],  # Truncate for display
                "metadata": result["metadata"],
                "relevance_score": result["score"]
            }
            for result in search_results
        ]

        # Calculate confidence based on relevance scores
        avg_score = sum(result["score"] for result in search_results) / len(search_results)
        confidence = min(avg_score, 1.0)  # Normalize to 0-1 range

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence
        }

# Global instance
rag_service = RAGService()