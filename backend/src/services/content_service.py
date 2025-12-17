from typing import List, Optional
from sqlalchemy.orm import Session
from backend.src.models.content import Content
from backend.src.services.embedding_service import embedding_service

class ContentService:
    @staticmethod
    def get_content(db: Session, content_id: int) -> Optional[Content]:
        return db.query(Content).filter(Content.id == content_id).first()

    @staticmethod
    def get_contents(db: Session, skip: int = 0, limit: int = 100) -> List[Content]:
        return db.query(Content).offset(skip).limit(limit).all()

    @staticmethod
    def create_content(db: Session, title: str, content: str, module_id: int, section: str) -> Content:
        db_content = Content(
            title=title,
            content=content,
            module_id=module_id,
            section=section
        )
        db.add(db_content)
        db.commit()
        db.refresh(db_content)

        # Store content in vector database for RAG
        embedding_service.store_content(
            content_id=str(db_content.id),
            text=content,
            metadata={
                "title": title,
                "module_id": module_id,
                "section": section
            }
        )

        return db_content

    @staticmethod
    def update_content(db: Session, content_id: int, title: str = None, content: str = None,
                      module_id: int = None, section: str = None) -> Optional[Content]:
        db_content = ContentService.get_content(db, content_id)
        if not db_content:
            return None

        if title is not None:
            db_content.title = title
        if content is not None:
            db_content.content = content
        if module_id is not None:
            db_content.module_id = module_id
        if section is not None:
            db_content.section = section

        db.commit()
        db.refresh(db_content)

        # Update content in vector database
        embedding_service.store_content(
            content_id=str(db_content.id),
            text=db_content.content,
            metadata={
                "title": db_content.title,
                "module_id": db_content.module_id,
                "section": db_content.section
            }
        )

        return db_content

    @staticmethod
    def delete_content(db: Session, content_id: int) -> bool:
        db_content = ContentService.get_content(db, content_id)
        if not db_content:
            return False

        db.delete(db_content)
        db.commit()

        # Delete from vector database
        embedding_service.delete_content(str(content_id))

        return True

    @staticmethod
    def search_content(query: str, limit: int = 5) -> List[dict]:
        """Search for content using vector similarity"""
        return embedding_service.search_content(query, limit)