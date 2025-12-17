from fastapi import APIRouter, Depends, HTTPException
from typing import List
from sqlalchemy.orm import Session

from backend.src.models.content import Content
from backend.src.api.deps import get_db

router = APIRouter()

@router.get("/", response_model=List[Content])
def get_contents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    contents = db.query(Content).offset(skip).limit(limit).all()
    return contents

@router.get("/{content_id}", response_model=Content)
def get_content(content_id: int, db: Session = Depends(get_db)):
    content = db.query(Content).filter(Content.id == content_id).first()
    if content is None:
        raise HTTPException(status_code=404, detail="Content not found")
    return content

@router.post("/", response_model=Content)
def create_content(content: Content, db: Session = Depends(get_db)):
    db_content = Content(**content.dict())
    db.add(db_content)
    db.commit()
    db.refresh(db_content)
    return db_content