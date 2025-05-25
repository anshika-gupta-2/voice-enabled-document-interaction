from enum import Enum
from pydantic import BaseModel


class RagUseCase(str, Enum):
    DEFAULT = "default"
    RESUME_FINDER = "resume_finder"
    DOC_RETRIEVAL = "doc_retrieval"
    REASON = "reason"


class RAGGenerationResponse(BaseModel):
    answer: str = ""

class Message(BaseModel):
    type: str
    content: str
