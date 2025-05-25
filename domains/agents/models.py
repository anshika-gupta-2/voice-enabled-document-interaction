import operator
from typing import Annotated, List, Literal, TypedDict
from langchain_core.documents import Document
from pydantic import BaseModel
from typing import Optional, Any

from domains.settings import config_settings


class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str
    query: Optional[str]
    namespace: Optional[str]
    documents: List[Document]


class SummaryState(TypedDict):
    content: str


class QueryRequest(BaseModel):
    query: str = None,
    namespace: Optional[str] = config_settings.PINECONE_DEFAULT_DEV_NAMESPACE
    thread_id: Optional[str] = None