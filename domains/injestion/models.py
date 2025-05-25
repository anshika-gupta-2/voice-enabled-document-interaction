from enum import Enum
from typing import Any, Literal, Optional, List, TypedDict, Dict
from pydantic import BaseModel
from domains.settings import config_settings
from domains.models import RequestStatus


FILE_TYPE = [
    "pdf",
    "txt",
    "docx"
]

class FileInjestionResponseDto(RequestStatus):
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    original_file_name: Optional[str] = None
    total_pages: Optional[int] = None
    error_detail: Optional[str] = None

class StatusRequestDto(BaseModel):
    request_id: int

class InjestRequestDto(StatusRequestDto):
    pre_signed_url: str
    file_name: str
    original_file_name: str
    file_type: str
    process_type: str
    namespace: Optional[str] = config_settings.PINECONE_DEFAULT_DEV_NAMESPACE