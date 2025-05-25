from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from domains.settings import config_settings
from dataclasses import dataclass


class ConnectionResponseDto(BaseModel):
    """Response DTO for connection status."""
    status: bool = False
    message: Optional[str] = None
    client: Optional[Any] = None
    manager: Optional[Any] = None


class ClientResponseDto(BaseModel):
    """Response DTO for client status."""
    status_code: bool = False
    client: Optional[Any] = None


class PushToDatabaseResponseDto(BaseModel):
    """Response DTO for pushing data to the database."""
    status: bool = False
    message: Optional[str] = None
    document_ids: Optional[List[str]] = None
    timestamp: Optional[str] = None
    index: Optional[str] = None
    namespace: Optional[str] = None


@dataclass
class PineconeConfig:
    index_name: str = config_settings.PINECONE_INDEX_NAME
    namespace: str = config_settings.PINECONE_DEFAULT_DEV_NAMESPACE
    dimension: int = 1536
    metric: str = config_settings.PINECONE_INDEX_METRIC_TYPE
    cloud: str = config_settings.PINECONE_INDEX_CLOUD_NAME
    region: str = config_settings.PINECONE_INDEX_REGION_NAME
