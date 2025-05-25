from pydantic import BaseModel
from enum import Enum


class RequestStatusEnum(str, Enum):
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ApiNameEnum(str, Enum):
    INJEST_DOC = "injest-doc"

class RequestStatus(BaseModel):
    request_id: int
    status: RequestStatusEnum
    api_name: ApiNameEnum
    data_json: dict = None
    error_detail: str = None