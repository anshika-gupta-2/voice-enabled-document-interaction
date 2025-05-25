import os

from enum import Enum
from pydantic_settings import BaseSettings
from typing import ClassVar


class LLMServiceType(str, Enum):
    OPENAI = "openai"
    GROQ = "groq"
    GEMINI = "gemini"


class VectorDBType(str, Enum):
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


class VectorDBServiceType(str, Enum):
    LOCAL = "local"
    ONLINE = "online"


class Settings(BaseSettings):
    # openai
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

    OPENAI_CHAT_BASE_URL: str = os.environ.get(
        "OPENAI_CHAT_BASE_URL", "https://api.openai.com/v1/chat/completions"
    )
    THRESHOLD_MESSAGE_TO_SUMMARIZE: int = int(
        os.environ.get("THRESHOLD_MESSAGE_TO_SUMMARIZE", 10)
    )
    API_HOSTNAME: str = os.environ.get("API_HOSTNAME", "https://dummyjson.com/c")

    MAX_TOKEN_LIMIT: int = os.environ.get("MAX_TOKEN_LIMIT", 1500)

    # weaviate
    WEAVIATE_VECTOR_DATABASE_SERVICE_TYPE: str = os.environ.get(
        "WEAVIATE_VECTOR_DATABASE_SERVICE_TYPE", VectorDBServiceType.LOCAL.value
    )

    WEAVIATE_API_KEY: str = os.environ.get("WEAVIATE_API_KEY", "")
    WEAVIATE_CLUSTER_URL: str = os.environ.get("WEAVIATE_CLUSTER_URL", "")

    WEAVIATE_AUTH_CREDENTIALS: str = os.environ.get("WEAVIATE_AUTH_CREDENTIALS", "")
    WEAVIATE_HOST: str = os.environ.get("WEAVIATE_HOST", "localhost")
    WEAVIATE_HOST_PORT: int = int(os.environ.get("WEAVIATE_HOST_PORT", 8080))
    WEAVIATE_GRPC_PORT: int = int(os.environ.get("WEAVIATE_GRPC_PORT", 50051))

    WEAVIATE_HYPERPARAMETER_HYBRID_SEARCH: float = float(
        os.environ.get("WEAVIATE_HYPERPARAMETER_HYBRID_SEARCH", 0.5)
    )

    WEAVIATE_MULTI_TENANCY_STATUS: bool = os.environ.get(
        "WEAVIATE_MULTI_TENANCY_STATUS", True
    )
    WEAVIATE_TEXT_KEY: str = os.environ.get("WEAVIATE_TEXT_KEY", "text")
    WEAVIATE_FILTER_RESULTS_PARAMETER: str = os.environ.get(
        "WEAVIATE_FILTER_RESULTS_PARAMETER", "file_name"
    )


    # pinecone
    PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")
    PINECONE_ENV: str = os.environ.get("PINECONE_ENV", "")
    PINECONE_INDEX_NAME: str = os.environ.get("PINECONE_INDEX", "voicechat")

    PINECONE_INDEX_METRIC_TYPE: str = os.environ.get("PINECONE_METRIC_TYPE", "cosine")
    PINECONE_INDEX_CLOUD_NAME: str = os.environ.get("PINECONE_INDEX_CLOUD_NAME", "aws")
    PINECONE_INDEX_REGION_NAME: str = os.environ.get("PINECONE_INDEX_REGION_NAME", "us-east-1")
    PINECONE_DEFAULT_DEV_NAMESPACE: str = os.environ.get("PINECONE_DEFAULT_DEV_NAMESPACE", "default_dev")
    PINECONE_DROP_INDEX_NAME_STATUS: bool = os.environ.get("PINECONE_DROP_INDEX_NAME_STATUS", False)
    DELETE_NAMESPACE_STATUS: bool = os.environ.get("DELETE_NAMESPACE_STATUS", True)
    PINECONE_TOTAL_DOCS_TO_RETRIEVE: int = os.environ.get(
        "PINECONE_TOTAL_DOCS_TO_RETRIEVE", 10
    )

    # chunk setting
    NUMBER_OF_RETRIEVAL_RESULTS: int = os.environ.get(
        "NUMBER_OF_RETRIEVAL_RESULTS", 10
    )
    CHUNK_SIZE: int = os.environ.get("CHUNK_SIZE", 1000)
    CHUNK_OVERLAP: int = os.environ.get("CHUNK_OVERLAP", 200)

    # classification
    CLASSIFICATION_MODEL: str = os.environ.get("CLASSIFICATION_MODEL", "gpt-4o")

    # conversion memory key
    CONVERSATIONAL_BUFFER_WINDOW_MEMORY_KEY: str = os.environ.get(
        "CONVERSATIONAL_BUFFER_WINDOW_MEMORY_KEY", "chat_history"
    )
    LANGCHAIN_MEMORY_BUFFER_WINDOW: int = os.environ.get(
        "LANGCHAIN_MEMORY_BUFFER_WINDOW", 10
    )
    CONVERSATIONAL_BUFFER_WINDOW_INPUT_KEY: str = os.environ.get(
        "CONVERSATIONAL_BUFFER_WINDOW_INPUT_KEY", "question"
    )
    CHAT_CONTEXT_HUMAN_MESSAGE_KEY: str = os.environ.get(
        "CHAT_CONTEXT_HUMAN_MESSAGE_KEY", "human"
    )
    CHAT_CONTEXT_AI_MESSAGE_KEY: str = os.environ.get(
        "CHAT_CONTEXT_AI_MESSAGE_KEY", "ai"
    )

    # optimized question
    OPTIMIZED_QUESTION_MODEL: str = os.environ.get("OPTIMIZED_QUESTION_MODEL", "gpt-4o-mini")
    MINIMUM_SCORE: float = float(os.environ.get("MINIMUM_SCORE", 0.5))

    UPLOAD_FOLDER: str = os.environ.get("UPLOAD_FOLDER", "uploads")
    LOGS_FOLDER: str = os.environ.get("LOGS_FOLDER", "logs")

    VECTOR_DATABASE_TO_USE: str = os.environ.get(
        "VECTOR_DATABASE_TO_USE",
        VectorDBType.WEAVIATE.value,
    )

    LLM_SERVICE: str = os.environ.get(
        "LLM_SERVICE", LLMServiceType.OPENAI.value
    )

    MAX_TOKENS: int = os.environ.get("MAX_TOKENS", 1500)

    # Modular Model Names
    LLMS: ClassVar[dict] = {
        "CHAT_MODEL_NAME": os.environ.get("OPENAI_CHAT_MODEL_NAME", "gpt-4o-mini"),
        "SUMMARIZE_LLM_MODEL": os.environ.get("SUMMARIZE_LLM_MODEL", "gpt-4o-mini"),
        "EMBEDDING_MODEL_NAME": os.environ.get(
            "EMBEDDING_MODEL_NAME", "text-embedding-3-small"
        ),
        "CLASSIFICATION_MODEL": os.environ.get("CLASSIFICATION_MODEL", "gpt-4o-mini"),
        "OPTIMIZED_QUESTION_MODEL": os.environ.get("OPTIMIZED_QUESTION_MODEL", "gpt-4o"),
        "CHAT_STREAMING_MODEL": os.environ.get("CHAT_STREAMING_MODEL", "gpt-4o-mini"),
    }

    GROQ_MODEL_SETTINGS: ClassVar[dict] = {
        "CHAT_MODEL_NAME": os.environ.get("CHAT_MODEL_NAME", "gpt-4o-mini"),
        "SUMMARIZE_LLM_MODEL": os.environ.get("SUMMARIZE_LLM_MODEL", "gpt-4o-mini"),
        "EMBEDDING_MODEL_NAME": os.environ.get(
            "EMBEDDING_MODEL_NAME", "text-embedding-3-small"
        ),
        "CLASSIFICATION_MODEL": os.environ.get("CLASSIFICATION_MODEL", "gpt-4o-mini"),
        "OPTIMIZED_QUESTION_MODEL": os.environ.get("OPTIMIZED_QUESTION_MODEL", "gpt-4o"),
        "CHAT_STREAMING_MODEL": os.environ.get("CHAT_STREAMING_MODEL", "gpt-4o-mini"),
    }

    GOOGLE_MODEL_SETTINGS: ClassVar[dict] = {
        "CHAT_MODEL_NAME": os.environ.get("CHAT_MODEL_NAME", "gemini-1.5-pro"),
        "SUMMARIZE_LLM_MODEL": os.environ.get("SUMMARIZE_LLM_MODEL", "gemini-1.5-pro"),
        "EMBEDDING_MODEL_NAME": os.environ.get(
            "EMBEDDING_MODEL_NAME", "text-embedding-3-small"
        ),
        "CLASSIFICATION_MODEL": os.environ.get("CLASSIFICATION_MODEL", "gemini-1.5-pro"),
        "OPTIMIZED_QUESTION_MODEL": os.environ.get("OPTIMIZED_QUESTION_MODEL", "gemini-1.5-pro"),
        "CHAT_STREAMING_MODEL": os.environ.get("CHAT_STREAMING_MODEL", "gemini-1.5-pro"),
    }

config_settings = Settings()
