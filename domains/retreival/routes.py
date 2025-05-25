import asyncio
import streamlit as st

from typing import List, Optional, Tuple, Any
from fastapi import WebSocket, HTTPException, status
from loguru import logger


from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from domains import retreival
from domains.retreival.rag_util import send_message_over_websocket
from domains.retreival.utils import (
    get_chat_model_with_streaming,
)
from domains.retreival.utils import transform_user_query_for_retrieval
from domains.vector_db.pinecone_utils import get_related_docs_without_context
from domains.retreival.initialize_memory import initialise_memory_from_chat_context
from domains.settings import config_settings
from domains.retreival.models import RagUseCase, RAGGenerationResponse, Message
from domains.retreival.prompts import (
    PROMPT_PREFIX_QNA,
    PROMPT_SUFFIX,
    initialise_doc_search_prompt_template,
)


class RAGError(Exception):
    """Base exception for RAG-related errors"""
    pass


class WebSocketConnectionError(RAGError):
    """Raised when WebSocket communication fails"""
    pass


class DocumentRetrievalError(RAGError):
    """Raised when document retrieval fails"""
    pass


class RAGError(Exception):
    """Base exception for RAG-related errors"""
    pass


class WebSocketConnectionError(RAGError):
    """Raised when WebSocket communication fails"""
    pass


class DocumentRetrievalError(RAGError):
    """Raised when document retrieval fails"""
    pass


async def run_rag(
        question: str,
        language: str,
        chat_context: Optional[List[Message]] = None,
        websocket: Optional[WebSocket] = None,
        namespace: Optional[str] = None,
) -> RAGGenerationResponse:
    """
    Main RAG pipeline function.

    Args:
        question: User's question
        language: Response language
        chat_context: Previous chat history
        websocket: WebSocket connection for streaming
        namespace: Pinecone namespace

    Returns:
        RAGGenerationResponse object

    Raises:
        RAGError: If the RAG pipeline fails
    """
    try:
        # Configuration
        minimum_score = config_settings.MINIMUM_SCORE or 0.8
        namespace = namespace or config_settings.PINECONE_DEFAULT_DEV_NAMESPACE

        # Initialize components
        prompt_qna = initialise_doc_search_prompt_template(
            PROMPT_PREFIX_QNA,
            PROMPT_SUFFIX
        )
        memory = initialise_memory_from_chat_context(chat_context or [])

        return await rag_with_streaming(
            websocket=websocket,
            language=language,
            question=question,
            minimum_score=minimum_score,
            prompt_template_ask_question=prompt_qna,
            memory=memory,
            namespace=namespace,
        )
    except Exception as e:
        logger.exception("RAG pipeline failed")
        raise RAGError(f"RAG pipeline failed: {str(e)}")


async def rag_with_streaming(
        websocket: Optional[WebSocket],
        question: str,
        language: str,
        minimum_score: float,
        prompt_template_ask_question: PromptTemplate,
        memory: Any,
        namespace: str,
        use_case: RagUseCase = RagUseCase.DEFAULT,
        citations_count: int = None,
) -> RAGGenerationResponse:
    """
    RAG pipeline with streaming support.
    """
    citations_count = citations_count or config_settings.PINECONE_TOTAL_DOCS_TO_RETRIEVE
    index_name = config_settings.PINECONE_INDEX_NAME

    try:
        if websocket:
            await send_message_over_websocket(
                websocket, "", retreival.MESSAGE_TYPE_START
            )

        # Get optimized retrieval query
        retreival_query = await transform_user_query_for_retrieval(
            question,
            "OPTIMIZED_QUESTION_MODEL",
            memory.buffer_as_str,
        )

        st.markdown(f"**Transformed user query for vector search {retreival_query}**")

        if not retreival_query or retreival_query == "None":
            logger.warning("Empty retrieval query")
            retreival_query = question

        # Retrieve related documents
        related_docs = await get_related_docs_without_context(
            index_name=index_name,
            namespace=namespace,
            question=retreival_query,
        )

        logger.debug(f"Retrieved {len(related_docs)} documents")

        # Generate response
        response = await generator_routing(
            memory=memory,
            language=language,
            optimised_question=retreival_query,
            prompt_template_ask_question=prompt_template_ask_question,
            websocket=websocket,
            route=RagUseCase.DEFAULT,
            citations_count=citations_count,
            minimum_score=minimum_score,
            related_docs_with_score=related_docs,
        )

        if websocket:
            await send_message_over_websocket(
                websocket, "", retreival.MESSAGE_TYPE_END
            )

        return response

    except Exception as e:
        logger.exception("Streaming RAG pipeline failed")
        if websocket:
            await send_message_over_websocket(
                websocket, str(e), retreival.MESSAGE_TYPE_ERROR
            )
        raise RAGError(f"Streaming RAG failed: {str(e)}")


async def generator_routing(
        memory: Any,
        language: str,
        optimised_question: str,
        prompt_template_ask_question: PromptTemplate,
        websocket: Optional[WebSocket],
        route: RagUseCase,
        citations_count: int,
        minimum_score: float,
        related_docs_with_score: List[Tuple[Document, float]],
) -> RAGGenerationResponse:
    """
    Routes the generation request to appropriate handler based on use case.
    """
    if route != RagUseCase.DEFAULT:
        raise ValueError(f"Unsupported route: {route}")

    return await run_doc_retrieval_flow(
        memory=memory,
        optimised_question=optimised_question,
        prompt_template_ask_question=prompt_template_ask_question,
        related_docs_with_score=related_docs_with_score[:citations_count],
        websocket=websocket,
        minimum_score=minimum_score,
        language=language,
    )


async def run_doc_retrieval_flow(
        memory: Any,
        optimised_question: str,
        prompt_template_ask_question: PromptTemplate,
        related_docs_with_score: List[Tuple[Document, float]],
        websocket: Optional[WebSocket],
        minimum_score: float,
        language: str,
) -> RAGGenerationResponse:
    """
    Executes the document retrieval and response generation flow.
    """
    try:
        document_count = len(related_docs_with_score)

        llm = get_chat_model_with_streaming(
            websocket,
        )
        if not llm:
            raise ValueError("Failed to initialize language model")

        llm_chain = prompt_template_ask_question | llm | StrOutputParser()

        response = await llm_chain.ainvoke(
            {
                "question": optimised_question,
                "chat_history": memory.buffer_as_str,
                "doc_count": str(document_count),
                "context": related_docs_with_score,
                "language": language
            }
        )

        return RAGGenerationResponse(answer=response)

    except Exception as e:
        logger.exception("Document retrieval flow failed")
        raise DocumentRetrievalError(f"Document retrieval failed: {str(e)}")
