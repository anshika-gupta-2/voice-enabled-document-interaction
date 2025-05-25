import asyncio
import pprint

import weaviate
from domains.settings import config_settings
from langchain_core.documents import Document
from langchain_community.vectorstores import Pinecone
from typing import Tuple, List, Optional, Union
from loguru import logger

from contextlib import asynccontextmanager
from functools import lru_cache

from weaviate.classes.query import Filter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_pinecone import PineconeVectorStore

from domains.vector_db.weaviate_utils import manager_client
from domains.injestion.utils import get_embeddings


@lru_cache(maxsize=32)
def load_index(
        index_name: str,
        namespace: str | None = None
) -> Union[Pinecone, WeaviateVectorStore]:

    """
    A function to load a vectorstore index by name and namespace.

    This function will load a vectorstore index based on the name and namespace provided.
    It will use the correct vectorstore type based on the setting for VECTOR_DATABASE_TO_USE.
    If the setting is "weaviate", it will return a WeaviateVectorStore object.
    If the setting is "pinecone", it will return a PineconeVectorStore object.

    :param index_name: The name of the index to load.
    :type index_name: str
    :param namespace: The namespace of the index to load. If None, the default namespace will be used.
    :type namespace: str | None
    :return: A vectorstore object.
    :rtype: Pinecone
    """
    if config_settings.VECTOR_DATABASE_TO_USE == "weaviate":

        if not manager_client.client:
            manager_client.connect()

        weaviate_client = manager_client.client

        # Weaviate logic here
        return WeaviateVectorStore(
            client=weaviate_client,
            index_name=index_name,
            embedding=get_embeddings(),
            use_multi_tenancy=config_settings.WEAVIATE_MULTI_TENANCY_STATUS,
            text_key=config_settings.WEAVIATE_TEXT_KEY,
        )

    elif config_settings.VECTOR_DATABASE_TO_USE == "pinecone":
        # load a pinecone index
        return Pinecone.from_existing_index(
            index_name=index_name,
            embedding=get_embeddings(model_key="EMBEDDING_MODEL"),
            namespace=namespace,
        )


@asynccontextmanager
async def get_docsearch(
        index_name: str,
        weaviate_client: Optional[weaviate.client] = None,
) -> PineconeVectorStore:
    """
    Context manager for handling document search initialization.
    """
    try:
        if config_settings.VECTOR_DATABASE_TO_USE == "weaviate":

            if not weaviate_client:
                weaviate_client = manager_client.client

            docsearch = WeaviateVectorStore(
                client=weaviate_client,
                index_name=index_name,
                embedding=get_embeddings(),
                use_multi_tenancy=config_settings.WEAVIATE_MULTI_TENANCY_STATUS,
                text_key=config_settings.WEAVIATE_TEXT_KEY,
            )

            yield docsearch

        elif config_settings.VECTOR_DATABASE_TO_USE == "pinecone":
            # Initialize PineconeVectorStore
            docsearch = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=get_embeddings(model_key="EMBEDDING_MODEL"),
            )
            yield docsearch

    except Exception as e:
        logger.error(f"Failed to initialize document search: {e}")
        raise


async def get_related_docs_with_score(
    index_name: str,
    namespace: str,
    question: str,
    total_docs_to_retrieve: int = config_settings.NUMBER_OF_RETRIEVAL_RESULTS,
    filter_value: Optional[str] = None,
) -> list[tuple[Document, float]]:
    try:
        docsearch = load_index(index_name=index_name)

        if not docsearch:
            raise ValueError("Document search object is None")

        if config_settings.VECTOR_DATABASE_TO_USE == "pinecone":
            # Perform similarity search without a filter
            related_docs_with_score = await docsearch.asimilarity_search_with_relevance_scores(
                query=question,
                namespace=namespace,
                k=total_docs_to_retrieve,
            )
            return related_docs_with_score

        elif config_settings.VECTOR_DATABASE_TO_USE == "weaviate":
            if filter_value is None:
                search_params = {
                    "query": question,
                    "k": total_docs_to_retrieve,
                    "tenant": (namespace if namespace is not None
                               else config_settings.WEAVIATE_DEFAULT_TENANT_NAME),
                    "alpha": (
                        config_settings.WEAVIATE_HYPERPARAMETER_HYBRID_SEARCH
                        if namespace is not None
                        else config_settings.WEAVIATE_HYPERPARAMETER_HYBRID_SEARCH
                    ),
                }
            else:
                search_params = {
                    "query": question,
                    "k": config_settings.NUMBER_OF_RETRIEVAL_RESULTS,
                    "tenant": (namespace if namespace is not None
                               else config_settings.WEAVIATE_DEFAULT_TENANT_NAME),
                    "alpha": (config_settings.WEAVIATE_HYPERPARAMETER_HYBRID_SEARCH
                              if namespace is not None
                              else config_settings.WEAVIATE_HYPERPARAMETER_HYBRID_SEARCH),
                    "filters": Filter.by_property(
                        config_settings.WEAVIATE_FILTER_RESULTS_PARAMETER
                    ).equal(
                        filter_value,
                    )
                }

            logger.debug(f"Executing similarity search with params: {search_params}")

            if docsearch:
                logger.debug(f"Retrieving {question} from {index_name}")
                results = await docsearch.asimilarity_search_with_relevance_scores(**search_params)

            if not results:
                logger.info(f"No results found for query: {question[:100]}...")
                return []

            return results

    except Exception as e:
        logger.error(f"Failed to get related docs without context: {e}")
        return []


async def get_related_docs_without_context(
        index_name: str,
        namespace: str,
        question: str,
        total_docs_to_retrieve: int = 10
) -> List[Tuple[Document, float]]:
    """
    Retrieve related documents using PineconeVectorStore retriever.
    """
    try:
        async with get_docsearch(index_name) as docsearch:
            if config_settings.VECTOR_DATABASE_TO_USE == "pinecone":
                # Perform similarity search without a filter
                retriever = docsearch.as_retriever(
                    search_kwargs={
                        "k": total_docs_to_retrieve,
                        "namespace": namespace
                    }
                )

            elif config_settings.VECTOR_DATABASE_TO_USE == "weaviate":
                logger.info(f"Using weaviate Database")
                # Perform similarity search without a filter
                retriever = docsearch.as_retriever(
                    search_kwargs={
                        "k": total_docs_to_retrieve,
                        "tenant": namespace
                    }
                )

            related_docs = await retriever.ainvoke(input=question)
            logger.info(f"Retrieved {len(related_docs)} documents")
            return related_docs

    except Exception as e:
        logger.error(f"Error in get_related_docs_without_context: {str(e)}")
        return []


async def main() -> None:
    """
    Example usage of the retrieval functions.
    """
    result = await get_related_docs_without_context(
        index_name=config_settings.PINECONE_INDEX_NAME,
        namespace="default_namespace",
        question="ACIO job requirements qualifications",
        total_docs_to_retrieve=config_settings.PINECONE_TOTAL_DOCS_TO_RETRIEVE
    )
    logger.info(f"Retrieved {len(result)} documents")


if __name__ == "__main__":
    asyncio.run(main())
