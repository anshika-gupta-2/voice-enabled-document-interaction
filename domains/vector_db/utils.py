# from pinecone import Pinecone, ServerlessSpec
# from domains.injestion.utils import get_embeddings
# from domains.settings import config_settings
# from loguru import logger
#
# from datetime import datetime
#
# from langchain_weaviate.vectorstores import WeaviateVectorStore
# from domains.vector_db.exception import DocumentRetrievalError, VectorDBOperationError
# from langchain_community.vectorstores import Pinecone as PineconeVectorStore
# from pinecone.exceptions import PineconeApiException
#
# from domains.vector_db.models import PushToDatabaseResponseDto
# from domains.vector_db.weaviate_utils import manager_client
# from domains.handler import retry_with_custom_backoff
#
#
# def initialize_pinecone() -> Pinecone:
#     try:
#         pc = Pinecone(api_key=config_settings.PINECONE_API_KEY)
#         logger.info("Successfully initialized Pinecone")
#         return pc
#     except Exception as e:
#         logger.error(f"Failed to initialize Pinecone: {e}")
#         raise
#
#
# @retry_with_custom_backoff()
# def validate_and_create_index(
#         index_name: str,
#         drop_index: bool=config_settings.PINECONE_DROP_INDEX_NAME_STATUS
# ) -> bool:
#     try:
#         if config_settings.VECTOR_DATABASE_TO_USE == "pinecone":
#             pc = initialize_pinecone()
#             indexes = [index.get("name", None) for index in pc.list_indexes()]
#
#             logger.info(f"Existing indexes: {indexes}")
#             def create_index(index_name: str) -> None:
#                 try:
#                     pc.create_index(
#                         name=index_name,
#                         dimension=1536,
#                         metric=config_settings.PINECONE_INDEX_METRIC_TYPE,
#                         spec=ServerlessSpec(
#                             cloud=config_settings.PINECONE_INDEX_CLOUD_NAME,
#                             region=config_settings.PINECONE_INDEX_REGION_NAME
#                         )
#                     )
#                     logger.info(f"Successfully created index: {index_name}")
#                 except PineconeApiException as e:
#                     logger.error(f"Pinecone API error: {e}")
#                     raise
#                 except Exception as e:
#                     logger.error(f"Failed to create index: {e}")
#                     raise
#
#             for idx in indexes:
#                 if idx is not None and idx == index_name:
#                     if drop_index:
#                         try:
#                             logger.info(f"Deleting index: {index_name}")
#                             pc.delete_index(index_name)
#                             create_index(index_name)
#                             logger.info(f"Successfully deleted index: {index_name}")
#                             return True
#
#                         except PineconeApiException as e:
#                             logger.error(f"Pinecone API error: {e}")
#                             return False
#                         except Exception as e:
#                             logger.error(f"Failed to delete index: {e}")
#                             return False
#                     else:
#                         logger.info(f"Index already exists: {index_name}")
#                         return True
#             create_index(index_name)
#             return True
#
#     except Exception as e:
#         logger.error(f"Failed to validate and create index: {e}")
#         return False
#
#
#
# import atexit
# import ssl
# from contextlib import suppress
#
# def cleanup_ssl_sockets():
#     with suppress(Exception):
#         for sock in ssl.SSLSocket._active:  # type: ignore
#             sock.close()
#
#
# def push_to_database(
#         texts: list,
#         index_name: str = config_settings.PINECONE_INDEX_NAME,
#         namespace: str = config_settings.PINECONE_DEFAULT_DEV_NAMESPACE,
#         drop_namespace=config_settings.DELETE_NAMESPACE_STATUS,
# ):
#     try:
#         meta_datas = [text.metadata for text in texts]
#
#         if namespace is None:
#             namespace = config_settings.PINECONE_DEFAULT_DEV_NAMESPACE
#
#         try:
#             if config_settings.VECTOR_DATABASE_TO_USE == "pinecone":
#                 if drop_namespace:
#                     pinecone_vs = initialize_pinecone()
#                     loaded_index = pinecone_vs.Index(index_name)
#
#                     if loaded_index is None:
#                         logger.error(f"Index {index_name} not found")
#                         return False
#
#                     list_namespaces = list(loaded_index.describe_index_stats(namespace=namespace)["namespaces"].keys())
#
#
#                     logger.info(
#                                 f"Namespaces in index: {index_name}: {list_namespaces}"
#                     )
#
#                     if namespace in list_namespaces:
#                         loaded_index.delete(
#                             delete_all=True,
#                             namespace=namespace,
#                         )
#                         logger.info(
#                             f"Successfully deleted namespace: {namespace} from index: {index_name}"
#                         )
#
#                     logger.info(f"Pushing data to Pinecone index: {index_name} and namespace: {namespace}")
#
#                     PineconeVectorStore.from_texts(
#                         [t.page_content for t in texts],
#                         get_embeddings(model_key="EMBEDDING_MODEL_NAME"),
#                         meta_datas,
#                         index_name=index_name,
#                         namespace=namespace,
#                     )
#                     logger.info("Successfully pushed data to Pinecone")
#
#                     return PushToDatabaseResponseDto(
#                         status=True,
#                         message="Documents pushed successfully",
#                         document_ids=None,
#                         timestamp=datetime.now().isoformat(),
#                         index=index_name,
#                         namespace=namespace
#                     )
#
#                 else:
#                     logger.info(f"Pushing data to Pinecone index: {index_name} and namespace: {namespace}")
#                     PineconeVectorStore.from_texts(
#                         [t.page_content for t in texts],
#                         get_embeddings(model_key="EMBEDDING_MODEL_NAME"),
#                         meta_datas,
#                         index_name=index_name,
#                         namespace=namespace,
#                     )
#                     logger.info("Successfully pushed data to Pinecone")
#
#                     return PushToDatabaseResponseDto(
#                         status=True,
#                         message="Documents pushed successfully",
#                         document_ids=None,
#                         timestamp=datetime.now().isoformat(),
#                         index=index_name,
#                         namespace=namespace
#                     )
#
#             elif config_settings.VECTOR_DATABASE_TO_USE == "weaviate":
#
#                 try:
#                     client = manager_client.manager.get_client()
#
#                     if manager_client.manager.validate_collection(
#                             collection_name=index_name
#                     ):
#                         logger.debug(f"Collection exists, proceeding with document ingestion")
#
#                         if drop_namespace:
#                             namespace_exist = manager_client.manager.validate_partition_name(
#                                 partition_name=namespace,
#                                 index_name=index_name
#                             )
#
#                             if namespace_exist:
#                                 update_status = manager_client.manager.handle_partition_update(
#                                     index_name=index_name,
#                                     partition_name=namespace,
#                                     delete_existing=True
#                                 )
#
#                                 if not update_status:
#                                     raise VectorDBOperationError("Failed to update/delete existing partition")
#
#                     vector_store = WeaviateVectorStore(
#                         client=client,
#                         index_name=index_name,
#                         embedding=get_embeddings(),
#                         use_multi_tenancy=True,
#                         text_key="text",
#                     )
#
#                     logger.info(f"Attempting to ingest {len(texts)} documents into Weaviate")
#
#                     document_ids = vector_store.add_documents(
#                         documents=texts,
#                         tenant=namespace,
#                     )
#
#                     if not document_ids or document_ids == []:
#                         logger.warning("Document ingestion completed but no document IDs were returned")
#                         return PushToDatabaseResponseDto(
#                             status=True,
#                             message="Document ingestion completed but no document IDs were returned",
#                             timestamp=datetime.now().isoformat(),
#                             index=index_name,
#                             namespace=namespace
#                         )
#
#                     logger.info(f"Successfully ingested {len(document_ids)} documents into Weaviate")
#
#                     return PushToDatabaseResponseDto(
#                         status=True,
#                         message="Documents ingested successfully",
#                         document_ids=document_ids,
#                         timestamp=datetime.now().isoformat(),
#                         index=index_name,
#                         namespace=namespace
#                     )
#
#                 finally:
#                     atexit.register(cleanup_ssl_sockets)
#
#             else:
#                 logger.error(f"Unsupported vector database: {config_settings.VECTOR_DATABASE_TO_USE}")
#                 return PushToDatabaseResponseDto(
#                     status=False,
#                     message="Unsupported vector database",
#                     timestamp=datetime.now().isoformat(),
#                     index=index_name,
#                     namespace=namespace
#                 )
#
#
#         except Exception as e:
#             logger.error(f"Failed to push data to Pinecone: {str(e)}")
#             return PushToDatabaseResponseDto(
#                 status=False,
#                 message=f"Failed to push data to Pinecone: {str(e)}",
#                 timestamp=datetime.now().isoformat(),
#                 index=index_name,
#                 namespace=namespace
#             )
#
#     except Exception as e:
#         logger.exception(f"Failed to push vectors to database: {str(e)}")
#         raise Exception(f"Vector database operation failed: {str(e)}")


from datetime import datetime
from typing import List, Optional, Union
import atexit
import ssl
from contextlib import suppress
from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from loguru import logger

from domains.vector_db.models import PineconeConfig
from domains.injestion.utils import get_embeddings
from domains.settings import config_settings
from domains.vector_db.exception import VectorDBOperationError
from domains.vector_db.models import PushToDatabaseResponseDto
from domains.vector_db.weaviate_utils import manager_client
from domains.handler import retry_with_custom_backoff


def initialize_pinecone() -> Pinecone:
    try:
        pc = Pinecone(api_key=config_settings.PINECONE_API_KEY)
        logger.info("Successfully initialized Pinecone")
        return pc
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")
        raise


def create_pinecone_index(pc: Pinecone, config: PineconeConfig) -> None:
    try:
        pc.create_index(
            name=config.index_name,
            dimension=config.dimension,
            metric=config.metric,
            spec=ServerlessSpec(cloud=config.cloud, region=config.region)
        )
        logger.info(f"Successfully created index: {config.index_name}")
    except (PineconeApiException, Exception) as e:
        logger.error(f"Failed to create index: {e}")
        raise


@retry_with_custom_backoff()
def validate_and_create_index(
        index_name: str,
        drop_index: bool = config_settings.PINECONE_DROP_INDEX_NAME_STATUS
) -> bool:
    if config_settings.VECTOR_DATABASE_TO_USE != "pinecone":
        return False

    try:
        pc = initialize_pinecone()
        indexes = [index.get("name") for index in pc.list_indexes()]
        logger.info(f"Existing indexes: {indexes}")

        config = PineconeConfig(index_name=index_name)

        if index_name in indexes:
            if drop_index:
                try:
                    pc.delete_index(index_name)
                    create_pinecone_index(pc, config)
                except Exception as e:
                    logger.error(f"Failed to handle existing index: {e}")
                    return False
            return True

        create_pinecone_index(pc, config)
        return True

    except Exception as e:
        logger.error(f"Failed to validate and create index: {e}")
        return False


def cleanup_ssl_sockets() -> None:
    with suppress(Exception):
        for sock in ssl.SSLSocket._active:  # type: ignore
            sock.close()


def handle_pinecone_push(
        texts: List,
        meta_datas: List,
        config: PineconeConfig,
        drop_namespace: bool
) -> PushToDatabaseResponseDto:
    if drop_namespace:
        pinecone_vs = initialize_pinecone()
        loaded_index = pinecone_vs.Index(config.index_name)

        if loaded_index is None:
            raise VectorDBOperationError(f"Index {config.index_name} not found")

        namespaces = list(loaded_index.describe_index_stats(namespace=config.namespace)["namespaces"].keys())

        if config.namespace in namespaces:
            loaded_index.delete(delete_all=True, namespace=config.namespace)
            logger.info(f"Deleted namespace: {config.namespace} from index: {config.index_name}")

    PineconeVectorStore.from_texts(
        [t.page_content for t in texts],
        get_embeddings(model_key="EMBEDDING_MODEL_NAME"),
        meta_datas,
        index_name=config.index_name,
        namespace=config.namespace,
    )

    return PushToDatabaseResponseDto(
        status=True,
        message="Documents pushed successfully",
        document_ids=None,
        timestamp=datetime.now().isoformat(),
        index=config.index_name,
        namespace=config.namespace
    )


def handle_weaviate_push(
        texts: List,
        index_name: str,
        namespace: str,
        drop_namespace: bool
) -> PushToDatabaseResponseDto:
    try:
        client = manager_client.manager.get_client()

        if manager_client.manager.validate_collection(collection_name=index_name):
            if drop_namespace:
                if manager_client.manager.validate_partition_name(
                        partition_name=namespace,
                        index_name=index_name
                ):
                    update_status = manager_client.manager.handle_partition_update(
                        index_name=index_name,
                        partition_name=namespace,
                        delete_existing=True
                    )
                    if not update_status:
                        raise VectorDBOperationError("Failed to update/delete existing partition")

        vector_store = WeaviateVectorStore(
            client=client,
            index_name=index_name,
            embedding=get_embeddings(),
            use_multi_tenancy=True,
            text_key="text",
        )

        document_ids = vector_store.add_documents(documents=texts, tenant=namespace)

        return PushToDatabaseResponseDto(
            status=True,
            message="Documents ingested successfully",
            document_ids=document_ids or None,
            timestamp=datetime.now().isoformat(),
            index=index_name,
            namespace=namespace
        )
    finally:
        atexit.register(cleanup_ssl_sockets)


def push_to_database(
        texts: List,
        index_name: str = config_settings.PINECONE_INDEX_NAME,
        namespace: str = config_settings.PINECONE_DEFAULT_DEV_NAMESPACE,
        drop_namespace: bool = config_settings.DELETE_NAMESPACE_STATUS,
) -> PushToDatabaseResponseDto:
    try:
        meta_datas = [text.metadata for text in texts]
        namespace = namespace or config_settings.PINECONE_DEFAULT_DEV_NAMESPACE

        if config_settings.VECTOR_DATABASE_TO_USE == "pinecone":
            config = PineconeConfig(index_name=index_name, namespace=namespace)
            return handle_pinecone_push(texts, meta_datas, config, drop_namespace)

        elif config_settings.VECTOR_DATABASE_TO_USE == "weaviate":
            return handle_weaviate_push(texts, index_name, namespace, drop_namespace)

        else:
            return PushToDatabaseResponseDto(
                status=False,
                message=f"Unsupported vector database: {config_settings.VECTOR_DATABASE_TO_USE}",
                timestamp=datetime.now().isoformat(),
                index=index_name,
                namespace=namespace
            )

    except Exception as e:
        logger.exception(f"Vector database operation failed: {str(e)}")
        return PushToDatabaseResponseDto(
            status=False,
            message=f"Operation failed: {str(e)}",
            timestamp=datetime.now().isoformat(),
            index=index_name,
            namespace=namespace
        )