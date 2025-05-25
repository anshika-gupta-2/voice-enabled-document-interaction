import weaviate
import atexit
from typing import Optional
from domains.settings import config_settings
from domains.vector_db.models import ConnectionResponseDto, ClientResponseDto
from weaviate.exceptions import WeaviateConnectionError
from loguru import logger
from weaviate.config import AdditionalConfig
from contextlib import contextmanager



class WeaviateConnectionManager:
    """Manages Weaviate database connections with proper resource cleanup."""

    def __init__(self) -> None:
        self._client: Optional[weaviate.WeaviateClient] = None
        self._additional_config = AdditionalConfig()
        atexit.register(self.close)

    def validate_collection(self, collection_name: str) -> bool:
        """Validate if a collection exists without using context manager."""
        try:
            if not self._client:
                self.connect()
            exists = self._client.collections.exists(collection_name)
            logger.debug(f"Collection {collection_name} {'exists' if exists else 'does not exist'}")

            return exists
        except Exception as e:
            logger.error(f"Error validating collection {collection_name}: {str(e)}")
            raise

    # Remove the managed_connection context manager since we want persistent connections
    def get_client(self) -> weaviate.WeaviateClient:
        """Get or create a client connection."""
        if not self._client:
            self.connect()
        return self._client

    def connect(self) -> ClientResponseDto:
        """Establish connection to Weaviate database using the appropriate connection method."""
        try:
            connection_type = config_settings.WEAVIATE_VECTOR_DATABASE_SERVICE_TYPE.lower()

            if connection_type == "local":
                self._client = weaviate.connect_to_local(
                    host=config_settings.WEAVIATE_HOST,
                    port=config_settings.WEAVIATE_HOST_PORT,
                    grpc_port=config_settings.WEAVIATE_GRPC_PORT,
                    additional_config=self._additional_config
                )
            elif connection_type == "online":
                self._client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=config_settings.WEAVIATE_CLUSTER_URL,
                    auth_credentials=config_settings.WEAVIATE_AUTH_CREDENTIALS,
                    additional_config=self._additional_config
                )
            else:
                raise ValueError(f"Unsupported connection type: {connection_type}")

            logger.info(f"Successfully connected to {connection_type} Weaviate instance")
            return ClientResponseDto(status_code=True, client=self._client)

        except WeaviateConnectionError as e:
            logger.error(f"Weaviate connection error: {str(e)}")
            raise e

        except Exception as e:
            logger.error(f"Unexpected error during connection: {str(e)}")
            raise e

    def close(self) -> None:
        """Safely close the Weaviate connection and cleanup resources."""
        if self._client:
            try:
                self._client.close()
                self._client = None
                logger.debug("Weaviate connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing Weaviate connection: {str(e)}")
            finally:
                self._client = None

    def get_partition_names(self, index_name: str) -> list:
        """Get all tenant names for a collection."""
        try:
            if self.validate_collection(index_name):
                tenants = self._client.collections.get(index_name).tenants.get()
                return list(tenants.keys())
            return []
        except Exception as e:
            logger.error(f"Error getting partition names: {str(e)}")
            raise

    def validate_partition_name(self, partition_name: str, index_name: str) -> bool:
        """Validate if a tenant exists in the collection."""
        try:
            if not self.validate_collection(index_name):
                return False

            return self._client.collections.get(index_name).tenants.exists(partition_name)
        except Exception as e:
            logger.error(f"Error validating partition {partition_name}: {str(e)}")
            raise

    def delete_partition(self, index_name: str, partition_name: str) -> bool:
        """Delete a tenant from the collection."""
        try:
            if not self.validate_partition_name(partition_name, index_name):
                logger.warning(f"Partition {partition_name} does not exist")
                return False

            self._client.collections.get(index_name).tenants.remove(partition_name)

            # Verify deletion
            exists = self.validate_partition_name(partition_name, index_name)
            if not exists:
                logger.info(f"Partition {partition_name} deleted successfully")
                return True
            else:
                logger.error(f"Failed to delete partition {partition_name}")
                return False

        except Exception as e:
            logger.error(f"Error deleting partition {partition_name}: {str(e)}")
            raise

    def delete_index_collection(self, index_name: str) -> bool:
        """Delete a collection from the database."""
        try:
            if not self.validate_collection(index_name):
                logger.warning(f"Collection {index_name} does not exist")
                return False

            self._client.collections.delete(index_name)
            logger.info(f"Collection {index_name} deleted successfully")

            if self.validate_collection(index_name):
                logger.error(f"Failed to delete collection {index_name}")
                return False
            return True

        except Exception as e:
            logger.error(f"Error deleting collection {index_name}: {str(e)}")
            raise

    def delete_partition_data(self, index_name: str, partition_name: str) -> bool:
        """Delete all data within a tenant while keeping the tenant."""
        try:
            if not self.validate_partition_name(partition_name, index_name):
                logger.warning(f"Partition {partition_name} does not exist")
                return False

            collection = self._client.collections.get(index_name)
            # Delete all objects in the tenant
            collection.data.delete(
                tenant=partition_name,
                where={"path": ["id"], "operator": "Like", "valueText": "*"}
            )

            logger.info(f"All data in partition {partition_name} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Error deleting data in partition {partition_name}: {str(e)}")
            raise


    def handle_partition_update(self, index_name: str, partition_name: str, delete_existing: bool = False) -> bool:
        """Handle partition update based on delete flag."""
        try:
            if not self.validate_collection(index_name):
                logger.error(f"Collection {index_name} does not exist")
                return False

            if delete_existing:
                return self.delete_partition(index_name, partition_name)
            else:
                return self.delete_partition_data(index_name, partition_name)

        except Exception as e:
            logger.error(f"Error in partition update handling: {str(e)}")
            raise


class DatabaseConnection:
    """Singleton class to manage database connection lifecycle."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._manager = WeaviateConnectionManager()
        return cls._instance

    def initialize(self) -> ConnectionResponseDto:
        """Initialize database connection with proper cleanup."""
        try:
            connection_response = self._manager.connect()

            return ConnectionResponseDto(
                status=connection_response.status_code,
                client=connection_response.client,
                manager=self._instance._manager,
                message="Connection established successfully" if connection_response.status_code else "Connection failed"
            )

        except WeaviateConnectionError as e:
            logger.error(f"Weaviate connection error: {str(e)}")
            return ConnectionResponseDto(
                status=False,
                client=None,
                manager=None,
                message=f"Connection error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {str(e)}")
            return ConnectionResponseDto(
                status=False,
                client=None,
                manager=None,
                message="Internal connection error occurred"
            )

    def cleanup(self):
        """Cleanup database resources."""
        if hasattr(self, '_manager'):
            self._manager.close()


# Initialize global connection with proper cleanup
db_connection = DatabaseConnection()
manager_client = db_connection.initialize()
atexit.register(db_connection.cleanup)


if __name__ == "__main__":
    try:
        with DatabaseConnection()._manager.get_client() as client:
            logger.info("Testing connection...")
            logger.info("Connection test complete")
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")