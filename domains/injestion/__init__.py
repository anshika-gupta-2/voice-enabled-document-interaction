from domains.vector_db.utils import validate_and_create_index
from domains.settings import config_settings
from loguru import logger


def start_injestion():
    validate_and_create_index(
        config_settings.PINECONE_INDEX_NAME,
        config_settings.PINECONE_DROP_INDEX_NAME_STATUS
    )


try:
    start_injestion()
except Exception as e:
    logger.error(f"Error occurred during injestion: {e}")