from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from domains.settings import config_settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from domains.models import RequestStatus
from domains.status_util import call_update_status_api


def split_text(text: list[Document], CHUNK_SIZE: int, CHUNK_OVERLAP: int) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(text)


def get_embeddings(
        model_key: str = "EMBEDDING_MODEL_NAME"
):
    if config_settings.LLM_SERVICE == "openai":
        return OpenAIEmbeddings(
            model=config_settings.LLMS.get(model_key, None),
            api_key=config_settings.OPENAI_API_KEY,
        )

    elif config_settings.LLM_SERVICE == "groq":
        return OpenAIEmbeddings(
            model=config_settings.LLMS.get(model_key, None),
            api_key=config_settings.GROQ_API_KEY,
        )

    elif config_settings.LLM_SERVICE == "google":
        return GoogleGenerativeAIEmbeddings(
            model=config_settings.GEMINI.get(model_key, None),
            google_api_key=config_settings.GOOGLE_API_KEY
        )



def update_status(api_path: str, request_status: RequestStatus, token: str=None) -> None:
    if api_path:
        call_update_status_api(api_path, request_status)

