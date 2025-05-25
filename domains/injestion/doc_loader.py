import os
import pprint
from os.path import expanduser, isfile
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader
)

from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader

from langchain_community.document_loaders import PyPDFLoader
from typing import Any, List
from langchain_core.documents import Document

from pathlib import Path
from loguru import logger
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from domains.settings import config_settings

from domains.injestion.models import FILE_TYPE
from langchain.chains import RetrievalQA, LLMChain
from typing import Any, Callable, IO, Dict, Iterator, get_args, Tuple, Callable

from urllib.parse import urlparse
from tempfile import NamedTemporaryFile
from os import remove
from domains.injestion.utils import split_text


class URLDownloaderMixin:
    @staticmethod
    def _is_valid_url(url: str) -> bool:
        parsed = urlparse(url)        ##focus on splitting a URL string into its components
        return bool(parsed.netloc) and bool(parsed.scheme)

    def __init__(self, file_path=None, *args: Any, **kwargs: Any) -> None:
        self._temp_file: IO[bytes] | None = None
        self.file_path = Path(str(file_path))
        if isinstance(self, PyMuPDFLoader):
            if file_path is None:
                raise TypeError("string argument file_path is needed for pdf loader")
            self.file_path = file_path
            self.web_path = None
            self.headers = kwargs.pop("headers", None)
            self.extract_images = kwargs.pop("extract_images", True)
            self.text_kwargs = kwargs

        if "~" in self.file_path:
            self.file_path = expanduser(self.file_path)

        if isfile(self.file_path):
            return

        if self._is_valid_url(self.file_path):
            self.web_path = self.file_path
            self._temp_file = NamedTemporaryFile(delete=False)
            self.file_path = self._temp_file.name
            return

        raise ValueError(f"File path {self.file_path} is not a valid file or url")

    def __del__(self) -> None:
        if self._temp_file:
            self._temp_file.close()
            remove(self._temp_file.name)


class PDFLoaderExtended(PyPDFLoader):
    """Extended PDF loader with additional functionality."""

    def __init__(
            self,
            file_path: str,
            extract_images: bool = False,
            **kwargs: Any
    ) -> None:
        # Convert Path to string if needed
        self.file_path = str(file_path) if hasattr(file_path, '__fspath__') else file_path
        super().__init__(self.file_path)
        self.extract_images = extract_images

    def load(self) -> List[Document]:
        try:
            documents = super().load()
            if not documents:
                raise ValueError("No documents loaded from PDF")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF {self.file_path}: {str(e)}")
            raise


# class PDFLoaderExtended(PyMuPDFLoader):
#     """Extended PDF loader with additional functionality."""
#
#     def __init__(self, file_path: str, extract_images: bool = False):
#         super().__init__(file_path)
#         self.extract_images = extract_images
#         self.parser = PDFMinerParser()  # Add the parser attribute
#
#     def load(self):
#         try:
#             documents = super().load()
#             if not documents:
#                 raise ValueError("No documents loaded from PDF")
#             return documents
#         except Exception as e:
#             logger.error(f"Error loading PDF: {e}")
#             raise


class DocLoaderExtended(URLDownloaderMixin, UnstructuredWordDocumentLoader):
    def __init__(
            self,
            file_path: str | Path,
            *,
            headers: dict[Any, Any] | None = None,
            mode: str = "single",
            **unstructured_kwargs: Any,
    ):
        # Convert Path to string if needed
        self.file_path = str(file_path) if hasattr(file_path, '__fspath__') else file_path
        URLDownloaderMixin.__init__(self, self.file_path)
        UnstructuredWordDocumentLoader.__init__(
            self,
            file_path=self.file_path,
            **unstructured_kwargs
        )
        self.mode = mode
        self.unstructured_kwargs = unstructured_kwargs or {}
        self.strategy = "fast" if self.mode == "single" else "accurate"
        self.unstructured_kwargs["strategy"] = self.strategy


class FileLoader(BaseLoader):
    def __init__(self, file_path: str, process_type: str = "txt"):
        self.file_path = str(file_path) if hasattr(file_path, '__fspath__') else file_path
        self.process_type = process_type
        self._validate_process_type()

    def _validate_process_type(self) -> None:
        valid_types = FILE_TYPE
        if self.process_type not in valid_types:
            raise ValueError(f"Invalid process type: {self.process_type}. Supported types are: {', '.join(valid_types)}")

    def _validate_file_path(self) -> None:
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def load(self) -> list[Document] | str:
        try:
            logger.info(f"{self.__class__.__name__}.load(): Attempting to load file from {self.file_path}")
            self._validate_file_path()

            if self.process_type == "txt":
                text_loader = TextLoader(file_path=self.file_path)
                file_contents = text_loader.load()
                logger.info(f"Successfully loaded file from {self.file_path} and total pages in file is {len(file_contents)}")
                return file_contents

            elif self.process_type == "pdf":
                pdf_loader = PDFLoaderExtended(file_path=self.file_path, extract_images=False)
                file_contents = pdf_loader.load()
                logger.info(f"Successfully loaded file from {self.file_path} and total pages in file is {len(file_contents)}")
                return file_contents

            elif self.process_type == "docx":
                doc_loader = DocLoaderExtended(file_path=self.file_path, extract_images=False)
                file_contents = doc_loader.load()
                logger.info(f"Successfully loaded file from {self.file_path} and total pages in file is {len(file_contents)}")
                return file_contents

            else:
                raise ValueError(f"Unsupported process type: {self.process_type}")

        except FileNotFoundError as fnf_error:
            logger.error(f"{self.__class__.__name__}.load(): File not found - {fnf_error}")
            return "Error: File does not exist."

        except ValueError as val_error:
            logger.error(f"{self.__class__.__name__}.load(): Value error - {val_error}")
            return str(val_error)
        except Exception as e:
            logger.error(f"{self.__class__.__name__}.load(): Unexpected error - {e}")
            return "Error: An unexpected error occurred while loading the file."


def file_loader(
    pre_signed_url: str,
    file_name: str,
    original_file_name: str,
    file_type: str,
    process_type: str,
    metadata: list[dict[str, str]] = [{}],
) -> Tuple[list[Document], Any]:

    if file_type not in FILE_TYPE:
        raise Exception(f"{file_type} is not a supported file type")

    loaders: dict[str, Callable[[], BaseLoader]] = {
        "txt": lambda: FileLoader(pre_signed_url, process_type="txt"),
        "pdf": lambda: FileLoader(pre_signed_url, process_type="pdf"),
        "docx": lambda: FileLoader(pre_signed_url, process_type="docx"),
    }

    if (loader := loaders.get(process_type)) is None:
        raise FileNotFoundError("Unsupported process_type")

    loaded_documents = loader().load()
    logger.info(f"documents loaded {len(loaded_documents)}")

    parsed_documents = split_text(
        text=loaded_documents,
        CHUNK_SIZE=config_settings.CHUNK_SIZE,
        CHUNK_OVERLAP=config_settings.CHUNK_OVERLAP
    )

    logger.info(f"Total number of document chunks {len(parsed_documents)}")

    additional_metadata = {
        "original_file_name": original_file_name,
        "file_name": file_name,
        "file_type": file_type,
        "process_type": process_type,
    }

    if metadata:
        for i in metadata:
            additional_metadata.update(i)

    for document in parsed_documents:
        document.metadata |= additional_metadata | {
            "title": document.metadata.get("title") or original_file_name
        }

    return parsed_documents, loaded_documents


if __name__ == "__main__":
    # Example usage
    file_path = str(Path(r"C:\Users\savit\Downloads\technology_environment_topics.docx")) # Replace with your file path
    process_type = "docx"  # or "txt", "docx"
    loader = UnstructuredWordDocumentLoader(file_path)
    documents = loader.load()
