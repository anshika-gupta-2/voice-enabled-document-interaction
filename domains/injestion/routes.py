import asyncio

from domains.injestion.doc_loader import file_loader
from domains.injestion.models import InjestRequestDto, FileInjestionResponseDto
from domains.models import RequestStatus, ApiNameEnum, RequestStatusEnum
from domains.injestion.utils import update_status
from domains.vector_db.utils import push_to_database
from domains.settings import config_settings
from domains.status_util import call_update_status_api

from loguru import logger

from fastapi import APIRouter, BackgroundTasks

router = APIRouter(tags=["injestion"])


@router.post(
    "/injest-doc",
    summary="Injests a document into the database",
    description="Injests a document into the database",
)
async def injest_doc(
        request: InjestRequestDto,
) -> FileInjestionResponseDto:
    logger.info(f"injest-doc request: {request.model_dump_json()}")

    try:
        logger.info(f"Processing file: {request.file_name}")

        response = await load_file_push_to_db(request)

        if response.status == RequestStatusEnum.COMPLETED:
            return FileInjestionResponseDto(
                request_id=request.request_id,
                status=RequestStatusEnum.COMPLETED,
                file_path=request.pre_signed_url,
                file_name=request.file_name,
                original_file_name=request.original_file_name,
                total_pages=response.data_json.get("total_pages", 0),
                api_name=ApiNameEnum.INJEST_DOC,
            )

        elif response.status == RequestStatusEnum.FAILED:
            return FileInjestionResponseDto(
                request_id=request.request_id,
                status=RequestStatusEnum.FAILED,
                file_name=request.file_name,
                original_file_name=request.original_file_name,
                error_detail=response.error_detail,
                total_pages=0,
                api_name=ApiNameEnum.INJEST_DOC
            )

    except Exception as e:
        logger.exception("Failed to process file")
        return FileInjestionResponseDto(
            request_id=request.request_id,
            status=RequestStatusEnum.FAILED,
            file_name=request.file_name,
            original_file_name=request.original_file_name,
            error_detail=str(e),
            total_pages=0,
            api_name=ApiNameEnum.INJEST_DOC
        )


def sanitize_file_path(file_path: str) -> str:
    if file_path.startswith("file:///"):
        return file_path[8:]  # Remove 'file:///' prefix
    return file_path


async def load_file_push_to_db(
        request: InjestRequestDto
):
    status = None
    try:
        logger.debug(f"load_file_push_to_db(): Attempting to load file from {request.pre_signed_url}")

        file_path = sanitize_file_path(request.pre_signed_url)

        chunked_documents, non_chunked_docs = file_loader(
            pre_signed_url=file_path,
            file_name=request.file_name,
            original_file_name=request.file_name,
            file_type=request.file_type,
            process_type=request.file_type,
        )
        logger.info(f"Successfully loaded file from {request.pre_signed_url} and total pages in file is {len(non_chunked_docs)}")

        push_status = push_to_database(
            texts=chunked_documents,
            index_name=config_settings.PINECONE_INDEX_NAME,
            namespace=request.namespace
        )

        if not push_status.status:
            return RequestStatus(
                request_id=request.request_id,
                api_name=ApiNameEnum.INJEST_DOC,
                status=RequestStatusEnum.FAILED,
                error_detail=push_status.message,
            )

        # Create success status
        status = RequestStatus(
            request_id=request.request_id,
            api_name=ApiNameEnum.INJEST_DOC,
            status=RequestStatusEnum.COMPLETED,
            data_json={"total_pages": len(non_chunked_docs)},
        )
        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Failed to load file from {request.pre_signed_url} and error is {e}")
        error_detail = f"Failed when process_type is {request.process_type}: {e}"
        status = RequestStatus(
            request_id=request.request_id,
            api_name=ApiNameEnum.INJEST_DOC,
            status=RequestStatusEnum.FAILED,
            error_detail=error_detail,
        )

    finally:
        if status:
            logger.info(
                f"Completed injest-doc for file_name: {request.file_name}"
                f" with status: {status.status}"
            )
            call_update_status_api(status_api_path="/injest-doc", request_status=status)
        else:
            logger.error("Status object was not created - this is unexpected")

    return status


if __name__ == "__main__":

    import os

    file_path = r"C:\Users\savit\Downloads\MAIN-DIFFUSION_IN_SOCIAL_NETWORKS.pdf"
    if os.path.isfile(file_path):
        print("File exists and is accessible.")
    else:
        print("File does not exist or is not accessible.")

    asyncio.run(
        load_file_push_to_db(
            request=InjestRequestDto(
                file_name="example.txt",
                pre_signed_url=r"C:\Users\savit\Downloads\ML Internship Interview Questions 1 (1).pdf",
                original_file_name="example.pdf",
                file_type="pdf",
                process_type="pdf",
                namespace="default_namespace",
                request_id="12345"
            )
        )
    )