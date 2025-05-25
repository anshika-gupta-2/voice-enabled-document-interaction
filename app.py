import base64
import os
import uuid
import docx
import io

import streamlit as st
from domains.injestion.doc_loader import file_loader
from dotenv import load_dotenv
from domains.injestion.routes import load_file_push_to_db
from domains.injestion.models import InjestRequestDto
from domains.agents.routes import react_orchestrator
from domains.retreival.routes import run_rag
from domains.settings import config_settings
from loguru import logger

load_dotenv()

# Check if the API keys are loaded correctly
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


if not openai_api_key:
    st.error("OPENAI_API_KEY is not set. Please check your .env file.")
if not pinecone_api_key:
    st.error("PINECONE_API_KEY is not set. Please check your .env file.")


st.title("RAG with Agents")
st.header("RAG with Multi-Agentic System")

st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["pdf", "docx", "txt"])


if uploaded_file is not None:
    output_folder_path = os.path.join(os.getcwd(), config_settings.STORAGE_FOLDER_NAME)

    x = [os.remove(f) for f in os.listdir(output_folder_path) if os.path.isfile(f)]

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)

    output_file_path = os.path.join(output_folder_path, uploaded_file.name)

    with open(output_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
        logger.info(f"File saved to {output_file_path}")

    file_name = uploaded_file.name
    original_file_name = uploaded_file.name
    file_type = uploaded_file.type.split("/")[1]
    process_type = file_type
    pre_signed_url = output_file_path
    params = {}
    metadata = []
    namespace = config_settings.PINECONE_DEFAULT_DEV_NAMESPACE

    st.sidebar.subheader('Document Preview')

    if file_type == "pdf":
        base64_pdf=base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
        pdf_display=f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="350" height="500" type="application/pdf"></iframe>'
        st.sidebar.markdown(pdf_display, unsafe_allow_html=True)

    elif file_type == "docx":
        docx = docx.Document(io.BytesIO(uploaded_file.getvalue()))
        text = '\n'.join([paragraph.text for paragraph in docx.paragraphs])
        st.sidebar.text(text)

    elif file_type == "txt":
        text_file_content=uploaded_file.getvalue().decode("utf-8")
        st.sidebar.text(text_file_content)

    if st.sidebar.button("Injest"):
        try:
            response = load_file_push_to_db(
                request=InjestRequestDto(
                    request_id=uuid.uuid4().int,
                    file_name=file_name,
                    original_file_name=original_file_name,
                    file_type=file_type,
                    process_type=process_type,
                    pre_signed_url=output_file_path,
                    params=params,
                    metadata=metadata,
                    namespace=namespace,
                    response_data_api_path="/injest-doc"
                )
            )

            st.sidebar.success("File loaded successfully")
        except Exception as e:
            st.sidebar.error(f"Failed to load file: {e}")