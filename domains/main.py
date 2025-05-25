import streamlit as st
import os
import asyncio
import json
import uuid
import websockets
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

from pathlib import Path
from streamlit_option_menu import option_menu
import time
from datetime import datetime
from functools import lru_cache
import logging
from logging.handlers import RotatingFileHandler

from domains.injestion.models import InjestRequestDto, FileInjestionResponseDto
from domains.models import RequestStatusEnum
from domains.injestion.routes import injest_doc
from domains.agents.routes import react_orchestrator
from domains.retreival.models import Message


# Logging setup
logging.basicConfig(
    handlers=[RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)],
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Must be the first Streamlit command
st.set_page_config(
    page_title="Document Chat System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
# Add this to your existing CSS section
st.markdown("""
<style>
    /* Global Styles */
    .main {
        padding: 2rem;
    }

    /* Header Styles */
    .stTitle {
        color: #1f447c;
        font-size: 2.5rem !important;
        margin-bottom: 2rem !important;
    }

    /* Login Form Styles */
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
    }

    /* Button Styles */
    .stButton button {
        background-color: #1f447c;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #15325c;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Chat Interface Styles */
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f8f9fa;
        margin-right: 2rem;
    }

    /* File Upload Styles */
    .upload-section {
        background-color: #f8f9fa;
        padding: 3rem;
        border: 2px dashed #1f447c;
        border-radius: 12px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .upload-section:hover {
        background-color: #e3f2fd;
        border-color: #1565c0;
    }

    /* Sidebar Styles */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 2rem;
    }

    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
    }

    /* Progress Bar Styles */
    .stProgress > div > div > div {
        background-color: #1f447c;
    }

    /* Chat Input Styles */
    .stChatInput {
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "user": None,
        "role": None,
        "chat_mode": "agent",
        "messages": [],
        "theme": "light",
        "upload_history": [],
        "ingested_files": set()  # Track ingested files
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@lru_cache(maxsize=100)
def get_user_settings(username: str) -> dict:
    return {"theme": "light", "language": "English"}


def validate_file(file) -> bool:
    max_size = 10 * 1024 * 1024  # 10MB
    allowed_types = ["pdf", "txt", "docx"]

    if file.size > max_size:
        st.error(f"File {file.name} exceeds maximum size of 10MB")
        return False
    if file.name.split('.')[-1].lower() not in allowed_types:
        st.error(f"File type not allowed for {file.name}")
        return False
    return True


def login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Welcome to Document Chat</h1>",
                    unsafe_allow_html=True)

        with st.container():
            st.markdown(
                "<div style='background-color: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>",
                unsafe_allow_html=True)
            username = st.text_input("Username", value="", max_chars=20, placeholder="Enter username")
            password = st.text_input("Password", type="password", max_chars=20, placeholder="Enter password")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", use_container_width=True):
                    if username == "admin" and password == "admin":
                        with st.spinner("Logging in..."):
                            time.sleep(0.5)  # Add slight delay for better UX
                            st.session_state.user = username
                            st.session_state.role = "admin"
                            st.rerun()
                    elif username == "user" and password == "user":
                        with st.spinner("Logging in..."):
                            time.sleep(0.5)
                            st.session_state.user = username
                            st.session_state.role = "user"
                            st.rerun()
                    else:
                        st.error("Invalid credentials")
            with col2:
                if st.button("Demo User", use_container_width=True, type="secondary"):
                    st.session_state.user = "demo"
                    st.session_state.role = "user"
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

class InjestionResponseDto(BaseSettings):
    request_id: int
    status: RequestStatusEnum
    message: str

async def handle_file_upload(file, file_path):
    # Generate a unique identifier for the file (using content hash or name+size)
    file_identifier = f"{file.name}_{file.size}"

    # Check if file was already ingested
    if file_identifier in st.session_state.ingested_files:
        return InjestionResponseDto(
            request_id=int(uuid.uuid4()),
            status=RequestStatusEnum.COMPLETED,
            message="File already ingested"
        )

    print(f"USER NAME : {st.session_state.user}")
    request = InjestRequestDto(
        request_id=int(uuid.uuid4()),
        pre_signed_url=str(file_path),
        file_name=file.name,
        original_file_name=file.name,
        file_type=file.name.split('.')[-1],
        namespace=str(st.session_state.user),
        process_type=file.name.split('.')[-1],
    )

    response = await injest_doc(request)

    # If ingestion was successful, add to tracked files
    if response.status == RequestStatusEnum.COMPLETED:
        st.session_state.ingested_files.add(file_identifier)

    return response


def upload_files():
    st.markdown("### Document Upload")
    st.markdown("Supported formats: PDF, TXT, DOCX")

    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_files = st.file_uploader(
                "Drag and drop files here",
                type=["pdf", "txt", "docx"],
                accept_multiple_files=True,
                key="file_uploader"
            )
        with col2:
            if uploaded_files:
                total_files = len(uploaded_files)
                st.metric("Files Selected", total_files)

    if uploaded_files:
        progress_bar = st.progress(0)
        for idx, file in enumerate(uploaded_files):
            if not validate_file(file):
                continue

            file_identifier = f"{file.name}_{file.size}"
            if file_identifier in st.session_state.ingested_files:
                st.info(f"📝 {file.name} already processed - skipping")
                progress_bar.progress((idx + 1) / total_files)
                continue

            try:
                with st.spinner(f'Processing {file.name}...'):
                    file_path = Path("temp") / file.name
                    file_path.parent.mkdir(exist_ok=True)

                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                    response = asyncio.run(handle_file_upload(file, file_path))

                    if response.status == RequestStatusEnum.COMPLETED:
                        st.success(f"✅ {file.name} processed successfully")
                    elif response.status == RequestStatusEnum.FAILED:
                        st.error(f"❌ {file.name}: {response.error_detail}")

                progress_bar.progress((idx + 1) / total_files)

            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")
            finally:
                if file_path.exists():
                    file_path.unlink()


async def chat_interface():
    with st.sidebar:
        st.markdown("### Chat Settings")
        chat_mode = option_menu(
            "Chat Mode",
            ["Agent-based RAG", "Streaming RAG"],
            icons=['robot', 'lightning'],
            default_index=0
        )
        st.session_state.chat_mode = chat_mode

        st.session_state.language = st.selectbox(
            "Language",
            ["English", "Spanish", "French", "Hindi", "German", "Italian", "Portuguese"],
            index=0
        )

        if st.button("Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("### Chat with Documents")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"{'🧑' if message['role'] == 'user' else '🤖'} {message['content']}")

    try:
        if prompt := st.chat_input("Type your question here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(f"🧑 {prompt}")

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if st.session_state.chat_mode == "Agent-based RAG":
                        response = await react_orchestrator(
                            query=prompt,
                            namespace=str(st.session_state.user),
                            id=str(uuid.uuid4()),
                            language=st.session_state.language,
                        )
                        st.markdown(f"🤖 {response}")
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        memory_messages = [
                            Message(
                                type="human" if msg["role"] in ("user", "human") else "ai",
                                content=msg["content"]
                            )
                            for msg in st.session_state.messages[-10:]
                        ]

                        message_placeholder = st.empty()
                        full_response = ""

                        async with websockets.connect('ws://localhost:8081/ws/run_rag') as websocket:
                            await websocket.send(json.dumps({
                                "question": prompt,
                                "language": st.session_state.language,
                                "namespace": st.session_state.user,
                                "chat_context": [m.model_dump() for m in memory_messages]
                            }))

                            async for message in websocket:
                                try:
                                    data = json.loads(message)
                                    if data["type"] == "stream":
                                        full_response += data["message"]
                                        message_placeholder.markdown(full_response + "▌")
                                    elif data["type"] == "end":
                                        message_placeholder.markdown(full_response)
                                        break
                                    elif data["type"] == "error":
                                        raise Exception(data["message"])
                                except json.JSONDecodeError:
                                    continue

                            if full_response:
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": full_response
                                })

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if st.button("Retry"):
            st.rerun()


def main():
    init_session_state()

    if not st.session_state.user:
        login()
    else:
        st.sidebar.title(f"Welcome {st.session_state.user}")
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()

        tab1, tab2 = st.tabs(["Upload Documents", "Chat"])

        with tab1:
            upload_files()

        with tab2:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(chat_interface())
            finally:
                loop.close()


if __name__ == "__main__":
    main()