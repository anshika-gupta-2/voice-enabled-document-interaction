from loguru import logger
from domains.settings import config_settings

from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from domains.retreival.models import Message
from langchain_core.messages import HumanMessage, AIMessage


def initialise_memory_from_chat_context(chat_context, input_key: str = None):
    return __load_chat_context(chat_context, input_key)


def __load_chat_context(chat_context, input_key: str = None):
    """Load chat context into ConversationBufferMemory."""
    memory = ConversationBufferWindowMemory(
        memory_key=config_settings.CONVERSATIONAL_BUFFER_WINDOW_MEMORY_KEY,
        return_messages=True,
        k=config_settings.LANGCHAIN_MEMORY_BUFFER_WINDOW,
        input_key=input_key or config_settings.CONVERSATIONAL_BUFFER_WINDOW_INPUT_KEY,
        chat_memory=ChatMessageHistory()
    )

    if not chat_context:
        return memory

    logger.info("Loading context from chat context")
    for message_dict in chat_context:
        # Convert dict to Message object if needed
        message = (
            message_dict if isinstance(message_dict, Message)
            else Message(**message_dict)
        )

        # Add messages to memory
        if message.type == config_settings.CHAT_CONTEXT_HUMAN_MESSAGE_KEY:
            memory.chat_memory.add_message(HumanMessage(content=message.content))
        elif message.type == config_settings.CHAT_CONTEXT_AI_MESSAGE_KEY:
            memory.chat_memory.add_message(AIMessage(content=message.content))

    return memory