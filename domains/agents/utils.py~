from langgraph.constants import Send
from loguru import logger
from typing import List, Literal
from langchain_core.documents import Document
from domains.utils import get_chat_model
from domains.settings import config_settings
from langchain_core.prompts import ChatPromptTemplate
from domains.agents.models import OverallState
from domains.agents.prompt import DOC_PARSER_PROMPT, DISTILL_SUMMARY_PROMPT

from langchain_core.output_parsers import StrOutputParser

def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents."""
    total_number_of_tokens = sum(get_chat_model(
        model_key=config_settings.LLMS.get("OPENAI_CHAT")
    ).get_num_tokens(doc.page_content) for doc in documents)

    logger.info("Total number of tokens: {}".format(total_number_of_tokens))
    return total_number_of_tokens


def initialize_doc_parser_chain():
    return (
            ChatPromptTemplate.from_messages(
                [("human", DOC_PARSER_PROMPT)]) |
            get_chat_model(
                model_key=config_settings.LLMS.get("OPENAI_CHAT", "OPENAI_CHAT")) |
            StrOutputParser()
            )

def reduce_summary_chain():
    return (
            ChatPromptTemplate.from_messages(
                [("human", DISTILL_SUMMARY_PROMPT)]) |
            get_chat_model(
                model_key=config_settings.LLMS.get("OPENAI_CHAT")) |
            StrOutputParser()
            )


def map_summaries(state: OverallState):
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]


def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }

def should_collapse(
    state: OverallState,
) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > config_settings.MAX_TOKEN_LIMIT:
        return "collapse_summaries"
    else:
        return "generate_final_summary"