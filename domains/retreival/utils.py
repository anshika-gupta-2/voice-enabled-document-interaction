import fastapi
from typing import Optional, List, Any
from loguru import logger

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from domains.utils import get_chat_model_with_streaming
from domains.utils import get_chat_model


async def transform_user_query_for_retrieval(
        question: str,
        model_key: str = "OPTIMIZED_QUESTION_MODEL",
        chat_history=List[Any],
):
    try:
        template = """
    Input Description: A natural language user query about a specific topic.

    Transformation Guidelines: Convert the user query into a more effective retrieval query by following these steps:

    Keyword Focus: Identify and use specific keywords that are central to the topic.
    Synonyms and Variants: Include synonyms or related terms to cover variations in how the information might be phrased.
    Streamlining: Eliminate common stop words and unnecessary punctuation to improve search focus.
    Conciseness: Ensure the query is concise, ideally between 4-12 words, to maintain focus without over-specifying.
    Exact Matches: Enclose terms in quotes to enforce exact phrase matching when necessary to capture precise information.
    Special Handling for Irrelevant Queries:

    If the query is casual or non-informative (e.g., "Hi, how are you"), return None to indicate that the query does not require a retrieval-based response.
    Output: Deliver the optimized query formatted according to the above guidelines, or None if the query is identified as irrelevant.

    Example:
    Original User Query: "What are some good resources for learning python programming?"
    Optimized Query: "Python programming tutorials resources"

    Original User Query: "Hi, how are you?"
    Optimized Query: None

    Pointers for Using This Prompt:
    Be vigilant about distinguishing between informational queries and non-informative interactions.
    Ensure that the language model understands the importance of keyword density and relevance to improve the specificity of the search results.
    Regularly update the synonyms or related terms based on evolving language usage or new developments in the subject area.
    Be mindful of the balance between conciseness and informativeness; overly broad queries may retrieve too much irrelevant information, while overly narrow queries might miss useful content.

    CHAT HISTORY: {chat_history}
    USER QUERY : {question}

    Note : Only return the transformed query or "None" if a user is trying to do a small talk.
        """

        prompt = PromptTemplate(template=template, input_variables=["question"])
        output_parser = StrOutputParser()
        llm = get_chat_model(model_key=model_key)

        if llm:
            llm_chain = prompt | llm | output_parser
            answer_from_model: str = await llm_chain.ainvoke(
                {
                    "question": question,
                    "chat_history": chat_history,
                }
            )
            logger.info(f"Transformed query for retrieval - {answer_from_model}")

            return answer_from_model

    except Exception as e:
        logger.error(f"Error in retrieval query transformation - {e}")
        return None


async def optimize_user_query(
        websocket: fastapi.WebSocket,
        question,
        memory,
        pre_grounding_prompt,
        model_key: str = "OPTIMIZED_QUESTION_MODEL",
) -> str:
    output_parser = StrOutputParser()
    llm_chain = (
            pre_grounding_prompt
            | get_chat_model_with_streaming(websocket, model_key=model_key)
            | output_parser
    )

    optimised_question = await llm_chain.ainvoke(
        {
            "question": question,
            "chat_history": memory.buffer_as_str
        },
    )
    logger.info(f"Optimised question generated: {optimised_question}")
    return optimised_question
