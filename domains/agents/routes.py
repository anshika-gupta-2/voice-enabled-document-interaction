from loguru import logger

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from fastapi import APIRouter, BackgroundTasks
from typing import List
from langchain_core.documents import Document

from domains.settings import config_settings
from domains.utils import get_chat_model
from domains.agents.tools import qna_tool, information_extraction_tool, summarize_content_tool


router = APIRouter(
    tags=["agents"],
    responses={404: {"description": "Not found"}},
    prefix="/agents",
)

@router.get("/react_orchestrator")
async def react_orchestrator(
        query: str,
        id: str,
        language: str = "english",
        namespace: str = config_settings.PINECONE_DEFAULT_DEV_NAMESPACE
):
    # Create partial functions for tools that need namespace
    async def qna_with_namespace(question: str) -> List[Document]:
        """
        Search the vector database for relevant documents based on the question.

        Args:
            question (str): The query to search for in the vector database

        Returns:
            List[Document]: List of relevant documents found in the database
        """
        return await qna_tool(question=question, namespace=namespace)

    # Create the tools list with the wrapped qna tool
    tools = [
        qna_with_namespace,  # Using namespace-aware version
        information_extraction_tool,
        summarize_content_tool
    ]

    # Get the model
    model = get_chat_model(model_key="CHAT_MODEL_NAME")

    # Create memory saver
    memory = MemorySaver()

    system_prompt = f"""
    You are a data retrieval agent that follows these precise rules:

    1. PRIMARY SEARCH (qna_with_namespace):
       - FIRST search vector database for ALL queries
       - If NO RESULTS found in vector database:
         * Explicitly state: "No results found in database"
         * AUTOMATICALLY proceed to web search
       - If results found:
         * Return database results immediately
         * Do not proceed to web search

    2. FALLBACK SEARCH (information_extraction_tool):
       - ONLY USE when vector database returns empty results
       - Execute web search for relevant information
       - Focus on factual, verifiable information
       - Always state: "Retrieved from web search:"

    3. CONTENT HANDLING (summarize_content_tool):
       - Use for results longer than 250 words
       - Keep key facts and details intact
       - Maintain clear source attribution

    4. LANGUAGE OUTPUT ({language}):
       - Always respond in {language}
       - Include translation notice if source is different
       - Format: "[Original Language] → {language}"

    RESPONSE FORMAT:
    - State source: "From Database:" or "From External Source:"
    - List tools used: "Tools: [tool names]"
    - Present exact retrieved information without interpretation
    - If no data found: "No relevant information found in [source]"
    - Give the answer in the same give language type [language]

    CORE RULES:
    - ALWAYS try vector database first
    - Auto-switch to web search if database empty
    - Never mix sources in single response
    - Only return tool-retrieved data
    - Maintain search order: Vector DB → Web
    """

    # Create the agent with built-in flow
    agent_executor = create_react_agent(
        model=model,
        tools=tools,
        state_modifier=system_prompt,
        checkpointer=memory,
        debug=True,
    )

    # Execute with config
    config = {"configurable": {"thread_id": id, "language": language}}
    final_result = None

    # Stream results
    async for step in agent_executor.astream(
            {
                "messages": [
                    HumanMessage(
                        content=query,
                        additional_kwargs={"thread_id": id, "language": language},
                        metadata={"namespace": namespace},
                    )
                ],
            },
            config
    ):
        if step.get("agent"):
            final_result = step.get("agent", {}).get("messages", [])[-1].content

    logger.info(f"Agent result: {final_result}")
    return final_result
