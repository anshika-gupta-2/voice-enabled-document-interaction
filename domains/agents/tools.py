import asyncio
from typing import List
from loguru import logger
from domains.agents.utils import (
    initialize_doc_parser_chain,
    length_function,
    reduce_summary_chain,
    collect_summaries,
    should_collapse,
    map_summaries
)
from domains.agents.models import SummaryState, OverallState, QueryRequest
from langchain.chains.combine_documents.reduce import acollapse_docs, split_list_of_docs
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, START, StateGraph

from domains.settings import config_settings
from domains.vector_db.pinecone_utils import get_related_docs_with_score
from langchain_core.documents import Document
from domains.retreival.utils import transform_user_query_for_retrieval


async def qna_tool(
        question: str,
        namespace: str = config_settings.PINECONE_DEFAULT_DEV_NAMESPACE,
) -> List[Document]:
    """
    Retrieves and filters personal data and documents from Pinecone vector database based on relevance score.

    Args:
        request (QueryRequest): Contains query and namespace information

    Returns:
        List[Document]: Filtered list of relevant documents

    Raises:
        ValueError: If request parameters are invalid
        Exception: For any other unexpected errors
    """
    try:
        # Validate input
        if not question or not namespace:
            raise ValueError("Invalid request parameters")

        # Transform query for optimal retrieval
        transformed_query = await transform_user_query_for_retrieval(
            question
        )
        if not transformed_query:
            logger.warning("Query transformation returned empty result")
            transformed_query = question

        # Retrieve documents with scores
        related_docs_with_score = await get_related_docs_with_score(
            index_name=config_settings.PINECONE_INDEX_NAME,
            namespace=namespace,
            question=transformed_query,
            total_docs_to_retrieve=config_settings.PINECONE_TOTAL_DOCS_TO_RETRIEVE
        )

        # Filter documents based on minimum score
        filtered_docs = [
            Document(
                metadata=doc[0].metadata,
                page_content=doc[0].page_content
            )
            for doc in related_docs_with_score
            if doc[1] > config_settings.MINIMUM_SCORE
        ]

        logger.info(f"Retrieved {len(filtered_docs)} relevant documents")
        return filtered_docs[:3]

    except ValueError as ve:
        logger.error(f"Validation error in qna_tool: {ve}")
        raise

    except Exception as e:
        logger.exception("Unexpected error in qna_tool")
        raise Exception(f"QnA tool failed: {str(e)}")


async def information_extraction_tool(query: str) -> List[Document]:
    """
    Performs web search using Tavily API and extracts relevant content.

    Args:
        query (str): Search query string

    Returns:
        List[Document]: List of documents containing search results

    Raises:
        ValueError: If query is empty or invalid
        Exception: For API or processing failures
    """
    try:
        # Validate input
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        # Initialize search tool with configuration
        tavily_tool = TavilySearchResults(
            max_results=2,
        )
        response = await tavily_tool.ainvoke(query)

        # Process results
        if not response:
            logger.info("No results found from Tavily search")
            return []

        # Extract and transform results
        documents = [
            Document(
                metadata={"url": result.get("url")},
                page_content=result.get("content", "").strip()
            )
            for result in response
            if result.get("content") and result.get("url")
        ]

        logger.info(f"Retrieved {len(documents)} documents from Tavily")
        return documents

    except ValueError as ve:
        logger.error(f"Invalid input: {ve}")
        raise

    except Exception as e:
        logger.exception("Failed to extract information from Tavily")
        raise Exception(f"Information extraction failed: {str(e)}")


async def summarize_content_tool(content: List[Document]) -> str:
    """
    Description:
    This function summarizes a list of documents using a state-based summarization pipeline. It extracts summaries from individual documents, merges them iteratively, and generates a final summary.

    Parameters:
    content (List[Document]): A list of Document objects to be summarized.

    Returns:
    A str containing the final summarized content.
    """
    async def generate_summary(state: SummaryState):
        try:
            logger.info("Generating summary for content")
            response = await initialize_doc_parser_chain().ainvoke(state["content"])
            logger.info("Generated summary successfully")
            return {"summaries": [response]}

        except Exception as e:
            logger.exception("Failed to generate summary")
            raise e

    async def collapse_summaries(state: OverallState):
        try:
            logger.info("Collapsing summaries")
            doc_lists = split_list_of_docs(
                state["collapsed_summaries"], length_function, config_settings.MAX_TOKENS
            )
            results = []
            for doc_list in doc_lists:
                results.append(await acollapse_docs(doc_list, reduce_summary_chain().ainvoke))
            logger.info("Collapsed summaries successfully")
            return {"collapsed_summaries": results}

        except Exception as e:
            logger.exception("Failed to collapse summaries")
            raise e

    async def generate_final_summary(state: OverallState):
        try:
            logger.info("Generating final summary")
            response = await reduce_summary_chain().ainvoke(state["collapsed_summaries"])
            logger.info("Generated final summary successfully")
            return {"final_summary": response}
        except Exception as e:
            logger.exception("Failed to generate final summary")
            raise e

    try:
        graph = StateGraph(OverallState)
        graph.add_node("generate_summary", generate_summary)
        graph.add_node("collect_summaries", collect_summaries)
        graph.add_node("collapse_summaries", collapse_summaries)
        graph.add_node("generate_final_summary", generate_final_summary)

        graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
        graph.add_edge("generate_summary", "collect_summaries")
        graph.add_conditional_edges("collect_summaries", should_collapse)
        graph.add_conditional_edges("collapse_summaries", should_collapse)
        graph.add_edge("generate_final_summary", END)

        app = graph.compile()

        async for step in app.astream(
                {"contents": [doc.page_content for doc in content]},
                {"recursion_limit": 10},
        ):
            if "generate_final_summary" in step:
                return step['generate_final_summary']["final_summary"]

        else:
            return step

    except Exception as e:
        logger.exception("Summarization tool failed")
        raise e


async def run_qna_tool(state: OverallState):
    """Fetches documents from the vector database."""
    try:
        request = QueryRequest(query=state['query'])
        documents = await qna_tool(request)
        state["documents"] = documents
        return {"documents": documents}
    except Exception as e:
        logger.exception("Failed to run qna_tool")
        raise e


async def run_information_extraction_tool(state: OverallState):
    """Runs information extraction if vector database results are insufficient."""
    try:
        extracted_docs = await information_extraction_tool(state["query"])
        return {"documents": state["documents"] + extracted_docs}
    except Exception as e:
        logger.exception("Failed to run information_extraction_tool")
        raise e


async def run_summarize_content_tool(state: OverallState):
    """Summarizes the final set of documents."""
    try:
        summary = await summarize_content_tool(state["documents"])
        return {"final_summary": summary}
    except Exception as e:
        logger.exception("Failed to run summarize_content_tool")
        raise e


async def orchestrator_agent(query: str) -> str:
    """
    Orchestrates the workflow based on vector database results.

    - This search the vector database first and if the infromation not found then fetches from internet and summarize them and give the
    consise summary of the documents.

    - All query needs to be handled by this agent.
    - It will call qna_tool, information_extraction_tool, and summarize_content_tool.
    """
    try:
        graph = StateGraph(OverallState)
        graph.add_node("run_qna_tool", run_qna_tool)
        graph.add_node("run_information_extraction_tool", run_information_extraction_tool)
        graph.add_node("run_summarize_content_tool", run_summarize_content_tool)

        graph.add_edge(START, "run_qna_tool")
        graph.add_conditional_edges(
            "run_qna_tool",
            lambda state: "run_summarize_content_tool" if len(
                state["documents"]) >= 5 else "run_information_extraction_tool"
        )
        graph.add_edge("run_information_extraction_tool", "run_summarize_content_tool")
        graph.add_edge("run_summarize_content_tool", END)

        app = graph.compile()
        async for step in app.astream({"query": query, "documents": []}, {"recursion_limit": 10}):
            if "final_summary" in step:
                return step["final_summary"]

        raise Exception("Failed to generate final summary")
    except Exception as e:
        logger.exception("Orchestrator agent failed")
        raise e


if __name__ == "__main__":
    res = asyncio.run(
        information_extraction_tool(
            "What is the capital of France?"
        )
    )
    print(res)