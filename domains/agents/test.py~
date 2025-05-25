import pprint
import asyncio
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from loguru import logger

from domains.agents.models import QueryRequest, OverallState


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
    """Orchestrates the workflow based on vector database results."""
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


memory = MemorySaver()


async def new(query: str, id):
    agent_executor = create_react_agent(
        model=get_chat_model(model_key="OPENAI_CHAT"),
        tools=[orchestrator_agent],
        checkpointer=memory,
    )
    config = {"configurable": {"thread_id": id}}
    async for step in agent_executor.astream(
            {"messages": [HumanMessage(content=query)]},
            config,
            stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from domains.utils import get_chat_model
from domains.agents.tools import qna_tool, information_extraction_tool, summarize_content_tool


async def create_react_orchestrator(query: str, id: str):
    # Create the tools
    tools = [qna_tool, information_extraction_tool, summarize_content_tool]

    # Get the model
    model = get_chat_model(model_key="OPENAI_CHAT")

    # Create memory saver
    memory = MemorySaver()

    # Define the system prompt
    system_prompt = """
    You are an orchestrator agent responsible for handling all queries using a structured retrieval and summarization process. Follow this workflow for every query:

    1. Search the vector database tool for relevant answers.
    2. If no relevant answer is found, fetch information from the internet using information tool.
    3. Summarize the retrieved documents into a concise and clear response.
    
    Additionally:
    - Always specify which tools were used to retrieve and summarize the information.
    - Do not make assumptions—base responses strictly on retrieved data.
    Ensure accuracy, relevance, and brevity in all responses.
    """

    # Create the agent with built-in flow
    agent_executor = create_react_agent(
        model=model,
        tools=tools,
        state_modifier=system_prompt,
        checkpointer=memory,
    )

    # Execute with config
    config = {"configurable": {"thread_id": id}}
    final_result = None

    # Stream results
    async for step in agent_executor.astream(
            {
                "messages": [HumanMessage(content=query)],
            },
            config
    ):
        if step.get("agent"):
            final_result = step.get("agent", {}).get("messages", [])[-1].content

    logger.info(f"Agent result: {final_result}")
    return final_result


if __name__ == "__main__":

    re=asyncio.run(
        create_react_orchestrator(
            query="candidate work experience?",
            id='asd12'
        )
    )

    print('NER RES')
    pprint.pprint(re)