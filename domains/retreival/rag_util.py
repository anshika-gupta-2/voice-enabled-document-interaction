import fastapi
from domains.retreival.chat_response import ChatResponse
from typing import Literal


async def send_message_over_websocket(
    websocket: fastapi.WebSocket,
    message: str,
    message_type: Literal["start", "stream", "end", "blob", "error", "info"] = "stream",
    content_type: (
        Literal[
            "optimised_question",
            "answer",
            "citations",
            "summary",
            "retrieval",
            "note",
            "resume_insight",
            "agent_interrupt",
            "intermittent_steps",
        ]
        | None
    ) = None,
):
    try:
        resp = ChatResponse(
            message=message, type=message_type, content_type=content_type
        )
        await websocket.send_json(resp.model_dump())
    except Exception as e:
        print(f"Error sending message over WebSocket: {e}")
        await websocket.close(code=1000)  #
