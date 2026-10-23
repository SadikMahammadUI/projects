import logging
import uuid

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from agent import invoke_crew_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_SESSIONS = 200
MAX_HISTORY_MESSAGES = 30

# session_id -> plain-text turns only: [{"role": "user"|"assistant", "content": str}, ...]
SESSIONS: dict[str, list[dict]] = {}

API_DESCRIPTION = """
A restaurant recommendation assistant powered by Claude and CrewAI. Use this page to chat with it.

**How to use**

1. `POST /chat` -> **Try it out** -> type a message -> **Execute**.
2. Copy the `session_id` from the response.
3. Paste it into `session_id` on your next request to continue the conversation.
4. `DELETE /chat/{session_id}` to clear the conversation and start over.
"""

app = FastAPI(title="Restaurant Assistant API", description=API_DESCRIPTION)


class ChatRequest(BaseModel):
    message: str = Field(examples=["Show me Italian restaurants"])
    session_id: str | None = Field(
        default=None,
        description=(
            "Leave empty to start a new conversation. To continue one, paste the "
            "session_id returned by a previous response."
        ),
    )


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict]
    session_id: str


def save_turn(session_id: str, history: list[dict], user_text: str, reply: str) -> None:
    """Store the plain-text turn, keep the newest sessions, and cap the history."""
    history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply},
    ]
    history = history[-MAX_HISTORY_MESSAGES:]
    # The API needs the conversation to start with a user message.
    while history and history[0]["role"] != "user":
        history = history[1:]

    SESSIONS.pop(session_id, None)  # re-insert so recent sessions stay newest
    SESSIONS[session_id] = history
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.pop(next(iter(SESSIONS)))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex
    history = SESSIONS.get(session_id, [])

    try:
        reply, tool_calls = await invoke_crew_agent(history, req.message)
    except Exception:
        logger.exception("CrewAI agent invocation failed")
        raise HTTPException(
            status_code=502,
            detail="The restaurant assistant could not process your request.",
        )

    if reply.strip():  # an empty assistant message would be invalid for the API
        save_turn(session_id, history, req.message, reply)

    return ChatResponse(reply=reply, tool_calls=tool_calls, session_id=session_id)


@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    """Clear a conversation so the next message starts fresh."""
    cleared = SESSIONS.pop(session_id, None) is not None
    return {"session_id": session_id, "cleared": cleared}


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "ok"}
