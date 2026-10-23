# Restaurant Assistant — Source Code

A FastAPI backend for a CrewAI- and Claude-powered restaurant recommendation assistant. There is no separate frontend: the interactive Swagger UI at `/docs` is the interface. Conversation memory is kept on the server, keyed by `session_id`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/chat` | Send a message. Optional `session_id` continues a conversation. Returns `reply`, `tool_calls`, `session_id`. |
| DELETE | `/chat/{session_id}` | Clear a conversation. |
| GET | `/health` | Health check. |
| GET | `/` | Redirects to `/docs`. |

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key. |
| `MODEL` | No | Claude model to use (default `claude-haiku-5-5`). |

## main.py

```python
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
```

## agent.py

```python
import asyncio
import json
import os

# CrewAI sends anonymous usage telemetry by default; keep it off unless the user opts in.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

import tools as restaurant_tools

MODEL = os.getenv("MODEL", "claude-haiku-5-5")

BACKSTORY = """
You are a helpful restaurant recommendation assistant.

You have access to these tools:

1. Search restaurants by cuisine
2. Get restaurant details
3. Check reservation availability

Use the appropriate tool whenever necessary.

Do not invent restaurant information.

If the available tools cannot answer the question,
clearly say that the requested information is outside
your available capabilities.

Be friendly, conversational, and concise.
"""


def _as_text(result) -> str:
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


def build_tools(tool_calls: list[dict]) -> list:
    """CrewAI tools wrapping the functions in tools.py; every call is recorded in tool_calls."""

    def record(name: str, output: str) -> str:
        tool_calls.append({"name": name, "output": output})
        return output

    @tool("search_restaurants_by_cuisine")
    def search_restaurants_by_cuisine(cuisine_type: str) -> str:
        """Search for restaurants by cuisine type. Supported cuisines: Indian, Italian, Chinese, Mexican."""
        result = restaurant_tools.search_restaurants_by_cuisine(cuisine_type)
        return record("search_restaurants_by_cuisine", _as_text(result))

    @tool("get_restaurant_details")
    def get_restaurant_details(restaurant_name: str) -> str:
        """Get detailed information about a restaurant, including address, phone, hours and specialties."""
        result = restaurant_tools.get_restaurant_details(restaurant_name)
        return record("get_restaurant_details", _as_text(result))

    @tool("check_reservation_availability")
    def check_reservation_availability(restaurant_name: str, date: str, party_size: int) -> str:
        """Check if a restaurant has reservation availability for a specific date (YYYY-MM-DD) and number of people."""
        result = restaurant_tools.check_reservation_availability(restaurant_name, date, party_size)
        return record("check_reservation_availability", _as_text(result))

    return [search_restaurants_by_cuisine, get_restaurant_details, check_reservation_availability]


def _run_crew(history: list[dict], message: str) -> tuple[str, list[dict]]:
    tool_calls: list[dict] = []

    agent = Agent(
        role="Restaurant recommendation assistant",
        goal="Help users find restaurants, get their details and check reservation availability.",
        backstory=BACKSTORY,
        tools=build_tools(tool_calls),
        llm=LLM(model=f"anthropic/{MODEL}", max_tokens=1024),
        max_iter=5,
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description=(
            "Conversation so far:\n{history}\n\n"
            "Latest user message:\n{message}\n\n"
            "Reply to the latest user message, using the conversation for context."
        ),
        expected_output="A friendly, concise reply to the user's latest message.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history) or "(none)"
    result = crew.kickoff(inputs={"history": transcript, "message": message})
    return result.raw, tool_calls


async def invoke_crew_agent(history: list[dict], message: str) -> tuple[str, list[dict]]:
    """Run the CrewAI agent (blocking) in a worker thread. Returns (reply, tool_calls)."""
    return await asyncio.to_thread(_run_crew, history, message)
```

## tools.py

```python
# ============================================================
# TOOL 1: Search Restaurants by Cuisine
# ============================================================

def search_restaurants_by_cuisine(cuisine_type: str) -> list:
    """
    Search for restaurants by cuisine type.
    Supported cuisines: Indian, Italian, Chinese, Mexican.
    """

    restaurants = {
        "indian": [
            {"name": "Spice Palace", "rating": 4.5, "price": "₹₹"},
            {"name": "Curry House", "rating": 4.3, "price": "₹₹"},
            {"name": "Taj Flavors", "rating": 4.7, "price": "₹₹₹"}
        ],
        "italian": [
            {"name": "Pizza Bella", "rating": 4.4, "price": "₹₹"},
            {"name": "Pasta Dreams", "rating": 4.6, "price": "₹₹₹"},
            {"name": "Roma Kitchen", "rating": 4.2, "price": "₹₹"}
        ],
        "chinese": [
            {"name": "Dragon House", "rating": 4.5, "price": "₹₹"},
            {"name": "Ming's Kitchen", "rating": 4.3, "price": "₹₹"},
            {"name": "Golden Wok", "rating": 4.8, "price": "₹₹₹"}
        ],
        "mexican": [
            {"name": "Taco Fiesta", "rating": 4.4, "price": "₹₹"},
            {"name": "Salsa Nights", "rating": 4.6, "price": "₹₹₹"},
            {"name": "Burrito House", "rating": 4.2, "price": "₹₹"}
        ]
    }

    cuisine_lower = cuisine_type.lower().strip()

    return restaurants.get(cuisine_lower, [])


# ============================================================
# TOOL 2: Get Restaurant Details
# ============================================================

def get_restaurant_details(restaurant_name: str) -> dict:
    """
    Get detailed information about a restaurant,
    including address, phone, hours and specialties.
    """

    details = {
        "Spice Palace": {
            "cuisine": "Indian",
            "rating": 4.5,
            "address": "123 Food Street, City Center",
            "phone": "9876543210",
            "hours": "11 AM - 11 PM",
            "speciality": "Butter Chicken, Biryani"
        },

        "Pizza Bella": {
            "cuisine": "Italian",
            "rating": 4.4,
            "address": "456 Dine Avenue, Downtown",
            "phone": "9123456789",
            "hours": "12 PM - 10:30 PM",
            "speciality": "Margherita Pizza, Tiramisu"
        },

        "Dragon House": {
            "cuisine": "Chinese",
            "rating": 4.5,
            "address": "789 Noodle Lane, Market Area",
            "phone": "9111222333",
            "hours": "11:30 AM - 11 PM",
            "speciality": "Hakka Noodles, Fried Rice"
        },

        "Taco Fiesta": {
            "cuisine": "Mexican",
            "rating": 4.4,
            "address": "321 Spice Row, Hip District",
            "phone": "9222333444",
            "hours": "1 PM - 12 AM",
            "speciality": "Fish Tacos, Guacamole"
        },

        "Curry House": {
            "cuisine": "Indian",
            "rating": 4.3,
            "address": "555 Flavor Street, Old Town",
            "phone": "9333444555",
            "hours": "11 AM - 10 PM",
            "speciality": "Paneer Tikka, Dal Makhani"
        },

        "Taj Flavors": {
            "cuisine": "Indian",
            "rating": 4.7,
            "address": "999 Taste Lane, Premium Zone",
            "phone": "9444555666",
            "hours": "12 PM - 11 PM",
            "speciality": "Biryani, Tandoori Chicken"
        },

        "Golden Wok": {
            "cuisine": "Chinese",
            "rating": 4.8,
            "address": "777 Wok Street, Central Market",
            "phone": "9555666777",
            "hours": "11 AM - 11 PM",
            "speciality": "Szechuan Noodles, Dim Sum"
        }
    }

    return details.get(
        restaurant_name.strip(),
        {"error": f"Restaurant '{restaurant_name}' not found"}
    )


# ============================================================
# TOOL 3: Check Reservation Availability
# ============================================================

def check_reservation_availability(
    restaurant_name: str,
    date: str,
    party_size: int
) -> str:
    """
    Check if a restaurant has reservation availability
    for a specific date and number of people.
    """

    available_slots = {
        "Spice Palace": {
            "2024-11-15": True,
            "2024-11-16": False,
            "2024-11-17": True
        },

        "Pizza Bella": {
            "2024-11-15": False,
            "2024-11-16": True,
            "2024-11-17": True
        },

        "Dragon House": {
            "2024-11-15": True,
            "2024-11-16": True,
            "2024-11-17": False
        },

        "Taco Fiesta": {
            "2024-11-15": True,
            "2024-11-16": True,
            "2024-11-17": True
        },

        "Curry House": {
            "2024-11-15": True,
            "2024-11-16": False,
            "2024-11-17": True
        },

        "Taj Flavors": {
            "2024-11-15": False,
            "2024-11-16": True,
            "2024-11-17": True
        }
    }

    if restaurant_name not in available_slots:
        return f"Restaurant '{restaurant_name}' not found."

    is_available = available_slots[restaurant_name].get(
        date,
        False
    )

    if is_available:
        return (
            f"✓ {restaurant_name} has availability "
            f"for {party_size} people on {date}"
        )

    return (
        f"✗ {restaurant_name} is fully booked "
        f"for {date}"
    )


# ============================================================
# Tool schemas (sent to Claude) and dispatch table
# ============================================================

tool_definitions = [
    {
        "name": "search_restaurants_by_cuisine",
        "description": (
            "Search for restaurants by cuisine type. "
            "Supported cuisines: Indian, Italian, Chinese, Mexican."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"cuisine_type": {"type": "string"}},
            "required": ["cuisine_type"],
        },
    },
    {
        "name": "get_restaurant_details",
        "description": (
            "Get detailed information about a restaurant, "
            "including address, phone, hours and specialties."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"restaurant_name": {"type": "string"}},
            "required": ["restaurant_name"],
        },
    },
    {
        "name": "check_reservation_availability",
        "description": (
            "Check if a restaurant has reservation availability "
            "for a specific date (YYYY-MM-DD) and number of people."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "restaurant_name": {"type": "string"},
                "date": {"type": "string"},
                "party_size": {"type": "integer"},
            },
            "required": ["restaurant_name", "date", "party_size"],
        },
    },
]

tool_functions = {
    "search_restaurants_by_cuisine": search_restaurants_by_cuisine,
    "get_restaurant_details": get_restaurant_details,
    "check_reservation_availability": check_reservation_availability,
}
```

## requirements.txt

```text
fastapi
uvicorn[standard]
crewai
anthropic
pydantic
python-dotenv
```

## .env.example

```bash
ANTHROPIC_API_KEY=your-api-key-here
MODEL=your-model-name-here
```
