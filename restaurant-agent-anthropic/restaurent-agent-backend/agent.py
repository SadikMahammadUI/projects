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
