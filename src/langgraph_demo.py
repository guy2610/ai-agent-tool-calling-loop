import json
import os
from typing import TypedDict, List, Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END

from schemas import FinalAnswer
from tools import TOOL_REGISTRY


MAX_ITERATIONS = 5


class AgentState(TypedDict):
    contents: List[Any]
    iteration: int
    final_answer: Optional[FinalAnswer]


def build_tools():
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="get_weather",
                    description="Get mocked weather information for a city.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                ),
                types.FunctionDeclaration(
                    name="calculator",
                    description="Evaluate a simple arithmetic expression.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                    },
                ),
                types.FunctionDeclaration(
                    name="get_current_time",
                    description="Get mocked/local current time for a location.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                ),
            ]
        )
    ]


def extract_function_calls(response):
    calls = []

    for candidate in response.candidates or []:
        content = candidate.content
        if not content:
            continue

        for part in content.parts or []:
            if part.function_call:
                calls.append(part.function_call)

    return calls

def clean_json_text(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()

    if text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return text

def extract_text(response):
    texts = []

    for candidate in response.candidates or []:
        content = candidate.content
        if not content:
            continue

        for part in content.parts or []:
            if part.text:
                texts.append(part.text)

    return "\n".join(texts).strip()


def call_tool(function_call):
    name = function_call.name
    args = dict(function_call.args or {})

    print(f"\n[langgraph tool call] {name}({args})")

    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        result = {"error": f"Unknown tool: {name}"}
    else:
        result = tool(**args)

    print(f"[langgraph tool result] {json.dumps(result, indent=2)}")

    return types.Part.from_function_response(
        name=name,
        response={"result": result},
    )


def agent_node(state: AgentState):
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("MODEL", "gemini-2.5-flash")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")

    client = genai.Client(api_key=api_key)

    iteration = state["iteration"] + 1
    print(f"\n========== langgraph iteration {iteration} ==========")

    response = client.models.generate_content(
        model=model,
        contents=state["contents"],
        config=types.GenerateContentConfig(
            tools=build_tools(),
            temperature=0,
        ),
    )

    function_calls = extract_function_calls(response)

    if function_calls:
        contents = list(state["contents"])
        contents.append(response.candidates[0].content)

        tool_parts = [call_tool(function_call) for function_call in function_calls]

        contents.append(
            types.Content(
                role="tool",
                parts=tool_parts,
            )
        )

        return {
            "contents": contents,
            "iteration": iteration,
            "final_answer": None,
        }

    final_text = extract_text(response)

    print("\n[langgraph final raw response]")
    print(final_text)

    parsed = FinalAnswer.model_validate_json(clean_json_text(final_text))

    print("\n[langgraph validated final answer]")
    print(parsed.model_dump_json(indent=2))

    return {
        "contents": state["contents"],
        "iteration": iteration,
        "final_answer": parsed,
    }


def should_continue(state: AgentState):
    if state["final_answer"] is not None:
        return END

    if state["iteration"] >= MAX_ITERATIONS:
        raise RuntimeError("LangGraph agent reached max iterations without a final answer.")

    return "agent"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)

    return graph.compile()


def run_langgraph_agent(user_prompt: str):
    initial_contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=(
                        "You are an AI agent. Use tools when needed. "
                        "When you have enough information, return ONLY valid JSON with this schema: "
                        '{"answer": "string", "tools_used": ["string"], "confidence": "low|medium|high"}. '
                        f"User request: {user_prompt}"
                    )
                )
            ],
        )
    ]

    app = build_graph()

    final_state = app.invoke(
        {
            "contents": initial_contents,
            "iteration": 0,
            "final_answer": None,
        }
    )

    return final_state["final_answer"]


if __name__ == "__main__":
    run_langgraph_agent(
        "What is the weather in Tel Aviv, what is 24 * 7, "
        "and what is the current time in Israel? Return a concise summary."
    )