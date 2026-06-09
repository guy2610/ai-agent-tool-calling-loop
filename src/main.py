import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from schemas import FinalAnswer
from tools import TOOL_REGISTRY


MAX_ITERATIONS = 5


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


def run_tool_call(function_call):
    name = function_call.name
    args = dict(function_call.args or {})

    print(f"\n[tool call] {name}({args})")

    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return {"error": f"Unknown tool: {name}"}

    result = tool(**args)
    print(f"[tool result] {json.dumps(result, indent=2)}")
    return result


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


def run_agent(user_prompt):
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("MODEL", "gemini-2.5-flash")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")

    client = genai.Client(api_key=api_key)

    contents = [
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

    config = types.GenerateContentConfig(
        tools=build_tools(),
        temperature=0,
    )

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n========== iteration {iteration} ==========")

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        function_calls = extract_function_calls(response)

        if function_calls:
            contents.append(response.candidates[0].content)

            tool_response_parts = []

            for function_call in function_calls:
                result = run_tool_call(function_call)
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=function_call.name,
                        response={"result": result},
                    )
                )

            contents.append(
                types.Content(
                    role="tool",
                    parts=tool_response_parts,
                )
            )
            continue

        final_text = extract_text(response)
        print("\n[final raw response]")
        print(final_text)

        try:
            parsed = FinalAnswer.model_validate_json(final_text)
        except ValidationError as exc:
            raise RuntimeError(f"Final response did not match schema: {exc}") from exc

        print("\n[validated final answer]")
        print(parsed.model_dump_json(indent=2))
        return parsed

    raise RuntimeError("Agent reached max iterations without a final answer.")


if __name__ == "__main__":
    run_agent(
        "What is the weather in Tel Aviv, what is 24 * 7, "
        "and what is the current time in Israel? Return a concise summary."
    )