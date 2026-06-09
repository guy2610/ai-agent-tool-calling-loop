# AI Agent Tool Calling Loop

## Overview

This project demonstrates an AI agent capable of:

- Calling external tools
- Executing multiple sequential tool calls
- Maintaining an agent loop
- Producing structured JSON output
- Validating responses using Pydantic

The implementation uses Gemini and a simple tool-calling architecture.

## Available Tools

### get_weather

Returns mocked weather information for a requested city.

Example:

```json
{
  "city": "Tel Aviv",
  "temperature_c": 28,
  "condition": "sunny"
}
```

### calculator

Evaluates simple arithmetic expressions.

Example:

```json
{
  "expression": "24 * 7",
  "result": 168
}
```

### get_current_time

Returns the current local system time for a requested location.

Example:

```json
{
  "location": "Israel",
  "current_time": "2026-06-09T20:37:17"
}
```

---

## Architecture

The application consists of four main components:

### Tool Registry

Contains all available tools and their implementations.

### Agent Loop

Responsible for:

1. Sending user requests to Gemini
2. Detecting tool calls
3. Executing requested tools
4. Returning tool results to Gemini
5. Receiving the final structured response

### Structured Output Schema

Final responses are validated using Pydantic:

```json
{
  "answer": "string",
  "tools_used": ["string"],
  "confidence": "low | medium | high"
}
```

### Logging

Every iteration is logged to the console including:

- Tool calls
- Tool results
- Final response
- Validation output

---

## Execution Flow

```text
User Prompt
      |
      v
 Gemini
      |
      v
 Tool Call Requested
      |
      v
 Execute Tool
      |
      v
 Return Result To Gemini
      |
      v
 Final JSON Response
      |
      v
 Pydantic Validation
```

---

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
MODEL=gemini-2.5-flash
```

Run:

```bash
python src/main.py
```

---

## Example Output

```text
========== iteration 1 ==========

[tool call] get_weather(...)
[tool call] calculator(...)
[tool call] get_current_time(...)

========== iteration 2 ==========

{
  "answer": "...",
  "tools_used": [
    "get_weather",
    "calculator",
    "get_current_time"
  ],
  "confidence": "high"
}
```

---

## Bonus - LangGraph

The same agent was reimplemented using LangGraph.

LangGraph abstracts:
- agent loop management
- state handling
- tool routing

This reduces orchestration code while preserving the same tool-calling behavior.

---


## AI Usage

AI tools were used for brainstorming implementation ideas, discussing architecture options, and reviewing design decisions.

The implementation, debugging, testing, and final validation were completed manually.