from datetime import datetime


def get_weather(city: str) -> dict:
    return {
        "city": city,
        "temperature_c": 28,
        "condition": "sunny",
        "source": "mocked_weather_provider"
    }


def calculator(expression: str) -> dict:
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not set(expression).issubset(allowed_chars):
            return {
                "error": "Expression contains unsupported characters"
            }

        result = eval(expression, {"__builtins__": {}})

        return {
            "expression": expression,
            "result": result
        }
    except Exception as exc:
        return {
            "error": str(exc)
        }


def get_current_time(location: str) -> dict:
    return {
        "location": location,
        "current_time": datetime.now().isoformat(),
        "source": "local_system_clock"
    }


TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculator": calculator,
    "get_current_time": get_current_time,
}