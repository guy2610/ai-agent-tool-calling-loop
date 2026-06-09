from typing import Literal
from pydantic import BaseModel


class FinalAnswer(BaseModel):
    answer: str
    tools_used: list[str]
    confidence: Literal["low", "medium", "high"]