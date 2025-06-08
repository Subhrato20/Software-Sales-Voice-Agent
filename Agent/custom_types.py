# Agent/custom_types.py
from pydantic import BaseModel
from typing import Any, Dict, List

class ConfigResponse(BaseModel):
    config: Dict[str, Any]
    response_id: int

class Utterance(BaseModel):
    role: str
    content: str

class ResponseRequiredRequest(BaseModel):
    interaction_type: str
    response_id: int
    transcript: List[Utterance]

class ResponseResponse(BaseModel):
    response_id: int
    content: str
    content_complete: bool
    end_call: bool
