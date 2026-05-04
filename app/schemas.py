from typing import Any
from pydantic import BaseModel,Field

class AgentChatRequest(BaseModel):
    message : str = Field(...,description="User task or question")#field是加规则，...表示该字段必填

class AgentChatResponse(BaseModel):
    user_message:str
    selected_tool:str
    tool_arguments: dict[str,Any]
    tool_result: Any
    final_answer: str

