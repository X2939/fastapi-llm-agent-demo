import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.schemas import AgentChatRequest,AgentChatResponse
from app.agent import run_agent

app=FastAPI(
    title="FastAPI LLM Agent Demo",
    description="A minimal tool-calling Agent demo built with FastAPI and an OpenAI-compatible LLM API",
    version="0.1.0",
)

@app.get("/health")
def health_check():
    return {
        "status":"ok",
        "message":"FastAPI service is running",
    }


@app.post("/agent/chat_fake",response_model=AgentChatResponse)
def agent_chat(request: AgentChatRequest):
    return AgentChatResponse(
        user_message=request.message,
        selected_tool="none",
        tool_arguments={},
        tool_result=None,
        final_answer="Agent is not implemented yet",
    )

@app.post("/agent/chat",response_model=AgentChatResponse)
def agent_chat(request:AgentChatRequest):
    return run_agent(request.message)

@app.post("/agent/stream")
def agent_stream(request:AgentChatRequest):
    def event_generator():
        yield _to_sse({"event":"start","user_message":request.message})

        response = run_agent(request.message)

        yield _to_sse(
            {
                "event":"tool_selected",
                "selected_tool":response.selected_tool,
                "tool_arguments":response.tool_arguments,
            }
        )

        yield _to_sse(
            {
                "event": "tool_result",
                "tool_result": response.tool_result,
            }
        )

        yield _to_sse(
            {
                "event":"final_answer",
                "final_answer":response.final_answer,
            }
        )

        yield _to_sse({"event":"done"})
    return StreamingResponse(event_generator(),media_type="text/event-stream")
    
def _to_sse(data:dict)->str:
        return f"data:{json.dumps(data,ensure_ascii=False)}\n\n"
