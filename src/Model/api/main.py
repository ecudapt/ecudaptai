# src/Model/api/main.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.agent import agent_reply  # your agent scaffold


# --------- FastAPI setup ---------

app = FastAPI(title="ECUDapt Agent API")

# Adjust origins once you know your exact frontend URLs
origins = [
    "http://localhost:5173",   # local dev (Vite)
    "https://ecudapt.com",     # production site
    "https://www.ecudapt.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------- Schemas ---------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str


# --------- Routes ---------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/agent/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    - session_id: string that groups a conversation (can be derived from user_id)
    - message: user's message
    """
    try:
        result = agent_reply(
            session_id=request.session_id,
            user_message=request.message,
        )
        return ChatResponse(
            session_id=request.session_id,
            answer=result["answer"],
        )
    except Exception as e:
        # Don't leak full stack traces to client, just message for now
        raise HTTPException(status_code=500, detail=str(e))
