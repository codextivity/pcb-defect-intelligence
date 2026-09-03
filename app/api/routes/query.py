# app/api/routes/query.py

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []

class QueryResponse(BaseModel):
    answer: str
    question: str

def parse_history(history: list[dict]) -> list:
    messages = []
    for item in history:
        if item["role"] == "human":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "ai":
            messages.append(AIMessage(content=item["content"]))
    return messages

@router.post("", response_model=QueryResponse)
async def query_quality_agent(request: Request, body: QueryRequest):
    """
    Ask the quality agent a question in natural language.

    Examples:
    - 'What is our yield rate today?'
    - 'Which defect type is most common?'
    - 'How many PCBs failed inspection?'
    - 'What improvements do you recommend?'
    """
    if request.app.state.agent is None:
        from app.core.agent import build_quality_agent
        request.app.state.agent = build_quality_agent()

    if not body.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    from app.core.agent import run_quality_agent
    chat_history = parse_history(body.history)
    answer = run_quality_agent(
        request.app.state.agent,
        body.question,
        chat_history
    )

    return QueryResponse(answer=answer, question=body.question)