"""
OpenAI-compatible API server with:
- Bearer token authentication
- Streaming and non-streaming chat completions
- Langfuse integration via utils.py
- Request validation with Pydantic
"""

from typing import List, Optional, Dict, Any
import uvicorn
import time
import os
import json
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from pydantic import BaseModel, Field
from datetime import datetime
from dotenv import load_dotenv

from inference import chat_completion_api
from utils import get_logger

load_dotenv()

# Set up a logger
logger = get_logger("openai_api_server", log_file="../logs/openai_api_server.log")

# Initialize FastAPI app
app = FastAPI(
    title="OpenAI Compatible Server",
    version="0.0.1",
    docs_url=None,
    redoc_url="/docs"
)


# ============================================================================
# Configuration
# ============================================================================

def get_valid_users() -> Dict[str, str]:
    """
    Load valid API keys from environment.
    Returns dict mapping api_key -> user_id
    """
    env_users = os.getenv("VALID_API_KEYS", None)
    if env_users:
        # Extract user_id from key by removing "sk-" prefix
        return {key.strip(): key.strip()[3:] for key in env_users.split(",") if key.strip()}

    # Fallback to defaults (development only)
    logger.warning("Using default API keys - not recommended for production!")
    return {
        "sk-coder": "coder",
        "sk-yoga": "yoga",
        "sk-pilates": "pilates",
        "sk-ski": "ski"
    }


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get a list of available models from environment or defaults.
    Format: Model_name1:Provider1,Model_name2:Provider2
    """
    models_env = os.getenv("AVAILABLE_MODELS", None)
    models = []

    if models_env:
        # extract model name and provider from the env
        for model_str in models_env.split(","):
            if ":" in model_str:
                name, owner = model_str.strip().split(":", 1)
                models.append({
                    "id": name,
                    "type": "oss",
                    "created": int(time.time()),
                    "owned_by": owner
                })

    # Fallback to defaults
    if not models:
        models = [
            {
                "id": "GPT-5.1",
                "object": "proprietary",
                "created": 1687882411,
                "owned_by": "OpenAI"
            }
        ]

    return models


# ============================================================================
# Pydantic Models
# ============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.6, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[List[str]] = None
    stream: Optional[bool] = True


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


# ============================================================================
# Authentication
# ============================================================================

def verify_api_key(authorization: str = Header(None)) -> str:
    """
    Verify Bearer token and return user_id.
    Raises HTTPException if authentication fails.
    """
    if not authorization:
        logger.warning("Request with missing Authorization header")
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )

    # Create a Bearer token authentication
    authorization = "Bearer " + authorization
    if not authorization.startswith("Bearer "):
        logger.warning(f"Invalid authorization format: {authorization[:20]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization format. Use 'Bearer <API_KEY>'"
        )

    # Extract api key
    api_key = authorization.split(" ", 1)[1]

    # Get all username (=api key) from config
    valid_users = get_valid_users()

    if api_key not in valid_users:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(status_code=403, detail="Invalid API key")

    user_id = valid_users[api_key]
    logger.info(f"Authenticated user: {user_id}")
    return user_id


# ============================================================================
# Helper Functions
# ============================================================================

def normalize_messages(messages_data: List[Dict]) -> None:
    """
    Normalize message content to a string format.
    Handles structured content (list of blocks) by flattening to text.
    Modifies messages in-place.
    """
    for msg in messages_data:
        content = msg.get("content")
        if isinstance(content, list):
            # Flatten structured content to a single string
            msg["content"] = "".join(
                block.get("text", "")
                for block in content
                if block.get("type") == "text"
            )


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (1 token ≈ 4 characters for English text).
    For production, use the model's tokenizer or library like a tiktoken.
    """
    return max(1, len(text) // 4)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with basic server info"""
    return {
        "message": "OpenAI Compatible Server",
        "version": "1.0.0",
        "endpoints": [
            "/v1/chat/completions",
            "/v1/models"
        ]
    }


@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {"data": get_available_models()}


@app.post("/v1/chat/completions")
async def create_chat_completion(
        request: Request,
        user_id: str = Depends(verify_api_key)
):
    """
    OpenAI-compatible chat completions endpoint.
    Supports both streaming and non-streaming responses.
    """
    # Parse raw JSON
    try:
        data = await request.json()
    except Exception as e:
        logger.exception("Failed to decode request JSON")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Normalize structured content to strings
    normalize_messages(data.get("messages", []))

    # Validate with Pydantic
    try:
        parsed_request = ChatCompletionRequest(**data)
    except Exception as e:
        logger.exception(f"Request validation failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    response_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    # Streaming response
    if parsed_request.stream:
        def generate_stream():
            try:
                for chunk_text in chat_completion_api(
                        model_nm=parsed_request.model,
                        messages=[msg.model_dump() for msg in parsed_request.messages],
                        user_id=user_id,
                        session_id=f"chat-{user_id}-{datetime.now().strftime('%Y-%m-%d')}",
                        **{
                            "temperature": parsed_request.temperature,
                            "top_p": parsed_request.top_p,
                            "frequency_penalty": parsed_request.frequency_penalty,
                            "presence_penalty": parsed_request.presence_penalty
                        }

                ):
                    chunk_data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": parsed_request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk_text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                # Final chunk
                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": parsed_request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception("Error during streaming")
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "server_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    # Non-streaming response
    else:
        response_text = ""
        try:
            for chunk in chat_completion_api(
                    messages=[msg.dict() for msg in parsed_request.messages],
                    user_id=user_id,
                    session_id=f"{time.strftime('%Y-%m-%d')}",
                    **{
                        "temperature": parsed_request.temperature ,
                        "top_p": parsed_request.top_p,
                        "frequency_penalty": parsed_request.frequency_penalty,
                        "presence_penalty": parsed_request.presence_penalty
                    }
            ):
                response_text += chunk
        except Exception as e:
            logger.exception("Error generating response")
            raise HTTPException(status_code=500, detail="Error generating response")

        # Estimate tokens
        prompt_text = " ".join([msg.content for msg in parsed_request.messages])
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(response_text)

        return ChatCompletionResponse(
            id=response_id,
            created=created,
            model=parsed_request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens+completion_tokens
            )
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
