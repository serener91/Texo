import os
from langfuse.openai import OpenAI
from langfuse import get_client, observe, propagate_attributes
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Generator
from dotenv import load_dotenv

from utils import get_logger

logger = get_logger(name="inference", log_file="../logs/inference.log")

load_dotenv()

# Load Langfuse client
langfuse = get_client()

def generate_deterministic_trace_id(session_id: str, turn_num: int) -> str:
    """
    Generate deterministic trace ID for a specific turn in a conversation.

    This ensures that retries for the same turn reuse the same trace container. (Idempotency)
    Format: {session_id}_{turn_number}

    Args:
        session_id: The conversation session identifier
        turn_num: The sequential turn number (1, 2, 3, ...)

    Returns:
        Deterministic trace ID string
    """
    seed = f"{session_id}_turn_{turn_num}"
    return langfuse.create_trace_id(seed=seed)


def get_conversation_turn(conversation):
    """
    Calculate the current turn in the conversation.
    Each turn consists of a user request and model response
    """

    return len(conversation) // 2+1


@observe()
def chat_completion_api(
        model_nm: str = None,
        messages: List[Dict[str, str]] = None,
        user_id: str = None,
        session_id: str = None,
        trace_name="assistant",
        **kwargs

) -> Generator[str, None, None]:
    """
    OpenAI Chat Completion API with proper multi-turn Langfuse tracing.

    Architecture:
    - session_id: Groups entire conversation (for Session Replay)
    - trace_id: Represents one user turn (format: {session_id}_{turn_number})

    Args:
        model_nm: model names to map to the host server
        messages: full conversation history : a list of message dicts with 'role' and 'content'
        user_id: User identifier for tracking
        session_id: Conversation thread identifier (chat session id from frontend)
        trace_name: Custom trace name for organization
        **kwargs: Additional OpenAI parameters (temperature, top_p, etc.)


    Yields:
        str: Text chunks from the streaming response
    """
    try:
        if os.getenv("OPENAI_API_KEY") is not None:
            client = OpenAI()
        else:
            # Add your model or use OpenAI (Highly recommend keeping configs separately for production)
            model_routers = {
                "gpt-oss-120b": os.getenv("LOCAL_SERVER")
            }

            client = OpenAI(
                base_url=model_routers.get(model_nm, os.getenv("LOCAL_SERVER")),
                api_key=os.getenv("LOCAL_API_KEY")
            )

        if messages is None:
            messages = [
                {"role": "system", "content": "You are a smart assistant."},
                {"role": "user", "content": "Say 'Ask me anything!'"},
            ]

        output_chunks = []
        reasoning_content = ""

        with propagate_attributes(
                user_id=user_id,
                session_id=session_id,
                metadata={"model_name": model_nm},
                tags=[model_nm],
                version="0.1",
        ):

            chat_response = client.chat.completions.create(
                model=os.getenv("LOCAL_MODEL", "gpt-5.1"),
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs
            )

            with langfuse.start_as_current_observation(
                    as_type="span",
                    name="reasoning-step",
                    input={"messages": messages}
            ) as reasoning_span:
                for chunk in chat_response:
                    if chunk.choices:
                        # Parse reasoning tokens
                        if hasattr(chunk.choices[0].delta, "reasoning") and chunk.choices[0].delta.reasoning is not None:
                            reasoning_content += chunk.choices[0].delta.reasoning

                        if chunk.choices[0].delta.content:
                            text = chunk.choices[0].delta.content
                            output_chunks.append(text)
                            yield text

                    else:
                        # Capture usage data from the final chunk
                        if hasattr(chunk, 'usage') and chunk.usage is not None:
                            token_usage = chunk.usage
                            logger.info("\n=====Tokens====\n")
                            logger.info(token_usage)

                reasoning_span.update(output=reasoning_content)

        langfuse.update_current_trace(
            input={"messages": messages},
            output="".join(output_chunks),
            session_id=session_id,
            user_id=user_id,
            tags=[model_nm],
            metadata={"model_name": model_nm},
            version="1.0",
        )

    except Exception as e:
        logger_error = f"Error in chat_completion_api: {str(e)}"
        logger.info(logger_error)
        traceback.print_exc()
        langfuse.update_current_generation(level="ERROR", status_message=str(e))
        yield f"[ERROR: {str(e)}]"
