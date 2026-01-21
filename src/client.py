"""
OpenAI-compatible API client for testing the server.
Supports both streaming and non-streaming chat completions.
"""

import os
import sys
import json
import requests
import argparse
from typing import List, Dict, Optional, Generator
from utils import get_logger

# Set up logger
logger = get_logger("openai_api_client", log_file="../logs/openai_api_client.log")


class OpenAICompatibleClient:
    """Client for testing OpenAI-compatible API servers."""

    def __init__(
            self,
            base_url: str = "http://localhost:8000/v1",
            api_key: str = "sk-master",
            model: str = "gpt-oss-120b",
            timeout: int = 30
    ):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API server (without trailing slash)
            api_key: API key for authentication
            model: Your model name
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.headers = {
            "Authorization": api_key,
            "Content-Type": "application/json"
        }

    def chat_completion(
            self,
            messages: List[Dict[str, str]],
            stream: bool = False,
            **kwargs
    ) -> Dict | Generator[str, None, None]:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            For non-streaming: dict with full response
            For streaming: generator yielding text chunks
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "frequency_penalty": kwargs.get("frequency_penalty"),
            "presence_penalty": kwargs.get("presence_penalty")
        }

        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        url = f"{self.base_url}/chat/completions"

        try:
            if stream:
                return self._stream_chat_completion(url, payload)
            else:
                return self._non_stream_chat_completion(url, payload)
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def _non_stream_chat_completion(self, url: str, payload: Dict) -> Dict:
        """Handle non-streaming chat completion."""
        logger.info(f"Sending non-streaming request to {url}")

        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout
        )

        if response.status_code != 200:
            logger.error(f"Request failed with status {response.status_code}: {response.text}")
            response.raise_for_status()

        return response.json()

    def _stream_chat_completion(self, url: str, payload: Dict) -> Generator[str, None, None]:
        """Handle streaming chat completion."""
        logger.info(f"Sending streaming request to {url}")

        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            stream=True,
            timeout=self.timeout
        )

        if response.status_code != 200:
            logger.error(f"Request failed with status {response.status_code}: {response.text}")
            response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            decoded_line = line.decode('utf-8')

            # Handle SSE data lines
            if decoded_line.startswith('data: '):
                data = decoded_line[6:].strip()

                # Check for stream end marker
                if data == '[DONE]':
                    logger.info("Stream completed successfully")
                    break

                # Parse JSON chunk
                try:
                    parsed_data = json.loads(data)

                    # Check for errors
                    if 'error' in parsed_data:
                        error_msg = parsed_data['error'].get('message', 'Unknown error')
                        logger.error(f"Server error: {error_msg}")
                        yield f"\n[ERROR: {error_msg}]\n"
                        break

                    # Extract content from delta
                    if 'choices' in parsed_data and parsed_data['choices']:
                        delta = parsed_data['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            yield content

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON: {data} - {e}")
                    continue
            else:
                # Handle non-SSE formatted errors
                try:
                    error_data = json.loads(decoded_line)
                    if 'detail' in error_data:
                        logger.error(f"API error: {error_data['detail']}")
                        yield f"\n[ERROR: {error_data['detail']}]\n"
                except json.JSONDecodeError:
                    logger.warning(f"Unexpected line format: {decoded_line}")

    def list_models(self) -> Dict:
        """List available models."""
        url = f"{self.base_url}/models"
        logger.info(f"Fetching models from {url}")

        response = requests.get(
            url,
            headers=self.headers,
            timeout=self.timeout
        )

        if response.status_code != 200:
            logger.error(f"Request failed with status {response.status_code}")
            response.raise_for_status()

        return response.json()


def test_streaming(client: OpenAICompatibleClient, prompt: str):
    """Test streaming chat completion."""
    print("\n"+"=" * 60)
    print("Testing Streaming Chat Completion")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")
    print("Response:", end=" ", flush=True)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        for chunk in client.chat_completion(
                messages=messages,
                stream=True,
                temperature=0.7
        ):
            print(chunk, end='', flush=True)
        print("\n")
    except Exception as e:
        logger.exception("Streaming test failed")
        print(f"\n[ERROR: {e}]\n")


def test_non_streaming(client: OpenAICompatibleClient, prompt: str):
    """Test non-streaming chat completion."""
    print("\n"+"=" * 60)
    print("Testing Non-Streaming Chat Completion")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            stream=False,
            temperature=0.7
        )

        print("Full Response:")
        print(json.dumps(response, indent=2))

        # Extract and print just the content
        if 'choices' in response and response['choices']:
            content = response['choices'][0]['message']['content']
            print(f"\nAssistant: {content}\n")
    except Exception as e:
        logger.exception("Non-streaming test failed")
        print(f"\n[ERROR: {e}]\n")


def test_list_models(client: OpenAICompatibleClient):
    """Test listing available models."""
    print("\n"+"=" * 60)
    print("Testing Model Listing")
    print("=" * 60)

    try:
        models = client.list_models()
        print(json.dumps(models, indent=2))
        print()
    except Exception as e:
        logger.exception("Model listing test failed")
        print(f"\n[ERROR: {e}]\n")


def main():
    """Main test runner with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Test OpenAI-compatible API server"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("API_BASE_URL", "http://localhost:30080/v1"),
        help="Base URL of the API server"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("API_KEY", "sk-master"),
        help="API key for authentication"
    )
    parser.add_argument(
        "--prompt",
        default="List important Python features related to system design.",
        help="Prompt to send to the model"
    )
    parser.add_argument(
        "--mode",
        choices=["streaming", "non-streaming", "models", "all"],
        default="streaming",
        help="Test mode to run"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds"
    )

    args = parser.parse_args()

    # Initialize client
    client = OpenAICompatibleClient(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout
    )

    print(f"\nConnecting to: {args.base_url}")
    print(f"Using API key: {args.api_key[:10]}...")

    # Run tests based on the mode
    try:
        if args.mode == "streaming" or args.mode == "all":
            test_streaming(client, args.prompt)

        if args.mode == "non-streaming" or args.mode == "all":
            test_non_streaming(client, args.prompt)

        if args.mode == "models" or args.mode == "all":
            test_list_models(client)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Test failed with unexpected error")
        print(f"\n[FATAL ERROR: {e}]\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
