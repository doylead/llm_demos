"""
Utility functions for AWS Bedrock LLM invocation using LangChain.

This module provides common functions for interacting with AWS Bedrock via LangChain,
including model creation and invocation methods.
"""

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def create_bedrock_model(region: str, model_id: str, max_tokens: int = 2048,
                         temperature: float = 1.0, profile: str = None):
    """
    Create and return a LangChain ChatBedrock model instance.

    Args:
        region: AWS region name (e.g., "us-east-1")
        model_id: Bedrock model ID or ARN
        max_tokens: Maximum tokens to generate (default: 2048)
        temperature: Sampling temperature 0.0 to 1.0 (default: 1.0)
        profile: AWS profile name, or None for default credentials

    Returns:
        ChatBedrock model instance
    """
    # Build credentials kwargs if profile is specified
    credentials_kwargs = {}
    if profile:
        credentials_kwargs["credentials_profile_name"] = profile

    # Create the ChatBedrock model
    model = ChatBedrock(
        model_id=model_id,
        region_name=region,
        model_kwargs={
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        **credentials_kwargs
    )

    return model


def invoke_llm(model, prompt: str = None, messages: list = None, system: str = None):
    """
    Invoke a language model via LangChain's ChatBedrock.

    This function provides a unified interface for both single-turn (prompt) and
    multi-turn (messages) interactions using LangChain.

    Args:
        model: ChatBedrock model instance
        prompt: Single user prompt text (for single-turn). Mutually exclusive with messages.
        messages: List of message dicts with 'role' and 'content' keys (for multi-turn).
                 Mutually exclusive with prompt.
        system: Optional system prompt (supported by most models)

    Returns:
        Response text from the model

    Raises:
        ValueError: If both prompt and messages are provided, or neither is provided
        Exception: If the API call fails

    Examples:
        # Single-turn usage
        response = invoke_llm(model, prompt="What is 2+2?")

        # Multi-turn usage
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What's the weather?"}
        ]
        response = invoke_llm(model, messages=messages)
    """
    # Validate input
    if prompt and messages:
        raise ValueError("Cannot provide both 'prompt' and 'messages'. Use one or the other.")
    if not prompt and not messages:
        raise ValueError("Must provide either 'prompt' or 'messages'.")

    # Convert to LangChain message format
    langchain_messages = []

    # Add system message if provided
    if system:
        langchain_messages.append(SystemMessage(content=system))

    # Convert prompt or messages to LangChain format
    if prompt:
        langchain_messages.append(HumanMessage(content=prompt))
    else:
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))

    try:
        # Invoke the model
        response = model.invoke(langchain_messages)

        # Extract the text content from the response
        return response.content

    except Exception as e:
        raise Exception(f"Error invoking model: {str(e)}") from e
