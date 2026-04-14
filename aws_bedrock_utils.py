"""
Utility functions for AWS Bedrock LLM invocation.

This module provides common functions for interacting with AWS Bedrock,
including client creation and model invocation methods.
"""

import json
import boto3
from botocore.exceptions import ClientError


def create_bedrock_client(region: str, profile: str = None):
    """
    Create and return a Bedrock Runtime client.

    Args:
        region: AWS region name (e.g., "us-east-1")
        profile: AWS profile name, or None for default credentials

    Returns:
        boto3 Bedrock Runtime client
    """
    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile

    session = boto3.Session(**session_kwargs)
    return session.client("bedrock-runtime")


def _detect_model_provider(model_id: str) -> str:
    """
    Detect the model provider based on the model ID.

    Args:
        model_id: Bedrock model ID

    Returns:
        Provider name: "anthropic", "amazon", "meta", "cohere", "ai21", or "mistral"
    """
    model_id_lower = model_id.lower()

    if "anthropic" in model_id_lower or "claude" in model_id_lower:
        return "anthropic"
    elif "amazon" in model_id_lower or "titan" in model_id_lower:
        return "amazon"
    elif "meta" in model_id_lower or "llama" in model_id_lower:
        return "meta"
    elif "cohere" in model_id_lower:
        return "cohere"
    elif "ai21" in model_id_lower:
        return "ai21"
    elif "mistral" in model_id_lower:
        return "mistral"
    else:
        raise ValueError(f"Unknown model provider for model ID: {model_id}")


def invoke_llm(client, model_id: str, prompt: str = None, messages: list = None,
               max_tokens: int = 2048, temperature: float = 1.0, system: str = None):
    """
    Invoke a language model via AWS Bedrock using the Converse API.

    This function uses the unified Converse API which works with all Bedrock models,
    including inference profiles. It supports both single-turn (prompt) and multi-turn
    (messages) interactions.

    Args:
        client: boto3 Bedrock Runtime client
        model_id: Bedrock model ID or ARN (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0"
                 or "arn:aws:bedrock:us-east-2:...:inference-profile/us.deepseek.r1-v1:0")
        prompt: Single user prompt text (for single-turn). Mutually exclusive with messages.
        messages: List of message dicts with 'role' and 'content' keys (for multi-turn).
                 Mutually exclusive with prompt.
        max_tokens: Maximum tokens to generate (default: 2048)
        temperature: Sampling temperature 0.0 to 1.0 (default: 1.0)
        system: Optional system prompt (supported by most models)

    Returns:
        Response text from the model

    Raises:
        ValueError: If both prompt and messages are provided, or neither is provided
        Exception: If the API call fails

    Examples:
        # Single-turn usage
        response = invoke_llm(client, model_id, prompt="What is 2+2?")

        # Multi-turn usage
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What's the weather?"}
        ]
        response = invoke_llm(client, model_id, messages=messages)
    """
    # Validate input
    if prompt and messages:
        raise ValueError("Cannot provide both 'prompt' and 'messages'. Use one or the other.")
    if not prompt and not messages:
        raise ValueError("Must provide either 'prompt' or 'messages'.")

    # Convert single prompt to messages format if needed
    if prompt:
        messages = [{"role": "user", "content": prompt}]

    # Use the modern Converse API which works with all models
    return _invoke_with_converse_api(client, model_id, messages, max_tokens, temperature, system)


def _invoke_with_converse_api(client, model_id: str, messages: list, max_tokens: int,
                              temperature: float, system: str = None):
    """
    Invoke a model using the unified Converse API.

    This API works with all Bedrock models and inference profiles, making it the
    recommended approach for new implementations.
    """
    # Format messages for Converse API (ensure content is in the right format)
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "role": msg["role"],
            "content": [{"text": msg["content"]}]
        })

    # Build the request
    request_params = {
        "modelId": model_id,
        "messages": formatted_messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        }
    }

    # Add system prompt if provided
    if system:
        request_params["system"] = [{"text": system}]

    try:
        response = client.converse(**request_params)

        # Extract the text from the response
        output_message = response["output"]["message"]
        text_content = output_message["content"][0]["text"]

        return text_content

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        raise Exception(f"AWS Bedrock Error ({error_code}): {error_message}") from e
    except Exception as e:
        raise Exception(f"Error invoking model: {str(e)}") from e


def _invoke_anthropic(client, model_id: str, messages: list, max_tokens: int,
                     temperature: float, system: str = None):
    """Invoke Anthropic Claude models."""
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages
    }

    if system:
        request_body["system"] = system

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )

        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        raise Exception(f"AWS Bedrock Error ({error_code}): {error_message}") from e


def _invoke_amazon_titan(client, model_id: str, messages: list, max_tokens: int,
                        temperature: float):
    """Invoke Amazon Titan models."""
    # Titan doesn't support multi-turn directly, so concatenate messages
    prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    request_body = {
        "inputText": prompt_text,
        "textGenerationConfig": {
            "maxTokenCount": max_tokens,
            "temperature": temperature,
        }
    }

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )

        response_body = json.loads(response["body"].read())
        return response_body["results"][0]["outputText"]

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        raise Exception(f"AWS Bedrock Error ({error_code}): {error_message}") from e


def _invoke_meta_llama(client, model_id: str, messages: list, max_tokens: int,
                      temperature: float, system: str = None):
    """Invoke Meta Llama models."""
    # Format messages in Llama's chat format
    prompt_parts = []

    if system:
        prompt_parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>")

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>")

    prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>")
    prompt = "".join(prompt_parts)

    request_body = {
        "prompt": prompt,
        "max_gen_len": max_tokens,
        "temperature": temperature,
    }

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )

        response_body = json.loads(response["body"].read())
        return response_body["generation"]

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        raise Exception(f"AWS Bedrock Error ({error_code}): {error_message}") from e


def _invoke_cohere(client, model_id: str, messages: list, max_tokens: int,
                  temperature: float, system: str = None):
    """Invoke Cohere Command models."""
    # Cohere uses chat_history format
    chat_history = []
    current_message = messages[-1]["content"] if messages else ""

    if len(messages) > 1:
        for msg in messages[:-1]:
            chat_history.append({
                "role": "USER" if msg["role"] == "user" else "CHATBOT",
                "message": msg["content"]
            })

    request_body = {
        "message": current_message,
        "chat_history": chat_history,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if system:
        request_body["preamble"] = system

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )

        response_body = json.loads(response["body"].read())
        return response_body["text"]

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        raise Exception(f"AWS Bedrock Error ({error_code}): {error_message}") from e


def _invoke_mistral(client, model_id: str, messages: list, max_tokens: int,
                   temperature: float, system: str = None):
    """Invoke Mistral models."""
    # Mistral format is similar to OpenAI
    formatted_messages = messages.copy()

    if system:
        formatted_messages.insert(0, {"role": "system", "content": system})

    request_body = {
        "prompt": _format_mistral_prompt(formatted_messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )

        response_body = json.loads(response["body"].read())
        return response_body["outputs"][0]["text"]

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        raise Exception(f"AWS Bedrock Error ({error_code}): {error_message}") from e


def _format_mistral_prompt(messages: list) -> str:
    """Format messages for Mistral models."""
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt_parts.append(f"<s>[INST] {content} [/INST]</s>")
        elif role == "user":
            prompt_parts.append(f"<s>[INST] {content} [/INST]")
        elif role == "assistant":
            prompt_parts.append(f" {content}</s>")
    return "".join(prompt_parts)
