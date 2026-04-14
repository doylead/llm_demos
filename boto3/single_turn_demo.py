#!/usr/bin/env python3
"""
Single-turn demo for AWS Bedrock LLM invocation.

Usage:
    python3 single_turn_demo.py

    You will be prompted to enter your question interactively.
"""

import sys
import textwrap
from aws_bedrock_utils import create_bedrock_client, invoke_llm
from config import AWS_REGION, AWS_PROFILE, MODEL_ID, MAX_TOKENS, TEMPERATURE, LINE_WIDTH


def main():
    """Main entry point for the script."""
    # Get user input interactively
    try:
        user_query = input("< ")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.", file=sys.stderr)
        sys.exit(0)

    if not user_query.strip():
        print("Error: Empty query provided.", file=sys.stderr)
        sys.exit(1)

    # Create Bedrock client
    try:
        bedrock_client = create_bedrock_client(AWS_REGION, AWS_PROFILE)
    except Exception as e:
        print(f"Error creating Bedrock client: {e}", file=sys.stderr)
        sys.exit(1)

    # Invoke the model
    try:
        response_text = invoke_llm(
            bedrock_client,
            MODEL_ID,
            prompt=user_query,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )

        # Format output with > prefix on each line, removing leading/trailing whitespace
        # Wrap text to a fixed width for better display in multi-turn scenarios
        response_text = response_text.strip()

        # Wrap each paragraph separately to preserve intentional line breaks
        wrapped_lines = []
        for line in response_text.split('\n'):
            if line.strip():  # Non-empty line
                wrapped = textwrap.fill(line, width=LINE_WIDTH, break_long_words=False, break_on_hyphens=False)
                wrapped_lines.extend(wrapped.split('\n'))
            else:  # Empty line (preserve paragraph breaks)
                wrapped_lines.append('')

        # Add > prefix to each line
        formatted_lines = [f"> {line}" for line in wrapped_lines]
        print('\n'.join(formatted_lines))

    except Exception as e:
        print(f"Error invoking model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
