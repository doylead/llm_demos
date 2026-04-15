#!/usr/bin/env python3
"""
Multi-turn demo for AWS Bedrock LLM invocation using LangChain.

This demo illustrates how LLM invocations are stateless by showing the
conversation history being replayed with each turn. The visual format makes
it clear that each API call includes all previous messages.

Usage:
    python3 multi_turn_demo.py

    You will be prompted to enter messages interactively.
    Type 'quit' or 'exit' to end the conversation.
"""

import sys
import textwrap
from langchain_bedrock_utils import create_bedrock_model, invoke_llm
from config import AWS_REGION, AWS_PROFILE, MODEL_ID, MAX_TOKENS, TEMPERATURE, LINE_WIDTH


def format_message(role: str, content: str, prefix: str = "") -> str:
    """
    Format a message with line wrapping and proper prefixes.

    Args:
        role: "user" or "assistant"
        content: The message content
        prefix: Additional prefix (e.g., "< " for replayed messages)

    Returns:
        Formatted string with appropriate prefixes
    """
    content = content.strip()
    role_marker = "<" if role == "user" else ">"

    # Wrap each paragraph separately to preserve intentional line breaks
    wrapped_lines = []
    for line in content.split('\n'):
        if line.strip():  # Non-empty line
            wrapped = textwrap.fill(line, width=LINE_WIDTH, break_long_words=False, break_on_hyphens=False)
            wrapped_lines.extend(wrapped.split('\n'))
        else:  # Empty line (preserve paragraph breaks)
            wrapped_lines.append('')

    # Add role marker and any additional prefix to each line
    formatted_lines = [f"{prefix}{role_marker} {line}" for line in wrapped_lines]
    return '\n'.join(formatted_lines)


def display_conversation_history(messages: list):
    """
    Display the conversation history with < prefix to show it's being replayed.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
    """
    for msg in messages:
        print(format_message(msg["role"], msg["content"], prefix="< "))


def main():
    """Main entry point for the script."""
    # Create LangChain Bedrock model once at the start
    try:
        bedrock_model = create_bedrock_model(
            AWS_REGION,
            MODEL_ID,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            profile=AWS_PROFILE
        )
    except Exception as e:
        print(f"Error creating Bedrock model: {e}", file=sys.stderr)
        sys.exit(1)

    # Conversation history
    messages = []

    print("Multi-turn conversation demo. Type 'quit' or 'exit' to end.\n")

    while True:
        # Display replayed conversation history (if any)
        if messages:
            display_conversation_history(messages)

        # Get user input
        try:
            user_input = input("< ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.", file=sys.stderr)
            sys.exit(0)

        # Check for exit commands
        if user_input.strip().lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not user_input.strip():
            print("Error: Empty input. Please enter a message or 'quit' to exit.", file=sys.stderr)
            continue

        # Add user message to conversation history
        messages.append({"role": "user", "content": user_input})

        # Invoke the model with full conversation history
        try:
            response_text = invoke_llm(
                bedrock_model,
                messages=messages
            )

            # Display the new response
            print(format_message("assistant", response_text))

            # Add assistant response to conversation history
            messages.append({"role": "assistant", "content": response_text})

            # Wait for user to press enter before showing the next turn
            print()
            try:
                input("Press Enter to continue...")
                print()  # Add blank line after pressing enter
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.", file=sys.stderr)
                sys.exit(0)

        except Exception as e:
            print(f"Error invoking model: {e}", file=sys.stderr)
            # Remove the user message that caused the error
            messages.pop()


if __name__ == "__main__":
    main()
