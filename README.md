# llm_demos

Explorations of using Large Language Models and AI

## Project Overview

This repository contains demonstrations and explorations of working with Large Language Models (LLMs) using various tools and frameworks. The examples cover fundamental concepts, practical implementations, and security considerations when building with AI.

## Tools and Frameworks

This project includes examples using multiple approaches:

- **boto3**: Direct AWS Bedrock API integration for low-level control
- **LangChain**: Higher-level abstractions and patterns for LLM applications
- Room for additional frameworks and tools as the project grows

## Key Concepts

**Stateless Execution**: LLMs do not maintain memory between requests. Each API call is independent and isolated. This is true for all LLM systems, including popular web applications like ChatGPT, Gemini, and Claude.

**Multi-turn Conversations**: To enable conversational interactions, the entire conversation history (context) must be explicitly passed with each request. The application is responsible for maintaining and transmitting this context.

**Web Application Wrappers**: While common LLM interfaces like ChatGPT, Gemini, and Claude appear to "remember" previous messages, they actually use wrapper layers that automatically manage and pass conversation context back and forth with each request. Under the hood, they work the same stateless way.

**Context Management**: Developers must manually track and include previous messages when making subsequent LLM calls to maintain conversation continuity.

## Repository Structure

- `langchain/`: Examples using the LangChain framework
  - Single-turn and multi-turn conversation demos
  - Configuration and utility modules
- Additional tool-specific directories as the project expands

## Educational Goals

- Demonstrate how LLM APIs handle stateless invocation
- Show the mechanics of passing conversation context in multi-turn interactions
- Illustrate the difference between how LLMs appear to "remember" versus how they actually process context
- Explore different tools and frameworks for building LLM applications
- Compare approaches between low-level APIs (boto3) and higher-level frameworks (LangChain)

## Future Exploration

This project may expand to explore:
- Additional LLM frameworks and tools
- Advanced prompting techniques and patterns
- Security implications of stateless context passing, including:
  - How conversation history could potentially be spoofed by malicious actors
  - Attack vectors related to context manipulation
  - Best practices for validating and securing conversation state
