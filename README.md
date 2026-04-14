# llm_demos

Showcase basic LLM functionality

## Project Overview

This project demonstrates the stateless nature of Large Language Model (LLM) invocation in both code and CLI environments.

### Key Concepts

**Stateless Execution**: LLMs do not maintain memory between requests. Each API call is independent and isolated. This is true for all LLM systems, including popular web applications like ChatGPT, Gemini, and Claude.

**Multi-turn Conversations**: To enable conversational interactions, the entire conversation history (context) must be explicitly passed with each request. The application is responsible for maintaining and transmitting this context.

**Web Application Wrappers**: While common LLM interfaces like ChatGPT, Gemini, and Claude appear to "remember" previous messages, they actually use wrapper layers that automatically manage and pass conversation context back and forth with each request. Under the hood, they work the same stateless way.

**Context Management**: Developers must manually track and include previous messages when making subsequent LLM calls to maintain conversation continuity.

### Educational Goals

- Demonstrate how LLM APIs handle stateless invocation
- Show the mechanics of passing conversation context in multi-turn interactions
- Illustrate the difference between how LLMs appear to "remember" versus how they actually process context

### Future Exploration

This project may expand to explore security implications of stateless context passing, including:
- How conversation history could potentially be spoofed by malicious actors
- Attack vectors related to context manipulation
- Best practices for validating and securing conversation state
