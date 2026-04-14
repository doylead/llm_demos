"""
Shared configuration for AWS Bedrock demos.

This file contains configuration settings used across all demo scripts.
"""

# ============================================================================
# AWS CONFIGURATION
# ============================================================================

# AWS Configuration
AWS_REGION = "us-east-2"  # Change to your preferred region
AWS_PROFILE = None  # Set to your AWS profile name, or None for default credentials

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model Configuration
# Examples of available models (now uses Converse API - works with all models):
# - "anthropic.claude-3-5-sonnet-20241022-v2:0" (Claude)
# - "us.anthropic.claude-3-5-sonnet-20241022-v2:0" (cross-region inference profile)
# - "us.deepseek.r1-v1:0" (DeepSeek via inference profile)
# - Full ARN also works: "arn:aws:bedrock:region:account:inference-profile/model-id"
MODEL_ID = "us.deepseek.r1-v1:0"

# Model Parameters
MAX_TOKENS = 2048
TEMPERATURE = 1.0

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

# Line width for text wrapping (total including "> " prefix will be LINE_WIDTH + 2)
LINE_WIDTH = 76
