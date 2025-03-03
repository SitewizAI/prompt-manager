"""Utilities for running AI completions."""

import os
import json
import time
from typing import List, Dict, Any, Optional, Union
import tiktoken

from .logging_utils import log_debug, log_error, ToolMessageTracker, measure_time

# System prompt and instructions
SYSTEM_PROMPT = """You are a helpful website optimization expert assistant assisting in creating an agentic workflow that automates digital experience optimization – from data analysis to insight/suggestion generation to code implementation. 
Your role is to analyze evaluations and provide recommendations to update the prompts and code files, thereby improving the quality and accuracy of outputs so that each evaluation is successful in a low number of turns. 
Use the provided context to generate specific, accurate, and traceable recommendations that update the code and prompt structure."""

PROMPT_INSTRUCTIONS = """1. Block-Level Prompt Optimization for Reasoning models (all agents use reasoning models)  
   - Techniques to Use:
     • Bootstrapped Demonstration Extraction: Analyze evaluation traces to identify 2–3 high-quality input/output demonstration examples and formatting that clarify task patterns.
     • Ensure your prompts are straightforward and easy to understand. Avoid ambiguity by specifying exactly what you need from the AI
     • Include specific details, constraints, and objectives to guide the model toward the desired output using domain specific knowledge of digital experience optimization and the agent role
     • Structure complex inputs with clear sections or headings
     • Specify end goal and desired output format explicitly"""

# Fallback model list
model_fallback_list = ["reasoning", "long"]

def initialize_vertex_ai():
    """Initialize AI service with credentials."""
    # In this implementation, just log the attempt
    log_debug("Initializing AI service")
    # Actual implementation would set up authentication credentials

@measure_time
def run_completion_with_fallback(
    messages=None, 
    prompt=None, 
    models=None, 
    response_format=None, 
    temperature=None, 
    num_tries=3,
    include_tool_messages: bool = True
) -> Optional[Union[str, Dict]]:
    """
    Run completion with fallback and tool message tracking.
    
    Args:
        messages: List of chat messages
        prompt: Alternative to messages - a single prompt string
        models: List of models to try in order
        response_format: Optional format for response validation
        temperature: Optional temperature parameter
        num_tries: Number of attempts to make per model
        include_tool_messages: Whether to include tracked tool messages in context
        
    Returns:
        The completion response or None if all attempts fail
    """
    log_debug("Starting completion with fallback")
    
    # Use default models if none provided
    if models is None:
        models = model_fallback_list
        
    # For testing/demo purposes, just return a mock response
    if prompt:
        return f"Mock response to prompt: {prompt[:50]}..."
    elif messages:
        last_msg = messages[-1]["content"] if messages else ""
        return f"Mock response to message: {last_msg[:50]}..."
    else:
        return "No input provided"

def count_tokens(text: str) -> int:
    """Count the number of tokens in the given text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        log_error(f"Error counting tokens: {str(e)}")
        # Fallback token counting: estimate 4 chars per token
        return len(text) // 4
