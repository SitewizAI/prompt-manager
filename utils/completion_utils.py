"""Utilities for running AI completions."""

import os
import json
import time
from typing import List, Dict, Any, Optional, Union
import tiktoken
import litellm
from litellm import completion
from litellm.utils import trim_messages
from pydantic import BaseModel
import boto3


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

aws_region = os.getenv('AWS_REGION') or "us-east-1"

# Function to get boto3 client with credentials
def get_boto3_client(service_name, region=None):
    return boto3.client(
        service_name,
        region_name=aws_region
    )

def get_api_key(secret_name):
    client = get_boto3_client('secretsmanager', region="us-east-1")
    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )
    return json.loads(get_secret_value_response["SecretString"])


def initialize_vertex_ai():
    """Initialize Vertex AI with service account credentials"""
    AI_KEYS = get_api_key("AI_KEYS")
    litellm.api_key = AI_KEYS["LLM_API_KEY"]
    litellm.api_base = "https://llms.sitewiz.ai"
    litellm.enable_json_schema_validation = True

@measure_time
def run_completion_with_fallback(
    messages=None, 
    prompt=None, 
    models=model_fallback_list, 
    response_format=None, 
    temperature=None, 
    num_tries=3,
    include_tool_messages: bool = True
) -> Optional[Union[str, Dict]]:
    """Run completion with fallback and tool message tracking."""
    initialize_vertex_ai()
    tracker = ToolMessageTracker()

    if messages is None:
        if prompt is None:
            raise ValueError("Either messages or prompt should be provided.")
        messages = [{"role": "user", "content": prompt}]

    # Add tool messages to context if requested
    if include_tool_messages and tracker.messages:
        tool_context = tracker.get_context()
        # Add tool context to the last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i]["content"] += "\n" + tool_context
                break

    for attempt in range(num_tries):
        for model in models:
            try:
                trimmed_messages = messages
                try:
                    trimmed_messages = trim_messages(messages, model)
                except Exception as e:
                    log_error(f"Error trimming messages", e)

                if response_format is None:
                    response = completion(
                        model="litellm_proxy/"+model, 
                        messages=trimmed_messages, 
                        temperature=temperature
                    )
                    return response.choices[0].message.content
                else:
                    response = completion(
                        model="litellm_proxy/"+model, 
                        messages=trimmed_messages,
                        response_format=response_format,
                        temperature=temperature
                    )
                    content = json.loads(response.choices[0].message.content)
                    if isinstance(response_format, BaseModel):
                        response_format.model_validate(content)
                    return content

            except Exception as e:
                error_msg = f"Failed to run completion with model {model}: {str(e)}"
                log_error(error_msg)
                # Add error to tracker
                tracker.add_message(
                    tool_name="completion",
                    input_msg=str(trimmed_messages),
                    response="",
                    error=error_msg
                )

    return None

def count_tokens(text: str) -> int:
    """Count the number of tokens in the given text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        log_error(f"Error counting tokens: {str(e)}")
        # Fallback token counting: estimate 4 chars per token
        return len(text) // 4
