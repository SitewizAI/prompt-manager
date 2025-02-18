import os
import requests
from dotenv import load_dotenv
import json
import re
from typing import List, Dict, Any
import boto3
from litellm import completion
from litellm.utils import trim_messages

load_dotenv()

def get_api_key(secret_name):
    region_name = "us-east-1"
    session = boto3.session.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
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

def run_completion_with_fallback(messages=None, prompt=None, models=["video"], response_format=None):
    """Run completion with fallback to evaluate."""
    initialize_vertex_ai()

    if messages is None:
        if prompt is None:
            raise ValueError("Either messages or prompt should be provided.")
        else:
            messages = [{"role": "user", "content": prompt}]

    trimmed_messages = messages
    try:
        trimmed_messages = trim_messages(messages, model)
    except Exception as e:
        pass

    for model in models:
        try:
            if response_format is None:
                response = completion(model=model, messages=trimmed_messages)
                content = response.choices[0].message.content
                return content
            else:
                response = completion(model=model, messages=trimmed_messages, response_format=response_format)
                content = json.loads(response.choices[0].message.content)
                return content
        except Exception as e:
            print(f"Failed to run completion with model {model}. Error: {str(e)}")
    return None

def analyze_issue_with_llm(issue_content: str) -> Dict[str, Any]:
    """Analyze an issue using LLM to understand how to fix it."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that analyzes GitHub issues and provides structured solutions."},
        {"role": "user", "content": f"Please analyze this issue and provide a structured response with root cause and solution steps:\n\n{issue_content}"}
    ]

    response_format = {
        "type": "object",
        "properties": {
            "root_cause": {"type": "string"},
            "solution_steps": {"type": "array", "items": {"type": "string"}},
            "estimated_effort": {"type": "string"}
        }
    }

    result = run_completion_with_fallback(
        messages=messages,
        models=["gpt-4"],
        response_format=response_format
    )

    return result

def create_github_issue(token: str, repo: str, title: str, body: str) -> Dict[str, Any]:
    """Create a new GitHub issue."""
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": title,
        "body": body
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 201:
        raise Exception(f"Failed to create issue: {response.text}")

    return response.json()

def lambda_handler(event, context):
    """AWS Lambda handler function."""
    try:
        # Extract parameters from the event
        issue_content = event.get('issue_content')
        github_token = event.get('github_token')
        repo = event.get('repo')
        title = event.get('title')

        if not all([issue_content, github_token, repo, title]):
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameters. Please provide issue_content, github_token, repo, and title.'
                })
            }

        # Analyze the issue
        analysis = analyze_issue_with_llm(issue_content)
        if not analysis:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': 'Failed to analyze issue with LLM.'
                })
            }

        # Create formatted body for the GitHub issue
        body = f"""
## Issue Analysis

### Root Cause
{analysis['root_cause']}

### Solution Steps
{chr(10).join(f'- {step}' for step in analysis['solution_steps'])}

### Estimated Effort
{analysis['estimated_effort']}

---
*This analysis was generated automatically by OpenHands AI*
"""

        # Create the GitHub issue
        issue = create_github_issue(github_token, repo, title, body)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully analyzed issue and created GitHub issue',
                'issue_url': issue['html_url'],
                'analysis': analysis
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
