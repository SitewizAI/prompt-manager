import os
import requests
from dotenv import load_dotenv
import json
import re
from typing import List, Dict, Any, Optional
import boto3
from litellm import completion
from litellm.utils import trim_messages
from pydantic import BaseModel, Field

load_dotenv()

class GithubIssue(BaseModel):
    title: str = Field(..., description="Title of the GitHub issue")
    body: str = Field(..., description="Body content of the GitHub issue")
    labels: List[str] = Field(default=["fix-me"], description="Labels to apply to the issue")

class PromptChange(BaseModel):
    ref: str = Field(..., description="Reference ID of the prompt")
    version: str = Field(..., description="Version of the prompt")
    content: str = Field(..., description="New content for the prompt")
    reason: str = Field(..., description="Reason for the prompt change")

class AnalysisResponse(BaseModel):
    github_issues: List[GithubIssue] = Field(..., description="List of GitHub issues to create")
    prompt_changes: List[PromptChange] = Field(..., description="List of prompt changes to make")

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

def run_completion_with_fallback(messages=None, prompt=None, models=["gpt-4"], response_format=None):
    """Run completion with fallback to evaluate."""
    initialize_vertex_ai()

    if messages is None:
        if prompt is None:
            raise ValueError("Either messages or prompt should be provided.")
        else:
            messages = [{"role": "user", "content": prompt}]

    for model in models:
        try:
            trimmed_messages = messages
            try:
                trimmed_messages = trim_messages(messages, model)
            except Exception as e:
                pass

            if response_format is None:
                response = completion(model=model, messages=trimmed_messages)
                content = response.choices[0].message.content
                return content
            else:
                response = completion(
                    model=model,
                    messages=trimmed_messages,
                    response_format={"type": "json_object", "schema": response_format}
                )
                content = json.loads(response.choices[0].message.content)
                return content
        except Exception as e:
            print(f"Failed to run completion with model {model}. Error: {str(e)}")
            if model == models[-1]:  # Only raise on last model attempt
                raise
    return None

def analyze_issue_with_llm(issue_content: str, context: str) -> AnalysisResponse:
    """Analyze an issue using LLM to understand how to fix it and suggest prompt changes."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that analyzes issues and provides structured solutions, including GitHub issues and prompt changes."},
        {"role": "user", "content": f"""Please analyze this issue and provide a structured response with GitHub issues to create and prompt changes to make.

Context:
{context}

Issue Content:
{issue_content}"""}
    ]

    response_format = {
        "type": "object",
        "properties": {
            "github_issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["fix-me"]
                        }
                    },
                    "required": ["title", "body"]
                }
            },
            "prompt_changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ref": {"type": "string"},
                        "version": {"type": "string"},
                        "content": {"type": "string"},
                        "reason": {"type": "string"}
                    },
                    "required": ["ref", "version", "content", "reason"]
                }
            }
        },
        "required": ["github_issues", "prompt_changes"]
    }

    result = run_completion_with_fallback(
        messages=messages,
        models=["gpt-4"],
        response_format=response_format
    )

    return AnalysisResponse(**result)

def get_github_issues(token: str, repo: str) -> List[Dict[str, Any]]:
    """Get all GitHub issues from a repository."""
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    params = {
        "state": "all",
        "per_page": 100
    }

    all_issues = []
    while url:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch issues: {response.text}")

        issues = response.json()
        all_issues.extend(issues)

        # Check for next page in Link header
        if "Link" in response.headers:
            links = response.headers["Link"].split(", ")
            next_link = [link for link in links if 'rel="next"' in link]
            url = next_link[0].split(";")[0][1:-1] if next_link else None
        else:
            url = None

    return all_issues

def create_github_issue(token: str, repo: str, title: str, body: str, labels: List[str] = None) -> Dict[str, Any]:
    """Create a new GitHub issue."""
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": title,
        "body": body,
        "labels": labels or []
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 201:
        raise Exception(f"Failed to create issue: {response.text}")

    # Add issue to project
    issue = response.json()
    project_url = "https://api.github.com/orgs/SitewizAI/projects/21/items"
    project_data = {
        "content_id": issue["id"],
        "content_type": "Issue"
    }
    project_response = requests.post(project_url, headers=headers, json=project_data)
    if project_response.status_code != 201:
        print(f"Warning: Failed to add issue to project: {project_response.text}")

    return issue

def lambda_handler(event, context):
    """AWS Lambda handler function."""
    try:
        # Extract parameters from the event
        issue_content = event.get('issue_content')
        github_token = event.get('github_token', os.getenv('GITHUB_TOKEN'))
        repo = event.get('repo')
        context_data = event.get('context', '')

        if not all([issue_content, github_token, repo]):
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameters. Please provide issue_content, github_token, and repo.'
                })
            }

        # Get existing GitHub issues for context
        try:
            existing_issues = get_github_issues(github_token, repo)
            issues_context = "\n".join([
                f"Issue #{issue['number']}: {issue['title']}\n{issue['body'][:200]}..."
                for issue in existing_issues[:5]  # Include last 5 issues for context
            ])
            context_data = f"{context_data}\n\nRecent GitHub Issues:\n{issues_context}"
        except Exception as e:
            print(f"Warning: Failed to fetch existing issues: {e}")

        # Analyze the issue
        analysis = analyze_issue_with_llm(issue_content, context_data)
        if not analysis:
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': 'Failed to analyze issue with LLM.'
                })
            }

        # Create GitHub issues
        created_issues = []
        for issue in analysis.github_issues:
            try:
                created_issue = create_github_issue(
                    github_token,
                    repo,
                    issue.title,
                    issue.body,
                    issue.labels
                )
                created_issues.append(created_issue)
            except Exception as e:
                print(f"Warning: Failed to create issue {issue.title}: {e}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully analyzed issue and created GitHub issues',
                'created_issues': [{'url': issue['html_url'], 'title': issue['title']} for issue in created_issues],
                'prompt_changes': [change.dict() for change in analysis.prompt_changes]
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
