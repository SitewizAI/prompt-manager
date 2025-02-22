import os
import requests
from dotenv import load_dotenv
import json
import re
from typing import List, Dict, Any, Optional
import boto3
from pydantic import BaseModel, Field
from datetime import datetime
from utils import SYSTEM_PROMPT, run_completion_with_fallback, get_context, get_most_recent_stream_key

load_dotenv()

SYSTEM_PROMPT_ADDITION = """
1. Analyze the provided context including:
   - Recent evaluations and their performance metrics
   - Current prompts and their versions
   - Python code files and their contents
   - Previous GitHub issues and their resolutions

2. Identify problems and suggest solutions through:
   - GitHub issues: Create detailed, actionable issues for bugs, improvements, or new features (if any new ones need to be created)
   - Prompt changes: Suggest specific updates to prompts to improve their effectiveness
   
3. Ensure all suggestions are:
   - Data-driven: Based on evaluation metrics and failure patterns
   - Specific: Include exact changes to make
   - Traceable: Reference specific files, prompts, or evaluations
   - Actionable: Provide clear steps for implementation

Format your response as structured JSON with github_issues and prompt_changes."""

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

class AnalysisResult(BaseModel):
    """Structured analysis result from the LLM."""
    github_issues: List[GithubIssue] = Field(..., description="List of GitHub issues to create")
    prompt_changes: List[PromptChange] = Field(..., description="List of prompt changes to make")
    evaluation_summary: Dict[str, Any] = Field(..., description="Summary of evaluation analysis")
    recommendations: List[str] = Field(..., description="List of general recommendations")
    impact_score: float = Field(..., ge=0, le=1, description="Impact score of the suggested changes")


def get_github_files(token, repo="SitewizAI/sitewiz", target_path="backend/agents/data_analyst_group"):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    def get_contents(path=""):
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error accessing {path}: {response.status_code}")
            return []

        contents = response.json()
        if not isinstance(contents, list):
            contents = [contents]

        return contents

    def process_contents(path=""):
        contents = get_contents(path)
        python_files = []

        for item in contents:
            full_path = os.path.join(path, item["name"])
            if item["type"] == "file" and item["name"].endswith(".py"):
                python_files.append({
                    "path": full_path,
                    "download_url": item["download_url"]
                })
            elif item["type"] == "dir":
                python_files.extend(process_contents(item["path"]))

        return python_files

    return process_contents(path=target_path)

def get_file_contents(file_info):
    response = requests.get(file_info["download_url"])
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error downloading {file_info['path']}")
        return ""

def get_recent_evaluations():
    """Get the most recent evaluation and its 5 previous evaluations."""
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('EvaluationsTable')
        
        # Get all stream keys' most recent evaluations
        response = table.scan(
            ProjectionExpression='streamKey, #ts',
            ExpressionAttributeNames={'#ts': 'timestamp'},
            Select='SPECIFIC_ATTRIBUTES'
        )
        
        if not response.get('Items'):
            return []
            
        # Find the most recent evaluation
        most_recent = max(response['Items'], key=lambda x: float(x['timestamp']))
        stream_key = most_recent['streamKey']
        
        # Get the 6 most recent evaluations for this stream key (current + 5 previous)
        response = table.query(
            KeyConditionExpression='streamKey = :sk',
            ExpressionAttributeValues={':sk': stream_key},
            ScanIndexForward=False,
            Limit=6
        )
        
        return sorted(response.get('Items', []), key=lambda x: float(x['timestamp']), reverse=True)
    except Exception as e:
        print(f"Error getting evaluations: {e}")
        return []

def analyze_system_with_llm() -> AnalysisResult:
    """Analyze the system using LLM based on evaluations and context."""
    # Get most recent stream key
    stream_key = get_most_recent_stream_key()
    if not stream_key:
        raise ValueError("No evaluations found")
    
    # Get context with GitHub issues included
    print(f"Analyzing system with stream key: {stream_key}")
    context = get_context(
        stream_key,
        return_type="string",
        include_github_issues=True
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + SYSTEM_PROMPT_ADDITION},
        {"role": "user", "content": f"Analyze the following system state and provide recommendations:\n{context}\n\n{SYSTEM_PROMPT_ADDITION}"}
    ]

    result = run_completion_with_fallback(
        messages=messages,
        response_format=AnalysisResponse
    )

    return AnalysisResult(**result)

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
        # Analyze the system
        analysis = analyze_system_with_llm()
        
        # Create GitHub issues
        github_token = os.getenv('GITHUB_TOKEN')
        created_issues = []
        
        for issue in analysis.github_issues:
            try:
                result = create_github_issue_with_project(
                    token=github_token,
                    title=issue.title,
                    body=issue.body,
                    labels=issue.labels
                )
                if result["success"]:
                    created_issues.append(result["issue"])
                else:
                    print(f"Warning: Failed to create issue {issue.title}: {result['error']}")
            except Exception as e:
                print(f"Warning: Failed to create issue {issue.title}: {e}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully analyzed system and created GitHub issues',
                'created_issues': [{'url': issue['url'], 'number': issue['number']} for issue in created_issues],
                'prompt_changes': [change.dict() for change in analysis.prompt_changes],
                'evaluation_summary': analysis.evaluation_summary,
                'recommendations': analysis.recommendations,
                'impact_score': analysis.impact_score
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
    
lambda_handler({}, None)