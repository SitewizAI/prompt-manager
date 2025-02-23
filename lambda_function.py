import os
from dotenv import load_dotenv
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from utils import (
    SYSTEM_PROMPT, 
    run_completion_with_fallback, 
    get_context, 
    get_most_recent_stream_key,
    create_github_issue_with_project,
    update_prompt
)
from litellm.utils import token_counter

load_dotenv()

SYSTEM_PROMPT_ADDITION = """
Analyze the provided context including recent evaluations, prompts, code files, and GitHub issues.
Identify potential improvements and issues that need addressing.
Our goal is to make each evaluation successful with a high quality output in a low number of turns. 

Format your response as JSON with:
1. github_issues: List of issues to create, each with:
   - title: Clear, specific issue title
   - body: Detailed description with context, specific files to change, and how to change them
   - labels: ["fix-me", "improvement", "bug"] etc. (you must always have "fix-me", so the issue is picked up by the AI)

2. prompt_changes: List of prompt updates, each with:
   - ref: Prompt reference ID - this must match the ref of an existing prompt
   - content: New prompt content
   - reason: Why this change is needed

Notes:
- A prompt change will directly change the prompt used in future evaluations.
- Opening a GitHub issue will create a task for the AI team to address the problem, they should be specific and actionable. Only open a github issue if you are sure what went wrong and how to fix it.
    If a similar issue exists, do not create a new one. Only create it if you can provide the specific code files and lines that need to be changed with the updated code.
    Github issues should fix bugs / systematic errors in tools, functions, or interactions
- If there is no success in the evaluations, focus on updating the question lists and prompts since those are the questions / weakening the threshholds which influence whether the output passes the evaluation. 
    Your first priority is ensuring that there is a non-zero success rate. Then focus on improving the quality of the output.
- If the quality of the output is low, focus on updating the prompts and the question lists.
- If updating the prompt of an object like a question list, return a valid JSON string of the same format as the original prompt.
- Please ensure all the agent prompts are block-level optimized with examples and clear instructions.
- You must update prompts by adding the PromptChange to the output with the ref of the prompt you want to change, not by creating a github issue.

The analysis should be data-driven based on evaluation metrics and failure patterns."""

class GithubIssue(BaseModel):
    title: str = Field(..., description="Issue title")
    body: str = Field(..., description="Issue description")
    labels: List[str] = Field(default=["fix-me"], description="Issue labels")

class PromptChange(BaseModel):
    ref: str = Field(..., description="Prompt reference")
    content: str = Field(..., description="New prompt content")
    reason: str = Field(..., description="Reason for change")

class AnalysisResponse(BaseModel):
    github_issues: List[GithubIssue] = Field(default=[])
    prompt_changes: List[PromptChange] = Field(default=[])

def lambda_handler(event, context):
    """AWS Lambda handler for system analysis and updates."""
    try:
        # Get most recent stream key
        stream_key = get_most_recent_stream_key()
        if not stream_key:
            raise ValueError("No evaluations found")
        
        # Get full context including GitHub issues
        system_context = get_context(
            stream_key=stream_key, 
            return_type="string",
            include_github_issues=True,
            include_code_files=True
        )
        
        # Count tokens in system prompt and context
        system_tokens = token_counter(messages=[{"role": "system", "content": SYSTEM_PROMPT}], model="gpt-4")
        addition_tokens = token_counter(messages=[{"role": "system", "content": SYSTEM_PROMPT_ADDITION}], model="gpt-4")
        context_tokens = token_counter(messages=[{"role": "user", "content": system_context}], model="gpt-4")
        
        print(f"Token counts:")
        print(f"- System prompt: {system_tokens}")
        print(f"- System addition: {addition_tokens}")
        print(f"- Context: {context_tokens}")
        print(f"- Total: {system_tokens + addition_tokens + context_tokens}")
        
        # Run analysis with LLM
        full_prompt = SYSTEM_PROMPT + "\n\n" + SYSTEM_PROMPT_ADDITION
        messages = [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": f"Analyze this system state and provide recommendations:\n\n{system_context}"}
        ]
        
        analysis = run_completion_with_fallback(
            messages=messages,
            response_format=AnalysisResponse
        )
        
        if not analysis:
            raise ValueError("Failed to get analysis from LLM")
        
        # Print the analysis for debugging
        print("\nAnalysis results:")
        print(analysis)
        
        # For now, return early to avoid making actual changes
        results = {
            'created_issues': [],
            'updated_prompts': [],
            'errors': []
        }
        
        # Create GitHub issues
        github_token = os.getenv('GITHUB_TOKEN')
        # convert analysis to pydatnic model
        analysis = AnalysisResponse(**analysis)
        if github_token and analysis.github_issues:
            for issue in analysis.github_issues:
                try:
                    result = create_github_issue_with_project(
                        token=github_token,
                        title=issue.title,
                        body=issue.body,
                        labels=issue.labels
                    )
                    if result["success"]:
                        results['created_issues'].append({
                            'number': result['issue']['number'],
                            'url': result['issue']['url']
                        })
                    else:
                        results['errors'].append(f"Failed to create issue: {result['error']}")
                except Exception as e:
                    results['errors'].append(f"Error creating issue: {str(e)}")
        
        # Update prompts
        if analysis.prompt_changes:
            for change in analysis.prompt_changes:
                try:
                    print(f"Attempting to update prompt {change.ref}")
                    print(f"New content: {change.content}")
                    
                    success = update_prompt(change.ref, change.content)
                    if success:
                        print(f"Successfully updated prompt {change.ref}")
                        results['updated_prompts'].append({
                            'ref': change.ref,
                            'reason': change.reason
                        })
                    else:
                        error_msg = f"Failed to update prompt {change.ref} - validation failed"
                        print(error_msg)
                        results['errors'].append(error_msg)
                except Exception as e:
                    error_msg = f"Error updating prompt {change.ref}: {str(e)}"
                    print(error_msg)
                    results['errors'].append(error_msg)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'System analysis complete',
                'stream_key': stream_key,
                'results': results
            })
        }

    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        }

if __name__ == "__main__":
    result = lambda_handler({}, None)
    print(json.dumps(result, indent=2))