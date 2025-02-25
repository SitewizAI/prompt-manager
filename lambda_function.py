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
    update_prompt,
    ToolMessageTracker  # Add import for ToolMessageTracker
)
from litellm.utils import token_counter

load_dotenv()

# Add lambda retry configuration
LAMBDA_MAX_RETRIES = 1  # Set to 1 for single retry

SYSTEM_PROMPT_ADDITION = """
Analyze the provided context including recent evaluations, prompts, code files, and GitHub issues.
Identify potential improvements and issues that need addressing.

Format your response as JSON with:
1. github_issues: List of issues to create, each with:
   - title: Clear, specific issue title
   - body: Detailed description with specific code changes needed
   - labels: ["fix-me", "improvement", "bug"] etc. (you must always have "fix-me")

2. prompt_changes: List of prompt updates, each with:
   - ref: Prompt reference ID - this must match the ref of an existing prompt
   - content: New prompt content
   - reason: Why this change is needed

Notes:
- A prompt change will directly modify the prompt used in future evaluations.
- GitHub issues should ONLY be created when ALL of these criteria are met:
  1. There is a clear bug or needed feature in the codebase (not in prompt content)
  2. You can specify exactly which files need to be modified
  3. You can provide the exact code changes needed (including file paths and line numbers)
  4. The issue cannot be fixed by updating prompts alone
  5. No similar issue already exists

- For most problems, prefer updating prompts rather than creating issues:
  - If success rate is low (<50%): Update evaluation questions lists ([type]_questions) and thresholds to be more permissive while ensuring no hallucinations
  - If output quality is poor: Update agent prompts and question lists
  - If agents make wrong tool calls: Add examples and clearer instructions
  - If reasoning is unclear: Update prompts to enforce better explanation format

- When updating prompts:
  - Return valid JSON matching the original prompt format
  - Include clear examples and constraints
  - Make reasoning requirements explicit
  - Focus on deriving clear insights from raw data
  - Add specific instructions for data queries and analysis

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
    for lambda_attempt in range(LAMBDA_MAX_RETRIES + 1):
        try:
            # Initialize/clear message tracker at start
            ToolMessageTracker().clear()
            
            # Get most recent stream key, optionally filtered by type
            eval_type = event.get('type')
            stream_key, timestamp = get_most_recent_stream_key(eval_type)
            
            if not stream_key:
                raise ValueError("No evaluations found")
            
            # Get full context including GitHub issues
            system_context = get_context(
                stream_key=stream_key,
                current_eval_timestamp=timestamp,
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
            
            # Build full prompt with any additional instructions
            full_prompt = SYSTEM_PROMPT + "\n\n" + SYSTEM_PROMPT_ADDITION
            if additional_instructions := event.get('additional_instructions'):
                full_prompt += f"\n\nAdditional Instructions:\n{additional_instructions}"
            
            # Run analysis with LLM with tool message tracking
            messages = [
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": f"Analyze this system state and provide recommendations:\n\n{system_context}"}
            ]
            
            analysis = None
            max_retries = 3
            for attempt in range(max_retries):
                analysis = run_completion_with_fallback(
                    messages=messages,
                    response_format=AnalysisResponse,
                    models=["long"],
                    include_tool_messages=True  # Include tool messages in context
                )
                if analysis:
                    break
                # If failed, add failure context to messages
                tool_context = ToolMessageTracker().get_context()
                messages.append({
                    "role": "user",
                    "content": f"Previous attempt failed. Here's the tool interaction history:\n{tool_context}\nPlease try again with this additional context."
                })
            
            if not analysis:
                raise ValueError("Failed to get analysis from LLM after all retries")
            
            # Print the analysis for debugging
            print("\nAnalysis results:")
            print(analysis)
            
            results = {
                'created_issues': [],
                'updated_prompts': [],
                'errors': [],
                'eval_type': eval_type,  # Include the type in response
                'additional_instructions': additional_instructions  # Include any additional instructions
            }
            
            # Create GitHub issues
            github_token = os.getenv('GITHUB_TOKEN')
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
                    'results': results,
                    'attempt': lambda_attempt  # Add attempt number to response
                })
            }

        except Exception as e:
            import traceback
            print(f"Error in attempt {lambda_attempt}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Add error to tracker
            ToolMessageTracker().add_message(
                tool_name="lambda",
                input_msg=f"lambda_handler attempt {lambda_attempt}",
                response="",
                error=str(e)
            )
            
            # If we have retries left, continue to next attempt
            if lambda_attempt < LAMBDA_MAX_RETRIES:
                print(f"Retrying lambda execution, attempt {lambda_attempt + 1}")
                continue
                
            # If we're out of retries, return error response
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'tool_messages': ToolMessageTracker().messages,
                    'final_attempt': lambda_attempt
                })
            }

if __name__ == "__main__":
    # Test with optional parameters
    test_event = {
        "type": "suggestions",  # Optional - filter by evaluation type
        "additional_instructions": "Focus on improving success rate by weakening [type]_question criteria"  # Optional - additional analysis instructions
    }
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))