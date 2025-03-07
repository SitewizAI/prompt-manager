import os
from dotenv import load_dotenv
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal
from pydantic import BaseModel, Field
from functools import wraps

# Import from refactored utils package
from utils import (
    SYSTEM_PROMPT,
    PROMPT_INSTRUCTIONS, 
    run_completion_with_fallback, 
    get_context, 
    get_most_recent_stream_key,
    create_github_issue_with_project,
    update_prompt,
    get_prompt_expected_parameters,
    get_all_prompt_versions,
    get_daily_metrics_from_table,
    get_dynamodb_table,
    log_debug, 
    log_error
)

load_dotenv()
is_local = os.getenv("IS_LOCAL") == "True"

SYSTEM_PROMPT_ADDITION = """
Analyze the provided context including recent evaluations, prompts, code files, and GitHub issues.
Identify potential improvements and issues that need addressing.

Format your response as JSON with:

1. prompt_changes: List of prompt updates, each with:
   - ref: Prompt reference ID - this must match the ref of an existing prompt
   - reason: Why this change is needed and how to improve it, be specific

Notes:
- A prompt change will directly modify the prompt used in future evaluations.
- For most problems, prefer updating prompts rather than creating issues:
  - If success rate is low (<50%): Update evaluation questions lists ([type]_questions) and thresholds to be more permissive while ensuring no hallucinations
  - If output quality is poor: Update agent prompts and question lists
  - If agents make wrong tool calls: Add examples and clearer instructions
  - If reasoning is unclear: Update prompts to enforce better explanation format

- Your response should be concrete on WHAT should be changed, not the exact content itself.
- Clearly explain the specific issues with each prompt and how it should be improved.
- The actual updated content will be generated in a separate step.

The analysis should be data-driven based on evaluation metrics and failure patterns."""

SYSTEM_PROMPT_ADDITION_NO_GITHUB = """
Analyze the provided context including recent evaluations, prompts, code files, and GitHub issues.
Identify potential improvements and issues that need addressing.

Format your response as JSON with:

1. prompt_changes: List of prompt updates, each with:
    - ref: Prompt reference ID - this must match the ref of an existing prompt
    - reason: Why this change is needed and detailed guidance on how to update it

Notes:
- A prompt change will directly modify the prompt used in future evaluations.

- Update the prompts in the following ways:
    - If success rate is low (<50%): Update evaluation questions lists ([type]_questions) and thresholds to be more permissive while ensuring no hallucinations. This can be done by removing questions unlikely to succeed, reducing threshholds, and making questions more permissive. We must ensure a high success rate (> 50%).
    - If output quality is poor: Update agent prompts and question lists
    - If agents make wrong tool calls: Add examples and clearer instructions
    - If reasoning is unclear: Update prompts to enforce better explanation format

- Your response should focus on identifying which prompts need changes and why
- Don't include the new content in this phase, just explain what needs improvement
- Be specific about what aspects of each prompt need to be changed, and how

The analysis should be data-driven based on evaluation metrics and failure patterns."""

SYSTEM_PROMPT_FINAL_INSTRUCTION = """Analyze this system state and identify prompts that need updates:

{system_context}



Be very detailed for how the prompt should be updated and include any necessary context because the prompt engineer that will update the prompt does not have access to the code files, other prompts, or the context you have.
Eg include the following details:
- All the variables used in the prompt and examples of what they look like
- Responsibility of agent in context of the workflow
- Examples to use in the prompt
- Exactly how the prompt should be updated
"""

PROMPT_UPDATE_INSTRUCTIONS = """
We are updating this prompt: {prompt_ref}

Current reason for update:
{reason}

Previous versions (from newest to oldest):
{previous_versions}

Historical performance data (past 7 days):
{historical_performance}

This prompt must use these required variables:
{variables}

These variables can optionally be in the prompt:
{optional_variables}

Usage in code:
File: {code_file}
Line: {code_line}
Function call: {function_call}

{format_instructions}

When updating the prompt, follow these instructions:
{PROMPT_INSTRUCTIONS}

Generate ONLY the new content for the prompt. Do not include any explanations or comments outside the prompt content. Do not prefix the prompt (eg by adding version numbers or suffix the prompt because the prompt will be provided as is to the LLM model)
"""

# Add specific format instructions for question arrays
QUESTIONS_FORMAT_INSTRUCTIONS = """
IMPORTANT: This is a _questions prompt that must be formatted as a valid JSON array of question objects.
Each question object must follow this structure:
{{
  "question": "The question text to evaluate",
  "output": ["field1", "field2"],  // Fields to check from the document
  "reference": ["field3", "field4"],  // Reference fields to compare against
  "confidence_threshold": 0.7,  // Number between 0.0 and 1.0
  "feedback": "Feedback message if this question fails"
}}

The document structure this evaluates against contains these fields:
{document_fields}

All output and reference fields in your questions MUST exist in the document structure.
"""

SYSTEM_PROMPT_ADDITION = SYSTEM_PROMPT_ADDITION_NO_GITHUB
print(SYSTEM_PROMPT_ADDITION)

class PromptChangeRequest(BaseModel):
    ref: str = Field(..., description="Prompt reference")
    reason: str = Field(..., description="Reason for change and how to update")

class AnalysisResponse(BaseModel):
    prompt_changes: List[PromptChangeRequest] = Field(default=[])

class DetailedPromptUpdate(BaseModel):
    content: str = Field(..., description="New prompt content")

def measure_time(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"⏱️ {func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper

@measure_time
def get_prompt_historical_performance(prompt_ref: str, eval_type: str, days: int = 7) -> str:
    """
    Get historical performance data for a specific prompt reference.
    
    Args:
        prompt_ref: The prompt reference ID
        eval_type: The evaluation type (okr, insights, etc.)
        days: Number of days to look back
        
    Returns:
        Formatted string with historical performance data
    """
    try:
        # Get metrics for the specified evaluation type
        metrics = get_daily_metrics_from_table(eval_type=eval_type, days=days, get_prompts=True)
        if not metrics or not metrics.get('daily_metrics'):
            return "No historical data available"
            
        # Get prompt versions from the past days
        prompt_versions = {}
        if metrics.get('prompt_versions'):
            for version in metrics['prompt_versions']:
                if version.get('ref') == prompt_ref:
                    date = version.get('date', 'unknown')
                    version_num = version.get('version', 'unknown')
                    if date not in prompt_versions:
                        prompt_versions[date] = version_num
        
        # Format daily metrics with prompt versions
        daily_data = metrics['daily_metrics']
        formatted_history = []
        
        for date, data in daily_data.items():
            version_text = f"Version: {prompt_versions.get(date, 'unknown')}" if date in prompt_versions else "No version data"
            formatted_history.append(
                f"Date: {date}\n"
                f"{version_text}\n"
                f"Success Rate: {data.get('success_rate', 0):.1f}%\n"
                f"Evaluations: {data.get('evaluations', 0)}\n"
                f"Successes: {data.get('successes', 0)}\n"
                f"Turns: {data.get('turns', 0)}\n"
                f"Attempts: {data.get('attempts', 0)}\n"
                f"Quality Metric: {data.get('quality_metric', 0):.2f}"
            )
            
        # Get all versions for this prompt
        all_versions = get_all_prompt_versions(prompt_ref)
        
        # Modified approach: Skip metadata summaries and directly include version 0 content
        if all_versions:
            # Sort by version
            all_versions.sort(key=lambda x: int(x.get('version', 0)))
            
            # Always include version 0 (the original version)
            first_version = next((v for v in all_versions if int(v.get('version', 0)) == 0), None)
            if first_version:
                version_num = first_version.get('version', '0')
                content = first_version.get('content', 'Content not available')
                formatted_history.append(
                    f"\nVersion {version_num} (Original):\n{content}"
                )
            
            # Include a few recent versions if they exist
            recent_versions = [v for v in all_versions if int(v.get('version', 0)) > 0]
            # Sort recent versions descending by version number
            recent_versions.sort(key=lambda x: int(x.get('version', 0)), reverse=True)
            # Get the 3 most recent versions
            recent_versions = recent_versions[:3]
            
            if recent_versions:
                formatted_history.append("\nRecent Versions:")
                for version in recent_versions:
                    version_num = version.get('version', 'unknown')
                    content = version.get('content', 'Content not available')
                    formatted_history.append(
                        f"Version {version_num}:\n{content}"
                    )
                
        return "\n\n".join(formatted_history)
    except Exception as e:
        log_error(f"Error getting historical performance: {str(e)}")
        import traceback
        log_debug(traceback.format_exc())
        return f"Failed to get historical data: {str(e)}"

def generate_updated_prompt_content(
    prompt_ref: str, 
    reason: str, 
    eval_type: str, 
    system_context: str = None,
    analysis_messages: List[Dict[str, str]] = None,
    analysis_output: Dict[str, Any] = None
) -> tuple:
    """
    Generate detailed prompt content for a specific prompt reference.
    
    Args:
        prompt_ref: The reference of the prompt to update
        reason: The reason for updating and guidance on how to update
        eval_type: The evaluation type (okr, insights, etc.)
        system_context: Optional system context from the analysis step
        analysis_messages: Original messages from the analysis step (for context)
        analysis_output: Analysis output to maintain context
        
    Returns:
        Tuple of (content, error)
    """
    try:
        # Get prompt usage information to understand required variables
        usage_info = get_prompt_expected_parameters(prompt_ref)
        if not usage_info['found']:
            return None, f"Could not find usage of prompt {prompt_ref} in code"
        
        # Get previous versions of the prompt
        previous_versions = get_all_prompt_versions(prompt_ref)
        if not previous_versions:
            return None, f"Could not retrieve previous versions of prompt {prompt_ref}"
        
        # Format previous versions as a string (up to 10)
        versions_text = ""
        for i, version in enumerate(previous_versions[:10]):
            content = version.get('content', '')
            version_num = version.get('version', 'N/A')
            versions_text += f"VERSION {version_num}:\n{content}\n\n"
        
        # Get historical performance data
        historical_performance = get_prompt_historical_performance(prompt_ref, eval_type)
            
        # Format the required and optional variables
        required_vars = ", ".join([f"{{{var}}}" for var in usage_info['parameters']]) or "None"
        optional_vars = ", ".join([f"{{{var}}}" for var in usage_info['optional_parameters']]) or "None"
        
        # Check if this is a _questions type prompt and add specific format instructions
        format_instructions = ""
        if prompt_ref.endswith('_questions'):
            # Get document structure for this question type
            from utils import get_document_structure
            doc_structure = get_document_structure(prompt_ref)
            document_fields = [f.strip() for f in list(doc_structure.keys())] if doc_structure else []
            
            # Add document fields to format instructions
            fields_text = ", ".join(f'"{f}"' for f in document_fields) if document_fields else "No fields found"
            format_instructions = QUESTIONS_FORMAT_INSTRUCTIONS.format(document_fields=fields_text)
        
        # Format the update instructions with the specific prompt information
        update_instructions = PROMPT_UPDATE_INSTRUCTIONS.format(
            prompt_ref=prompt_ref,
            reason=reason,
            previous_versions=versions_text,
            historical_performance=historical_performance,
            variables=required_vars,
            optional_variables=optional_vars,
            code_file=usage_info['file'],
            code_line=usage_info['line'],
            function_call=usage_info['function_call'],
            format_instructions=format_instructions,
            PROMPT_INSTRUCTIONS=PROMPT_INSTRUCTIONS
        )
        
        # Build messages with enhanced context from the analysis step
        messages = []
        
        # Use the original analysis messages if provided, otherwise use basic system prompt
        if analysis_messages:
            messages += analysis_messages
            
            # Add context about the previous analysis output with cache_control
            if analysis_output:
                analysis_summary = f"Based on the previous analysis that identified these prompt changes: {json.dumps(analysis_output, indent=2)}\n\n"
                messages.append({
                    "role": "assistant", 
                    "content": [
                        {
                            "type": "text",
                            "text": analysis_summary,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                })
        elif system_context:
            # Fallback to just using system context if analysis messages not provided
            context_text = f"{SYSTEM_PROMPT}\n\n{SYSTEM_PROMPT_ADDITION}\n\nPrevious System Context:\n{system_context}"
            messages.append({
                "role": "system", 
                "content": [
                    {
                        "type": "text",
                        "text": context_text,
                    }
                ]
            })
        else:
            # Basic fallback if no context provided
            messages.append({
                "role": "system", 
                "content": "You are a prompt engineering expert. Generate only the updated prompt content based on the provided instructions."
            })
            
        # Add the update instructions as user message
        messages.append({
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": update_instructions,
                }
            ]
        })

        # save to file
        if is_local:
            with open("detailed_prompt_update.txt", "w") as f:
                # write all the messages
                for msg in messages:
                    f.write(f"Role: {msg['role']}\n")
                    content = msg['content']
                    if isinstance(content, list):
                        for item in content:
                            f.write(f"Content: {item['text']}\n\n")
                    else:
                        f.write(f"Content: {content}\n\n")
        
        content = run_completion_with_fallback(
            messages=messages,
            models=["long"]  # Use the reasoning model for prompt generation
        )
        
        return content, None
    except Exception as e:
        import traceback
        return None, f"Error generating prompt content: {str(e)}\n{traceback.format_exc()}"

def lambda_handler(event, context):
    """AWS Lambda handler for system analysis and updates."""
    try:
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
            include_code_files=True
        )
        
        # Count tokens in system prompt and context
        from litellm.utils import token_counter
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
        
        # STEP 1: Run initial analysis with LLM to identify which prompts need updates and why
        # Use cache_control to enable prompt caching for large context
        analysis_messages = [
            {
                "role": "system", 
                "content": [
                    {
                        "type": "text",
                        "text": full_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_FINAL_INSTRUCTION.format(system_context=system_context),
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        ]

        # save to file
        if is_local:
            with open("system_analysis.txt", "w") as f:
                for msg in analysis_messages:
                    f.write(f"Role: {msg['role']}\n")
                    content = msg['content']
                    if isinstance(content, list):
                        for item in content:
                            f.write(f"Content: {item['text']}\n\n")
                    else:
                        f.write(f"Content: {content}\n\n")
        
        analysis = run_completion_with_fallback(
            messages=analysis_messages,
            response_format=AnalysisResponse,
            models=["long"]
        )
        
        if not analysis:
            raise ValueError("Failed to get analysis from LLM")
        
        # Print the analysis for debugging
        print("\nAnalysis results (Step 1):")
        print(analysis)
        
        results = {
            'updated_prompts': [],
            'errors': [],
            'eval_type': eval_type,  # Include the type in response
            'additional_instructions': additional_instructions  # Include any additional instructions
        }
        
        # Parse response
        analysis = AnalysisResponse(**analysis)
        
        # STEP 2: For each prompt identified, generate detailed content updates
        if analysis.prompt_changes:
            import asyncio
            for change_request in analysis.prompt_changes:
                prompt_ref = change_request.ref
                reason = change_request.reason
                
                print(f"\nGenerating detailed content update for prompt: {prompt_ref}")
                
                # Try up to 3 times to generate valid content
                max_attempts = 3
                success = False
                validation_errors = []
                
                for attempt in range(1, max_attempts + 3):
                    print(f"Attempt {attempt}/{max_attempts} for prompt {prompt_ref}")
                    
                    if attempt > 1 and validation_errors:
                        # Add validation errors to the reason for better guidance
                        validation_summary = "\n\nPrevious validation errors:"
                        for i, error in enumerate(validation_errors, 1):
                            validation_summary += f"\n{i}. {error}"
                        reason += validation_summary
                    
                    # Pass the original analysis messages and output along with other parameters
                    content, error = generate_updated_prompt_content(
                        prompt_ref=prompt_ref, 
                        reason=reason, 
                        eval_type=eval_type,
                        system_context=system_context,
                        analysis_messages=analysis_messages,  # Pass the cached messages
                        analysis_output=analysis.dict()       # Pass the analysis output
                    )
                    
                    if error:
                        print(f"Error in content generation: {error}")
                        results['errors'].append(f"Error generating content for {prompt_ref}: {error}")
                        break
                    
                    # Attempt to update the prompt with the new content
                    update_result = update_prompt(prompt_ref, content)
                    
                    if update_result:  # Direct boolean check instead of using .get()
                        # Success!
                        print(f"Successfully updated prompt {prompt_ref}")
                        results['updated_prompts'].append({
                            'ref': prompt_ref,
                            'reason': reason,
                            'version': 'new',  # We don't have access to the exact version number here
                            'attempts': attempt
                        })
                        success = True
                        break
                    else:
                        # Failed validation - collect errors and try again
                        error_msg = "Validation failed for prompt update"
                        print(f"Validation failed for prompt {prompt_ref} - {error_msg}")
                        validation_errors.append(error_msg)
                        
                        # Note: We can't get detailed validation info since update_prompt just returns bool
                
                # Record final error if all attempts failed
                if not success:
                    error_summary = f"Failed to update prompt {prompt_ref} after {max_attempts} attempts"
                    print(error_summary)
                    if validation_errors:
                        error_summary += f": {', '.join(validation_errors)}"
                    results['errors'].append(error_summary)
        
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
    # Test with optional parameters
    test_event = {
        "type": "okr",  # Optional - filter by evaluation type
    }
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))
