import os
from dotenv import load_dotenv
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal
from pydantic import BaseModel, Field
from functools import wraps
import asyncio

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
    get_daily_metrics_from_table,
    get_dynamodb_table,
    get_top_prompt_content,
    log_debug, 
    log_error,
    PROMPT_TYPES,
    AGENT_TOOLS
)

load_dotenv()
is_local = os.getenv("IS_LOCAL") == "True"

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

Focus on updating these prompts since they are the ones that affect task completion and output quality:

{prompts}

Here are the tools available to the agents in the group for reference:

{tools}

Look at the most successful versions of each prompt (marked with 🏆) and identify patterns that make them effective. 
Prioritize keeping elements that contributed to successful evaluations.

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

Previous versions (from newest to oldest, with successful versions highlighted):
{previous_versions}

Historical performance data (past 7 days):
{historical_performance}

This prompt must include required variables (only use once) which will be substituted with their actual value in the prompt:
{variables}

These variables can optionally be in the prompt (use max once) which will be substituted with their actual value in the prompt:
{optional_variables}

Usage in code:
File: {code_file}
Line: {code_line}
Function call: {function_call}

{format_instructions}

When updating the prompt, follow these instructions:
1. Look at the successful versions (marked with 🏆) and understand what made them work well
2. Keep patterns and language that appear in successful versions
3. Incorporate improvements from the reason for update
{PROMPT_INSTRUCTIONS}

Generate ONLY the new content for the prompt. Do not include any explanations or comments outside the prompt content. Do not prefix the prompt (eg by adding version numbers or suffix the prompt because the prompt will be provided as is to the LLM model. Do not add a ``` or ```python at the start of the prompt since the prompt should not be wrapped)
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
            # Calculate score based on the formula
            successes = data.get('successes', 0)
            attempts = data.get('attempts', 0)
            quality_metric = data.get('quality_metric', 0)
            
            if successes == 0 or quality_metric == 0:
                score = min(10, attempts)
            else:
                score = 10 + 10 * quality_metric
                
            version_text = f"Version: {prompt_versions.get(date, 'unknown')}" if date in prompt_versions else "No version data"
            formatted_history.append(
                f"Date: {date}\n"
                f"{version_text}\n"
                f"Score: {score:.2f}\n"
                f"Success Rate: {data.get('success_rate', 0):.1f}%\n"
                f"Evaluations: {data.get('evaluations', 0)}\n"
                f"Successes: {data.get('successes', 0)}\n"
                f"Turns: {data.get('turns', 0)}\n"
                f"Attempts: {data.get('attempts', 0)}\n"
                f"Quality Metric: {data.get('quality_metric', 0):.2f}"
            )
            
        # Get top performing evaluations for this prompt instead of all versions
        top_evaluations = get_top_prompt_content(prompt_ref, max_evaluations=10, eval_type=eval_type)
        
        if top_evaluations:
            # Format successful evaluations first
            successful_evals = [e for e in top_evaluations if e.get('successes', 0) > 0]
            other_evals = [e for e in top_evaluations if e.get('successes', 0) == 0]
            
            # Add Version 0 (original version) for reference if available
            first_version = next((e for e in top_evaluations if e.get('prompt_version') == Decimal('0')), None)
            if first_version:
                version_num = first_version.get('prompt_version', '0')
                content = first_version.get('prompt_content', 'Content not available')
                
                formatted_history.append(
                    f"\nVersion {version_num} (Original - It might reference inaccurate information - eg agents that don't exist in workflow or responsibilites / tools that have changed -  so don't use this if so):\n{content}"
                )
            
            # Include successful evaluations
            if successful_evals:
                formatted_history.append("\nMOST SUCCESSFUL VERSIONS:")
                for eval_data in successful_evals:
                    version_num = eval_data.get('prompt_version', 'unknown')
                    content = eval_data.get('prompt_content', 'Content not available')
                    score = eval_data.get('score', 0)
                    date = eval_data.get('date', 'unknown')
                    
                    formatted_history.append(
                        f"🏆 Version {version_num} (Score: {score:.2f}, Date: {date}):\n{content}"
                    )
            
            # Add a few recent unsuccessful evaluations for comparison
            if other_evals and len(formatted_history) < 10:
                formatted_history.append("\nRecent Versions (Not Yet Successful):")
                for eval_data in other_evals[:2]:  # Just show a couple
                    version_num = eval_data.get('prompt_version', 'unknown')
                    content = eval_data.get('prompt_content', 'Content not available')
                    score = eval_data.get('score', 0)
                    date = eval_data.get('date', 'unknown')
                    
                    formatted_history.append(
                        f"Version {version_num} (Score: {score:.2f}, Date: {date}):\n{content}"
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
    analysis_output: Dict[str, Any] = None,
    validation_errors: List[str] = None
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
        validation_errors: List of validation errors from previous attempts
        
    Returns:
        Tuple of (content, error)
    """
    try:
        # Get prompt usage information to understand required variables
        usage_info = get_prompt_expected_parameters(prompt_ref)
        if not usage_info['found']:
            return None, f"Could not find usage of prompt {prompt_ref} in code"
    
        
        # Format previous versions as a string, emphasizing successful ones
        versions_text = ""
        top_evaluations = get_top_prompt_content(prompt_ref, max_evaluations=10, eval_type=eval_type)
        # Separate successful and other evaluations
        successful_evals = [e for e in top_evaluations if e.get('successes', 0) > 0]
        other_evals = [e for e in top_evaluations if e.get('successes', 0) == 0]
        
        # First show successful versions if any
        if successful_evals:
            versions_text += "SUCCESSFUL VERSIONS:\n\n"
            for eval_data in successful_evals:
                version_num = eval_data.get('prompt_version', 'N/A')
                content = eval_data.get('prompt_content', 'Content not available')
                score = eval_data.get('score', 0)
                date = eval_data.get('date', 'unknown')
                
                versions_text += f"🏆 VERSION {version_num} - SUCCESSFUL (Score: {score:.2f}, Date: {date}):\n{content}\n\n"
        
        # Then show other versions
        versions_text += "OTHER VERSIONS:\n\n"
        
        # Try to find version 0 (original version)
        # version_zero = next((e for e in top_evaluations if e.get('prompt_version') == Decimal('0')), None)
        # if version_zero:
        #     content = version_zero.get('prompt_content', 'Content not available')
        #     date = version_zero.get('date', 'unknown')
        #     versions_text += f"ORIGINAL VERSION (Version 0, Date: {date}):\n{content}\n\n"
        
        # Add a few more recent versions if available
        for eval_data in other_evals[:3]:  # Show up to 3 other versions
            if eval_data.get('prompt_version') != Decimal('0'):  # Skip version 0 as we already included it
                version_num = eval_data.get('prompt_version', 'N/A')
                content = eval_data.get('prompt_content', 'Content not available')
                score = eval_data.get('score', 0)
                date = eval_data.get('date', 'unknown')
                
                versions_text += f"VERSION {version_num} (Score: {score:.2f}, Date: {date}):\n{content}\n\n"
        
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
        
        # Add validation error feedback if provided
        validation_feedback = ""
        if validation_errors:
            # Create detailed feedback from all validation errors
            validation_feedback = "\n\nIMPORTANT: Previous update attempts failed validation with these errors:\n"
            for i, error in enumerate(validation_errors, 1):
                validation_feedback += f"\n{i}. {error}"
                
            # Add specific guidance for common error types
            if any("Missing output fields:" in err or "Missing reference fields:" in err for err in validation_errors):
                from utils import get_document_structure
                doc_structure = get_document_structure(prompt_ref) if prompt_ref.endswith('_questions') else None
                available_fields = list(doc_structure.keys()) if doc_structure else []
                
                validation_feedback += f"""
\nDOCUMENT STRUCTURE ERROR: You must ONLY use field names that exist in the document structure.
Available document fields you can use (use exact spelling and case):
{", ".join(available_fields)}

Please fix these issues by:
1. Ensuring all field names in "output" and "reference" arrays match exactly the field names listed above
2. Do not use any field names that aren't in the available fields list
3. Make sure each question has valid output and reference fields
4. Ensure the JSON format is valid and properly formatted
"""
            elif any("Invalid JSON" in err for err in validation_errors):
                validation_feedback += """
\nJSON FORMAT ERROR: Your response must be valid JSON. Please ensure:
1. All quotes are properly paired and escaped
2. All brackets and braces are properly balanced
3. All commas are correctly placed
4. For _questions prompts: The content must be a valid JSON array of objects
"""
        
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

        # Add validation errors as explicit messages if they exist
        if validation_errors:
            # Add a system message with the validation error details
            messages.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"Your previous response had validation errors that need to be fixed. Focus specifically on addressing these errors:\n\n{validation_feedback}\n\nFollow the instructions exactly and avoid introducing any explanations or comments. Output ONLY the corrected prompt content.",
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            })
            
            # Add a user message with the error details to create a more conversational flow
            messages.append({
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": f"Please fix these validation errors in your response:\n\n{validation_feedback}\n\nMake sure your response contains ONLY the corrected prompt content with no explanations or surrounding text.",
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

async def generate_updated_prompt_content_async(
    prompt_ref: str, 
    reason: str, 
    eval_type: str, 
    system_context: str = None,
    analysis_messages: List[Dict[str, str]] = None,
    analysis_output: Dict[str, Any] = None,
    validation_errors: List[str] = None
) -> tuple:
    """
    Truly asynchronous implementation of prompt content generation
    
    Args:
        prompt_ref: The prompt reference ID
        reason: Reason for update
        eval_type: Evaluation type
        system_context: System context
        analysis_messages: Analysis messages
        analysis_output: Analysis output
        validation_errors: List of validation errors from previous attempts
        
    Returns:
        Tuple of (content, error)
    """
    try:
        # Get prompt usage information using the async version
        from utils.prompt_utils import get_prompt_expected_parameters_async
        usage_info = await get_prompt_expected_parameters_async(prompt_ref)
        
        if not usage_info['found']:
            return None, f"Could not find usage of prompt {prompt_ref} in code"
        
        # Add validation_errors to generate_updated_prompt_content call
        return generate_updated_prompt_content(
            prompt_ref=prompt_ref,
            reason=reason,
            eval_type=eval_type,
            system_context=system_context,
            analysis_messages=analysis_messages,
            analysis_output=analysis_output,
            validation_errors=validation_errors
        )
    except Exception as e:
        import traceback
        return None, f"Error in async prompt content generation: {str(e)}\n{traceback.format_exc()}"

async def update_prompts_in_parallel(
    prompt_changes: List[PromptChangeRequest],
    eval_type: str,
    system_context: str,
    analysis_messages: List[Dict[str, str]],
    analysis_output: Dict[str, Any]
) -> Dict[str, List]:
    """
    Process multiple prompt updates in parallel
    
    Args:
        prompt_changes: List of prompt change requests
        eval_type: Evaluation type
        system_context: System context string
        analysis_messages: Original analysis messages
        analysis_output: Analysis output dictionary
        
    Returns:
        Dictionary with updated prompts and errors
    """
    parallel_start_time = time.time()
    print(f"⏰ Starting parallel prompt updates")
    
    updated_prompts = []
    errors = []
    tasks = []
    
    # Group requests by prompt reference to avoid race conditions
    prompt_refs_seen = set()
    prioritized_changes = []
    
    # Prioritize unique prompts - only process first request for each prompt ref
    for change in prompt_changes:
        if change.ref not in prompt_refs_seen:
            prompt_refs_seen.add(change.ref)
            prioritized_changes.append(change)
        else:
            print(f"Skipping duplicate update request for {change.ref}")
    
    print(f"⏰ Processing {len(prioritized_changes)} unique prompt updates in parallel after {time.time() - parallel_start_time:.2f}s of prep")
    
    # Create a task for each prompt update
    for change_request in prioritized_changes:
        prompt_ref = change_request.ref
        reason = change_request.reason
        
        # Create a task to process this prompt update
        task = asyncio.create_task(
            process_single_prompt_update(
                prompt_ref=prompt_ref,
                reason=reason,
                eval_type=eval_type,
                system_context=system_context,
                analysis_messages=analysis_messages,
                analysis_output=analysis_output
            )
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    tasks_start = time.time()
    print(f"⏰ Waiting for {len(tasks)} tasks to complete")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"⏰ All tasks completed in {time.time() - tasks_start:.2f}s (total parallel time: {time.time() - parallel_start_time:.2f}s)")
    
    # Process results
    for result in results:
        if isinstance(result, Exception):
            # Handle exceptions from tasks
            errors.append(f"Task error: {str(result)}")
        else:
            # Process successful result
            success, data = result
            if success:
                updated_prompts.append(data)
            else:
                errors.append(data)
    
    print(f"⏰ Prompt parallel processing complete: {len(updated_prompts)} succeeded, {len(errors)} failed")
    print(f"⏰ Total parallel update time: {time.time() - parallel_start_time:.2f}s")
    
    return {
        'updated_prompts': updated_prompts,
        'errors': errors
    }

async def process_single_prompt_update(
    prompt_ref: str,
    reason: str,
    eval_type: str,
    system_context: str,
    analysis_messages: List[Dict[str, str]],
    analysis_output: Dict[str, Any]
) -> tuple:
    """
    Process a single prompt update with retries
    
    Args:
        prompt_ref: Prompt reference ID
        reason: Reason for the update
        eval_type: Evaluation type
        system_context: System context
        analysis_messages: Analysis messages
        analysis_output: Analysis output
        
    Returns:
        Tuple of (success, result) where result is either the updated prompt data or an error message
    """
    start_time = time.time()
    print(f"⏰ Starting update for prompt: {prompt_ref}")
    
    # Try up to 3 times to generate valid content
    max_attempts = 3
    validation_errors = []
    
    for attempt in range(1, max_attempts + 1):
        attempt_start = time.time()
        print(f"⏰ Attempt {attempt}/{max_attempts} for prompt {prompt_ref} at {attempt_start - start_time:.2f}s")
        
        current_reason = reason
        if attempt > 1 and validation_errors:
            # Add validation errors to the reason for better guidance
            validation_summary = "\n\nPrevious validation errors:"
            for i, error in enumerate(validation_errors, 1):
                validation_summary += f"\n{i}. {error}"
            current_reason += validation_summary
        
        # Generate content for this prompt using the async version
        # Pass the validation_errors to the content generation function
        generation_start = time.time()
        content, error = await generate_updated_prompt_content_async(
            prompt_ref=prompt_ref, 
            reason=current_reason, 
            eval_type=eval_type,
            system_context=system_context,
            analysis_messages=analysis_messages,
            analysis_output=analysis_output,
            validation_errors=validation_errors if validation_errors else None
        )
        print(f"⏰ Content generation for {prompt_ref} (attempt {attempt}) took {time.time() - generation_start:.2f}s")
        
        if error:
            print(f"⏰ Error in content generation for {prompt_ref} after {time.time() - attempt_start:.2f}s: {error}")
            return (False, f"Error generating content for {prompt_ref}: {error}")
        
        # Attempt to update the prompt with the new content
        update_start = time.time()
        prompt_update_success, error_msg = update_prompt(prompt_ref, content)
        print(f"⏰ Prompt update operation for {prompt_ref} took {time.time() - update_start:.2f}s")
        
        if prompt_update_success:
            # Success!
            total_time = time.time() - start_time
            print(f"⏰ Successfully updated prompt {prompt_ref} in {total_time:.2f}s after {attempt} attempt(s)")
            return (True, {
                'ref': prompt_ref,
                'reason': reason,
                'version': 'new',
                'attempts': attempt,
                'time_taken': f"{total_time:.2f}s"
            })
        else:
            # Failed validation - collect errors and try again
            print(f"⏰ Validation failed for prompt {prompt_ref} after {time.time() - attempt_start:.2f}s - {error_msg}")
            validation_errors.append(error_msg)
    
    # If we get here, all attempts failed
    total_time = time.time() - start_time
    error_summary = f"Failed to update prompt {prompt_ref} after {max_attempts} attempts and {total_time:.2f}s"
    if validation_errors:
        error_summary += f": {', '.join(validation_errors)}"
    
    print(f"⏰ {error_summary}")
    return (False, error_summary)

def lambda_handler(event, context):
    """AWS Lambda handler for system analysis and updates."""
    start_time = time.time()
    print(f"⏰ Lambda handler started")
    
    try:
        # Get most recent stream key, optionally filtered by type
        eval_type = event.get('type')
        if not eval_type or eval_type not in PROMPT_TYPES:
            print(f"Invalid evaluation type: {eval_type}")
            eval_type = "okr"  # Default to OKR
        stream_key, timestamp = get_most_recent_stream_key(eval_type)
        print(f"⏰ Got stream key: {stream_key} in {time.time() - start_time:.2f}s")
        
        if not stream_key:
            raise ValueError("No evaluations found")
        
        # Get full context including GitHub issues
        context_start = time.time()
        print(f"⏰ Starting context retrieval at {context_start - start_time:.2f}s")
        system_context = get_context(
            stream_key=stream_key,
            current_eval_timestamp=timestamp,
            return_type="string",
            include_code_files=True
        )
        print(f"⏰ Context retrieval completed in {time.time() - context_start:.2f}s (total: {time.time() - start_time:.2f}s)")
        
        # Count tokens in system prompt and context
        token_start = time.time()
        from litellm.utils import token_counter
        system_tokens = token_counter(messages=[{"role": "system", "content": SYSTEM_PROMPT}], model="gpt-4")
        addition_tokens = token_counter(messages=[{"role": "system", "content": SYSTEM_PROMPT_ADDITION}], model="gpt-4")
        context_tokens = token_counter(messages=[{"role": "user", "content": system_context}], model="gpt-4")
        print(f"⏰ Token counting completed in {time.time() - token_start:.2f}s (total: {time.time() - start_time:.2f}s)")
        
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
        analysis_prep_start = time.time()
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
                        "text": SYSTEM_PROMPT_FINAL_INSTRUCTION.format(system_context=system_context, prompts=PROMPT_TYPES[eval_type], tools=AGENT_TOOLS[eval_type]),
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }
        ]
        print(f"⏰ Analysis message preparation completed in {time.time() - analysis_prep_start:.2f}s (total: {time.time() - start_time:.2f}s)")

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
        
        analysis_start = time.time()
        print(f"⏰ Starting LLM analysis at {analysis_start - start_time:.2f}s")
        analysis = run_completion_with_fallback(
            messages=analysis_messages,
            response_format=AnalysisResponse,
            models=["long"]
        )
        print(f"⏰ LLM analysis completed in {time.time() - analysis_start:.2f}s (total: {time.time() - start_time:.2f}s)")
        
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
        
        # STEP 2: For each prompt identified, generate detailed content updates in parallel
        if analysis.prompt_changes:
            prompt_updates_start = time.time()
            print(f"⏰ Starting prompt updates for {len(analysis.prompt_changes)} prompts at {prompt_updates_start - start_time:.2f}s")
            
            # Use asyncio to run updates in parallel
            if is_local:
                # When running locally, need to create a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                parallel_results = loop.run_until_complete(
                    update_prompts_in_parallel(
                        prompt_changes=analysis.prompt_changes,
                        eval_type=eval_type,
                        system_context=system_context,
                        analysis_messages=analysis_messages,
                        analysis_output=analysis.dict()
                    )
                )
                loop.close()
            else:
                # In Lambda, we can just run the async function directly
                parallel_results = asyncio.run(
                    update_prompts_in_parallel(
                        prompt_changes=analysis.prompt_changes,
                        eval_type=eval_type,
                        system_context=system_context,
                        analysis_messages=analysis_messages,
                        analysis_output=analysis.dict()
                    )
                )
            
            # Merge results from parallel processing
            results['updated_prompts'] = parallel_results['updated_prompts']
            results['errors'] = parallel_results['errors']
            print(f"⏰ Prompt updates completed in {time.time() - prompt_updates_start:.2f}s (total: {time.time() - start_time:.2f}s)")
        else:
            print(f"⏰ No prompt changes identified, skipping updates")
        
        total_time = time.time() - start_time
        print(f"⏰ Total execution time: {total_time:.2f}s")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'System analysis complete',
                'stream_key': stream_key,
                'execution_time': f"{total_time:.2f}s",
                'results': results
            })
        }

    except Exception as e:
        error_time = time.time() - start_time
        import traceback
        print(f"⏰ Error after {error_time:.2f}s: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'execution_time': f"{error_time:.2f}s",
                'traceback': traceback.format_exc()
            })
        }

if __name__ == "__main__":
    # Test with optional parameters
    script_start = time.time()
    print(f"⏰ Script started")
    
    task_types = ["okr", "insights", "suggestions", "design", "code"]
    task_type = task_types[0]
    test_event = {
        "type": task_type,  # Optional - filter by evaluation type
        "additional_instructions": f"Update all the {task_type} task prompts to ensure all the prompts follow the right format and all the agents have the right context to complete the task."
    }
    
    print(f"⏰ Calling lambda_handler at {time.time() - script_start:.2f}s")
    result = lambda_handler(test_event, None)
    print(f"⏰ Lambda handler returned after {time.time() - script_start:.2f}s")
    print(json.dumps(result, indent=2))
    print(f"⏰ Total script execution time: {time.time() - script_start:.2f}s")
