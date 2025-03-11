import os
import json
import boto3
from typing import List, Dict, Any, Optional, Tuple
import traceback
import time
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.completion_utils import (
    PROMPT_INSTRUCTIONS,
    SYSTEM_PROMPT,
    run_completion_with_fallback,
    get_boto3_client
)

from utils.prompt_utils import PROMPT_TYPES, AGENT_TOOLS, update_prompt

from utils.logging_utils import log_debug, log_error, measure_time
from utils import get_dynamodb_table

PROMPT_RESET_SYSTEM = """You are an expert prompt engineer responsible for creating high-quality prompts.
Your task is to reset a prompt to version 1 based on the provided instructions and context.

Review the current version 0 of the prompt and create a new improved version following the best practices in the instructions.
Your goal is to create a prompt that:
1. Follows all formatting requirements
2. Uses the required variables correctly
3. Provides clear instructions to the model
4. Avoids any anti-patterns or problematic instructions
5. Improves upon the original version based on the provided instructions

Output ONLY the new prompt content. Do not include any explanations, comments, or metadata."""

PROMPT_RESET_INSTRUCTIONS = """
We are resetting this prompt: {prompt_ref} to version 1

Current version 0:
{current_content}

This prompt must include these required variables:
{variables}

These variables are optional and can be included if needed:
{optional_variables}

{tools_information}

Usage in code:
File: {code_file}
Line: {code_line}
Function call: {function_call}

{format_instructions}

When creating the new version 1, follow these instructions:
{SYSTEM_PROMPT}

Generate ONLY the new prompt content. Do not include any explanations or comments outside the prompt content. Do not prefix the prompt (eg by adding version numbers or suffix the prompt because the prompt will be provided as is to the LLM model. Do not add a ``` or ```python at the start of the prompt since the prompt should not be wrapped)
"""

@measure_time
def get_prompt_from_dynamodb(prompt_ref: str, version: Optional[int] = None) -> Dict[str, Any]:
    """
    Get prompt from DynamoDB by reference and optional version.
    
    Args:
        prompt_ref: The prompt reference ID
        version: Optional specific version to retrieve (default: latest)
        
    Returns:
        Dictionary with prompt data or empty dict if not found
    """
    try:
        table = get_dynamodb_table('PromptsTable')
        
        if version is not None:
            response = table.get_item(
                Key={
                    'ref': prompt_ref,
                    'version': version
                }
            )
        else:
            # Get the latest version
            response = table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('ref').eq(prompt_ref),
                ScanIndexForward=False,  # Sort in descending order (newest first)
                Limit=1
            )
            if 'Items' in response and response['Items']:
                return response['Items'][0]
            return {}
            
        return response.get('Item', {})
    except Exception as e:
        log_error(f"Error retrieving prompt from DynamoDB: {str(e)}")
        return {}

@measure_time
def get_all_prompt_versions(prompt_ref: str) -> List[Dict[str, Any]]:
    """
    Get all versions of a prompt.
    
    Args:
        prompt_ref: The prompt reference ID
        
    Returns:
        List of prompt versions ordered by version number
    """
    try:
        table = get_dynamodb_table('PromptsTable')
        
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key('ref').eq(prompt_ref),
            ScanIndexForward=True  # Sort in ascending order by version
        )
        
        items = response.get('Items', [])
        
        # Sort by version just to be sure
        items.sort(key=lambda x: int(x.get('version', 0)))
        
        return items
    except Exception as e:
        log_error(f"Error retrieving prompt versions: {str(e)}")
        return []

@measure_time
def delete_prompt_version(prompt_ref: str, version: int) -> bool:
    """
    Delete a specific version of a prompt.
    
    Args:
        prompt_ref: The prompt reference ID
        version: Version to delete
        
    Returns:
        Boolean indicating success
    """
    try:
        table = get_dynamodb_table('PromptsTable')
        
        table.delete_item(
            Key={
                'ref': prompt_ref,
                'version': version
            }
        )
        
        log_debug(f"Deleted prompt {prompt_ref} version {version}")
        return True
    except Exception as e:
        log_error(f"Error deleting prompt version: {str(e)}")
        return False

@measure_time
def get_prompt_expected_parameters(prompt_ref: str) -> Dict[str, Any]:
    """
    Get expected parameters for a prompt by analyzing code usage.
    
    Args:
        prompt_ref: The prompt reference ID
        
    Returns:
        Dictionary with usage information
    """
    # This is a simplified version - in a real implementation, 
    # we would analyze the codebase to find where the prompt is used
    try:
        # Placeholder implementation
        # In a real scenario, we might scan source files to find usage
        return {
            'found': True,
            'file': 'unknown_file.py',
            'line': 0,
            'function_call': f'get_prompt_from_dynamodb("{prompt_ref}")',
            'parameters': [],
            'optional_parameters': []
        }
    except Exception as e:
        log_error(f"Error getting prompt parameters: {str(e)}")
        return {'found': False}

@measure_time
def get_all_code_files(directory: str = "/Users/ram/Github/prompt-manager") -> str:
    """
    Get all code files in the specified directory for context.
    
    Args:
        directory: The directory to scan for code files
        
    Returns:
        String containing code file contents
    """
    try:
        code_files_content = []
        extensions = ['.py', '.js', '.tsx', '.ts']
        exclude_dirs = ['node_modules', 'venv', '__pycache__', '.git']
        
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            code_files_content.append(f"# File: {file_path}\n{content}")
                    except Exception as e:
                        log_error(f"Error reading file {file_path}: {str(e)}")
        
        return "\n\n".join(code_files_content)
    except Exception as e:
        log_error(f"Error getting code files: {str(e)}")
        return ""

@measure_time
def generate_reset_prompt_content(prompt_ref: str, validation_error: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate new content for version 1 of the prompt based on version 0.
    Uses cache control with a two-step message approach for better performance.
    Includes retry logic for completion failures.
    
    Args:
        prompt_ref: The prompt reference ID
        validation_error: Optional validation error from previous attempt
        
    Returns:
        Tuple of (content, error)
    """
    try:
        # Get version 0 of the prompt
        version_0 = get_prompt_from_dynamodb(prompt_ref, 0)
        if not version_0:
            return None, f"Could not find version 0 of prompt {prompt_ref}"
        
        # ... existing code for preparation ...
        current_content = version_0.get('content', '')
        
        # Get prompt usage information
        usage_info = get_prompt_expected_parameters(prompt_ref)
        if not usage_info['found']:
            return None, f"Could not find usage of prompt {prompt_ref} in code"
        
        # Format the required and optional variables
        required_vars = ", ".join([f"{{{var}}}" for var in usage_info['parameters']]) or "None"
        optional_vars = ", ".join([f"{{{var}}}" for var in usage_info['optional_parameters']]) or "None"
        
        # Format instructions based on prompt type
        format_instructions = ""
        is_question_prompt = prompt_ref.endswith('_questions')
        if is_question_prompt:
            # Get document structure for this question type for more precise instructions
            from utils import get_document_structure
            doc_structure = get_document_structure(prompt_ref) if prompt_ref.endswith('_questions') else None
            document_fields = list(doc_structure.keys()) if doc_structure else []
            
            # Include available document fields in the format instructions
            fields_list = "\n".join([f"- {field}" for field in document_fields])
            format_instructions = f"""
            IMPORTANT: This is a _questions prompt that must be formatted as a valid JSON array of question objects.
            Each question object must follow this structure:
            {{
                "question": "The question text to evaluate",
                "output": ["field1", "field2"],  // Fields to check from the document
                "reference": ["field3", "field4"],  // Reference fields to compare against
                "confidence_threshold": 0.7,  // Number between 0.0 and 1.0
                "feedback": "Feedback message if this question fails"
            }}
            
            Your response MUST be valid JSON that can be parsed - enclose the entire array in square brackets [ ].
            
            AVAILABLE DOCUMENT FIELDS (use ONLY these exact field names in output and reference arrays):
            {fields_list}
            """

        # Add validation error feedback if provided
        validation_feedback = ""
        if validation_error:
            # Enhanced validation feedback specifically for questions prompts with document field errors
            if "Missing output fields:" in validation_error or "Missing reference fields:" in validation_error:
                from utils import get_document_structure
                doc_structure = get_document_structure(prompt_ref) if prompt_ref.endswith('_questions') else None
                available_fields = list(doc_structure.keys()) if doc_structure else []
                
                validation_feedback = f"""
IMPORTANT: The previous attempt to update this prompt failed validation with the following error:
{validation_error}

DOCUMENT STRUCTURE ERROR: You must ONLY use field names that exist in the document structure.
Available document fields you can use (use exact spelling and case):
{", ".join(available_fields)}

Please fix these issues by:
1. Ensuring all field names in "output" and "reference" arrays match exactly the field names listed above
2. Do not use any field names that aren't in the available fields list
3. Make sure each question has valid output and reference fields
4. Ensure the JSON format is valid and properly formatted
"""
            else:
                validation_feedback = f"""
IMPORTANT: The previous attempt to update this prompt failed validation with the following error:
{validation_error}

Please fix these issues in your response. Pay special attention to:
1. Using only the required and optional variables specified
2. Correct JSON formatting (for question prompts) - ensure it's valid JSON that can be parsed
3. Ensuring all required variable placeholders are present
4. Avoiding extra variables not listed in the requirements

For _questions prompts: Make sure to return ONLY the JSON array without any surrounding text, markdown or code blocks.
"""

        # Determine which group this prompt belongs to and get tools information
        tools_information = ""
        prompt_group = None
        is_agent_prompt = False
        agent_name = None
        
        # Find which group this prompt belongs to
        for group, prompts in PROMPT_TYPES.items():
            if prompt_ref in prompts:
                prompt_group = group
                break
        
        # For agent system messages, provide tools information
        if prompt_ref.endswith('_system_message'):
            is_agent_prompt = True
            # Extract agent name from prompt ref (remove _system_message suffix)
            agent_name = prompt_ref.replace('_system_message', '')
            
        if prompt_group and is_agent_prompt and agent_name:
            # Get tools for this agent in this group
            if prompt_group in AGENT_TOOLS and agent_name in AGENT_TOOLS[prompt_group]:
                agent_tools = AGENT_TOOLS[prompt_group][agent_name]
                if agent_tools:
                    tools_information = f"""
Available tools for this agent in the {prompt_group} group:
{', '.join(agent_tools)}

Make sure the prompt only references these tools and no others.
"""
                else:
                    tools_information = f"This agent does not have any tools available in the {prompt_group} group."
            else:
                tools_information = f"Could not find tools information for agent {agent_name} in group {prompt_group}."
        elif prompt_group:
            # For non-agent prompts, provide general tools info for the group
            tools_info = []
            if prompt_group in AGENT_TOOLS:
                for agent, tools in AGENT_TOOLS[prompt_group].items():
                    if tools:
                        tools_info.append(f"- {agent}: {', '.join(tools)}")
                
                if tools_info:
                    tools_information = f"""
Tools available to agents in the {prompt_group} group (if this agent isn't listed, they don't have access to any tools):
{chr(10).join(tools_info)}
"""
        # Get all code files for context (will be cached)
        code_files = get_all_code_files()
        
        # Format the reset instructions for the second message
        reset_instructions = PROMPT_RESET_INSTRUCTIONS.format(
            prompt_ref=prompt_ref,
            current_content=current_content,
            variables=required_vars,
            optional_variables=optional_vars,
            tools_information=tools_information,
            code_file=usage_info['file'],
            code_line=usage_info['line'],
            function_call=usage_info['function_call'],
            format_instructions=format_instructions + validation_feedback,  # Add validation feedback
            SYSTEM_PROMPT=SYSTEM_PROMPT
        )

        print("Reset Instructions:")
        print(reset_instructions)
        
        # Two-step message approach with cache control
        messages = [
            # First message: System prompt + all code files (cached)
            {
                "role": "system", 
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_RESET_SYSTEM,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here are all the code files for context:\n" + code_files,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            },
            # Second message: Specific prompt information (not cached)
            {
                "role": "user", 
                "content": reset_instructions
            }
        ]
        
        # Include explicit system instructions to override any previous messaging
        if validation_error:
            # Add a clear instruction to guide the LLM in fixing the error
            messages.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Your previous response had validation errors that need to be fixed. Focus on correcting the specific issues mentioned in the error message. Follow the instructions exactly and avoid introducing any explanations or comments. Output ONLY the corrected prompt content.",
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            })

        # Run completion with retries
        max_retries = 3
        retry_count = 0
        content = None
        last_error = None

        while retry_count < max_retries:
            try:
                retry_count += 1
                log_debug(f"Completion attempt {retry_count}/{max_retries} for prompt {prompt_ref}")
                
                content = run_completion_with_fallback(
                    messages=messages,
                    models=["long"]
                )
                
                if content:
                    break
                else:
                    last_error = f"Empty response on attempt {retry_count}"
                    log_error(last_error)
            except Exception as e:
                last_error = f"Completion attempt {retry_count} failed: {str(e)}"
                log_error(last_error)
                
            if retry_count < max_retries:
                # Exponential backoff: 2^retry_count seconds
                backoff_time = 2 ** retry_count
                log_debug(f"Waiting {backoff_time} seconds before retry {retry_count+1}")
                time.sleep(backoff_time)

        if not content:
            return None, f"Failed to generate prompt content after {max_retries} attempts. Last error: {last_error}"
        
        # Process the content based on prompt type before returning
        if content:
            # For _questions prompts, ensure the content is valid JSON
            if is_question_prompt:
                try:
                    # First, try to parse the response as JSON
                    content = content.strip()
                    
                    # Remove any markdown code block formatting if present
                    if content.startswith("```json") or content.startswith("```"):
                        # Find the first ``` and the last ```
                        start_idx = content.find("```") + 3
                        # Skip the word "json" if it's there
                        if content[start_idx:start_idx+4] == "json":
                            start_idx += 4
                        end_idx = content.rfind("```")
                        # Extract just the JSON part
                        content = content[start_idx:end_idx].strip()
                    
                    # Parse to validate it's proper JSON
                    parsed_json = json.loads(content)
                    
                    # Ensure it's an array
                    if not isinstance(parsed_json, list):
                        return None, f"Questions content must be a JSON array, got {type(parsed_json).__name__}"
                    
                    # Return the cleaned JSON string with proper formatting
                    content = json.dumps(parsed_json, indent=2)
                    log_debug(f"Successfully parsed and formatted JSON content for {prompt_ref}")
                except json.JSONDecodeError as e:
                    return None, f"Invalid JSON format in generated content: {str(e)}"
                except Exception as e:
                    return None, f"Error processing JSON content: {str(e)}"

        return content, None
    except Exception as e:
        return None, f"Error generating prompt content: {str(e)}\n{traceback.format_exc()}"

def reset_prompt(prompt_ref: str) -> Dict[str, Any]:
    """
    Reset a prompt to version 1 and delete all versions after 1.
    
    This function:
    1. First deletes all versions >= 1 of the prompt
    2. Generates new optimized content for version 1 using AI
    3. Updates version 1 with the new content using update_prompt from prompt_utils
    
    If validation fails, it retries with feedback about the validation errors.
    
    Args:
        prompt_ref: The prompt reference ID
        
    Returns:
        Dictionary with results including success status and versions deleted
    """
    results = {
        'prompt_ref': prompt_ref,
        'success': False,
        'error': None,
        'versions_deleted': [],
    }
    
    try:
        log_debug(f"Resetting prompt {prompt_ref}")
        
        # Get all versions of the prompt
        versions = get_all_prompt_versions(prompt_ref)
        if not versions:
            results['error'] = f"No versions found for prompt {prompt_ref}"
            return results
        
        # STEP 1: First delete all versions >= 1
        deleted_versions = []
        for version in versions:
            version_num = version.get('version')
            if version_num is not None and int(version_num) >= 1:
                success = delete_prompt_version(prompt_ref, int(version_num))
                if success:
                    deleted_versions.append(int(version_num))
                else:
                    log_error(f"Failed to delete version {version_num} of prompt {prompt_ref}")
        
        log_debug(f"Deleted {len(deleted_versions)} versions of prompt {prompt_ref}: {deleted_versions}")
        
        # STEP 2 & 3: Generate content and update prompt with retries
        max_retries = 3
        retry_count = 0
        update_success = False
        last_error = None
        validation_error = None
        
        while retry_count < max_retries and not update_success:
            try:
                retry_count += 1
                log_debug(f"Prompt update attempt {retry_count}/{max_retries} for prompt {prompt_ref}")
                
                # Generate content with validation error feedback if this is a retry
                content, error = generate_reset_prompt_content(prompt_ref, validation_error)
                if error:
                    last_error = error
                    log_error(f"Content generation failed: {error}")
                    continue
                
                # Check if we need to parse as JSON for _questions prompts
                # Note: This is a double-check in case generate_reset_prompt_content didn't handle it
                if prompt_ref.endswith('_questions') and isinstance(content, str):
                    try:
                        # First try to parse as JSON to validate
                        parsed_json = json.loads(content)
                        # Use the parsed object directly to ensure update_prompt knows it's an object
                        content_to_update = parsed_json
                        log_debug(f"Using parsed JSON object for {prompt_ref}")
                    except json.JSONDecodeError as e:
                        validation_error = f"Invalid JSON format: {str(e)}. Make sure to provide a valid JSON array of question objects."
                        last_error = f"Invalid JSON format: {str(e)}"
                        log_error(last_error)
                        continue
                else:
                    content_to_update = content
                
                # Use the imported update_prompt from prompt_utils - returns (success, error_message)
                prompt_update_success, error_msg = update_prompt(prompt_ref, content_to_update)
                
                # Check if update was successful
                if prompt_update_success:
                    update_success = True
                    print(f"Update successful for prompt {prompt_ref}")
                    print(f"New content:\n{content}")
                    break
                else:
                    # Store validation error for next retry
                    validation_error = error_msg
                    last_error = f"Update failed: {error_msg}"
                    print(f"Validation failed for prompt {prompt_ref}: {error_msg}")
                    log_error(last_error)
            except Exception as e:
                last_error = f"Update attempt {retry_count} failed: {str(e)}"
                log_error(last_error)
            
            if retry_count < max_retries and not update_success:
                # Exponential backoff: 2^retry_count seconds
                backoff_time = 2 ** retry_count
                log_debug(f"Waiting {backoff_time} seconds before retry {retry_count+1}")
                time.sleep(backoff_time)
        
        if not update_success:
            results['error'] = f"Failed to update prompt {prompt_ref} after {max_retries} attempts. Last error: {last_error}"
            return results
        
        results['success'] = True
        results['versions_deleted'] = deleted_versions
        
        log_debug(f"Successfully reset prompt {prompt_ref} to version 1")
        log_debug(f"Deleted versions: {deleted_versions}")
        
        return results
    except Exception as e:
        results['error'] = f"Error resetting prompt: {str(e)}"
        log_error(f"Error in reset_prompt: {str(e)}")
        log_error(traceback.format_exc())
        return results

def process_prompt(prompt_ref: str) -> Dict[str, Any]:
    """
    Process a single prompt - wrapper for reset_prompt to use with ThreadPoolExecutor.
    
    Args:
        prompt_ref: The prompt reference ID
        
    Returns:
        Result dictionary with status information
    """
    try:
        result = reset_prompt(prompt_ref)
        if result['success']:
            print(f"Successfully reset prompt {prompt_ref} to version 1")
            print(f"Deleted versions: {result['versions_deleted']}")
        else:
            print(f"Failed to reset prompt: {result['error']}")
        return result
    except Exception as e:
        error_msg = f"Error processing prompt {prompt_ref}: {str(e)}"
        log_error(error_msg)
        print(error_msg)
        return {
            'prompt_ref': prompt_ref,
            'success': False,
            'error': error_msg,
            'versions_deleted': []
        }

def process_prompt_group(prompt_refs: List[str], max_workers: int = 5) -> List[Dict[str, Any]]:
    """
    Process multiple prompts in parallel using ThreadPoolExecutor.
    
    Args:
        prompt_refs: List of prompt reference IDs to process
        max_workers: Maximum number of parallel workers (default: 5)
        
    Returns:
        List of result dictionaries for each prompt
    """
    results = []
    
    print(f"Processing {len(prompt_refs)} prompts with {max_workers} parallel workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_prompt = {executor.submit(process_prompt, ref): ref for ref in prompt_refs}
        
        # Process results as they complete
        for future in as_completed(future_to_prompt):
            prompt_ref = future_to_prompt[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                error_msg = f"Exception processing prompt {prompt_ref}: {str(e)}"
                log_error(error_msg)
                print(error_msg)
                results.append({
                    'prompt_ref': prompt_ref,
                    'success': False,
                    'error': error_msg,
                    'versions_deleted': []
                })
    
    return results

if __name__ == "__main__":
    # Process all prompts in parallel by type
    prompt_types = ["okr", "insights", "suggestions"]
    prompt_types = ["okr", "insights", "suggestions", "design", "code"]
    # Process each type with parallel execution
    all_results = {}
    for prompt_type in prompt_types:
        print(f"\nProcessing {prompt_type} prompts...")
        prompt_refs = PROMPT_TYPES[prompt_type]
        
        # Determine an appropriate number of workers (adjust based on your system and API limits)
        max_workers = min(20, len(prompt_refs))  # No more than 10 workers, or fewer if fewer prompts
        
        # Process this group of prompts in parallel
        results = process_prompt_group(prompt_refs, max_workers)
        all_results[prompt_type] = results
        
        # Print summary for this group
        success_count = sum(1 for r in results if r['success'])
        print(f"Completed {prompt_type} prompts: {success_count} successful, {len(results) - success_count} failed")
    
    # Print overall summary
    total = sum(len(results) for results in all_results.values())
    successful = sum(sum(1 for r in results if r['success']) for results in all_results.values())
    print(f"\nOverall results: {successful}/{total} prompts successfully reset")
    
    # Print details about failed prompts
    failed_prompts = []
    for prompt_type, results in all_results.items():
        for result in results:
            if not result['success']:
                failed_prompts.append({
                    'type': prompt_type,
                    'ref': result['prompt_ref'],
                    'error': result['error']
                })
    
    if failed_prompts:
        print("\nFailed prompt resets:")
        print("-" * 80)
        
        # Group by error message for better readability
        error_groups = {}
        for failed in failed_prompts:
            error_msg = failed['error']
            if error_msg not in error_groups:
                error_groups[error_msg] = []
            error_groups[error_msg].append(failed)
        
        # Print each group
        for error_msg, prompts in error_groups.items():
            print(f"\nError: {error_msg}")
            for prompt in prompts:
                print(f"  - [{prompt['type']}] {prompt['ref']}")
        
        print("-" * 80)
    
    # process_prompt('okr_questions')
