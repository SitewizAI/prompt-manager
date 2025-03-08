"""Utilities for prompt management."""

from datetime import datetime
from decimal import Decimal
import json
import time
import os
import re
from typing import Dict, List, Any, Tuple, Optional, Union
from botocore.exceptions import ClientError
import traceback
import asyncio

from .db_utils import get_dynamodb_table
from .logging_utils import log_debug, log_error
from .validation_utils import (
    validate_prompt_format, 
    validate_question_objects_with_documents,
    find_prompt_usage_in_code
)

# Define prompt types and their corresponding refs
PROMPT_TYPES = {
    "all": [],  # Will be populated with all refs
    "okr": [
        "okr_evaluation_prompt",
        "okr_evaluation_questions",
        "okr_system_prompt"
    ],
    "insights": [
        "insights_evaluation_prompt",
        "insights_evaluation_questions",
        "insights_system_prompt"
    ],
    "suggestion": [
        "suggestion_evaluation_prompt",
        "suggestion_evaluation_questions",
        "suggestion_system_prompt"
    ],
    "design": [
        "design_evaluation_prompt",
        "design_evaluation_questions",
        "design_system_prompt"
    ],
    "code": [
        "code_evaluation_prompt",
        "code_evaluation_questions",
        "code_system_prompt"
    ]
}

# Global cache for prompt expected parameters
_prompt_usage_cache = {}

def measure_time(func):
    """Decorator to measure function execution time."""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"⏱️ {func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper

@measure_time
def get_all_prompts() -> List[Dict[str, Any]]:
    """
    Fetch all prompts from DynamoDB PromptsTable, retrieving only the latest version of each prompt reference.
    This significantly improves performance by reducing the amount of data fetched.
    """
    try:
        log_debug("Attempting to get all prompts...")
        table = get_dynamodb_table('PromptsTable')
        
        # First, scan to get all unique refs
        response = table.scan(
            ProjectionExpression='#r',
            ExpressionAttributeNames={
                '#r': 'ref'
            }
        )
        
        # Extract and deduplicate prompt refs
        refs = list(set(item['ref'] for item in response.get('Items', [])))
        log_debug(f"Found {len(refs)} unique prompt references")
        
        # Handle pagination for the scan operation if necessary
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                ProjectionExpression='#r',
                ExpressionAttributeNames={
                    '#r': 'ref'
                },
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            new_refs = list(set(item['ref'] for item in response.get('Items', [])))
            refs.extend(new_refs)
            refs = list(set(refs))  # Deduplicate again
            
        # Now fetch only the latest version for each ref
        latest_prompts = []
        start_time = time.time()
        
        # Use batch operation to reduce network calls
        for i in range(0, len(refs), 25):  # Process in batches of 25 for better performance
            batch_refs = refs[i:i+25]
            log_debug(f"Processing batch of {len(batch_refs)} refs ({i+1}-{i+len(batch_refs)} of {len(refs)})")
            
            # Process each ref in the batch
            batch_results = []
            
            for ref in batch_refs:
                # Query for the latest version of this ref
                response = table.query(
                    KeyConditionExpression='#r = :ref',
                    ExpressionAttributeNames={'#r': 'ref'},
                    ExpressionAttributeValues={':ref': ref},
                    ScanIndexForward=False,  # Sort in descending order (newest first)
                    Limit=1  # Get only the latest version
                )
                
                if response.get('Items'):
                    batch_results.append(response['Items'][0])
            
            latest_prompts.extend(batch_results)
            log_debug(f"Batch processed in {time.time() - start_time:.2f}s - Total prompts: {len(latest_prompts)}")
            start_time = time.time()
        
        log_debug(f"Retrieved {len(latest_prompts)} latest prompt versions")
        return latest_prompts
        
    except Exception as e:
        log_error("Error getting prompts", e)
        print(f"Traceback: {traceback.format_exc()}")
        return []

@measure_time
def get_all_prompt_versions(ref: str) -> List[Dict[str, Any]]:
    """
    Fetch all versions of a specific prompt reference from DynamoDB.
    
    Args:
        ref: The prompt reference ID
        
    Returns:
        List of prompt versions sorted by version number (newest first)
    """
    try:
        log_debug(f"Fetching all versions for prompt ref: {ref}")
        table = get_dynamodb_table('PromptsTable')
        
        # Query for all versions of this ref
        response = table.query(
            KeyConditionExpression='#r = :ref',
            ExpressionAttributeNames={'#r': 'ref'},
            ExpressionAttributeValues={':ref': ref},
            ScanIndexForward=False  # Sort in descending order (newest first)
        )
        
        versions = response.get('Items', [])
        log_debug(f"Found {len(versions)} versions for prompt ref: {ref}")
        
        # Handle pagination if needed
        while 'LastEvaluatedKey' in response:
            response = table.query(
                KeyConditionExpression='#r = :ref',
                ExpressionAttributeNames={'#r': 'ref'},
                ExpressionAttributeValues={':ref': ref},
                ExclusiveStartKey=response['LastEvaluatedKey'],
                ScanIndexForward=False
            )
            versions.extend(response.get('Items', []))
        
        # Sort by version (descending)
        versions.sort(key=lambda x: int(x.get('version', 0)), reverse=True)
        return versions
        
    except Exception as e:
        log_error(f"Error getting all versions for prompt {ref}", e)
        log_debug(f"Traceback: {traceback.format_exc()}")
        return []

@measure_time
def get_prompt_versions_by_date(date_str: str, eval_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetch prompt versions used on a specific date from DateEvaluationsTable.

    Args:
        date_str: The date string in format 'YYYY-MM-DD'
        eval_type: The evaluation type (e.g., 'okr', 'insights', etc.)

    Returns:
        Dictionary mapping prompt refs to their content as used on the specified date
    """
    try:
        log_debug(f"Fetching prompt versions for date {date_str} and type {eval_type}")
        table = get_dynamodb_table('DateEvaluationsTable')

        # Query for the specific date and type
        response = table.query(
            KeyConditionExpression='#type = :type_val AND #date = :date_val',
            ExpressionAttributeNames={
                '#type': 'type',
                '#date': 'date'
            },
            ExpressionAttributeValues={
                ':type_val': eval_type,
                ':date_val': date_str
            }
        )

        items = response.get('Items', [])
        if not items:
            log_debug(f"No data found for date {date_str} and type {eval_type}")
            return {}

        # Extract prompt versions from the item
        prompt_versions = {}
        for item in items:
            if 'promptVersions' in item:
                for prompt_version in item['promptVersions']:
                    ref = prompt_version.get('ref')
                    if ref:
                        prompt_versions[ref] = prompt_version

        log_debug(f"Found {len(prompt_versions)} prompt versions for date {date_str}")
        return prompt_versions

    except Exception as e:
        log_error(f"Error getting prompt versions for date {date_str}", e)
        log_debug(f"Traceback: {traceback.format_exc()}")
        return {}

@measure_time
def get_available_prompt_dates(eval_type: str) -> List[str]:
    """
    Get a list of dates for which prompt versions are available in DateEvaluationsTable.

    Args:
        eval_type: The evaluation type (e.g., 'okr', 'insights', etc.)

    Returns:
        List of date strings in format 'YYYY-MM-DD', sorted in descending order (newest first)
    """
    try:
        log_debug(f"Fetching available prompt dates for type {eval_type}")
        table = get_dynamodb_table('DateEvaluationsTable')

        # Query for all items of the specified type
        response = table.query(
            KeyConditionExpression='#type = :type_val',
            ProjectionExpression='#date',
            ExpressionAttributeNames={
                '#type': 'type',
                '#date': 'date'
            },
            ExpressionAttributeValues={
                ':type_val': eval_type
            }
        )

        items = response.get('Items', [])

        # Extract and sort dates
        dates = [item['date'] for item in items if 'date' in item]
        dates = sorted(list(set(dates)), reverse=True)  # Remove duplicates and sort

        log_debug(f"Found {len(dates)} available dates for type {eval_type}")
        return dates

    except Exception as e:
        log_error(f"Error getting available prompt dates: {str(e)}")
        log_debug(f"Traceback: {traceback.format_exc()}")
        return []

@measure_time
def revert_prompts_to_date(date_str: str, eval_type: str) -> Tuple[bool, str, List[str]]:
    """
    Revert all prompts to the versions used on a specific date.

    Args:
        date_str: The date string in format 'YYYY-MM-DD'
        eval_type: The evaluation type (e.g., 'okr', 'insights', etc.)

    Returns:
        Tuple of (success, message, updated_refs)
        - success: Boolean indicating if the operation was successful
        - message: Success or error message
        - updated_refs: List of prompt refs that were updated
    """
    try:
        log_debug(f"Reverting prompts to date {date_str} for type {eval_type}")

        # Get prompt versions for the specified date
        prompt_versions = get_prompt_versions_by_date(date_str, eval_type)

        if not prompt_versions:
            return False, f"No prompt versions found for date {date_str}", []

        # Update each prompt to the version from the specified date
        updated_refs = []
        failed_refs = []

        for ref, version_data in prompt_versions.items():
            content = version_data.get('content')
            if content is None:
                log_error(f"No content found for prompt {ref} on date {date_str}")
                failed_refs.append(ref)
                continue

            # Update the prompt
            update_result = update_prompt(ref, content)

            # Handle both return types (boolean or tuple)
            if isinstance(update_result, tuple):
                success, error_msg = update_result
            else:
                success, error_msg = update_result, None

            if success:
                updated_refs.append(ref)
                log_debug(f"Successfully reverted prompt {ref} to version from {date_str}")
            else:
                failed_refs.append(ref)
                log_error(f"Failed to revert prompt {ref}: {error_msg}")

        # Generate result message
        if not failed_refs:
            return True, f"Successfully reverted {len(updated_refs)} prompts to versions from {date_str}", updated_refs
        elif updated_refs:
            return True, f"Partially successful: Updated {len(updated_refs)} prompts, but failed to update {len(failed_refs)} prompts", updated_refs
        else:
            return False, f"Failed to revert any prompts to date {date_str}", []

    except Exception as e:
        log_error(f"Error reverting prompts to date {date_str}: {str(e)}")
        log_debug(f"Traceback: {traceback.format_exc()}")
        return False, f"Error: {str(e)}", []

def get_prompt_from_dynamodb(ref: str, substitutions: Dict[str, Any] = None) -> str:
    """
    Get prompt with highest version from DynamoDB PromptsTable by ref.

    Args:
        ref: The reference ID of the prompt to retrieve
        substitutions: Optional dictionary of variables to substitute in the prompt

    Returns:
        The prompt content with substitutions applied if provided
    """
    try:
        table = get_dynamodb_table('PromptsTable')
        # Query the table for all versions of this ref
        response = table.query(
            KeyConditionExpression='#r = :ref',
            ExpressionAttributeNames={'#r': 'ref'},
            ExpressionAttributeValues={':ref': ref},
            ScanIndexForward=False,  # Sort in descending order
            Limit=1  # Only get the most recent version
        )
        
        if not response['Items']:
            print(f"No prompt found for ref: {ref}")
            return ""

        content = response['Items'][0]['content']

        # If substitutions are provided, apply them to the prompt
        if substitutions:
            try:
                content = content.format(**substitutions)
            except KeyError as e:
                error_msg = f"Missing substitution key in prompt {ref}: {e}"
                log_error(error_msg)
                raise ValueError(error_msg)
            except Exception as e:
                error_msg = f"Error applying substitutions to prompt {ref}: {e}"
                log_error(error_msg)
                raise ValueError(error_msg)

        return content
    except Exception as e:
        if not isinstance(e, ValueError):
            print(f"Error getting prompt {ref} from DynamoDB: {e}")
        raise

def update_prompt(ref: str, content: Union[str, Dict[str, Any], List]) -> Union[bool, Tuple[bool, Optional[str]]]:    
    """
    Update or create a prompt in DynamoDB PromptsTable with versioning and validation.
    
    Args:
        ref: The prompt reference ID
        content: The prompt content to update
        
    Returns:
        If IS_DETAILED_ERRORS is False: A boolean indicating success
        If IS_DETAILED_ERRORS is True: A tuple of (success, error_message)
    """
    # Check if we should provide detailed errors (controlled by environment variable)
    IS_DETAILED_ERRORS = os.environ.get("DETAILED_PROMPT_ERRORS", "true").lower() == "true"
    
    try:
        table = get_dynamodb_table('PromptsTable')
        
        # Get latest version of the prompt
        response = table.query(
            KeyConditionExpression='#r = :ref',
            ExpressionAttributeNames={'#r': 'ref'},
            ExpressionAttributeValues={':ref': ref},
            ScanIndexForward=False,
            Limit=1
        )

        # Get current version and type information
        if not response.get('Items'):
            error_msg = f"No prompt found for ref: {ref}"
            log_error(error_msg)
            return (False, error_msg) if IS_DETAILED_ERRORS else False
            
        latest_prompt = response['Items'][0]
        current_version = int(latest_prompt.get('version', 0))
        is_object_original = latest_prompt.get('is_object', False)
        created_at = latest_prompt.get('createdAt', datetime.now().isoformat())
        
        log_debug(f"Updating prompt {ref} (current version: {current_version}, is_object: {is_object_original})")
        
        # If content provided is a string, check if it's JSON (either dict or list)
        is_object_new = isinstance(content, (dict, list))  # Now handles lists too
        
        if isinstance(content, str) and not is_object_new:
            try:
                # Try parsing as JSON
                content_obj = json.loads(content)
                # Check if result is a dict or list
                if isinstance(content_obj, (dict, list)):
                    is_object_new = True
                    content = content_obj
                    log_debug(f"String content parsed as JSON for prompt {ref}")
            except json.JSONDecodeError:
                # Not valid JSON, keep as string
                is_object_new = False
                log_debug(f"Content for prompt {ref} is a regular string")
        
        # For backwards compatibility: if original is object but provided as string 
        # and the string is valid JSON, parse it
        if is_object_original and isinstance(content, str) and not is_object_new:
            try:
                content = json.loads(content)
                is_object_new = True
                log_debug(f"Parsed string content into JSON for object-type prompt {ref}")
            except json.JSONDecodeError:
                error_msg = f"Content provided as string but prompt {ref} requires JSON object"
                log_error(error_msg)
                return (False, error_msg) if IS_DETAILED_ERRORS else False
        
        # Validate the prompt using validate_prompt_parameters in all cases
        is_valid, error_message, details = validate_prompt_parameters(ref, content)
        if not is_valid:
            error_msg = f"Prompt validation failed for ref: {ref} - {error_message}"
            log_error(error_msg)
            return (False, error_msg) if IS_DETAILED_ERRORS else False
        
        # Create new version
        new_version = current_version + 1
        
        # Prepare item for DynamoDB
        item = {
            'ref': ref,
            'content': json.dumps(content) if is_object_new else content,
            'version': new_version,
            'is_object': is_object_new,
            'updatedAt': datetime.now().isoformat(),
            'createdAt': created_at
        }
        
        log_debug(f"Creating new version {new_version} for prompt {ref}")
        
        # Store the content
        table.put_item(Item=item)
        
        log_debug(f"Successfully updated prompt {ref} to version {new_version}")
        return (True, None) if IS_DETAILED_ERRORS else True
        
    except ClientError as e:
        error_msg = f"DynamoDB error updating prompt {ref}: {str(e)}"
        log_error(error_msg)
        print(f"DynamoDB error: {str(e)}")
        return (False, error_msg) if IS_DETAILED_ERRORS else False
    except Exception as e:
        error_msg = f"Error updating prompt {ref}: {str(e)}"
        log_error(error_msg)
        print(f"Traceback: {traceback.format_exc()}")
        return (False, error_msg) if IS_DETAILED_ERRORS else False

def get_prompt_expected_parameters(prompt_ref: str) -> Dict[str, Any]:
    """
    Get information about how a prompt is used in code, including expected parameters.
    
    Args:
        prompt_ref: The prompt reference ID
        
    Returns:
        Dictionary with usage information including:
        - parameters: List of parameter names expected by the function call
        - optional_parameters: List of standard optional parameters used
        - file: The file where the prompt is used
        - line: The line number where the prompt is used
        - function_call: The actual function call text
        - found: Whether the prompt reference was found in the code
    """
    global _prompt_usage_cache
    
    # Check cache first
    if prompt_ref in _prompt_usage_cache:
        return _prompt_usage_cache[prompt_ref]
    
    # Define standard optional parameters - extended list
    common_optional_params = [
        'stream_key', 
        'context', 
        'business_context', 
        'question'
    ]
    
    # Find usages of the prompt in the code using the validation_utils function
    usage = find_prompt_usage_in_code(prompt_ref)
    print(f"Found usage for prompt {prompt_ref}: {usage}")
    
    # If no usages found, return empty info
    if not usage or usage[0] is None:
        result = {
            'parameters': [],
            'optional_parameters': common_optional_params,
            'file': None,
            'line': None,
            'function_call': None,
            'found': False,
            'is_questions': prompt_ref.endswith('_questions')
        }
        _prompt_usage_cache[prompt_ref] = result
        return result
    
    found_ref, found_params = usage
    
    # Calculate optional parameters (intersection of found params and common optional params)
    optional_params = [p for p in found_params if p in common_optional_params]
    
    # All other parameters are considered required
    required_params = [p for p in found_params if p not in optional_params]
    
    # Create and cache the usage info
    usage_info = {
        'parameters': required_params,
        'optional_parameters': optional_params,
        'file': found_ref,
        'line': None,
        'function_call': None,
        'found': True,
        'is_questions': prompt_ref.endswith('_questions')
    }
    
    _prompt_usage_cache[prompt_ref] = usage_info
    return usage_info

@measure_time
def validate_prompt_parameters(prompt_ref, content):
    """
    Validate that a prompt string only uses variables that are passed to it.
    For object prompts, validate against the expected schema.
    
    Args:
        prompt_ref: The prompt reference ID
        content: The prompt content to validate
        
    Returns:
        Tuple of (is_valid, error_message, details)
    """
    try:
        # First, check for questions prompt with dictionary wrapping issue
        if prompt_ref.endswith("_questions"):
            # Check if content is a dictionary with nested questions
            if isinstance(content, dict):
                # Check for wrappers like "evaluation_questions"
                wrapped_key = None
                for key in content.keys():
                    if key.endswith('_questions'):
                        wrapped_key = key
                        break
                
                if wrapped_key:
                    error_msg = f"Questions must be a direct array, not wrapped in an object with '{wrapped_key}'"
                    log_error(error_msg)
                    return False, error_msg, {"validation_error": error_msg}
            
            # If it's a string, check if it parses to a dictionary with wrapper
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        wrapped_key = None
                        for key in parsed.keys():
                            if key.endswith('_questions'):
                                wrapped_key = key
                                break
                        
                        if wrapped_key:
                            error_msg = f"Questions must be a direct array, not wrapped in an object with '{wrapped_key}'"
                            log_error(error_msg)
                            return False, error_msg, {"validation_error": error_msg}
                except json.JSONDecodeError:
                    # Not valid JSON, handled in the next sections
                    pass
        
        # Check if this is a JSON object that needs schema validation
        is_object = False
        if prompt_ref.endswith("_questions"):
            # For a string, try to parse as JSON
            if isinstance(content, str):
                try:
                    content_obj = json.loads(content)
                    # We expect a direct array for questions
                    if isinstance(content_obj, list):
                        is_object = True
                        content = content_obj  # Use parsed content
                    elif isinstance(content_obj, dict):
                        # Check if there's a key like "evaluation_questions"
                        for k in content_obj.keys():
                            if k.endswith('_questions'):
                                error_msg = f"Questions must be a direct array, not wrapped in object with key '{k}'"
                                log_error(error_msg)
                                return False, error_msg, {"validation_error": error_msg}
                        # If no specific wrapper key found, still not valid format
                        error_msg = "Questions content must be a direct array, not a dictionary"
                        log_error(error_msg)
                        return False, error_msg, {"validation_error": error_msg}
                except json.JSONDecodeError as e:
                    # Not valid JSON
                    error_msg = f"Invalid JSON format: {str(e)}"
                    log_error(error_msg)
                    return False, error_msg, {"validation_error": error_msg}
            # If already an object type, check direct structure
            elif isinstance(content, list):
                is_object = True
            elif isinstance(content, dict):
                # Again check for wrapper keys
                for k in content.keys():
                    if k.endswith('_questions'):
                        error_msg = f"Questions must be a direct array, not wrapped in object with key '{k}'"
                        log_error(error_msg)
                        return False, error_msg, {"validation_error": error_msg}
                error_msg = "Questions content must be a direct array, not a dictionary"
                log_error(error_msg)
                return False, error_msg, {"validation_error": error_msg}
            
            # Now validate questions with document fields
            if is_object:
                try:
                    # Import here to avoid circular imports
                    from .validation_models import QuestionsArray
                    
                    # Update to use Pydantic v2 parsing
                    questions = QuestionsArray(root=content)
                    
                    # Then check if document fields match output/reference fields
                    doc_valid, doc_error, doc_details = validate_question_objects_with_documents(prompt_ref, content)
                    
                    # If validation against document structure failed, return that error
                    if not doc_valid:
                        return doc_valid, doc_error, doc_details
                    
                    # Otherwise, return success with combined details
                    return True, None, {
                        "object_validated": True,
                        "question_count": len(questions),
                        "type": "questions_array",
                        "document_validation": doc_details
                    }
                except Exception as e:
                    import traceback
                    error_msg = f"Invalid questions format: {str(e)}"
                    error_details = traceback.format_exc()
                    log_error(error_msg)
                    log_debug(f"Validation error details: {error_details}")
                    return False, error_msg, {"validation_error": str(e)}
            else:
                error_msg = "Questions content must be a JSON array"
                log_error(error_msg)
                return False, error_msg, {"validation_error": error_msg}
        
        # For string prompts, validate variables
        if not is_object:
            # Find all format variables in the content using regex
            # This updated regex only matches {var} patterns that aren't part of {{var}} or other structures
            # Matches {variable} but not {{variable}} or more complex structures like {var: value}
            format_vars = set(re.findall(r'(?<!\{)\{([a-zA-Z0-9_]+)\}(?!\})', content))
            
            # if not format_vars:
            #     # If no variables found, the prompt is valid
            #     return True, None, {"used_vars": [], "unused_vars": [], "extra_vars": []}
            
            # Get expected parameters
            prompt_usage = get_prompt_expected_parameters(prompt_ref)
            if not prompt_usage['found']:
                # Add standard optional parameters to default assumptions
                standard_optional_params = [
                    'stream_key', 'context', 'business_context', 'question', 
                    'function_details'
                ]
                
                # If we can't find usage, assume all are valid but treat standard ones as optional
                all_expected_params = format_vars
                optional_params = [v for v in format_vars if v in standard_optional_params]
                
                return True, None, {
                    "used_vars": list(format_vars),
                    "unused_vars": [],
                    "extra_vars": [],
                    "note": "Couldn't find usage in code, all variables assumed valid"
                }
            
            # Combine required and optional parameters for validation
            all_expected_params = set(prompt_usage['parameters'] + prompt_usage['optional_parameters'])
            
            # Check for mismatch between format variables and expected parameters
            missing_vars = set(prompt_usage['parameters']) - format_vars  # Required variables expected but not in prompt
            extra_vars = format_vars - all_expected_params    # Variables in prompt but not expected at all
            used_vars = format_vars.intersection(all_expected_params)  # Variables properly used
            
            # Modified validation logic: fail if any required variables are missing
            if missing_vars:
                error_message = f"Missing required parameters in prompt: {', '.join(['{'+v+'}' for v in missing_vars])}"
                log_error(error_message)
                return False, error_message, {
                    "file": prompt_usage['file'],
                    "line": prompt_usage['line'],
                    "used_vars": list(used_vars),
                    "unused_vars": list(missing_vars),
                    "extra_vars": list(extra_vars),
                    "validation_error": "required_params_missing"
                }
            
            if extra_vars:
                error_messages = []
                details = {
                    "file": prompt_usage['file'],
                    "line": prompt_usage['line'],
                    "used_vars": list(used_vars),
                    "unused_vars": list(missing_vars),
                    "extra_vars": list(extra_vars)
                }
                
                extra_list = ", ".join([f"{{{v}}}" for v in extra_vars])
                error_messages.append(f"Extra parameters in prompt that aren't provided: {extra_list}")
                    
                # Include file location in error message
                file_info = f"Error in {prompt_usage['file']}:{prompt_usage['line']}" if prompt_usage['file'] else ""
                error_message = (file_info + "\n" if file_info else "") + "\n".join(error_messages)
                
                return False, error_message, details
            
            return True, None, {
                "file": prompt_usage['file'],
                "line": prompt_usage['line'],
                "used_vars": list(used_vars),
                "unused_vars": list(missing_vars),
                "extra_vars": []
            }
    
    except Exception as e:
        error_msg = f"Unexpected error in prompt validation: {str(e)}"
        log_error(error_msg)
        log_debug(f"Validation error trace: {traceback.format_exc()}")
        return False, error_msg, {}
