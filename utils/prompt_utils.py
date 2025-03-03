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
    validate_question_objects_with_documents
)

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
        
        # Special validation for _questions type prompts - must be arrays of question objects
        if ref.endswith('_questions'):
            # If content is a dict with a wrapper key (like "evaluation_questions"), reject it
            if isinstance(content, dict):
                question_keys = [k for k in content.keys() if k.endswith('_questions')]
                if question_keys:
                    error_msg = f"Questions must be a direct array, not wrapped in a dict with '{question_keys[0]}'"
                    log_error(error_msg)
                    return (False, error_msg) if IS_DETAILED_ERRORS else False
                
            # For questions, content must be a list/array
            if not isinstance(content, list):
                try:
                    # Try to extract a list if it's wrapped in an object with a single key
                    if isinstance(content, dict) and len(content) == 1:
                        # Try to get the first value if it's a list
                        first_key = next(iter(content))
                        if isinstance(content[first_key], list):
                            content = content[first_key]
                            log_debug(f"Extracted questions array from wrapper object key '{first_key}'")
                        else:
                            error_msg = f"Questions content must be an array/list, got {type(content)} (dict with non-list value)"
                            log_error(error_msg)
                            return (False, error_msg) if IS_DETAILED_ERRORS else False
                    else:
                        error_msg = f"Questions content must be an array/list, got {type(content)}"
                        log_error(error_msg)
                        return (False, error_msg) if IS_DETAILED_ERRORS else False
                except Exception as e:
                    error_msg = f"Error processing questions content: {str(e)}"
                    log_error(error_msg)
                    return (False, error_msg) if IS_DETAILED_ERRORS else False
                
            # Now validate with the proper schema
            is_valid, error_message, details = validate_prompt_parameters(ref, content)
            if not is_valid:
                error_msg = f"Questions validation failed for ref: {ref} - {error_message}"
                log_error(error_msg)
                return (False, error_msg) if IS_DETAILED_ERRORS else False
        elif isinstance(content, str):  # Regular string prompt validation
            is_valid, error_message = validate_prompt_format(content)
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

def find_prompt_usage_with_context(prompt_ref: str, code_files: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """
    Find where a prompt reference is used in code and extract the full context.
    
    Args:
        prompt_ref: The prompt reference ID to search for
        code_files: Dictionary of file paths to file contents, or None to scan directories
        
    Returns:
        Dictionary with parameters and optional_parameters or None if not found
    """
    global _prompt_usage_cache
    
    # Check cache first
    if prompt_ref in _prompt_usage_cache:
        return _prompt_usage_cache[prompt_ref]
    
    # If code_files parameter is None, scan directories looking for Python files
    if code_files is None:
        code_files = {}
        base_dir = '/Users/ram/Github/prompt-manager'  # Base directory for scanning
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            code_files[file_path] = f.read()
                    except Exception as e:
                        log_error(f"Error reading file {file_path}: {str(e)}")
    
    # Define standard optional parameters - extended list
    common_optional_params = [
        'stream_key', 
        'context', 
        'business_context', 
        'question',
        'function_details',
        'questions',  # Add "questions" as a common optional parameter
        'sample_query_data',  # Add demonstration data parameters 
        'another_sample_query_data'
    ]
    
    # Special handling for prompts ending with _questions - they use run_evaluation
    if prompt_ref.endswith('_questions'):
        # This is a questions object, not a string template prompt
        for file_path, content in code_files.items():
            if not isinstance(content, str):
                continue
            
            # Look for run_evaluation with this prompt
            eval_pattern = rf'run_evaluation\s*\(\s*([a-zA-Z0-9_]+)\s*,\s*{prompt_ref}'
            eval_matches = re.search(eval_pattern, content)
            
            if eval_matches:
                # Found run_evaluation call
                doc_var = eval_matches.group(1)
                line_number = content[:eval_matches.start()].count('\n') + 1
                
                # Create usage info for _questions prompt
                usage_info = {
                    'file': file_path,
                    'line': line_number,
                    'function_call': f"run_evaluation({doc_var}, {prompt_ref})",
                    'parameters': [],  # No string formatting parameters for questions
                    'optional_parameters': [],
                    'doc_variable': doc_var,
                    'found': True,
                    'is_questions': True
                }
                _prompt_usage_cache[prompt_ref] = usage_info
                return usage_info
    
    # Regular prompt with string formatting - regex to find get_prompt_from_dynamodb calls
    ref_pattern = rf"get_prompt_from_dynamodb\(['\"]({prompt_ref})['\"](?:,\s*({{[^}}]+}}))?(?:,\s*([^)]+))?\)"
    
    for file_path, content in code_files.items():
        if not isinstance(content, str):
            continue
            
        # Find all matches in the file
        matches = re.finditer(ref_pattern, content)
        
        for match in matches:
            # Get the entire matched function call
            function_call = match.group(0)
            
            # Get the parameters dictionary if it exists
            param_dict_str = match.group(2) if len(match.groups()) > 1 else None
            
            # Extract parameter names from the dictionary
            parameters = {}
            if param_dict_str:
                # Parse parameter keys with regex
                param_matches = re.finditer(r"['\"]([\w_]+)['\"]:\s*([^,}]+)", param_dict_str)
                for param_match in param_matches:
                    param_name = param_match.group(1)
                    param_value = param_match.group(2).strip()
                    parameters[param_name] = param_value
            
            # Get list of parameter names
            param_names = list(parameters.keys())
            
            # Calculate optional parameters (intersection of found params and common optional params)
            optional_params = [p for p in param_names if p in common_optional_params]
            
            # Move all common optional params from required to optional
            required_params = [p for p in param_names if p not in optional_params]
            
            # Find line number
            line_number = content[:match.start()].count('\n') + 1
            
            # Create usage info and store in cache
            usage_info = {
                'file': content,
                'line': line_number,
                'function_call': function_call,
                'parameters': required_params,
                'optional_parameters': optional_params,
                'found': True,
                'is_questions': False
            }
            _prompt_usage_cache[prompt_ref] = usage_info
            
            # Return the first match with parameters and optional parameters
            return usage_info
    
    # No matches found
    empty_result = {
        'parameters': [],
        'optional_parameters': common_optional_params,  # Always include standard optional params
        'file': None,
        'line': None,
        'function_call': None,
        'found': False,
        'is_questions': prompt_ref.endswith('_questions')
    }
    _prompt_usage_cache[prompt_ref] = empty_result
    return empty_result

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
    # Find usages of the prompt in the code
    usage = find_prompt_usage_with_context(prompt_ref)
    
    # If no usages found, return empty info
    if not usage or not usage['found']:
        return {
            'parameters': [],
            'optional_parameters': [],
            'file': None,
            'line': None,
            'function_call': None,
            'found': False
        }
    
    # Get required and optional parameters
    return usage

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
            
            if not format_vars:
                # If no variables found, the prompt is valid
                return True, None, {"used_vars": [], "unused_vars": [], "extra_vars": []}
            
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
            
            if missing_vars and not extra_vars:
                # If we only have missing variables but no extra ones, the prompt is technically valid
                # (not all required variables need to be used in the prompt)
                return True, None, {
                    "file": prompt_usage['file'],
                    "line": prompt_usage['line'],
                    "used_vars": list(used_vars),
                    "unused_vars": list(missing_vars),
                    "extra_vars": []
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