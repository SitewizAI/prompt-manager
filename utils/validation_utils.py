"""Utilities for validating prompts and schemas."""

import asyncio
import json
import re
from typing import Dict, Any, Tuple, Optional, List, Union
import traceback

from .logging_utils import log_debug, log_error
from .validation_models import QuestionObject, QuestionsArray

# Cache of code files, initialized when needed
_code_file_cache = {}

async def fetch_and_cache_code_files(token=None, repo="SitewizAI/sitewiz", refresh=False):
    """
    Fetch and cache code files from GitHub repository.
    
    Args:
        token: GitHub API token (if None, uses environment variable)
        repo: Repository name in format "owner/repo"
        refresh: Whether to refresh the cache
        
    Returns:
        Dictionary of file paths to file contents
    """
    global _code_file_cache
    
    # Return from cache if available and not refreshing
    if _code_file_cache and not refresh:
        log_debug(f"Using cached code files ({len(_code_file_cache)} files)")
        return _code_file_cache
        
    # Use token from environment if not provided
    if not token:
        import os
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            log_error("No GitHub token available")
            return {}
    
    log_debug(f"Fetching code files from {repo}")
    
    # Placeholder for async file fetching implementation
    # In a real implementation, this would use aiohttp
    
    # For now, return empty dictionary to avoid errors
    _code_file_cache = {}
    log_debug(f"Cached {len(_code_file_cache)} code files")
    return _code_file_cache

def get_test_parameters() -> List[str]:
    """Get list of all possible prompt format parameters."""
    return [
        "question",
        "business_context",
        "stream_key",
        "insight_example",
        "insight_notes",
        "insight_criteria",
        "okrs",
        "insights",
        "suggestions",
        "additional_instructions",
        "function_details",
        "functions_module",
        "name",
        "all_okr_prompts",
        "suggestion_example",
        "suggestion_notes",
        "suggestion_criteria",
        "questions",
        "okr_criteria",
        "okr_code_example",
        "okr_notes",
        "reach_example",
        "criteria",
        "code_example",
        "notes"
    ]

def find_prompt_usage_in_code(content: str) -> Optional[Tuple[str, List[str]]]:
    """
    Find where a prompt is used in the codebase and what parameters are passed to it.
    
    Args:
        content: The prompt content to search for
        
    Returns:
        Tuple of (prompt_ref, list_of_parameters) or None if not found
    """
    global _code_file_cache
    
    try:
        if not _code_file_cache:
            _code_file_cache = asyncio.run(fetch_and_cache_code_files())
        
        # Look for get_prompt_from_dynamodb calls with parameters
        pattern = r"get_prompt_from_dynamodb\(['\"]([^'\"]+)['\"](?:,\s*({[^}]+}))?\)"
        
        for file_path, file_content in _code_file_cache.items():
            if not isinstance(file_content, str):
                continue
                
            matches = re.findall(pattern, file_content)
            for match in matches:
                prompt_ref, params_dict = match
                
                # If we found the prompt reference we're looking for
                if prompt_ref in content:
                    # Extract parameter names from the dictionary
                    if params_dict:
                        # Parse the parameter dictionary
                        param_pattern = r"['\"]([\w_]+)['\"]:"
                        params = re.findall(param_pattern, params_dict)
                        return prompt_ref, params
                    return prompt_ref, []
        
        return None
    except Exception as e:
        log_error(f"Error finding prompt usage: {e}")
        return None

def validate_prompt_format(content: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a prompt string can be formatted with test parameters.
    
    Args:
        content: The prompt content to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Create test parameters dictionary with empty strings
        test_params = {param: "" for param in get_test_parameters()}
        
        # First, identify and exclude code blocks from validation
        code_block_pattern = r'```(?:python)?\s*\n([\s\S]*?)```|(?:^    .*?$)+'
        
        # Replace content of code blocks with placeholders to protect them during validation
        code_blocks = []
        
        def replace_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"
        
        content_without_code = re.sub(code_block_pattern, replace_code_block, content, flags=re.MULTILINE)
        
        # Now find format variables only in the content outside code blocks
        format_vars = re.findall(r'{([^{}]*)}', content_without_code)

        # Check if all format variables are in the test parameters
        unknown_vars = [var for var in format_vars if var not in test_params]
        if unknown_vars:
            error_msg = f"Unknown variables in prompt: {', '.join(unknown_vars)}"
            log_error(error_msg)
            return False, error_msg

        # Try formatting the content
        formatted = content.format(**test_params)
        log_debug("Prompt format validation successful")
        return True, None
        
    except KeyError as e:
        error_msg = f"Invalid format key in prompt: {e}"
        log_error(error_msg)
        return False, error_msg
    except ValueError as e:
        error_msg = f"Invalid format value in prompt: {e}"
        log_error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error in prompt validation: {e}"
        log_error(error_msg)
        return False, error_msg

def validate_question_objects_with_documents(prompt_ref: str, content: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Validate that question objects in a '[type]_questions' prompt only reference fields 
    that exist in the documents passed to run_evaluation.
    
    Args:
        prompt_ref: The prompt reference ID (should end with '_questions')
        content: The prompt content containing question objects
        
    Returns:
        Tuple of (is_valid, error_message, details)
    """
    try:
        if not prompt_ref.endswith('_questions'):
            return False, "Not a questions prompt - must end with '_questions'", {}
            
        # Try parsing as JSON to get question objects
        try:
            questions = json.loads(content)
            if not isinstance(questions, list) or not questions:
                return False, "Content is not a valid list of question objects", {}
        except json.JSONDecodeError as e:
            return False, f"Content is not valid JSON: {str(e)}", {}
        
        # Extract document structure from codebase
        global _code_file_cache
        
        # Ensure code files are loaded
        if not _code_file_cache:
            _code_file_cache = asyncio.run(fetch_and_cache_code_files())
        
        # Find run_evaluation calls with this question type
        document_fields = []
        eval_type = prompt_ref.split('_')[0]  # Extract type from prompt_ref (e.g., 'okr' from 'okr_questions')
        log_debug(f"Searching for {eval_type} document structure and run_evaluation calls")
        
        # Regex patterns for finding document structure and run_evaluation calls
        doc_pattern = r'documents\s*=\s*{([^}]+)}'
        eval_pattern = rf'run_evaluation\s*\(\s*documents\s*,\s*{prompt_ref}\s*\)'
        
        for file_path, content in _code_file_cache.items():
            if not isinstance(content, str):
                continue
                
            # Find run_evaluation calls with this question type
            eval_matches = re.search(eval_pattern, content)
            if eval_matches:
                log_debug(f"Found run_evaluation call with {prompt_ref} in {file_path}")
                
                # Find document structure definition nearby
                # First, try looking before the run_evaluation call
                file_content_before_eval = content[:eval_matches.start()]
                doc_matches = re.search(doc_pattern, file_content_before_eval)
                
                if doc_matches:
                    doc_def = doc_matches.group(1)
                    # Extract field names from the documents dictionary
                    field_pattern = r'"([^"]+)":\s*{[^}]*}'
                    field_matches = re.findall(field_pattern, doc_def)
                    document_fields.extend(field_matches)
                    log_debug(f"Found document structure with fields: {field_matches}")
                    break  # Stop after finding the first valid match
        
        if not document_fields:
            return False, f"Could not find document structure for {prompt_ref} in code", {}
        
        # Now extract output and reference fields from questions
        output_fields = set()
        reference_fields = set()
        
        for question in questions:
            if not isinstance(question, dict):
                continue
                
            # Collect output fields
            if 'output' in question and isinstance(question['output'], list):
                output_fields.update(question['output'])
                
            # Collect reference fields
            if 'reference' in question and isinstance(question['reference'], list):
                reference_fields.update(question['reference'])
        
        # Check if all fields exist in documents
        missing_output_fields = [field for field in output_fields if field not in document_fields]
        missing_reference_fields = [field for field in reference_fields if field not in document_fields]
        
        valid = not (missing_output_fields or missing_reference_fields)
        
        # Build validation details
        details = {
            "document_fields": document_fields,
            "output_fields": list(output_fields),
            "reference_fields": list(reference_fields),
            "missing_output_fields": missing_output_fields,
            "missing_reference_fields": missing_reference_fields
        }
        
        if not valid:
            error_message = "Questions reference fields not found in documents:"
            if missing_output_fields:
                error_message += f"\nMissing output fields: {', '.join(missing_output_fields)}"
            if missing_reference_fields:
                error_message += f"\nMissing reference fields: {', '.join(missing_reference_fields)}"
            return False, error_message, details
        
        return True, None, details
    
    except Exception as e:
        error_msg = f"Error validating questions against documents: {str(e)}"
        log_error(error_msg)
        log_debug(f"Validation error trace: {traceback.format_exc()}")
        return False, error_msg, {}
