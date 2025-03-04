"""Utilities for validating prompts and schemas."""

import asyncio
import json
import re
from typing import Dict, Any, Tuple, Optional, List, Union
import traceback

from .logging_utils import log_debug, log_error
from .validation_models import QuestionObject, QuestionsArray
from .db_utils import get_dynamodb_table
from .github_utils import fetch_and_cache_code_files

_code_file_cache = {}

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
        
        log_debug(f"Searching for prompt reference in code files: {content}")
        
        # More specific pattern matching for exact prompt references with their parameters
        patterns = [
            # Direct call with optional parameters: get_prompt_from_dynamodb("prompt_ref", {...})
            rf'get_prompt_from_dynamodb\([\'"]({re.escape(content)})[\'"](?:,\s*({{[^}}]+}}))?(?:,\s*([^)]+))?\)',
            
            # Assignment with optional parameters: var = get_prompt_from_dynamodb("prompt_ref", {...})
            rf'\w+\s*=\s*get_prompt_from_dynamodb\([\'"]({re.escape(content)})[\'"](?:,\s*({{[^}}]+}}))?',
        ]
        
        for file_path, file_content in _code_file_cache.items():
            if not isinstance(file_content, str):
                continue
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, file_content))
                
                for match in matches:
                    # Extract params if available in the direct call
                    params_dict = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                    params = []
                    
                    if params_dict:
                        # Extract parameter names from the dictionary
                        param_pattern = r'[\'"]([a-zA-Z0-9_]+)[\'"]:'
                        params = re.findall(param_pattern, params_dict)
                        log_debug(f"Found direct parameters for {content}: {params}")
                    
                    # Return only parameters explicitly passed to this prompt reference
                    return content, params
        
        # If we get here, search for just the prompt reference without parameters
        simple_patterns = [
            rf'get_prompt_from_dynamodb\([\'"]({re.escape(content)})[\'"]',
            rf'\w+\s*=\s*get_prompt_from_dynamodb\([\'"]({re.escape(content)})[\'"]'
        ]
        
        for file_path, file_content in _code_file_cache.items():
            if not isinstance(file_content, str):
                continue
                
            for pattern in simple_patterns:
                if re.search(pattern, file_content):
                    log_debug(f"Found prompt reference '{content}' in {file_path} with no parameters")
                    return content, []
                    
        # If we get here, the prompt reference wasn't found
        log_debug(f"Could not find prompt reference '{content}' in any code file")
        return None
    except Exception as e:
        log_error(f"Error finding prompt usage: {str(e)}")
        log_debug(traceback.format_exc())
        return None

def validate_prompt_format(content: str, variables: Dict[str, Any] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate that a prompt string can be formatted with provided variables.
    
    Args:
        content: The prompt content to validate
        variables: Dictionary of variables to use for validation
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
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

        # If no variables provided, create test params with dummy values
        if variables is None:
            test_params = {var: "test" for var in format_vars}
        else:
            test_params = variables
            # Check if all required format variables are in the provided variables
            missing_vars = [var for var in format_vars if var not in test_params]
            if missing_vars:
                error_msg = f"Missing variables for validation: {', '.join(missing_vars)}"
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

def validate_question_objects_with_documents(prompt_ref: str, content: Union[str, List]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Validate that question objects in a '[type]_questions' prompt only reference fields 
    that exist in the documents passed to run_evaluation.
    
    Args:
        prompt_ref: The prompt reference ID (should end with '_questions')
        content: The prompt content (JSON string or parsed list of question objects)
        
    Returns:
        Tuple of (is_valid, error_message, details)
    """
    try:
        if not prompt_ref.endswith('_questions'):
            return False, "Not a questions prompt - must end with '_questions'", {}
            
        # Parse content if it's a string, or use directly if already parsed
        if isinstance(content, str):
            try:
                questions = json.loads(content)
            except json.JSONDecodeError as e:
                return False, f"Content is not valid JSON: {str(e)}", {}
        else:
            questions = content
            
        # Ensure we have a list of questions, not wrapped in an object
        if isinstance(questions, dict):
            # Check if it's a wrapper with a key like "evaluation_questions" 
            question_keys = [k for k in questions.keys() if k.endswith('_questions')]
            if question_keys:
                # Found a key like "evaluation_questions", check if its value is a list
                key = question_keys[0]
                if isinstance(questions[key], list):
                    questions = questions[key]  # Extract the actual questions array
                    log_debug(f"Extracted questions from wrapper key '{key}'")
                else:
                    return False, f"'{key}' value is not a valid list of question objects", {}
            else:
                return False, "Content is not a valid list of question objects", {}
            
        # Now validate that questions is a list
        if not isinstance(questions, list) or not questions:
            return False, "Content is not a valid list of question objects", {}
        
        # Validate that each item in the list is a question object with required fields
        for i, question in enumerate(questions):
            if not isinstance(question, dict):
                return False, f"Question at index {i} is not a valid question object", {}
            
            # Check required fields
            if 'question' not in question:
                return False, f"Question at index {i} is missing 'question' field", {}
            if 'output' not in question:
                return False, f"Question at index {i} is missing 'output' field", {}
            if not isinstance(question.get('output'), list):
                return False, f"Question at index {i} 'output' must be a list", {}
            if 'reference' in question and not isinstance(question.get('reference'), list):
                return False, f"Question at index {i} 'reference' must be a list", {}
            if 'confidence_threshold' not in question:
                return False, f"Question at index {i} is missing 'confidence_threshold' field", {}
            try:
                thresh = float(question.get('confidence_threshold'))
                if thresh < 0.0 or thresh > 1.0:
                    return False, f"Question at index {i} 'confidence_threshold' must be between 0.0 and 1.0", {}
            except (ValueError, TypeError):
                return False, f"Question at index {i} 'confidence_threshold' must be a number", {}
            if 'feedback' not in question:
                return False, f"Question at index {i} is missing 'feedback' field", {}
                
        # Get document structure using our document structure finder
        document_structure = get_document_structure(prompt_ref)
        document_fields = list(document_structure.keys()) if document_structure else []
        
        # Log what fields we found
        log_debug(f"Found {len(document_fields)} document fields for {prompt_ref}: {', '.join(document_fields)}")
        
        # If no document structure was found, return an error immediately
        if not document_fields:
            error_msg = f"Could not find document structure for {prompt_ref} in code"
            log_debug(error_msg)
            return False, error_msg, {"document_structure": {}}
        
        # Now extract output and reference fields from questions
        output_fields = set()
        reference_fields = set()
        
        for i, question in enumerate(questions):
            if not isinstance(question, dict):
                continue
                
            # Collect output fields
            if 'output' in question and isinstance(question['output'], list):
                output_fields.update(question['output'])
                log_debug(f"Question {i+1} output fields: {question['output']}")
                
            # Collect reference fields
            if 'reference' in question and isinstance(question['reference'], list):
                reference_fields.update(question['reference'])
                log_debug(f"Question {i+1} reference fields: {question['reference']}")
        
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
            "missing_reference_fields": missing_reference_fields,
            "document_structure": document_structure
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

def get_document_structure(prompt_ref: str) -> Dict[str, Dict[str, Any]]:
    """
    Find and parse the document structure used with a specific questions prompt.
    
    Args:
        prompt_ref: The prompt reference ID (e.g., 'okr_questions', 'insights_questions')
        
    Returns:
        Dictionary containing the document structure or empty dict if not found
    """
    try:
        # Handle special case for known document structures
        if prompt_ref == "data_questions":
            # Return the document structure for data_questions
            return {
                "Insight": {"type": "text", "description": "Insight generalized from the data"},
                "explanation": {"type": "text", "description": "Explanation of the data connection to the insight"},
                "Data": {"type": "mixed", "description": "Evidence data (image or text)"}
            }
        
        if prompt_ref == "suggestion_questions":
            # Return the document structure for suggestion_questions
            return {
                "suggestion_markdown": {"type": "text", "description": "The full suggestion content"},
                "Insights": {"type": "text", "description": "Data insights"},
                "Expanded": {"type": "text", "description": "Expanded details"},
                "Tags": {"type": "text", "description": "Suggestion tags"},
                "Shortened": {"type": "text", "description": "Suggestion header"},
                "previous_suggestions": {"type": "text", "description": "Previously stored suggestions"},
                "business_context": {"type": "text", "description": "Business context"},
                "suggestion_summary": {"type": "text", "description": "Summary of previous suggestions"}
            }
        
        # For other cases, use the generic document structure finder
        global _code_file_cache
        if not _code_file_cache:
            _code_file_cache = asyncio.run(fetch_and_cache_code_files())
        
        log_debug(f"Searching for document structure used with {prompt_ref}")
        
        # First, look for specific pattern: validation_results = run_evaluation(documents, prompt_ref)
        validation_pattern = rf'validation_results\s*=\s*run_evaluation\s*\(\s*([a-zA-Z0-9_]+)\s*,\s*{prompt_ref}'
        
        # If not found, try other common patterns
        eval_patterns = [
            validation_pattern,
            rf'run_evaluation\s*\(\s*([a-zA-Z0-9_]+)\s*,\s*{prompt_ref}\s*[,)]',
            rf'run_evaluation\s*\(\s*([a-zA-Z0-9_]+)\s*,\s*[\'"]?{prompt_ref}[\'"]?\s*[,)]',
            rf'run_evaluation\s*\(\s*([a-zA-Z0-9_]+)\s*,\s*{prompt_ref}\s*,.+\)',
        ]
        
        files_scanned = 0
        
        for file_path, content in _code_file_cache.items():
            if not isinstance(content, str):
                continue
                
            files_scanned += 1
            
            # Try each pattern to find run_evaluation call
            for eval_pattern in eval_patterns:
                eval_matches = re.search(eval_pattern, content)
                if not eval_matches:
                    continue
                
                # Found a run_evaluation call with our prompt_ref
                docs_var_name = eval_matches.group(1)
                log_debug(f"Found run_evaluation call in {file_path} using document variable: {docs_var_name}")
                print(f"FOUND: {eval_matches.group(0)}")
                
                # Get the context before this line to find the document structure
                file_content_before_eval = content[:eval_matches.start()]
                
                # Look for document structure using regex with proper brace matching
                doc_pattern = re.compile(
                    rf'{docs_var_name}\s*=\s*{{', 
                    re.MULTILINE
                )
                
                start_match = doc_pattern.search(file_content_before_eval)
                if not start_match:
                    log_debug(f"Couldn't find document definition for {docs_var_name}")
                    continue
                    
                # Found the start of the document definition
                start_pos = start_match.end() - 1  # Position of the opening brace
                log_debug(f"Found document definition start at position {start_pos}")
                
                # Extract the full dictionary with proper brace matching
                brace_count = 1
                end_pos = start_pos + 1
                for i in range(start_pos + 1, len(file_content_before_eval)):
                    char = file_content_before_eval[i]
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                if brace_count != 0:
                    log_debug(f"Couldn't find balanced closing brace for document definition")
                    continue
                
                # Extract the full document definition including braces
                doc_def = file_content_before_eval[start_pos:end_pos]
                print(f"FOUND DOCUMENT STRUCTURE: {doc_def}")
                log_debug(f"Extracted document structure: {len(doc_def)} characters")
                
                # Parse the document structure to extract field definitions
                document_structure = {}
                
                # Pattern to find field definitions like "field": {"type": "text", ...}
                field_pattern = re.compile(
                    r'"([^"]+)":\s*{([^{}]*(?:{[^{}]*}[^{}]*)*)}',
                    re.DOTALL
                )
                
                field_matches = field_pattern.finditer(doc_def)
                fields_found = 0
                
                for match in field_matches:
                    field_name = match.group(1)
                    field_def = match.group(2)
                    fields_found += 1
                    log_debug(f"Found field definition: {field_name}")
                    
                    # Extract attributes from the field definition
                    field_attrs = {}
                    
                    # Extract type
                    type_match = re.search(r'"type":\s*"([^"]+)"', field_def)
                    if type_match:
                        field_attrs['type'] = type_match.group(1)
                    
                    # Extract description
                    desc_match = re.search(r'"description":\s*"([^"]+)"', field_def)
                    if desc_match:
                        field_attrs['description'] = desc_match.group(1)
                    
                    # Add to document structure
                    document_structure[field_name] = field_attrs
                
                if fields_found > 0:
                    log_debug(f"Successfully parsed {fields_found} fields in document structure")
                    log_debug(f"Field names: {', '.join(document_structure.keys())}")
                    return document_structure
        
        # If we've gone through all files and found nothing
        log_debug(f"No document structure found for {prompt_ref} after scanning {files_scanned} files")
        
        # Return empty dict if structure not found
        return {}
        
    except Exception as e:
        log_error(f"Error finding document structure: {str(e)}")
        log_debug(traceback.format_exc())
        return {}