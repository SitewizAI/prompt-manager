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
    Fetch and cache code files from local directory or GitHub repository.
    
    Args:
        token: GitHub API token (if None, uses environment variable)
        repo: Repository name in format "owner/repo" (used only if fetching from GitHub)
        refresh: Whether to refresh the cache
        
    Returns:
        Dictionary of file paths to file contents
    """
    global _code_file_cache
    
    # Return from cache if available and not refreshing
    if _code_file_cache and not refresh:
        log_debug(f"Using cached code files ({len(_code_file_cache)} files)")
        return _code_file_cache
    
    # Initialize cache
    _code_file_cache = {}
    
    try:
        # Scan local project directory
        import os
        
        # Start with current directory for the project
        base_dir = '/Users/ram/Github/prompt-manager'
        log_debug(f"Scanning local directory for Python files: {base_dir}")
        
        # Count scanned files for logging
        file_count = 0
        
        # Walk through directory structure recursively
        for root, _, files in os.walk(base_dir):
            for filename in files:
                if filename.endswith('.py'):
                    try:
                        file_path = os.path.join(root, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                            _code_file_cache[file_path] = file_content
                            file_count += 1
                    except Exception as e:
                        log_error(f"Error reading file {file_path}: {str(e)}")
                        
        log_debug(f"Scanned {file_count} Python files in local directory")
    except Exception as e:
        log_error(f"Error scanning local directory: {str(e)}")
    
    log_debug(f"Cached {len(_code_file_cache)} code files")
    return _code_file_cache

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
        patterns = [
            # Common patterns for direct calls
            r'get_prompt_from_dynamodb\([\'"]([^\'"]+)[\'"](?:,\s*({[^}]+}))?\)',  # Standard pattern
            r'get_prompt_from_dynamodb\([\'"]([^\'"]+)[\'"]', # Simple reference
            
            # Patterns for variable assignments
            r'(\w+)\s*=\s*get_prompt_from_dynamodb\([\'"]([^\'"]+)[\'"]',
            r'(\w+)\s*=\s*get_prompt_from_dynamodb\([\'"]([^\'"]+)[\'"],\s*({[^}]+})',
            
            # Pattern for return statements
            r'return\s+get_prompt_from_dynamodb\([\'"]([^\'"]+)[\'"]',
            
            # Direct variable assignments
            r'prompt_ref\s*=\s*[\'"]([^\'"]+)[\'"]'
        ]
        
        log_debug(f"Searching for prompt reference in code files: {content}")
        
        for file_path, file_content in _code_file_cache.items():
            if not isinstance(file_content, str):
                continue
            
            # First try to find by direct reference match
            for pattern in patterns:
                matches = list(re.finditer(pattern, file_content))
                
                for match in matches:
                    # Different patterns have different group structures
                    if 'get_prompt_from_dynamodb' in pattern:
                        if '=' in pattern:  # Assignment pattern
                            prompt_ref = match.group(2)
                        else:
                            prompt_ref = match.group(1)
                    else:
                        prompt_ref = match.group(1)
                    
                    # Check if this is the prompt reference we're looking for
                    if prompt_ref != content and prompt_ref not in content:
                        continue
                        
                    log_debug(f"Found prompt reference '{prompt_ref}' in {file_path}")
                    
                    # Extract parameters if available
                    params = []
                    
                    # For assignment patterns with parameters
                    if '=' in pattern and len(match.groups()) > 2:
                        params_dict = match.group(3) if len(match.groups()) > 2 else None
                        if params_dict:
                            param_pattern = r'[\'"]([a-zA-Z0-9_]+)[\'"]:'
                            params = re.findall(param_pattern, params_dict)
                    
                    # For standard call with parameters
                    elif '=' not in pattern and len(match.groups()) > 1:
                        params_dict = match.group(2) if len(match.groups()) > 1 else None
                        if params_dict:
                            param_pattern = r'[\'"]([a-zA-Z0-9_]+)[\'"]:'
                            params = re.findall(param_pattern, params_dict)
                    
                    # If we couldn't find params in the immediate pattern, look for the context
                    if not params:
                        # Look for context with parameter dictionary
                        context_range = 20  # Lines to check before/after
                        lines = file_content.splitlines()
                        match_line = file_content[:match.start()].count('\n')
                        
                        start_line = max(0, match_line - context_range)
                        end_line = min(len(lines), match_line + context_range)
                        
                        context_block = '\n'.join(lines[start_line:end_line])
                        
                        # Look for dictionaries, variable assignments, etc.
                        dict_patterns = [
                            r'({[^{}]*"[^"]+"\s*:[^{}]+(?:{[^{}]*}[^{}]*)*})',  # Complex nested dictionaries
                            r'({(?:\s*"[^"]+"\s*:[^,{}]+,?)+\s*})',             # Simple dictionaries
                            r'substitutions\s*=\s*({[^}]+})'                    # Named substitutions
                        ]
                        
                        for dict_pattern in dict_patterns:
                            dict_matches = re.finditer(dict_pattern, context_block)
                            for dict_match in dict_matches:
                                dict_content = dict_match.group(1)
                                param_pattern = r'[\'"]([a-zA-Z0-9_]+)[\'"]:'
                                found_params = re.findall(param_pattern, dict_content)
                                if found_params:
                                    params.extend(found_params)
                                    log_debug(f"Found parameters in context: {found_params}")
                                    
                        # Deduplicate parameters
                        params = list(set(params))
                    
                    return prompt_ref, params
                    
        # If we get here, the prompt reference wasn't found
        log_debug(f"Could not find prompt reference '{content}' in any code file")
        return None
    except Exception as e:
        log_error(f"Error finding prompt usage: {str(e)}")
        log_debug(traceback.format_exc())
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
        # create test params with dummy values using the format variables
        test_params = {var: "test" for var in format_vars}
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
        global _code_file_cache
        if not _code_file_cache:
            _code_file_cache = asyncio.run(fetch_and_cache_code_files())
        
        log_debug(f"Searching for document structure used with {prompt_ref}")
        
        # Multiple patterns to match run_evaluation calls
        eval_patterns = [
            rf'run_evaluation\s*\(\s*([a-zA-Z0-9_]+)\s*,\s*{prompt_ref}\s*[,)]',
            rf'run_evaluation\s*\(\s*([a-zA-Z0-9_]+)\s*,\s*[\'"]?{prompt_ref}[\'"]?\s*[,)]',
            rf'run_evaluation\s*\(\s*([a-zA-Z0-9_]+)\s*,\s*{prompt_ref}\s*,.+\)',
        ]
        
        files_scanned = 0
        
        for file_path, content in _code_file_cache.items():
            if not isinstance(content, str):
                continue
                
            files_scanned += 1
            
            for eval_pattern in eval_patterns:
                eval_matches = re.search(eval_pattern, content)
                if not eval_matches:
                    continue
                
                # Found a run_evaluation call with our prompt_ref
                docs_var_name = eval_matches.group(1)
                log_debug(f"Found run_evaluation call in {file_path} using document variable: {docs_var_name}")
                
                # Get the context before this line to find the document structure
                file_content_before_eval = content[:eval_matches.start()]
                
                # Try different block-capturing approaches to handle multi-line definitions
                document_structure = {}
                
                # Try to find the document definition with balanced braces
                doc_start_pattern = re.compile(rf'{docs_var_name}\s*=\s*{{')
                start_match = doc_start_pattern.search(file_content_before_eval)
                
                if start_match:
                    start_pos = start_match.end() - 1  # Position of the opening '{'
                    # Find the corresponding closing brace with brace matching
                    brace_count = 1
                    doc_def = "{"
                    
                    for i in range(start_pos + 1, len(file_content_before_eval)):
                        char = file_content_before_eval[i]
                        doc_def += char
                        
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                break  # Found the closing brace
                    
                    if brace_count == 0:
                        log_debug(f"Found document structure definition with balanced braces")
                        
                        # Extract field definitions using regex for entries like:
                        # "name": {"type": "text", "content": name, "description": "OKR Name"}
                        field_pattern = re.compile(r'"([^"]+)":\s*{([^{}]|{[^{}]*})*?}', re.DOTALL)
                        field_matches = field_pattern.finditer(doc_def)
                        
                        for match in field_matches:
                            field_name = match.group(1)
                            field_def = match.group(0)
                            
                            # Extract attributes
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
                
                # If we found fields in the structure, return it
                if document_structure:
                    log_debug(f"Successfully parsed document structure with {len(document_structure)} fields")
                    for field, attrs in document_structure.items():
                        log_debug(f"  - Field: {field}, Type: {attrs.get('type')}, Desc: {attrs.get('description')}")
                    return document_structure
                
                # Try an alternative approach using line-by-line analysis
                if not document_structure:
                    lines = file_content_before_eval.splitlines()
                    in_doc_def = False
                    doc_def_lines = []
                    
                    # Look for document definition in the last 50 lines
                    for line in lines[-50:]:
                        if re.search(rf'{docs_var_name}\s*=\s*{{', line):
                            in_doc_def = True
                            doc_def_lines.append(line)
                        elif in_doc_def:
                            doc_def_lines.append(line)
                            if re.search(r'^\s*}\s*$', line):
                                # End of document definition
                                break
                    
                    if doc_def_lines:
                        doc_def = '\n'.join(doc_def_lines)
                        log_debug(f"Found document structure definition from lines")
                        
                        # Parse field entries
                        field_pattern = re.compile(r'"([^"]+)":\s*{([^}]+)}', re.DOTALL)
                        field_matches = field_pattern.finditer(doc_def)
                        
                        for match in field_matches:
                            field_name = match.group(1)
                            field_attrs_str = match.group(2)
                            
                            # Extract attributes
                            field_attrs = {}
                            
                            # Extract type
                            type_match = re.search(r'"type":\s*"([^"]+)"', field_attrs_str)
                            if type_match:
                                field_attrs['type'] = type_match.group(1)
                                
                            # Extract description
                            desc_match = re.search(r'"description":\s*"([^"]+)"', field_attrs_str)
                            if desc_match:
                                field_attrs['description'] = desc_match.group(1)
                                
                            # Add to document structure
                            document_structure[field_name] = field_attrs
                
                # If we found fields in the structure now, return it
                if document_structure:
                    log_debug(f"Successfully parsed document structure with {len(document_structure)} fields")
                    for field, attrs in document_structure.items():
                        log_debug(f"  - Field: {field}, Type: {attrs.get('type')}, Desc: {attrs.get('description')}")
                    return document_structure
                
        # If we've gone through all files and found nothing
        log_debug(f"No document structure found for {prompt_ref} after scanning {files_scanned} files")
        
        # Return empty dict if structure not found
        return {}
        
    except Exception as e:
        log_error(f"Error finding document structure: {str(e)}")
        log_debug(traceback.format_exc())
        return {}