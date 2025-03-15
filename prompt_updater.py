"""
Tool for updating prompts with improvements based on their version history.
This differs from prompt_resetter.py in that it analyzes version history instead of
just resetting to version 0.
"""

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

from utils.prompt_utils import PROMPT_TYPES, AGENT_TOOLS, update_prompt, get_all_prompt_versions
from utils.logging_utils import log_debug, log_error, measure_time
from utils import get_dynamodb_table

PROMPT_UPDATE_SYSTEM = """You are an expert prompt engineer responsible for creating high-quality updated prompts.
Your task is to create an improved version of a prompt based on its version history and the provided instructions.

Review the versions of the prompt (especially version 0 and the most recent version) to understand how it has evolved.
Then create a new improved version that builds on the positive changes while addressing any issues.

Your goal is to create a prompt that:
1. Follows all formatting requirements
2. Uses the required variables correctly
3. Provides clear instructions to the model
4. Avoids any anti-patterns or problematic instructions
5. Builds on the strengths of the previous versions
6. Adds new improvements based on the provided instructions

Output ONLY the new prompt content. Do not include any explanations, comments, or metadata."""

PROMPT_UPDATE_INSTRUCTIONS = """
We are updating the prompt: {prompt_ref} to a new improved version

Version history:
{version_history}

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

When creating the new improved version, follow these instructions:
{SYSTEM_PROMPT}

Additional specific instructions for this update:
{additional_instructions}

Generate ONLY the new prompt content. Do not include any explanations or comments outside the prompt content. Do not prefix the prompt (eg by adding version numbers or suffix the prompt because the prompt will be provided as is to the LLM model. Do not add a ``` or ```python at the start of the prompt since the prompt should not be wrapped)
"""

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
            return response.get('Item', {})
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
            
    except Exception as e:
        log_error(f"Error retrieving prompt from DynamoDB: {str(e)}")
        return {}

def get_version_history(prompt_ref: str, max_versions: int = 5) -> Dict[str, Any]:
    """
    Get version history of a prompt, focusing on key versions.
    
    This function gets:
    1. Version 0 (original)
    2. Latest version
    3. Some intermediate versions if available
    
    Args:
        prompt_ref: The prompt reference ID
        max_versions: Maximum number of versions to include
        
    Returns:
        Dictionary with version history information
    """
    try:
        versions = get_all_prompt_versions(prompt_ref)
        
        if not versions:
            return {"error": f"No versions found for prompt {prompt_ref}"}
            
        # Sort versions by version number
        versions.sort(key=lambda x: int(x.get('version', 0)))
        
        # We always want version 0 and the latest version
        history = {}
        
        # Get version 0
        version_0 = next((v for v in versions if int(v.get('version', 0)) == 0), None)
        if version_0:
            history["v0"] = {
                "content": version_0.get('content', ''),
                "updatedAt": version_0.get('updatedAt', 'Unknown'),
                "is_object": version_0.get('is_object', False)
            }
        
        # Get latest version
        latest_version = versions[-1] if versions else None
        if latest_version and int(latest_version.get('version', 0)) != 0:
            version_num = int(latest_version.get('version', 0))
            history[f"v{version_num}"] = {
                "content": latest_version.get('content', ''),
                "updatedAt": latest_version.get('updatedAt', 'Unknown'),
                "is_object": latest_version.get('is_object', False)
            }
        
        # If we have more than 2 versions and max_versions > 2, add some intermediate versions
        if len(versions) > 2 and max_versions > 2:
            # Calculate how many intermediate versions we can include
            num_intermediate = min(max_versions - 2, len(versions) - 2)
            
            if num_intermediate > 0:
                # Pick evenly spaced intermediate versions
                step = (len(versions) - 2) // (num_intermediate + 1)
                for i in range(1, num_intermediate + 1):
                    idx = min(i * step, len(versions) - 2)
                    intermediate = versions[idx]
                    version_num = int(intermediate.get('version', 0))
                    history[f"v{version_num}"] = {
                        "content": intermediate.get('content', ''),
                        "updatedAt": intermediate.get('updatedAt', 'Unknown'),
                        "is_object": intermediate.get('is_object', False)
                    }
        
        # Create a formatted version history string
        formatted_history = ""
        for version, data in sorted(history.items(), key=lambda x: int(x[0][1:]) if x[0][1:].isdigit() else -1):
            formatted_history += f"\n--- {version.upper()} (Updated: {data['updatedAt']}) ---\n"
            formatted_history += data['content']
            formatted_history += "\n\n"
            
        return {
            "versions": history,
            "formatted_history": formatted_history,
            "latest_version": int(latest_version.get('version', 0)) if latest_version else 0
        }
    except Exception as e:
        log_error(f"Error getting version history: {str(e)}")
        return {"error": str(e)}

@measure_time
def get_prompt_expected_parameters(prompt_ref: str) -> Dict[str, Any]:
    """
    Get information about how a prompt is used in code, including expected parameters.
    
    Args:
        prompt_ref: The prompt reference ID
        
    Returns:
        Dictionary with usage information
    """
    # Import here to avoid circular imports
    from utils.prompt_utils import get_prompt_expected_parameters as utils_get_params
    
    try:
        return utils_get_params(prompt_ref)
    except Exception as e:
        log_error(f"Error getting prompt parameters: {str(e)}")
        return {'found': False}

def analyze_prompt_for_tools_info(prompt_ref: str, version_history: Dict[str, Any]) -> str:
    """
    Analyze prompt to determine if it's an agent system message and generate tools information.
    
    Args:
        prompt_ref: The prompt reference ID
        version_history: Version history data
        
    Returns:
        String with tools information or empty string
    """
    from utils.prompt_utils import PROMPT_TYPES, AGENT_TOOLS
    
    try:
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
Tools available to agents in the {prompt_group} group:
{chr(10).join(tools_info)}
"""
        return tools_information
    except Exception as e:
        log_error(f"Error analyzing prompt for tools: {str(e)}")
        return ""

@measure_time
def generate_updated_prompt_content(
    prompt_ref: str, 
    additional_instructions: str = "",
    validation_error: str = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate new content for an updated version of the prompt based on its version history.
    
    Args:
        prompt_ref: The prompt reference ID
        additional_instructions: Additional specific instructions for this update
        validation_error: Optional validation error from previous attempt
        
    Returns:
        Tuple of (content, error)
    """
    try:
        # Get version history
        version_history = get_version_history(prompt_ref, max_versions=5)
        if "error" in version_history:
            return None, f"Error getting version history: {version_history['error']}"
        
        # Format the version history
        formatted_history = version_history.get('formatted_history', 'No version history available.')
        
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
            validation_feedback = f"""
IMPORTANT: The previous attempt to update this prompt failed validation with the following error:
{validation_error}

Please fix these issues in your response. Pay special attention to:
1. Using only the required and optional variables specified
2. Correct formatting (especially for JSON objects)
3. Ensuring all required variable placeholders are present
4. Avoiding extra variables not listed in the requirements
"""

        # Get tools information
        tools_information = analyze_prompt_for_tools_info(prompt_ref, version_history)
        
        # Format the update instructions
        update_instructions = PROMPT_UPDATE_INSTRUCTIONS.format(
            prompt_ref=prompt_ref,
            version_history=formatted_history,
            variables=required_vars,
            optional_variables=optional_vars,
            tools_information=tools_information,
            code_file=usage_info.get('file', 'Unknown'),
            code_line=usage_info.get('line', 'Unknown'),
            function_call=usage_info.get('function_call', f"get_prompt_from_dynamodb('{prompt_ref}')"),
            format_instructions=format_instructions + validation_feedback,
            SYSTEM_PROMPT=SYSTEM_PROMPT,
            additional_instructions=additional_instructions
        )

        # Get all code files for context (will be cached in memory)
        from prompt_resetter import get_all_code_files
        code_files = get_all_code_files()
        
        # Two-step message approach with cache control
        messages = [
            # First message: System prompt + all code files (cached)
            {
                "role": "system", 
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_UPDATE_SYSTEM,
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
                "content": update_instructions
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

def update_prompt_with_improvements(
    prompt_ref: str,
    additional_instructions: str = ""
) -> Dict[str, Any]:
    """
    Update a prompt with improvements based on version history and additional instructions.
    
    Args:
        prompt_ref: The prompt reference ID
        additional_instructions: Additional instructions for the update
        
    Returns:
        Dictionary with results including success status
    """
    results = {
        'prompt_ref': prompt_ref,
        'success': False,
        'error': None,
        'previous_version': None,
        'new_version': None,
    }
    
    try:
        log_debug(f"Updating prompt {prompt_ref} with improvements")
        
        # Get version history to check if prompt exists and get latest version
        version_history = get_version_history(prompt_ref)
        if "error" in version_history:
            results['error'] = version_history["error"]
            return results
        
        # Record the previous version number
        previous_version = version_history.get('latest_version')
        results['previous_version'] = previous_version
        
        # Generate improved content with retries
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
                content, error = generate_updated_prompt_content(
                    prompt_ref, 
                    additional_instructions,
                    validation_error
                )
                if error:
                    last_error = error
                    log_error(f"Content generation failed: {error}")
                    continue
                
                # Check if we need to parse as JSON for _questions prompts
                if prompt_ref.endswith('_questions') and isinstance(content, str):
                    try:
                        # First try to parse as JSON to validate
                        parsed_json = json.loads(content)
                        # Use the parsed object directly to ensure update_prompt knows it's an object
                        content_to_update = parsed_json
                        log_debug(f"Using parsed JSON object for {prompt_ref}")
                    except json.JSONDecodeError as e:
                        validation_error = f"Invalid JSON format: {str(e)}. Make sure to provide a valid JSON array."
                        last_error = f"Invalid JSON format: {str(e)}"
                        log_error(last_error)
                        continue
                else:
                    content_to_update = content
                
                # Use update_prompt from prompt_utils to create a new version
                prompt_update_success, error_msg = update_prompt(prompt_ref, content_to_update)
                
                # Check if update was successful
                if prompt_update_success:
                    update_success = True
                    log_debug(f"Successfully updated prompt {prompt_ref}")
                    
                    # Get the new version number
                    latest_prompt = get_prompt_from_dynamodb(prompt_ref)
                    new_version = int(latest_prompt.get('version', 0))
                    results['new_version'] = new_version
                    
                    break
                else:
                    # Store validation error for next retry
                    validation_error = error_msg
                    last_error = f"Update failed: {error_msg}"
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
        log_debug(f"Successfully updated prompt {prompt_ref} from version {previous_version} to {results['new_version']}")
        
        return results
    except Exception as e:
        results['error'] = f"Error updating prompt: {str(e)}"
        log_error(f"Error in update_prompt_with_improvements: {str(e)}")
        log_error(traceback.format_exc())
        return results

def process_prompt(prompt_ref: str, additional_instructions: str = "") -> Dict[str, Any]:
    """
    Process a single prompt - wrapper for update_prompt_with_improvements to use with ThreadPoolExecutor.
    
    Args:
        prompt_ref: The prompt reference ID
        additional_instructions: Additional instructions for the update
        
    Returns:
        Result dictionary with status information
    """
    try:
        result = update_prompt_with_improvements(prompt_ref, additional_instructions)
        if result['success']:
            print(f"Successfully updated prompt {prompt_ref} from v{result['previous_version']} to v{result['new_version']}")
        else:
            print(f"Failed to update prompt {prompt_ref}: {result['error']}")
        return result
    except Exception as e:
        error_msg = f"Error processing prompt {prompt_ref}: {str(e)}"
        log_error(error_msg)
        print(error_msg)
        return {
            'prompt_ref': prompt_ref,
            'success': False,
            'error': error_msg,
            'previous_version': None,
            'new_version': None
        }

def process_prompt_group(
    prompt_refs: List[str], 
    additional_instructions: str = "",
    max_workers: int = 5
) -> List[Dict[str, Any]]:
    """
    Process multiple prompts in parallel using ThreadPoolExecutor.
    
    Args:
        prompt_refs: List of prompt reference IDs to process
        additional_instructions: Additional instructions for updates
        max_workers: Maximum number of parallel workers (default: 5)
        
    Returns:
        List of result dictionaries for each prompt
    """
    results = []
    
    print(f"Processing {len(prompt_refs)} prompts with {max_workers} parallel workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_prompt = {
            executor.submit(process_prompt, ref, additional_instructions): ref for ref in prompt_refs
        }
        
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
                    'previous_version': None,
                    'new_version': None
                })
    
    return results

if __name__ == "__main__":
    # Example usage:
    # 1. Update a single prompt
    # result = update_prompt_with_improvements(
    #     "example_prompt",
    #     "Please ensure the prompt is more specific about output format requirements."
    # )
    # print(f"Update result: {'Success' if result['success'] else 'Failed'}")
    
    # 2. Process a group of prompts with the same additional instructions
    # For more targeted updates, you would run this script multiple times with different
    # prompt selections and instructions
    
    # Example for updating OKR prompts
    prompts = PROMPT_TYPES["okr"]
    # prompts = ["insight_analyst_agent_description", "insight_analyst_agent_system_message", "insights_analyst_group_instructions"]
    instructions = """
    Please update these prompts by:
    1. You must ensure that calculate_reach and calculate_metrics follow the function signature given by the examples. Eg:
    - def calculate_reach(start_date: str, end_date: str) -> ReachOutput:
    - def calculate_metrics(start_date: str, end_date: str) -> MetricOutput:
    """
    # prompts = ["okr_store_agent_system_message", "okr_store_agent_description", "okr_store_group_instructions"]
    # instructions = """
    # Please update these prompts by:
    # 1. The OKR store agent should both create and store the OKR. 
    # """
    # instructions = """
    # Please update these prompts by:
    # 1. Remove all references to research_analyst, python_analyst_interpreter
    # 2. Make sure all agent are aware of the new groupchat workflow
    # """
    
    results = process_prompt_group(
        prompts,  # Process just one prompt for testing
        instructions,
        max_workers=10
    )
    
    # Print summary
    success_count = sum(1 for r in results if r['success'])
    print(f"Updates completed: {success_count}/{len(results)} successful")
    
    # Print details of failed updates
    failed = [r for r in results if not r['success']]
    if failed:
        print("\nFailed updates:")
        for f in failed:
            print(f"- {f['prompt_ref']}: {f['error']}")
