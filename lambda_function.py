import os
from dotenv import load_dotenv
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from utils import (
    SYSTEM_PROMPT,
    PROMPT_INSTRUCTIONS, 
    run_completion_with_fallback, 
    get_context, 
    get_most_recent_stream_key,
    create_github_issue_with_project,
    update_prompt,
    get_prompt_expected_parameters,
    get_all_prompt_versions
)
from litellm.utils import token_counter
import asyncio

load_dotenv()

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



Be very detailed for how the prompt should be updated and include any necessary context because the prompt engineer that will update the prompt does not have access to the code files not including the prompt, previous evaluations, and other prompts.
Eg include the following details:
- All the variables used in the prompt and examples of what they look like
- Responsibility of agent in context of the workflow-
- Minified examples of what the prompt should look like
- Potential examples to use in the prompt
"""

PROMPT_UPDATE_INSTRUCTIONS = """
We are updating this prompt: {prompt_ref}

Current reason for update:
{reason}

Previous versions (from newest to oldest):
{previous_versions}

This prompt must use these required variables:
{variables}

These variables can optionally be in the prompt:
{optional_variables}

Usage in code:
File: {code_file}
Line: {code_line}
Function call: {function_call}

When updating the prompt, follow these instructions:
{PROMPT_INSTRUCTIONS}

Generate ONLY the new content for the prompt. Do not include any explanations or comments outside the prompt content.
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

async def generate_updated_prompt_content(prompt_ref: str, reason: str) -> str:
    """
    Generate detailed prompt content for a specific prompt reference.
    
    Args:
        prompt_ref: The reference of the prompt to update
        reason: The reason for updating and guidance on how to update
        
    Returns:
        The updated prompt content
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
            
        # Format the required and optional variables
        required_vars = ", ".join([f"{{{var}}}" for var in usage_info['parameters']]) or "None"
        optional_vars = ", ".join([f"{{{var}}}" for var in usage_info['optional_parameters']]) or "None"
        
        # Format the update instructions with the specific prompt information
        update_instructions = PROMPT_UPDATE_INSTRUCTIONS.format(
            prompt_ref=prompt_ref,
            reason=reason,
            previous_versions=versions_text,
            variables=required_vars,
            optional_variables=optional_vars,
            code_file=usage_info['file'],
            code_line=usage_info['line'],
            function_call=usage_info['function_call'],
            PROMPT_INSTRUCTIONS=PROMPT_INSTRUCTIONS
        )
        
        # Generate the updated prompt content
        messages = [
            {"role": "system", "content": "You are a prompt engineering expert. Generate only the updated prompt content based on the provided instructions."},
            {"role": "user", "content": update_instructions}
        ]
        
        content = run_completion_with_fallback(
            messages=messages,
            models=["reasoning"]  # Use the reasoning model for prompt generation
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
        messages = [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": SYSTEM_PROMPT_FINAL_INSTRUCTION.format(system_context=system_context)}
        ]

        
        analysis = run_completion_with_fallback(
            messages=messages,
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
                    
                    # Generate updated content
                    content, error = asyncio.run(generate_updated_prompt_content(prompt_ref, reason))
                    
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
