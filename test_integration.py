"""Test script to verify integration of the prompt manager components."""

import os
from utils import (
    get_all_prompts,
    validate_prompt_parameters,
    get_prompt_expected_parameters,
    log_debug
)

def test_prompt_validation():
    """Test prompt validation functionality."""
    print("\n===== Testing Prompt Validation =====")
    
    # Example prompt with variables
    test_prompt = """
    This is a test prompt with variables: {question}, {business_context}.
    It should be able to validate correctly.
    """
    
    # Test prompt validation
    is_valid, error_message, details = validate_prompt_parameters("test_prompt_ref", test_prompt)
    
    print(f"Is valid: {is_valid}")
    if error_message:
        print(f"Error message: {error_message}")
    print(f"Details: {details}")
    
    return is_valid

def test_parameter_extraction():
    """Test extraction of expected parameters from code."""
    print("\n===== Testing Parameter Extraction =====")
    
    # Test parameter extraction
    params = get_prompt_expected_parameters("test_prompt_ref")
    
    print(f"Parameters: {params['parameters']}")
    print(f"Optional parameters: {params['optional_parameters']}")
    print(f"File: {params['file']}")
    
    return len(params['parameters']) > 0 or params['found']

if __name__ == "__main__":
    # Set up working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run tests
    validation_result = test_prompt_validation()
    parameter_result = test_parameter_extraction()
    
    # Summary
    print("\n===== Test Summary =====")
    print(f"Prompt validation test: {'PASSED' if validation_result else 'FAILED'}")
    print(f"Parameter extraction test: {'PASSED' if parameter_result else 'FAILED'}")
