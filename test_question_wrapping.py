"""Test script specifically for dictionary-wrapped question arrays validation."""

import json
from utils.prompt_utils import validate_prompt_parameters, update_prompt

def test_wrapped_questions_validation():
    """Test that wrapped questions are rejected."""
    print("\n===== Testing Wrapped Questions Validation =====")
    
    prompt_ref = "okr_questions"
    
    # Create a wrapped questions object - this should fail validation
    wrapped_questions = {
        "evaluation_questions": [
            {
                "question": "Is this OKR unique from previous OKRs?",
                "output": ["okr_markdown"],
                "reference": ["prev_okr_markdowns"],
                "confidence_threshold": 0.9,
                "feedback": "This OKR is not unique."
            },
            {
                "question": "Does the code implement requirements?",
                "output": ["code", "query_execution_output"],
                "reference": ["okr_criteria"],
                "confidence_threshold": 0.7,
                "feedback": "The code doesn't meet requirements."
            }
        ],
        "storing_function_modifications": "Some other content here...",
        "demonstrations": [
            {
                "input": "{sample_query_data}",
                "output": "Some output data",
                "explanation": "Explanation here"
            }
        ]
    }
    
    # Test validation with the dictionary directly
    print("\nTest 1: Validating wrapped questions as dictionary")
    is_valid, error_msg, details = validate_prompt_parameters(prompt_ref, wrapped_questions)
    print(f"Valid? {is_valid}")
    print(f"Error message: {error_msg}")
    
    # Test validation with the dictionary converted to JSON string
    print("\nTest 2: Validating wrapped questions as JSON string")
    wrapped_json = json.dumps(wrapped_questions)
    is_valid, error_msg, details = validate_prompt_parameters(prompt_ref, wrapped_json)
    print(f"Valid? {is_valid}")
    print(f"Error message: {error_msg}")
    
    # Test update_prompt with wrapped questions
    print("\nTest 3: Updating prompt with wrapped questions (should fail)")
    result = update_prompt(prompt_ref, wrapped_questions)
    print(f"Update succeeded? {result}")
    
    # Test with unwrapped questions array (should pass)
    print("\nTest 4: Validating properly formatted questions array (should pass)")
    unwrapped_questions = [
        {
            "question": "Is this OKR unique from previous OKRs?",
            "output": ["okr_markdown"],
            "reference": ["prev_okr_markdowns"],
            "confidence_threshold": 0.9,
            "feedback": "This OKR is not unique."
        },
        {
            "question": "Does the code implement requirements?",
            "output": ["code", "query_execution_output"],
            "reference": ["okr_criteria"],
            "confidence_threshold": 0.7,
            "feedback": "The code doesn't meet requirements."
        }
    ]
    is_valid, error_msg, details = validate_prompt_parameters(prompt_ref, unwrapped_questions)
    print(f"Valid? {is_valid}")
    if not is_valid:
        print(f"Error message (unexpected): {error_msg}")

    print("\nTest 5: Updating prompt with unwrapped questions array (should pass)")
    result = update_prompt(prompt_ref, unwrapped_questions)
    print(f"Update succeeded? {result}")
    
    
    # Print overall results
    print("\nAll tests completed. Dictionary-wrapped questions should have failed validation.")

if __name__ == "__main__":
    test_wrapped_questions_validation()
