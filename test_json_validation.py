"""Test script for validating JSON question arrays."""

import json
from utils.validation_utils import validate_question_objects_with_documents
from utils.prompt_utils import update_prompt

def test_valid_questions():
    """Test with valid questions array."""
    prompt_ref = "okr_questions"
    
    # Valid questions format
    valid_questions = [
        {
            "question": "Is this OKR unique from previous OKRs?",
            "output": ["okr_markdown"],
            "reference": ["prev_okr_markdowns"],
            "confidence_threshold": 0.9,
            "feedback": "This OKR is not unique. Please propose a different OKR."
        },
        {
            "question": "Does the code strictly implement the OKR requirements?",
            "output": ["code", "query_execution_output"],
            "reference": ["okr_criteria"],
            "confidence_threshold": 0.7,
            "feedback": "The provided code does not meet all the requirements."
        }
    ]
    
    print("\nTesting valid questions array:")
    is_valid, error_msg, details = validate_question_objects_with_documents(prompt_ref, valid_questions)
    
    print(f"Valid? {is_valid}")
    if not is_valid:
        print(f"Error: {error_msg}")
    
    return is_valid

def test_invalid_wrapped_questions():
    """Test with questions wrapped in another object."""
    prompt_ref = "okr_questions"
    
    # Invalid format - questions wrapped in an object
    invalid_questions = {
        "evaluation_questions": [
            {
                "question": "Is this OKR unique?",
                "output": ["okr_markdown"],
                "reference": ["prev_okr_markdowns"],
                "confidence_threshold": 0.9,
                "feedback": "This OKR is not unique."
            }
        ],
        "storing_function_modifications": "Some text...",
        "demonstrations": []
    }
    
    print("\nTesting invalid wrapped questions:")
    is_valid, error_msg, details = validate_question_objects_with_documents(prompt_ref, invalid_questions)
    
    print(f"Valid? {is_valid}")
    if not is_valid:
        print(f"Error: {error_msg}")
    
    return not is_valid  # Should be invalid

def test_invalid_field_reference():
    """Test with questions referencing fields that don't exist in document structure."""
    prompt_ref = "okr_questions"
    
    # Invalid field references
    invalid_field_questions = [
        {
            "question": "Is this OKR unique?",
            "output": ["okr_markdown"],
            "reference": ["prev_okr_markdowns"],
            "confidence_threshold": 0.9,
            "feedback": "This OKR is not unique."
        },
        {
            "question": "Does the code have good metrics?",
            "output": ["values"],  # This field doesn't exist in document structure
            "reference": ["queries"],
            "confidence_threshold": 0.3,
            "feedback": "The values aren't good."
        }
    ]
    
    print("\nTesting invalid field references:")
    is_valid, error_msg, details = validate_question_objects_with_documents(prompt_ref, invalid_field_questions)
    
    print(f"Valid? {is_valid}")
    if not is_valid:
        print(f"Error: {error_msg}")
        print(f"Missing output fields: {details.get('missing_output_fields', [])}")
        print(f"Missing reference fields: {details.get('missing_reference_fields', [])}")
    
    return not is_valid  # Should be invalid

def test_update_prompt():
    """Test updating a prompt with different formats."""
    prompt_ref = "okr_questions"
    
    # Valid questions format
    valid_questions = [
        {
            "question": "Is this OKR unique from previous OKRs?",
            "output": ["okr_markdown"],
            "reference": ["prev_okr_markdowns"],
            "confidence_threshold": 0.9,
            "feedback": "This OKR is not unique. Please propose a different OKR."
        },
        {
            "question": "Does the code implement requirements?",
            "output": ["code", "query_execution_output"],
            "reference": ["okr_criteria"],
            "confidence_threshold": 0.7,
            "feedback": "The provided code does not meet all the requirements."
        }
    ]
    
    # Invalid wrapped format
    invalid_wrapped = {
        "evaluation_questions": [
            {
                "question": "Is this OKR unique?",
                "output": ["okr_markdown"],
                "reference": ["prev_okr_markdowns"],
                "confidence_threshold": 0.9,
                "feedback": "This OKR is not unique."
            }
        ]
    }
    
    print("\nTesting update_prompt with valid questions array:")
    result = update_prompt(prompt_ref, valid_questions)
    print(f"Update with valid questions: {'Success' if result else 'Failed'}")
    
    print("\nTesting update_prompt with invalid wrapped questions:")
    result = update_prompt(prompt_ref, invalid_wrapped)
    print(f"Update with invalid wrapped questions: {'Success' if result else 'Failed (Expected)'}")
    
    # Convert valid array to JSON string
    valid_json_str = json.dumps(valid_questions)
    print("\nTesting update_prompt with valid JSON string:")
    result = update_prompt(prompt_ref, valid_json_str)
    print(f"Update with valid JSON string: {'Success' if result else 'Failed'}")

if __name__ == "__main__":
    # Run all tests
    print("===== JSON Question Validation Tests =====")
    
    all_passed = True
    all_passed &= test_valid_questions()
    all_passed &= test_invalid_wrapped_questions()
    all_passed &= test_invalid_field_reference()
    
    print("\n===== Update Prompt Tests =====")
    test_update_prompt()
    
    print(f"\n===== Test Results =====")
    print(f"All validation tests: {'PASSED' if all_passed else 'FAILED'}")
