
# Example code for suggestions
from utils import get_prompt_from_dynamodb, run_evaluation

def process_suggestion(question, business_context, stream_key):
    # Get examples and criteria
    suggestion_example = "Example suggestion"
    suggestion_notes = "Important notes"
    suggestion_criteria = "Criteria for good suggestions"
    
    # Get system message with multiple parameters
    system_message = get_prompt_from_dynamodb("suggestions_user_proxy_system_message", {
        "question": question,
        "business_context": business_context,
        "stream_key": stream_key,
        "suggestion_example": suggestion_example,
        "suggestion_notes": suggestion_notes,
        "suggestion_criteria": suggestion_criteria,
        "stream_key": stream_key
    })
    
    # For document validation
    suggestion_doc = {
        "suggestion": {"type": "text", "content": "Sample suggestion", "description": "The suggestion"},
        "reasoning": {"type": "text", "content": "Why this works", "description": "Reasoning"},
        "context": {"type": "text", "content": business_context, "description": "Business context"}
    }
    
    # Run evaluation
    validation = run_evaluation(suggestion_doc, suggestion_questions)
    
    return system_message, validation
