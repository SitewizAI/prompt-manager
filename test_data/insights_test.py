
# Example code for insights questions
from utils import get_prompt_from_dynamodb, run_evaluation

def process_insights():
    # Get questions with parameters
    insights_questions = get_prompt_from_dynamodb("insights_questions", {
        "business_context": "Test context",
        "stream_key": "test_key"
    })
    
    # Prepare document structure for evaluation
    verification_object = {
        "Insight": {"type": "text", "content": "Sample insight", "description": "Main insight"},
        "Data": {"type": "text", "content": "Analytics data", "description": "Source data"},
        "evidence": {"type": "text", "content": "Evidence supporting insight", "description": "Supporting evidence"}
    }
    
    # Run evaluation with different variable name
    results = run_evaluation(verification_object, insights_questions, partition="insights#123")
    
    return results
