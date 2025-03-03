
# Example code for OKR questions
from utils import get_prompt_from_dynamodb, run_evaluation

def process_okr():
    # Get questions
    okr_questions = get_prompt_from_dynamodb("okr_questions")
    
    # Prepare document structure for evaluation
    documents = {
        "name": {"type": "text", "content": "Sample OKR", "description": "OKR Name"},
        "description": {"type": "text", "content": "This is a sample OKR", "description": "OKR Description"},
        "okr_markdown": {"type": "text", "content": "# Sample OKR", "description": "OKR Markdown"}
    }
    
    # Run evaluation
    validation_results = run_evaluation(documents, okr_questions)
    
    return validation_results
