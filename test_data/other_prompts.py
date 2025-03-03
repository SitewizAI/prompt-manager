
# Example code for other prompt types
from utils import get_prompt_from_dynamodb

def get_okr_criteria():
    return get_prompt_from_dynamodb("okr_criteria")
    
def get_suggestion_example():
    return get_prompt_from_dynamodb("suggestion_example")
