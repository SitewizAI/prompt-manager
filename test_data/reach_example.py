
# Example code for reach example prompt
from utils import get_prompt_from_dynamodb

def get_reach_content(stream_key):
    return get_prompt_from_dynamodb("reach_example", {
        "stream_key": stream_key
    })
