import boto3
from datetime import datetime
from typing import Dict, Any, List
import json

def get_recent_evaluations(stream_key: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch the most recent evaluations for a given stream key.

    Args:
        stream_key: The stream key to query
        limit: Maximum number of items to return

    Returns:
        List of evaluation records
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('EvaluationsTable')

    response = table.query(
        KeyConditionExpression='streamKey = :sk',
        ExpressionAttributeValues={
            ':sk': stream_key
        },
        ScanIndexForward=False,  # Sort in descending order (most recent first)
        Limit=limit
    )

    return response.get('Items', [])

def get_prompt_by_ref(ref: str) -> Dict[str, Any]:
    """
    Fetch a prompt by its reference ID.

    Args:
        ref: The reference ID of the prompt

    Returns:
        Prompt data including ref, content, and version
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('PromptsTable')

    response = table.get_item(
        Key={
            'ref': ref
        }
    )

    return response.get('Item', {})

def format_evaluation(eval_data: Dict[str, Any]) -> str:
    """Format a single evaluation record for display."""
    timestamp = datetime.fromtimestamp(eval_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')

    formatted = f"\nEvaluation at {timestamp}\n"
    formatted += "=" * 50 + "\n"

    # Extract core evaluation data
    formatted += f"Question: {eval_data.get('question', 'N/A')}\n"
    formatted += f"Type: {eval_data.get('type', 'N/A')}\n"
    formatted += f"Success: {eval_data.get('success', False)}\n"
    formatted += f"Number of Turns: {eval_data.get('num_turns', 0)}\n"

    # Add failure reasons if present
    failure_reasons = eval_data.get('failure_reasons', [])
    if failure_reasons:
        formatted += "\nFailure Reasons:\n"
        for reason in failure_reasons:
            formatted += f"- {reason}\n"

    # Add summary if present
    if eval_data.get('summary'):
        formatted += f"\nSummary:\n{eval_data['summary']}\n"

    # Add prompt information if available
    prompt_ref = eval_data.get('prompt_ref')
    if prompt_ref:
        prompt_data = get_prompt_by_ref(prompt_ref)
        if prompt_data:
            formatted += "\nPrompt Information:\n"
            formatted += f"Ref: {prompt_data.get('ref', 'N/A')}\n"
            formatted += f"Version: {prompt_data.get('version', 'N/A')}\n"
            formatted += f"Content: {prompt_data.get('content', 'N/A')}\n"

    conversation = eval_data.get('conversation', '')
    if conversation:
        formatted += "\nConversation:\n"
        formatted += conversation

    return formatted

def main():
    # Example stream key - replace with actual key
    stream_key = "VPFnNcTxE78nD7fMcxfcmnKv2C5coD92vdcYBtdf"

    print(f"Fetching recent evaluations for stream key: {stream_key}")
    evaluations = get_recent_evaluations(stream_key)

    if not evaluations:
        print("No evaluations found")
        return

    print(f"Found {len(evaluations)} evaluations\n")

    for eval_data in evaluations:
        print(format_evaluation(eval_data))

if __name__ == "__main__":
    main()
