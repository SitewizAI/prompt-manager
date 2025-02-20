import boto3
from datetime import datetime
from typing import Dict, Any, List
import json
import argparse

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

def get_prompt_details(ref: str) -> Dict[str, Any]:
    """
    Fetch prompt details from the PromptsTable.

    Args:
        ref: The prompt reference ID

    Returns:
        Prompt details as a dictionary
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

    # Add prompt details if ref is present
    prompt_ref = eval_data.get('prompt_ref')
    if prompt_ref:
        prompt_details = get_prompt_details(prompt_ref)
        if prompt_details:
            formatted += "\nPrompt Details:\n"
            formatted += f"Ref: {prompt_ref}\n"
            for key, value in prompt_details.items():
                if key != 'ref':
                    formatted += f"{key}: {value}\n"

    conversation = eval_data.get('conversation', '')
    if conversation:
        formatted += "\nConversation:\n"
        formatted += conversation

    return formatted

def save_to_file(data: List[Dict[str, Any]], filename: str):
    """Save evaluation data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Analyze recent evaluations from DynamoDB')
    parser.add_argument('--stream-key', type=str, required=True,
                      help='Stream key to query evaluations for')
    parser.add_argument('--limit', type=int, default=10,
                      help='Maximum number of evaluations to fetch')
    parser.add_argument('--output', type=str, default='output/recent_evals.json',
                      help='Output file path for JSON data')

    args = parser.parse_args()

    print(f"Fetching recent evaluations for stream key: {args.stream_key}")
    evaluations = get_recent_evaluations(args.stream_key, args.limit)

    if not evaluations:
        print("No evaluations found")
        return

    print(f"Found {len(evaluations)} evaluations\n")

    # Save raw data to file
    save_to_file(evaluations, args.output)
    print(f"Raw evaluation data saved to {args.output}")

    # Display formatted evaluations
    for eval_data in evaluations:
        print(format_evaluation(eval_data))

if __name__ == "__main__":
    main()
