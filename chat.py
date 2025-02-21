from typing import Dict, List, Any
import boto3
from datetime import datetime, timezone
from utils import get_data, get_dynamodb_table, get_prompt_from_dynamodb, run_completion_with_fallback

def get_recent_evaluations(stream_key: str, count: int = 5) -> List[Dict[str, Any]]:
    """Get the most recent evaluations for a stream key."""
    eval_table = get_dynamodb_table('website-evaluations')

    response = eval_table.query(
        KeyConditionExpression='streamKey = :sk',
        ExpressionAttributeValues={
            ':sk': stream_key
        },
        ScanIndexForward=False,  # Sort in descending order (most recent first)
        Limit=count + 1  # Get one extra to include the most recent
    )

    return response.get('Items', [])

def get_chat_context(stream_key: str) -> Dict[str, Any]:
    """Get comprehensive context for chat including recent evaluations and current data."""
    # Get evaluations
    evaluations = get_recent_evaluations(stream_key)
    if not evaluations:
        return None

    # Most recent evaluation
    most_recent = evaluations[0]
    # Previous 5 evaluations
    previous_evals = evaluations[1:6]

    # Get current data
    current_data = get_data(stream_key)
    if not current_data:
        return None

    # Get prompts from DynamoDB
    prompts = {}
    prompt_refs = most_recent.get('prompt_refs', [])
    for ref in prompt_refs:
        prompts[ref] = get_prompt_from_dynamodb(ref)

    # Compile failure reasons and summaries
    failure_summaries = []
    for eval_data in previous_evals:
        failure_summary = {
            'timestamp': eval_data.get('timestamp'),
            'failure_reason': eval_data.get('failure_reason'),
            'summary': eval_data.get('summary')
        }
        failure_summaries.append(failure_summary)

    # Compile context
    context = {
        'most_recent_evaluation': {
            'timestamp': most_recent.get('timestamp'),
            'conversation': most_recent.get('conversation', []),
            'results': most_recent.get('results', {})
        },
        'previous_evaluations': failure_summaries,
        'current_data': current_data,
        'prompts': prompts
    }

    return context

def chat_with_context(stream_key: str, user_message: str) -> str:
    """Chat with AI using comprehensive context."""
    # Get context
    context = get_chat_context(stream_key)
    if not context:
        return "Error: Could not retrieve context for chat."

    # Format context into prompt
    context_str = f"""
Context for {stream_key}:

Most Recent Evaluation (Timestamp: {datetime.fromtimestamp(context['most_recent_evaluation']['timestamp']/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}):
{context['most_recent_evaluation']['conversation']}

Previous Evaluation Summaries:
{'\n'.join([f"- Timestamp: {datetime.fromtimestamp(eval['timestamp']/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n  Failure Reason: {eval['failure_reason']}\n  Summary: {eval['summary']}" for eval in context['previous_evaluations']])}

Current Dashboard Data:
OKRs:
{'\n'.join([okr['markdown'] for okr in context['current_data']['okrs']])}

Insights:
{'\n'.join([insight['markdown'] for insight in context['current_data']['insights']])}

Suggestions:
{'\n'.join([suggestion['markdown'] for suggestion in context['current_data']['suggestions']])}

Active Prompts:
{'\n'.join([f"- {ref}: {prompt}" for ref, prompt in context['prompts'].items()])}

Previous Evaluation Results:
{context['most_recent_evaluation']['results']}
"""

    # Use run_completion_with_fallback for chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes website optimization data and provides insights. Use the provided context to answer questions accurately."},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_message}"}
    ]

    response = run_completion_with_fallback(messages=messages)
    return response if response else "Error: Could not generate response."
