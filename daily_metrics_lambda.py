import boto3
from datetime import datetime, timedelta
import json

def aggregate_daily_metrics(event, context):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('DateEvaluationsTable')

    # Get yesterday's date since we're aggregating completed day
    yesterday = datetime.now() - timedelta(days=1)
    target_date = yesterday.strftime('%Y-%m-%d')
    start_timestamp = int(yesterday.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    end_timestamp = int(yesterday.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp())

    # Query all evaluations for yesterday
    response = table.query(
        KeyConditionExpression='#date = :date AND #timestamp BETWEEN :start AND :end',
        ExpressionAttributeNames={
            '#date': 'date',
            '#timestamp': 'timestamp'
        },
        ExpressionAttributeValues={
            ':date': target_date,
            ':start': start_timestamp,
            ':end': end_timestamp
        }
    )

    # Initialize metrics by type
    metrics_by_type = {}

    # Process all items
    for item in response.get('Items', []):
        data = item.get('data', {})
        eval_type = data.get('type', 'unknown')

        if eval_type not in metrics_by_type:
            metrics_by_type[eval_type] = {
                'turns': 0,
                'evaluations': 0,
                'successes': 0,
                'attempts': 0,
                'quality_metric': 0
            }

        metrics = metrics_by_type[eval_type]
        metrics['evaluations'] += 1
        metrics['turns'] += data.get('turns', 0)
        metrics['attempts'] += data.get('attempts', 0)
        metrics['successes'] += 1 if data.get('success', False) else 0

    # Store cumulative metrics for each type
    now = datetime.now()
    current_timestamp = int(now.timestamp())

    for eval_type, metrics in metrics_by_type.items():
        # Store with special marker to identify as cumulative metrics
        metrics['is_cumulative'] = True
        metrics['type'] = eval_type

        table.put_item(
            Item={
                'date': target_date,
                'timestamp': current_timestamp,
                'data': metrics
            }
        )

    return {
        'statusCode': 200,
        'body': json.dumps('Daily metrics aggregation completed')
    }
