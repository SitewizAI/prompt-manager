import boto3
from datetime import datetime, timedelta
import json
from decimal import Decimal
from typing import Dict, Any, List

def query_evaluations_by_type(evaluations_table, eval_type: str, start_time: int, end_time: int) -> list:
    """Query evaluations for a specific type within a time range using type-timestamp-index."""
    try:
        # Use type-timestamp-index GSI for efficient querying
        query_params = {
            'IndexName': 'type-timestamp-index',
            'KeyConditionExpression': '#type = :type_val AND #ts BETWEEN :start AND :end',
            'ExpressionAttributeNames': {
                '#type': 'type',
                '#ts': 'timestamp'
            },
            'ExpressionAttributeValues': {
                ':type_val': eval_type,
                ':start': Decimal(str(start_time)),
                ':end': Decimal(str(end_time))
            },
            'ScanIndexForward': False  # Get most recent first
        }
        
        evaluations = []
        response = evaluations_table.query(**query_params)
        evaluations.extend(response.get('Items', []))
        
        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = evaluations_table.query(
                **query_params,
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            evaluations.extend(response.get('Items', []))
            
        print(f"Found {len(evaluations)} evaluations for type {eval_type}")
        return evaluations
    except Exception as e:
        print(f"Error querying evaluations for type {eval_type}: {str(e)}")
        return []

def convert_decimal(obj):
    """Convert Decimal objects to float/int recursively."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimal(i) for i in obj]
    return obj

def calculate_metrics(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate metrics from evaluations with proper type handling."""
    metrics = {
        'turns': 0,
        'evaluations': len(evaluations),
        'successes': 0,
        'attempts': 0,
        'quality_metric': 0.0
    }
    
    for eval in evaluations:
        # Convert Decimal types before calculations
        eval = convert_decimal(eval)
        
        # Sum up metrics
        metrics['turns'] += eval.get('num_turns', 0)
        metrics['attempts'] += eval.get('attempts', 0)
        metrics['successes'] += 1 if eval.get('success', False) else 0
        
    # Calculate quality metric
    if metrics['evaluations'] > 0:
        success_rate = metrics['successes'] / metrics['evaluations']
        avg_turns = metrics['turns'] / metrics['evaluations'] if metrics['turns'] > 0 else 0
        metrics['quality_metric'] = float(format(success_rate * (1.0 / (1.0 + avg_turns)), '.3f'))
    
    return metrics

def check_existing_metrics(metrics_table, eval_type: str, target_date: str) -> bool:
    """Check if metrics already exist for given type and date."""
    try:
        response = metrics_table.get_item(
            Key={
                'type': eval_type,
                'date': target_date
            }
        )
        return 'Item' in response
    except Exception as e:
        print(f"Error checking existing metrics: {str(e)}")
        return False

def aggregate_daily_metrics(event, context):
    """Aggregate daily metrics from EvaluationsTable to DateEvaluationsTable."""
    try:
        # Get the days_back parameter from the event, default to 1 if not specified
        days_back = event.get('days_back', 1)
        print(f"Processing metrics for {days_back} days back")

        dynamodb = boto3.resource('dynamodb')
        evaluations_table = dynamodb.Table('EvaluationsTable')
        metrics_table = dynamodb.Table('DateEvaluationsTable')

        # Create list of dates to process based on days_back parameter
        dates_to_process = []
        for day_offset in range(1, days_back + 1):
            target_date = datetime.now() - timedelta(days=day_offset)
            dates_to_process.append({
                'date': target_date.strftime('%Y-%m-%d'),
                'start_timestamp': int(target_date.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()),
                'end_timestamp': int(target_date.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp())
            })

        # List of evaluation types to aggregate
        eval_types = ['okr', 'insights', 'suggestion', 'code', 'design']
        metrics_stored = []

        # Process each date in the range
        for date_info in dates_to_process:
            target_date = date_info['date']
            start_timestamp = date_info['start_timestamp']
            end_timestamp = date_info['end_timestamp']
            
            print(f"Processing date: {target_date}")

            for eval_type in eval_types:
                try:
                    # Check if metrics exist first
                    if check_existing_metrics(metrics_table, eval_type, target_date):
                        print(f"Metrics already exist for {eval_type} on {target_date}")
                        continue

                    # Query evaluations for this type
                    evaluations = query_evaluations_by_type(
                        evaluations_table, 
                        eval_type,
                        start_timestamp,
                        end_timestamp
                    )

                    # Calculate metrics - if no evaluations, still create an entry with zeros
                    if not evaluations:
                        print(f"No evaluations found for type {eval_type} on {target_date} - creating empty record")
                        metrics = {
                            'turns': 0,
                            'evaluations': 0,
                            'successes': 0,
                            'attempts': 0,
                            'quality_metric': 0.0
                        }
                    else:
                        metrics = calculate_metrics(evaluations)
                    
                    # Convert all numbers to Decimal for DynamoDB
                    store_metrics = json.loads(
                        json.dumps(metrics), 
                        parse_float=Decimal
                    )

                    # Don't store type in data since it's redundant with the partition key
                    store_item = {
                        'type': eval_type,
                        'date': target_date,
                        'timestamp': Decimal(str(int(datetime.now().timestamp()))),
                        'data': store_metrics,
                        'ttl': Decimal(str(int((datetime.now() + timedelta(days=90)).timestamp())))
                    }

                    metrics_table.put_item(Item=store_item)
                    metrics_stored.append(f"{eval_type}_{target_date}")
                    print(f"Successfully stored metrics for {eval_type} on {target_date}")

                except Exception as e:
                    print(f"Error processing type {eval_type} on {target_date}: {str(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    continue

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Daily metrics aggregation completed',
                'dates_processed': [date_info['date'] for date_info in dates_to_process],
                'days_back': days_back,
                'metrics_stored': metrics_stored
            })
        }

    except Exception as e:
        print(f"Error in daily metrics aggregation: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Failed to aggregate daily metrics'
            })
        }

if __name__ == "__main__":
    # Test with days_back parameter
    test_event = {'days_back': 3}
    result = aggregate_daily_metrics(test_event, {})
    print(json.dumps(result, indent=2))