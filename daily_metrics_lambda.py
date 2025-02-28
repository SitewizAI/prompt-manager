import boto3
from datetime import datetime, timedelta
import json
from decimal import Decimal
from typing import Dict, Any, List, Optional

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
        metrics['successes'] += eval.get('successes', 0)
        
    # Calculate quality metric
    if metrics['evaluations'] > 0:
        success_rate = metrics['successes'] / metrics['evaluations']
        avg_turns = metrics['turns'] / metrics['evaluations'] if metrics['turns'] > 0 else 0
        metrics['quality_metric'] = float(format(success_rate * (1.0 / (1.0 + avg_turns)), '.3f'))
    
    return metrics

def get_evaluations_for_date(evaluations_table, start_time: int, end_time: int, eval_types: List[str]) -> Dict[str, List[Dict]]:
    """
    Get evaluations for all specified types for a specific date range.
    Returns a dictionary with eval_type as key and list of evaluations as value.
    """
    evaluations_by_type = {}
    for eval_type in eval_types:
        evaluations = query_evaluations_by_type(evaluations_table, eval_type, start_time, end_time)
        evaluations_by_type[eval_type] = evaluations
    return evaluations_by_type

def get_prompt_versions_from_evaluation(evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract prompt versions from an evaluation.
    The 'prompts' field contains a list of dictionaries with 'ref' and 'version' keys.
    """
    # Extract the prompts array from the evaluation
    prompts = evaluation.get('prompts', [])
    
    # Check if we have prompts data
    if not prompts:
        print("No prompts found in evaluation")
        return []
    
    # Log found prompts for debugging
    print(f"Found {len(prompts)} prompts in evaluation: {json.dumps(prompts, default=str)[:100]}...")
    
    # Return the prompts as they are stored - maintain the original structure
    # Each item should already have 'ref' and 'version' keys
    return prompts

def get_prompt_versions_for_date(evaluations_by_type: Dict[str, List[Dict]], eval_types: List[str]) -> List[Dict[str, Any]]:
    """
    Get prompt versions from the first evaluation of any type found.
    Prioritizes types in the order they are provided in eval_types.
    """
    for eval_type in eval_types:
        evaluations = evaluations_by_type.get(eval_type, [])
        if evaluations:
            # Sort evaluations by timestamp to get the most recent one first
            sorted_evals = sorted(
                evaluations,
                key=lambda x: float(x.get('timestamp', 0)),
                reverse=True
            )
            
            for eval_data in sorted_evals:
                prompts = get_prompt_versions_from_evaluation(eval_data)
                if prompts:
                    print(f"Using prompts from {eval_type} evaluation from {datetime.fromtimestamp(float(eval_data.get('timestamp', 0))).strftime('%Y-%m-%d %H:%M:%S')}")
                    return prompts
    
    print("No prompts found in any evaluation")
    return []

def aggregate_daily_metrics(event, context):
    """
    Aggregate daily metrics from EvaluationsTable to DateEvaluationsTable.
    Uses type as hash key and timestamp (start of day) as range key.
    """
    try:
        # Get the days_back parameter from the event, default to 1 if not specified
        days_back = event.get('days_back', 1)
        print(f"Processing metrics for {days_back} days back")

        # Calculate target date
        target_date = datetime.now() - timedelta(days=days_back)
        date_str = target_date.strftime('%Y-%m-%d')
        
        # Calculate timestamp for the beginning of the target date (midnight UTC)
        # This will be used as the range key in DynamoDB
        start_of_day_timestamp = int(target_date.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        end_timestamp = int(target_date.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp())
        
        print(f"Processing date: {date_str} (timestamp range: {start_of_day_timestamp} to {end_timestamp})")

        dynamodb = boto3.resource('dynamodb')
        evaluations_table = dynamodb.Table('EvaluationsTable')
        metrics_table = dynamodb.Table('DateEvaluationsTable')

        # List of evaluation types to aggregate
        eval_types = ['okr', 'insights', 'suggestion', 'code', 'design']
        metrics_stored = []

        # Get all evaluations for this date across all types
        evaluations_by_type = get_evaluations_for_date(
            evaluations_table,
            start_of_day_timestamp,
            end_timestamp,
            eval_types
        )
        
        # Log the number of evaluations found for each type
        for eval_type, evals in evaluations_by_type.items():
            print(f"Found {len(evals)} evaluations for type {eval_type}")
        
        # Get prompt versions for this date from evaluations
        prompt_versions = get_prompt_versions_for_date(evaluations_by_type, eval_types)
        print(f"Found {len(prompt_versions)} prompt versions to store")
        
        # Convert prompt versions to a serializable format
        serialized_prompt_versions = json.loads(
            json.dumps(prompt_versions, default=str)
        )

        for eval_type in eval_types:
            try:
                # Query evaluations for this type
                evaluations = evaluations_by_type.get(eval_type, [])

                # Calculate metrics - if no evaluations, still create an entry with zeros
                if not evaluations:
                    print(f"No evaluations found for type {eval_type} on {date_str} - creating empty record")
                    metrics = {
                        'turns': 0,
                        'evaluations': 0,
                        'successes': 0,
                        'attempts': 0,
                        'quality_metric': 0.0
                    }
                else:
                    metrics = calculate_metrics(evaluations)
                    print(f"Calculated metrics for {eval_type}: {json.dumps(metrics, default=str)}")
                
                # Convert all numbers to Decimal for DynamoDB
                store_metrics = json.loads(
                    json.dumps(metrics), 
                    parse_float=Decimal
                )

                # Store the metrics with prompt versions for all types
                store_item = {
                    'type': eval_type,
                    'timestamp': Decimal(str(start_of_day_timestamp)),  # Use start of day timestamp as range key
                    'date': date_str,  # Store date as a separate attribute for readability
                    'data': store_metrics,
                    'promptVersions': serialized_prompt_versions,  # Store all prompt versions
                    'ttl': Decimal(str(int((datetime.now() + timedelta(days=90)).timestamp())))
                }

                # Always put item, which will overwrite if it exists
                metrics_table.put_item(Item=store_item)
                metrics_stored.append(f"{eval_type}_{date_str}")
                print(f"Successfully stored metrics for {eval_type} on {date_str} with {len(serialized_prompt_versions)} prompts")

            except Exception as e:
                print(f"Error processing type {eval_type} on {date_str}: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                continue

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Daily metrics aggregation completed',
                'date_processed': date_str,
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
    for i in range(1, 8):
        test_event = {'days_back': i}
        result = aggregate_daily_metrics(test_event, {})
        print(json.dumps(result, default=str, indent=2))
    # test_event = {'days_back': 1}
    # result = aggregate_daily_metrics(test_event, {})
    # print(json.dumps(result, default=str, indent=2))