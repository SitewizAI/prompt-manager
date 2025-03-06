"""Utilities for fetching and analyzing evaluation metrics."""

import time
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
from functools import wraps
import traceback

from .db_utils import get_dynamodb_table, get_boto3_resource, get_boto3_client
from .logging_utils import log_debug, log_error

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"⏱️ {func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper

@measure_time
def get_conversation_history(stream_key: str, timestamp: float, eval_type: Optional[str] = None) -> str:
    """
    Fetch full conversation history for a specific evaluation from S3.
    
    Args:
        stream_key: The stream key identifier
        timestamp: Evaluation timestamp
        eval_type: Optional evaluation type filter
        
    Returns:
        Complete conversation history as string or empty string if not found
    """
    try:
        log_debug(f"Fetching conversation history for {stream_key} at {timestamp}")
        table = get_dynamodb_table('EvaluationsTable')
        
        # Query for the specific evaluation using primary key to get the conversation_key
        response = table.get_item(
            Key={
                'streamKey': stream_key,
                'timestamp': Decimal(str(timestamp))
            },
            ProjectionExpression='conversation_key,#t', # Use expression attribute name for 'type'
            ExpressionAttributeNames={
                '#t': 'type'  # Map the reserved keyword
            }
        )

        print(f"Response: {response}")
        
        if 'Item' not in response:
            log_debug(f"No evaluation found for timestamp {timestamp}")
            return ""
            
        item = response['Item']
        
        # If eval_type specified, check if it matches
        if eval_type and item.get('type') != eval_type:
            log_debug(f"Found evaluation with wrong type: {item.get('type')} vs {eval_type}")
            return ""
            
        # Get the conversation_key from the evaluation object
        conversation_key = item.get('conversation_key')
        if not conversation_key:
            log_debug(f"No conversation_key found for evaluation at timestamp {timestamp}")
            return ""

        # Fetch the conversation from S3
        try:
            s3_client = get_boto3_client('s3')
            bucket_name = 'sitewiz-websites'

            log_debug(f"Fetching conversation from S3 bucket {bucket_name} with key {conversation_key}")
            response = s3_client.get_object(Bucket=bucket_name, Key=conversation_key)
            conversation = response['Body'].read().decode('utf-8')
            return conversation
        except Exception as s3_error:
            log_error(f"Error fetching conversation from S3: {str(s3_error)}")
            log_debug(traceback.format_exc())
            return ""
        
    except Exception as e:
        log_error(f"Error fetching conversation history: {str(e)}")
        log_debug(traceback.format_exc())
        return ""

@measure_time
def get_evaluation_by_timestamp(stream_key: str, timestamp: float, eval_type: str = None) -> Dict[str, Any]:
    """
    Fetch a single evaluation by its timestamp and type.
    
    Args:
        stream_key: The stream key for the evaluation
        timestamp: The timestamp of the evaluation
        eval_type: Optional evaluation type
    
    Returns:
        The full evaluation including conversation history or empty dict if not found
    """
    try:
        table = get_dynamodb_table('EvaluationsTable')
        print(f"Fetching evaluation for {stream_key} at {timestamp}")
        # Query for the specific evaluation
        response = table.get_item(
            Key={
                'streamKey': stream_key,
                'timestamp': Decimal(str(timestamp))
            }
        )
        
        if 'Item' in response:
            item = response['Item']
            
            # If eval_type is provided, check if it matches
            if eval_type and item.get('type') != eval_type:
                return {}
                
            return item
        else:
            log_debug(f"No evaluation found for timestamp {timestamp}")
            return {}
    except Exception as e:
        log_error(f"Error fetching evaluation by timestamp {timestamp}", e)
        log_debug(traceback.format_exc())
        return {}

@measure_time
def get_most_recent_stream_key(eval_type: Optional[str] = None) -> Tuple[Optional[str], Optional[float]]:
    """
    Get the stream key and timestamp with the most recent evaluation, optionally filtered by type.
    Uses 'type-timestamp-index' GSI with partition key 'type' and sort key 'timestamp'.
    
    Args:
        eval_type: Optional type to filter by (e.g. 'okr', 'insights', etc)
    
    Returns:
        Tuple of (stream_key, timestamp) or (None, None) if not found
    """
    try:
        table = get_dynamodb_table('EvaluationsTable')
        
        if eval_type:
            # Query just for the specified type
            response = table.query(
                IndexName='type-timestamp-index',
                KeyConditionExpression='#type = :type_val',
                ExpressionAttributeNames={
                    '#type': 'type'
                },
                ExpressionAttributeValues={
                    ':type_val': eval_type
                },
                ScanIndexForward=False,  # Get in descending order
                Limit=1
            )
            
            items = response.get('Items', [])
            if items:
                item = items[0]
                return item['streamKey'], float(item['timestamp'])
            return None, None
        
        # If no type specified, check all types
        types = ['okr', 'insights', 'suggestion', 'code']
        most_recent_item = None
        
        for type_val in types:
            response = table.query(
                IndexName='type-timestamp-index',
                KeyConditionExpression='#type = :type_val',
                ExpressionAttributeNames={
                    '#type': 'type'
                },
                ExpressionAttributeValues={
                    ':type_val': type_val
                },
                ScanIndexForward=False,
                Limit=1
            )
            
            items = response.get('Items', [])
            if items:
                item = items[0]
                if most_recent_item is None or float(item['timestamp']) > float(most_recent_item['timestamp']):
                    most_recent_item = item
        
        if most_recent_item:
            return most_recent_item['streamKey'], float(most_recent_item['timestamp'])
        return None, None
        
    except Exception as e:
        log_error(f"Error getting most recent stream key", e)
        return None, None

@measure_time
def get_all_evaluations(limit_per_stream: int = 1000, eval_type: str = None) -> List[Dict[str, Any]]:
    """
    Fetch all recent evaluations using type-timestamp-index.
    Uses the GSI to get evaluations efficiently.
    """
    try:
        table = get_dynamodb_table('EvaluationsTable')
        
        # Calculate timestamp for filtering
        one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        one_week_ago_timestamp = Decimal(one_week_ago.timestamp())
        
        # Query params using type-timestamp-index
        query_params = {
            'IndexName': 'type-timestamp-index',
            'KeyConditionExpression': '#type = :type AND #ts >= :one_week_ago',
            'ExpressionAttributeNames': {
                '#type': 'type',
                '#ts': 'timestamp'
            },
            'ExpressionAttributeValues': {
                ':type': eval_type,
                ':one_week_ago': one_week_ago_timestamp
            },
            'ScanIndexForward': False,  # Get most recent first
            'Limit': limit_per_stream
        }
        
        # Single query to get all evaluations
        evaluations = []
        response = table.query(**query_params)
        evaluations.extend(response.get('Items', []))
        
        # Handle pagination if needed
        while 'LastEvaluatedKey' in response and len(evaluations) < limit_per_stream and len(evaluations) > 0:
            response = table.query(
                **query_params,
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            if len(response.get('Items', [])) == 0:
                break
            evaluations.extend(response.get('Items', []))
            if len(evaluations) >= limit_per_stream:
                evaluations = evaluations[:limit_per_stream]
                break
                
        log_debug(f"Retrieved {len(evaluations)} evaluations using type-timestamp-index")
        return evaluations
        
    except Exception as e:
        log_error("Error getting evaluations", e)
        log_debug(traceback.format_exc())
        return []

@measure_time
def get_stream_evaluations(stream_key: str, limit: int = 6, eval_type: str = None) -> List[Dict[str, Any]]:
    """
    Fetch recent evaluations for specific stream key, optionally filtered by type.
    Uses filter expression instead of index for type filtering.
    """
    try:
        table = get_dynamodb_table('EvaluationsTable')
        
        # Validate inputs
        if not stream_key:
            raise ValueError("stream_key cannot be empty")
            
        # Base query parameters with proper key condition expression
        query_params = {
            'KeyConditionExpression': 'streamKey = :streamKey',
            'ExpressionAttributeValues': {
                ':streamKey': stream_key
            },
            'ScanIndexForward': False,  # Get most recent first
            'Limit': limit
        }

        # Add type filter if specified
        if eval_type:
            query_params.update({
                'FilterExpression': '#type = :type_val',
                'ExpressionAttributeNames': {
                    '#type': 'type'
                },
            })
            query_params['ExpressionAttributeValues'][':type_val'] = eval_type

        # Execute query
        response = table.query(**query_params)
        evaluations = response.get('Items', [])
        
        # Get more items if needed
        while len(evaluations) < limit and 'LastEvaluatedKey' in response and len(evaluations) > 0:
            query_params['ExclusiveStartKey'] = response['LastEvaluatedKey']
            response = table.query(**query_params)
            if len(response.get('Items', [])) == 0:
                break
            evaluations.extend(response.get('Items', []))

        # Sort by timestamp and limit results
        evaluations.sort(key=lambda x: float(x.get('timestamp', 0)), reverse=True)
        return evaluations[:limit]
        
    except Exception as e:
        log_error(f"Error getting evaluations for stream key {stream_key}", e)
        log_debug(traceback.format_exc())
        return []

@measure_time
def get_recent_evaluations(eval_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get most recent evaluations using type-timestamp-index more efficiently.
    """
    try:
        table = get_dynamodb_table('EvaluationsTable')
        log_debug(f"Fetching {limit} recent evaluations for type: {eval_type}")
        
        # Single query to get complete data using GSI
        query_params = {
            'IndexName': 'type-timestamp-index',
            'KeyConditionExpression': '#type = :type',
            'ExpressionAttributeNames': {
                '#type': 'type',
                '#fr': 'failure_reasons',
                '#q': 'question',
                '#ts': 'timestamp'  # Add timestamp to expression attribute names
            },
            'ExpressionAttributeValues': {
                ':type': eval_type
            },
            'ProjectionExpression': 'streamKey, #type, successes, attempts, num_turns, #ts, prompts, #fr, #q, summary',
            'ScanIndexForward': False,  # Most recent first
            'Limit': limit
        }
        
        response = table.query(**query_params)
        evaluations = response.get('Items', [])
        log_debug(f"Retrieved {len(evaluations)} evaluations")
        
        return evaluations
        
    except Exception as e:
        log_error(f"Error getting recent evaluations for {eval_type}", e)
        log_debug(traceback.format_exc())
        return []

@measure_time
def get_evaluation_metrics(days: int = 30, eval_type: str = None) -> Dict[str, Any]:
    """
    Get evaluation metrics for the last N days using the type-timestamp schema.
    Returns daily and total metrics with proper formatting for visualization.
    """
    try:
        if not eval_type:
            raise ValueError("eval_type must be specified")
            
        # Get daily metrics from DateEvaluationsTable
        dynamodb = get_boto3_resource('dynamodb')
        date_table = dynamodb.Table('DateEvaluationsTable')
        
        # Calculate timestamp range for the query - use start of day timestamps
        n_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        start_timestamp = int(n_days_ago.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        
        # Query metrics by type and timestamp range
        response = date_table.query(
            KeyConditionExpression='#type = :type_val AND #ts >= :start_ts',
            ExpressionAttributeNames={
                '#type': 'type',
                '#ts': 'timestamp'
            },
            ExpressionAttributeValues={
                ':type_val': eval_type,
                ':start_ts': Decimal(str(start_timestamp))
            }
        )
        
        # Process daily metrics
        daily_metrics = {}
        total_metrics = {
            'total_evaluations': 0,
            'total_successes': 0,
            'total_attempts': 0,
            'total_turns': 0
        }
        
        # Initialize dates for complete date range
        current_date = n_days_ago.date()
        end_date_obj = datetime.now(timezone.utc).date()
        while current_date <= end_date_obj:
            date_str = current_date.strftime('%Y-%m-%d')
            daily_metrics[date_str] = {
                'evaluations': 0,
                'successes': 0,
                'attempts': 0,
                'turns': 0,
                'quality_metric': 0
            }
            current_date += timedelta(days=1)
        
        # Process metrics from query response
        for item in response.get('Items', []):
            date = item['date']  # Now using the date attribute
            data = item['data']
            
            # Populate metrics dictionary
            metrics = daily_metrics.get(date, {
                'evaluations': 0,
                'successes': 0,
                'attempts': 0, 
                'turns': 0,
                'quality_metric': 0
            })
            
            metrics['evaluations'] = data.get('evaluations', 0)
            metrics['successes'] = data.get('successes', 0)
            metrics['attempts'] = data.get('attempts', 0)
            metrics['turns'] = data.get('turns', 0)
            metrics['quality_metric'] = data.get('quality_metric', 0)
            daily_metrics[date] = metrics
            
            # Update total metrics
            total_metrics['total_evaluations'] += metrics['evaluations']
            total_metrics['total_successes'] += metrics['successes']
            total_metrics['total_attempts'] += metrics['attempts']
            total_metrics['total_turns'] += metrics['turns']
        
        # Calculate success rate for total metrics
        total_metrics['success_rate'] = (
            (total_metrics['total_successes'] / total_metrics['total_evaluations'] * 100)
            if total_metrics['total_evaluations'] > 0 else 0.0
        )
        
        # Calculate success rate and add to daily metrics
        for date, metrics in daily_metrics.items():
            metrics['success_rate'] = (
                (metrics['successes'] / metrics['evaluations'] * 100)
                if metrics['evaluations'] > 0 else 0.0
            )
        
        return {
            'total_metrics': total_metrics,
            'daily_metrics': daily_metrics
        }
    except Exception as e:
        log_error(f"Error getting evaluation metrics: {str(e)}")
        return {
            'total_metrics': {},
            'daily_metrics': {}
        }
    
def get_daily_metrics_from_table(eval_type: str, days: int = 30, get_prompts: bool = False) -> Dict[str, Any]:
    """
    Fetch metrics directly from DateEvaluationsTable using the updated schema.
    This uses the primary key structure of type + timestamp.
    
    Args:
        eval_type: The evaluation type to fetch metrics for
        days: Number of days to look back
        get_prompts: Whether to include prompt versions in the results
    
    Returns:
        Dictionary with total metrics, daily metrics, and optionally prompt versions
    """
    try:
        if not eval_type:
            raise ValueError("eval_type must be specified")
            
        # Get daily metrics from DateEvaluationsTable
        dynamodb = get_boto3_resource('dynamodb')
        date_table = dynamodb.Table('DateEvaluationsTable')
        
        # Calculate timestamp range for the query - use start of day timestamps
        n_days_ago = datetime.now(timezone.utc) - timedelta(days=days)
        start_timestamp = int(n_days_ago.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        
        # Query metrics by type and timestamp range using the primary key (type + timestamp)
        response = date_table.query(
            KeyConditionExpression='#type = :type_val AND #ts >= :start_ts',
            ExpressionAttributeNames={
                '#type': 'type',
                '#ts': 'timestamp'
            },
            ExpressionAttributeValues={
                ':type_val': eval_type,
                ':start_ts': Decimal(str(start_timestamp))
            }
        )
        
        items = response.get('Items', [])
        print(f"Found {len(items)} metrics entries for {eval_type} in the last {days} days")
        
        if not items:
            return {
                'total_metrics': {
                    'total_evaluations': 0,
                    'total_successes': 0,
                    'total_attempts': 0,
                    'total_turns': 0,
                    'success_rate': 0.0
                },
                'daily_metrics': {},
                'prompt_versions': [] if get_prompts else None
            }
            
        # Process daily metrics
        daily_metrics = {}
        total_metrics = {
            'total_evaluations': 0,
            'total_successes': 0,
            'total_attempts': 0,
            'total_turns': 0
        }

        # Store prompt versions if requested
        prompt_versions = [] if get_prompts else None
        
        for item in items:
            # Extract date and data
            date = item['date']  # Using the date attribute directly
            data = item['data']
            
            # Convert Decimal values to float for calculations
            data_dict = {k: float(v) if isinstance(v, Decimal) else v for k, v in data.items()}
            
            # Populate metrics dictionary
            daily_metrics[date] = {
                'evaluations': data_dict.get('evaluations', 0),
                'successes': data_dict.get('successes', 0),
                'attempts': data_dict.get('attempts', 0), 
                'turns': data_dict.get('turns', 0),
                'quality_metric': data_dict.get('quality_metric', 0)
            }
            
            # Update total metrics
            total_metrics['total_evaluations'] += daily_metrics[date]['evaluations']
            total_metrics['total_successes'] += daily_metrics[date]['successes']
            total_metrics['total_attempts'] += daily_metrics[date]['attempts']
            total_metrics['total_turns'] += daily_metrics[date]['turns']
            
            # Extract prompt versions if requested and available
            if get_prompts and 'promptVersions' in item:
                # Add date to each prompt version for context
                for prompt_version in item['promptVersions']:
                    prompt_version['date'] = date
                prompt_versions.extend(item['promptVersions'])
        
        # Calculate success rate for total metrics
        total_metrics['success_rate'] = (
            (total_metrics['total_successes'] / total_metrics['total_evaluations'] * 100)
            if total_metrics['total_evaluations'] > 0 else 0.0
        )
        
        # Calculate success rate for each day
        for date, metrics in daily_metrics.items():
            metrics['success_rate'] = (
                (metrics['successes'] / metrics['evaluations'] * 100)
                if metrics['evaluations'] > 0 else 0.0
            )
        
        result = {
            'total_metrics': total_metrics,
            'daily_metrics': daily_metrics
        }
        
        # Add prompt versions if requested
        if get_prompts:
            result['prompt_versions'] = prompt_versions
            
        return result
        
    except Exception as e:
        log_error(f"Error getting daily metrics: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {
            'total_metrics': {},
            'daily_metrics': {},
            'prompt_versions': [] if get_prompts else None
        }
