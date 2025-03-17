"""Database utilities for DynamoDB operations."""

import boto3
import os
import json
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import time
from functools import wraps

from .logging_utils import log_debug, log_error, measure_time

# Check if AWS credentials are set
aws_region = os.getenv('AWS_REGION') or "us-east-1"
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)

def get_boto3_resource(service_name='dynamodb'):
    """Get DynamoDB table resource with debug logging."""
    log_debug(f"Creating boto3 resource for {service_name}")
    try:
        resource = boto3.resource(
            service_name,
            region_name=aws_region
        )
        log_debug(f"Successfully created {service_name} resource")
        return resource
    except Exception as e:
        log_error(f"Failed to create {service_name} resource", e)
        raise

def get_boto3_client(service_name, region=None):
    """Get a boto3 client with credentials."""
    return boto3.client(
        service_name,
        region_name=aws_region
    )

def get_api_key(secret_name):
    """Get API key from AWS Secrets Manager."""
    client = get_boto3_client('secretsmanager', region="us-east-1")
    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )
    return json.loads(get_secret_value_response["SecretString"])

# Cache for DynamoDB resources and tables
_resource_cache = {}
_table_cache = {}

def get_dynamodb_table(table_name: str):
    """Get DynamoDB table resource with caching."""
    # Check if table already exists in cache
    if table_name in _table_cache:
        log_debug(f"Using cached table resource for {table_name}")
        return _table_cache[table_name]
    
    # Get or create the DynamoDB resource
    if 'dynamodb' not in _resource_cache:
        _resource_cache['dynamodb'] = get_boto3_resource('dynamodb')
    
    # Create the table resource and cache it
    log_debug(f"Creating and caching table resource for {table_name}")
    table = _resource_cache['dynamodb'].Table(table_name)
    _table_cache[table_name] = table
    return table

def convert_decimal(value):
    """Convert Decimal values to float/int for Streamlit metrics."""
    if isinstance(value, Decimal):
        return float(value)
    return value

def debug_dynamodb_table(table):
    """Debug helper to print table information."""
    try:
        log_debug(f"Table name: {table.name}")
        log_debug(f"Table ARN: {table.table_arn}")
        
        # Get table description
        description = table.meta.client.describe_table(TableName=table.name)['Table']
        
        # Log key schema
        log_debug("Key Schema:")
        for key in description.get('KeySchema', []):
            log_debug(f"- {key['AttributeName']}: {key['KeyType']}")
            
        # Log GSIs
        log_debug("Global Secondary Indexes:")
        for gsi in description.get('GlobalSecondaryIndexes', []):
            log_debug(f"- {gsi['IndexName']}:")
            for key in gsi['KeySchema']:
                log_debug(f"  - {key['AttributeName']}: {key['KeyType']}")
                
    except Exception as e:
        log_error(f"Error getting table info", e)

@measure_time
def parallel_dynamodb_query(queries: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Execute multiple DynamoDB queries in parallel.
    
    Args:
        queries: List of dictionaries with keys 'table', 'key', and 'params'
            - table: DynamoDB table resource
            - key: A key to identify the result in the output dictionary
            - params: Parameters to pass to the query method
    
    Returns:
        Dictionary with query keys mapped to results
    """
    try:
        log_debug(f"Executing {len(queries)} parallel DynamoDB queries")
        results = {}
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=min(10, len(queries))) as executor:
            # Create a future for each query
            future_to_key = {}
            
            for query in queries:
                table = query['table']
                key = query['key']
                params = query['params']
                
                # Submit query to executor
                future = executor.submit(
                    lambda t, p: t.query(**p).get('Items', []),
                    table,
                    params
                )
                future_to_key[future] = key
            
            # Process results as they complete
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    items = future.result()
                    results[key] = items
                    log_debug(f"Query for key {key} returned {len(items)} items")
                except Exception as e:
                    log_error(f"Query for key {key} failed", e)
                    results[key] = []
        
        return results
    except Exception as e:
        log_error(f"Error in parallel DynamoDB query", e)
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {}
