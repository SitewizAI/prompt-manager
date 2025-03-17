import boto3
import json
import time
import concurrent.futures
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from boto3.dynamodb.conditions import Key
import argparse

def get_dynamodb_table(table_name: str):
    """Get a DynamoDB table resource."""
    dynamodb = boto3.resource('dynamodb')
    return dynamodb.Table(table_name)

def query_evaluations_by_type_and_timestamp(
    table, 
    eval_type: str, 
    cutoff_timestamp: int, 
    batch_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Query evaluations by type and timestamp using the GSI.
    This is much more efficient than scanning the table.
    
    Args:
        table: DynamoDB table reference
        eval_type: The evaluation type to filter by (okr, insights, etc.)
        cutoff_timestamp: Timestamp to filter evaluations before
        batch_size: Maximum items to return per query
        
    Returns:
        List of evaluation items to delete
    """
    print(f"Querying {eval_type} evaluations before {datetime.fromtimestamp(cutoff_timestamp).strftime('%Y-%m-%d %H:%M:%S')} using type-timestamp-index")
    
    items_to_delete = []
    last_evaluated_key = None
    query_count = 0
    
    # Query params using type-timestamp-index GSI
    query_params = {
        'IndexName': 'type-timestamp-index',
        'KeyConditionExpression': Key('type').eq(eval_type) & Key('timestamp').lt(Decimal(str(cutoff_timestamp))),
        'ProjectionExpression': 'streamKey, #ts',  # Only request keys needed for deletion
        'ExpressionAttributeNames': {'#ts': 'timestamp'},
        'Limit': batch_size
    }
    
    # Keep querying until we've gone through all results
    more_pages = True
    while more_pages:
        query_count += 1
        
        # Add pagination token if we have one
        if last_evaluated_key:
            query_params['ExclusiveStartKey'] = last_evaluated_key
        
        # Execute the query
        start_time = time.time()
        response = table.query(**query_params)
        query_duration = time.time() - start_time
        
        # Get results
        batch_items = response.get('Items', [])
        items_to_delete.extend(batch_items)
        
        # Update pagination info
        last_evaluated_key = response.get('LastEvaluatedKey')
        more_pages = 'LastEvaluatedKey' in response
        
        print(f"Query #{query_count}: Found {len(batch_items)} items to delete, took {query_duration:.2f}s")
        
        # Be nice to DynamoDB and add a small delay between queries
        if more_pages:
            time.sleep(0.5)
    
    print(f"Completed query: Found {len(items_to_delete)} total {eval_type} evaluations to delete")
    return items_to_delete

def scan_evaluations_before_timestamp(table, cutoff_timestamp: int, batch_size: int = 100) -> List[Dict[str, Any]]:
    """
    Scan for evaluations older than the specified cutoff timestamp.
    Use this when no specific evaluation type is provided.
    
    Args:
        table: DynamoDB table reference
        cutoff_timestamp: Timestamp to filter evaluations before
        batch_size: Maximum items to return per scan
        
    Returns:
        List of evaluation items to delete
    """
    print(f"Scanning for evaluations before {datetime.fromtimestamp(cutoff_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    
    items_to_delete = []
    last_evaluated_key = None
    scan_count = 0
    
    # Scan params with filter for timestamp < cutoff
    scan_params = {
        'FilterExpression': Key('timestamp').lt(Decimal(str(cutoff_timestamp))),
        'ProjectionExpression': 'streamKey, #ts',  # Only request keys needed for deletion
        'ExpressionAttributeNames': {'#ts': 'timestamp'},
        'Limit': batch_size
    }
    
    # Keep scanning until we've gone through the entire table
    more_pages = True
    while more_pages:
        scan_count += 1
        
        # Add pagination token if we have one
        if last_evaluated_key:
            scan_params['ExclusiveStartKey'] = last_evaluated_key
        
        # Execute the scan
        start_time = time.time()
        response = table.scan(**scan_params)
        scan_duration = time.time() - start_time
        
        # Get results
        batch_items = response.get('Items', [])
        items_to_delete.extend(batch_items)
        
        # Update pagination info
        last_evaluated_key = response.get('LastEvaluatedKey')
        more_pages = 'LastEvaluatedKey' in response
        
        print(f"Scan #{scan_count}: Found {len(batch_items)} items to delete, took {scan_duration:.2f}s")
        
        # Be nice to DynamoDB and add a small delay between scans
        if more_pages:
            time.sleep(0.5)
    
    print(f"Completed scan: Found {len(items_to_delete)} total evaluations to delete")
    return items_to_delete

def delete_batch(batch: List[Dict[str, Any]], table, batch_idx: int) -> Tuple[int, float]:
    """
    Delete a batch of items from the table.
    
    Args:
        batch: List of items to delete
        table: DynamoDB table reference
        batch_idx: Batch index for logging
        
    Returns:
        Tuple of (number_deleted, duration_seconds)
    """
    start_time = time.time()
    deleted_count = 0
    
    try:
        with table.batch_writer() as batch_writer:
            for item in batch:
                stream_key = item.get('streamKey')
                timestamp = item.get('timestamp')
                
                if stream_key and timestamp:
                    batch_writer.delete_item(
                        Key={
                            'streamKey': stream_key,
                            'timestamp': timestamp
                        }
                    )
                    deleted_count += 1
                else:
                    print(f"Warning: Skipping item with missing keys: {item}")
        
        duration = time.time() - start_time
        print(f"Batch #{batch_idx}: Deleted {deleted_count} items in {duration:.2f}s")
        return deleted_count, duration
    except Exception as e:
        duration = time.time() - start_time
        print(f"Error in batch #{batch_idx}: {str(e)}, duration: {duration:.2f}s")
        return 0, duration

def delete_evaluations(items_to_delete: List[Dict[str, Any]], table, dry_run: bool = False, 
                      batch_size: int = 25, parallel: int = 5):
    """
    Delete evaluations in batches with parallel processing.
    
    Args:
        items_to_delete: List of items to delete with streamKey and timestamp
        table: DynamoDB table reference
        dry_run: If True, don't actually delete anything
        batch_size: How many items to delete in each batch request
        parallel: Maximum number of parallel deletion threads
    """
    if not items_to_delete:
        print("No items to delete")
        return
    
    if dry_run:
        print(f"DRY RUN: Would delete {len(items_to_delete)} evaluations")
        return
    
    total_items = len(items_to_delete)
    total_batches = (total_items + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"Deleting {total_items} items in {total_batches} batches using {parallel} parallel threads")
    
    # Create batches
    batches = []
    for i in range(0, total_items, batch_size):
        batches.append(items_to_delete[i:i + batch_size])
    
    # Use ThreadPoolExecutor for parallel deletions
    start_time = time.time()
    deleted_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        # Submit all batches as tasks
        future_to_batch = {
            executor.submit(delete_batch, batch, table, i): i 
            for i, batch in enumerate(batches)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_deleted, _ = future.result()
                deleted_count += batch_deleted
                # Print progress update
                progress = (batch_idx + 1) / total_batches * 100
                elapsed = time.time() - start_time
                print(f"Progress: {progress:.1f}% - {batch_idx + 1}/{total_batches} batches, {deleted_count}/{total_items} items deleted, elapsed: {elapsed:.2f}s")
            except Exception as e:
                print(f"Batch #{batch_idx} generated an exception: {str(e)}")
    
    total_time = time.time() - start_time
    print(f"Deletion complete: {deleted_count}/{total_items} evaluations deleted in {total_time:.2f}s")
    print(f"Average delete rate: {deleted_count / total_time:.2f} items/second")

def delete_old_evaluations(days: int, eval_type: Optional[str] = None, dry_run: bool = True, 
                          batch_size: int = 25, parallel: int = 5):
    """
    Delete evaluations older than the specified number of days.
    If eval_type is provided, only delete evaluations of that type.
    
    Args:
        days: Number of days before which to delete evaluations
        eval_type: Optional evaluation type to filter by (okr, insights, etc.)
        dry_run: If True, only scan for items but don't delete
        batch_size: How many items to delete in each batch
        parallel: Maximum number of parallel deletion threads
    """
    try:
        # Calculate cutoff timestamp (now minus specified days)
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_timestamp = int(cutoff_date.timestamp())
        
        type_str = f"'{eval_type}' " if eval_type else ""
        print(f"Deleting {type_str}evaluations older than {days} days ({cutoff_date.strftime('%Y-%m-%d %H:%M:%S')})")
        
        # Get DynamoDB table
        table = get_dynamodb_table('EvaluationsTable')
        
        # Get items to delete either by query or scan
        if eval_type:
            # Use query with GSI - much more efficient
            items_to_delete = query_evaluations_by_type_and_timestamp(
                table, 
                eval_type, 
                cutoff_timestamp, 
                batch_size
            )
        else:
            # Use scan - less efficient but works for all items
            items_to_delete = scan_evaluations_before_timestamp(
                table, 
                cutoff_timestamp, 
                batch_size
            )
        
        if not items_to_delete:
            print("No evaluations found to delete")
            return
        
        # Print stats before deleting
        print(f"Found {len(items_to_delete)} evaluations to delete")
        
        # Delete the evaluations with parallel processing
        delete_evaluations(items_to_delete, table, dry_run, batch_size, parallel)
        
    except Exception as e:
        print(f"Error deleting old evaluations: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

def list_evaluation_types(table, days: int = 30, limit: int = 100):
    """
    List all evaluation types in the table with their counts.
    Helps users know what types are available to delete.
    
    Args:
        table: DynamoDB table reference
        days: Number of recent days to scan (to limit scan size)
        limit: Maximum items per scan page
        
    Returns:
        Dictionary of evaluation types and their counts
    """
    print(f"Scanning for evaluation types in the past {days} days...")
    
    # Calculate cutoff timestamp
    cutoff_date = datetime.now() - timedelta(days=days)
    cutoff_timestamp = int(cutoff_date.timestamp())
    
    last_evaluated_key = None
    type_counts = {}
    scan_count = 0
    
    # Scan params - project only the type field to minimize data transfer
    scan_params = {
        'FilterExpression': Key('timestamp').gt(Decimal(str(cutoff_timestamp))),
        'ProjectionExpression': '#type',
        'ExpressionAttributeNames': {'#type': 'type'},
        'Limit': limit
    }
    
    # Keep scanning until we've gone through entire table or hit item limit
    more_pages = True
    total_items = 0
    while more_pages and total_items < 1000:  # Limit to 1000 items for quick sampling
        scan_count += 1
        
        # Add pagination token if we have one
        if last_evaluated_key:
            scan_params['ExclusiveStartKey'] = last_evaluated_key
        
        # Execute the scan
        response = table.scan(**scan_params)
        
        # Count each type
        for item in response.get('Items', []):
            eval_type = item.get('type', 'unknown')
            if eval_type in type_counts:
                type_counts[eval_type] += 1
            else:
                type_counts[eval_type] = 1
        
        total_items += len(response.get('Items', []))
        
        # Update pagination info
        last_evaluated_key = response.get('LastEvaluatedKey')
        more_pages = 'LastEvaluatedKey' in response
        
        # Be nice to DynamoDB
        if more_pages:
            time.sleep(0.5)
    
    # Print results
    print("\nEvaluation Types Found:")
    for eval_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {eval_type}: {count} evaluations")
    
    return type_counts

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Delete old evaluations from DynamoDB')
    parser.add_argument('--days', type=int, default=30,
                        help='Delete evaluations older than this many days (default: 30)')
    parser.add_argument('--type', type=str, default=None,
                        help='Only delete evaluations of this type (e.g., okr, insights)')
    parser.add_argument('--list-types', action='store_true',
                        help='List available evaluation types and their counts')
    parser.add_argument('--no-dry-run', action='store_true', 
                        help='Actually perform deletion (default is dry run)')
    parser.add_argument('--batch-size', type=int, default=25,
                        help='Batch size for operations (default: 25)')
    parser.add_argument('--parallel', type=int, default=5,
                        help='Maximum number of parallel deletion threads (default: 5)')
    
    args = parser.parse_args()
    
    # Get DynamoDB table
    table = get_dynamodb_table('EvaluationsTable')
    
    # List types if requested
    if args.list_types:
        list_evaluation_types(table, days=args.days)
        exit(0)
    
    # Get confirmation before proceeding with actual deletion
    dry_run = not args.no_dry_run
    if not dry_run:
        type_str = f"{args.type} " if args.type else ""
        confirm = input(f"WARNING: This will DELETE ALL {type_str}evaluations older than {args.days} days. Type 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("Operation cancelled")
            exit(1)
    
    # Delete evaluations
    delete_old_evaluations(args.days, args.type, dry_run, args.batch_size, args.parallel)
