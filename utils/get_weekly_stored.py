"""Utilities for retrieving and analyzing weekly stored data."""

import boto3
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
from tabulate import tabulate
import traceback

from utils.db_utils import get_dynamodb_table
from utils.logging_utils import log_debug, log_error, measure_time
from utils.context_utils import okr_to_markdown, insight_to_markdown, suggestion_to_markdown

@measure_time
def get_weekly_storage_stats(include_verified_only: bool = True) -> Dict[str, Any]:
    """
    Get counts of OKRs, insights, and suggestions stored from the beginning of the week, 
    including totals and breakdowns per stream key.
    
    Args:
        include_verified_only: If True, only count verified items
        
    Returns:
        Dictionary with storage statistics
    """
    try:
        # Calculate timestamp for start of current week (Sunday)
        today = datetime.now()
        start_of_week = today - timedelta(days=today.weekday() + 1)  # +1 because weekday() considers Monday as 0
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        start_of_week_ms = int(start_of_week.timestamp() * 1000)
        start_of_week_s = int(start_of_week.timestamp())
        
        # Initialize result structure
        results = {
            "totals": {
                "okrs": 0,
                "insights": 0,
                "suggestions": 0
            },
            "by_stream_key": {},
            "week_start": start_of_week.isoformat(),
            "week_end": today.isoformat()
        }
        
        # Get tables
        okr_table = get_dynamodb_table('website-okrs')
        insight_table = get_dynamodb_table('website-insights')
        suggestion_table = get_dynamodb_table('WebsiteReports')
        
        # Scan for all stream keys first to have a complete list
        all_stream_keys = set()
        
        # Scan OKRs table for stream keys
        okr_scan = okr_table.scan(
            ProjectionExpression='streamKey',
            FilterExpression=Attr('timestamp').gte(start_of_week_ms)
        )
        for item in okr_scan.get('Items', []):
            if 'streamKey' in item:
                all_stream_keys.add(item['streamKey'])
        
        # Handle pagination for OKRs
        while 'LastEvaluatedKey' in okr_scan:
            okr_scan = okr_table.scan(
                ProjectionExpression='streamKey',
                FilterExpression=Attr('timestamp').gte(start_of_week_ms),
                ExclusiveStartKey=okr_scan['LastEvaluatedKey']
            )
            for item in okr_scan.get('Items', []):
                if 'streamKey' in item:
                    all_stream_keys.add(item['streamKey'])
        
        # Scan insights table for stream keys
        insight_scan = insight_table.scan(
            ProjectionExpression='streamKey',
            FilterExpression=Attr('timestamp').gte(start_of_week_ms)
        )
        for item in insight_scan.get('Items', []):
            if 'streamKey' in item:
                all_stream_keys.add(item['streamKey'])
        
        # Handle pagination for insights
        while 'LastEvaluatedKey' in insight_scan:
            insight_scan = insight_table.scan(
                ProjectionExpression='streamKey',
                FilterExpression=Attr('timestamp').gte(start_of_week_ms),
                ExclusiveStartKey=insight_scan['LastEvaluatedKey']
            )
            for item in insight_scan.get('Items', []):
                if 'streamKey' in item:
                    all_stream_keys.add(item['streamKey'])
        
        # Scan suggestions table for stream keys
        suggestion_scan = suggestion_table.scan(
            ProjectionExpression='streamKey',
            FilterExpression=Attr('timestamp').gte(start_of_week_s)
        )
        for item in suggestion_scan.get('Items', []):
            if 'streamKey' in item:
                all_stream_keys.add(item['streamKey'])
        
        # Handle pagination for suggestions
        while 'LastEvaluatedKey' in suggestion_scan:
            suggestion_scan = suggestion_table.scan(
                ProjectionExpression='streamKey',
                FilterExpression=Attr('timestamp').gte(start_of_week_s),
                ExclusiveStartKey=suggestion_scan['LastEvaluatedKey']
            )
            for item in suggestion_scan.get('Items', []):
                if 'streamKey' in item:
                    all_stream_keys.add(item['streamKey'])
        
        # Initialize stream key data
        for stream_key in all_stream_keys:
            results["by_stream_key"][stream_key] = {
                "okrs": 0,
                "insights": 0,
                "suggestions": 0
            }
        
        # Now query for actual counts for each stream key
        for stream_key in all_stream_keys:
            # Query OKRs - Use ts as the sort key in KeyConditionExpressionon
            if include_verified_only:
                okr_response = okr_table.query(
                    KeyConditionExpression=Key('streamKey').eq(stream_key),
                    FilterExpression=Attr('verified').eq(True) & Attr('timestamp').gte(start_of_week_ms)
                )
            else:
                okr_response = okr_table.query(
                    KeyConditionExpression=Key('streamKey').eq(stream_key),
                    FilterExpression=Attr('timestamp').gte(start_of_week_ms)
                )
                
            okr_count = len(okr_response.get('Items', []))
            
            # Handle pagination for OKRs
            while 'LastEvaluatedKey' in okr_response:
                if include_verified_only:
                    okr_response = okr_table.query(
                        KeyConditionExpression=Key('streamKey').eq(stream_key),
                        FilterExpression=Attr('verified').eq(True) & Attr('timestamp').gte(start_of_week_ms),
                        ExclusiveStartKey=okr_response['LastEvaluatedKey']
                    )
                else:
                    okr_response = okr_table.query(
                        KeyConditionExpression=Key('streamKey').eq(stream_key),
                        FilterExpression=Attr('timestamp').gte(start_of_week_ms),
                        ExclusiveStartKey=okr_response['LastEvaluatedKey']
                    )
                okr_count += len(okr_response.get('Items', []))
            
            # Update OKR counts
            results["by_stream_key"][stream_key]["okrs"] = okr_count
            results["totals"]["okrs"] += okr_count
            
            # Query insights - Use timestamp in KeyConditionExpression as in get_data()
            if include_verified_only:
                insight_response = insight_table.query(
                    KeyConditionExpression=Key('streamKey').eq(stream_key) & Key('timestamp').gte(start_of_week_ms),
                    FilterExpression=Attr('verified').eq(True)
                )
            else:
                insight_response = insight_table.query(
                    KeyConditionExpression=Key('streamKey').eq(stream_key) & Key('timestamp').gte(start_of_week_ms)
                )
                
            insight_count = len(insight_response.get('Items', []))
            
            # Handle pagination for insights
            while 'LastEvaluatedKey' in insight_response:
                if include_verified_only:
                    insight_response = insight_table.query(
                        KeyConditionExpression=Key('streamKey').eq(stream_key) & Key('timestamp').gte(start_of_week_ms),
                        FilterExpression=Attr('verified').eq(True),
                        ExclusiveStartKey=insight_response['LastEvaluatedKey']
                    )
                else:
                    insight_response = insight_table.query(
                        KeyConditionExpression=Key('streamKey').eq(stream_key) & Key('timestamp').gte(start_of_week_ms),
                        ExclusiveStartKey=insight_response['LastEvaluatedKey']
                    )
                insight_count += len(insight_response.get('Items', []))
            
            # Update insight counts
            results["by_stream_key"][stream_key]["insights"] = insight_count
            results["totals"]["insights"] += insight_count
            
            # Query suggestions - Use timestamp in KeyConditionExpression as in get_data()
            if include_verified_only:
                suggestion_response = suggestion_table.query(
                    KeyConditionExpression=Key('streamKey').eq(stream_key) & Key('timestamp').gte(start_of_week_s),
                    FilterExpression=Attr('verified').eq(True)
                )
            else:
                suggestion_response = suggestion_table.query(
                    KeyConditionExpression=Key('streamKey').eq(stream_key) & Key('timestamp').gte(start_of_week_s)
                )
                
            suggestion_count = len(suggestion_response.get('Items', []))
            
            # Handle pagination for suggestions
            while 'LastEvaluatedKey' in suggestion_response:
                if include_verified_only:
                    suggestion_response = suggestion_table.query(
                        KeyConditionExpression=Key('streamKey').eq(stream_key) & Key('timestamp').gte(start_of_week_s),
                        FilterExpression=Attr('verified').eq(True),
                        ExclusiveStartKey=suggestion_response['LastEvaluatedKey']
                    )
                else:
                    suggestion_response = suggestion_table.query(
                        KeyConditionExpression=Key('streamKey').eq(stream_key) & Key('timestamp').gte(start_of_week_s),
                        ExclusiveStartKey=suggestion_response['LastEvaluatedKey']
                    )
                suggestion_count += len(suggestion_response.get('Items', []))
            
            # Update suggestion counts
            results["by_stream_key"][stream_key]["suggestions"] = suggestion_count
            results["totals"]["suggestions"] += suggestion_count
        
        return results
    
    except Exception as e:
        log_error(f"Error getting weekly storage stats: {str(e)}")
        import traceback
        log_error(traceback.format_exc())
        return {
            "error": str(e),
            "totals": {"okrs": 0, "insights": 0, "suggestions": 0},
            "by_stream_key": {}
        }

@measure_time
def get_data(stream_key: str, task: str = None) -> Dict[str, Any]:
    """
    Get OKRs, insights and suggestions with markdown representations and relationship counts.
    Each OKR includes the number of insights connected.
    Each insight includes the number of suggestions connected.
    Suggestions include design status.
    The 'code' list is a subset of suggestions that include a Code field.

    Args:
        stream_key: The stream key to get data for
        task: Optional task type. If 'OKR', will also return all previous OKRs in the 'all_okrs' field
        
    Returns:
        Dictionary with processed data or None if an error occurred
    """
    try:
        # Use resource tables
        okr_table = get_dynamodb_table('website-okrs')
        insight_table = get_dynamodb_table('website-insights')
        suggestion_table = get_dynamodb_table('WebsiteReports')

        # Calculate timestamp for start of current week (Sunday)
        today = datetime.now()
        start_of_week = today - timedelta(days=today.weekday() + 1)  # +1 because weekday() considers Monday as 0
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        start_of_week_ms = int(start_of_week.timestamp() * 1000)
        start_of_week_s = int(start_of_week.timestamp())

        # Get all OKRs for the stream key from start of week
        okr_response = okr_table.query(
            KeyConditionExpression=Key('streamKey').eq(stream_key),
            FilterExpression=Attr('verified').eq(True) & Attr('timestamp').gte(start_of_week_ms)
        )
        okrs = okr_response.get('Items', [])
        
        # Handle pagination for OKRs
        while 'LastEvaluatedKey' in okr_response:
            okr_response = okr_table.query(
                KeyConditionExpression=Key('streamKey').eq(stream_key),
                FilterExpression=Attr('verified').eq(True) & Attr('timestamp').gte(start_of_week_ms),
                ExclusiveStartKey=okr_response['LastEvaluatedKey']
            )
            okrs.extend(okr_response.get('Items', []))

        # Get insights from start of week that are connected to an OKR
        insight_response = insight_table.query(
            KeyConditionExpression=Key('streamKey').eq(stream_key),
            FilterExpression=Attr('verified').eq(True) & Attr('timestamp').gte(start_of_week_ms)
        )
        insights = [item for item in insight_response.get('Items', []) if 'okr_name' in item]
        
        # Handle pagination for insights
        while 'LastEvaluatedKey' in insight_response:
            insight_response = insight_table.query(
                KeyConditionExpression=Key('streamKey').eq(stream_key),
                FilterExpression=Attr('verified').eq(True) & Attr('timestamp').gte(start_of_week_ms),
                ExclusiveStartKey=insight_response['LastEvaluatedKey']
            )
            insights.extend([item for item in insight_response.get('Items', []) if 'okr_name' in item])

        # Get suggestions from start of week
        suggestion_response = suggestion_table.query(
            KeyConditionExpression=Key('streamKey').eq(stream_key),
            FilterExpression=Attr('verified').eq(True) & Attr('timestamp').gte(start_of_week_s)
        )
        # Filter suggestions that have an associated InsightConnectionTimestamp
        suggestions = [
            item for item in suggestion_response.get('Items', [])
            if 'InsightConnectionTimestamp' in item
        ]
        
        # Handle pagination for suggestions
        while 'LastEvaluatedKey' in suggestion_response:
            suggestion_response = suggestion_table.query(
                KeyConditionExpression=Key('streamKey').eq(stream_key),
                FilterExpression=Attr('verified').eq(True) & Attr('timestamp').gte(start_of_week_s),
                ExclusiveStartKey=suggestion_response['LastEvaluatedKey']
            )
            suggestions.extend([
                item for item in suggestion_response.get('Items', [])
                if 'InsightConnectionTimestamp' in item
            ])

        processed_data = {
            "okrs": [],
            "insights": [],
            "suggestions": [],
            "code": [],
            "trajectories": []
        }

        # Process OKRs: each OKR gets an insight_count field.
        okr_map = {}
        for okr in okrs:
            okr_name = okr.get('name', 'N/A')
            okr_record = {
                "markdown": okr_to_markdown(okr),
                "name": okr_name,
                "insight_count": 0
            }
            # Add trajectory if available
            if 'trajectory' in okr:
                okr_record["trajectory"] = okr.get('trajectory')
                processed_data["trajectories"].append({
                    "type": "okr",
                    "name": okr_name,
                    "trajectory": okr.get('trajectory')
                })

            processed_data["okrs"].append(okr_record)
            okr_map[okr_name] = okr_record

        # Process insights: each insight gets a suggestion_count field.
        insight_map = {}
        for insight in insights:
            okr_name = insight.get('okr_name', 'N/A')
            insight_id = str(insight.get('timestamp', '0'))
            insight_record = {
                "markdown": insight_to_markdown(insight),
                "okr_name": okr_name,
                "timestamp": insight_id,
                "suggestion_count": 0
            }
            # Add trajectory if available
            if 'trajectory' in insight:
                insight_record["trajectory"] = insight.get('trajectory')
                processed_data["trajectories"].append({
                    "type": "insight",
                    "timestamp": insight_id,
                    "okr_name": okr_name,
                    "trajectory": insight.get('trajectory')
                })

            processed_data["insights"].append(insight_record)
            insight_map[insight_id] = insight_record
            # Update the corresponding OKR's insight count
            if okr_name in okr_map:
                okr_map[okr_name]["insight_count"] += 1

        # Process suggestions and update corresponding insight counts.
        for suggestion in suggestions:
            insight_id = str(suggestion.get('InsightConnectionTimestamp', '0'))
            # Determine if the suggestion includes a Code field or design
            has_code = suggestion.get('Code') is not None
            has_design = suggestion.get('Design') is not None
            suggestion_record = {
                "markdown": suggestion_to_markdown(suggestion, timestamp=True),
                "timestamp": suggestion["timestamp"],
                "InsightConnectionTimestamp": insight_id,
                "has_code": has_code,
                "has_design": has_design,
                "suggestion_id": suggestion.get("suggestionId", "")
            }
            processed_data["suggestions"].append(suggestion_record)
            # Update suggestion count for the associated insight
            if insight_id in insight_map:
                insight_map[insight_id]["suggestion_count"] += 1
            # Add to code list if it includes a Code field
            if has_code:
                processed_data["code"].append(suggestion_record)

        # If task is OKR, get all previous OKRs
        if task == 'OKR':
            # Get all OKRs for the stream key (without time filter)
            all_okr_response = okr_table.query(
                KeyConditionExpression=Key('streamKey').eq(stream_key),
                FilterExpression=Attr('verified').eq(True)
            )
            all_okrs = all_okr_response.get('Items', [])
            
            # Handle pagination for all OKRs
            while 'LastEvaluatedKey' in all_okr_response:
                all_okr_response = okr_table.query(
                    KeyConditionExpression=Key('streamKey').eq(stream_key),
                    FilterExpression=Attr('verified').eq(True),
                    ExclusiveStartKey=all_okr_response['LastEvaluatedKey']
                )
                all_okrs.extend(all_okr_response.get('Items', []))

            # Process all OKRs
            all_okrs_processed = []
            for okr in all_okrs:
                okr_record = {
                    "markdown": okr_to_markdown(okr),
                    "name": okr.get('name', 'N/A'),
                    "timestamp": okr.get('timestamp', 0),
                    "description": okr.get('description', '')
                }
                all_okrs_processed.append(okr_record)

            # Sort by timestamp (newest first)
            all_okrs_processed.sort(key=lambda x: x["timestamp"], reverse=True)

            # Add all_okrs to processed_data
            processed_data["all_okrs"] = all_okrs_processed

        return processed_data
    except Exception as e:
        log_error(f"Error processing data for stream key {stream_key}: {str(e)}")
        log_error(traceback.format_exc())
        return None

def format_storage_stats_table(stats: Dict[str, Any]) -> str:
    """
    Format storage statistics as a nice table.
    
    Args:
        stats: The statistics dictionary from get_weekly_storage_stats
        
    Returns:
        Formatted table as a string
    """
    if "error" in stats:
        return f"Error retrieving statistics: {stats['error']}"
    
    # Create a DataFrame for prettier display
    data = []
    for stream_key, counts in stats["by_stream_key"].items():
        data.append({
            "Stream Key": stream_key,
            "OKRs": counts["okrs"],
            "Insights": counts["insights"],
            "Suggestions": counts["suggestions"],
            "Total": counts["okrs"] + counts["insights"] + counts["suggestions"]
        })
    
    # Add total row
    data.append({
        "Stream Key": "TOTAL",
        "OKRs": stats["totals"]["okrs"],
        "Insights": stats["totals"]["insights"],
        "Suggestions": stats["totals"]["suggestions"],
        "Total": stats["totals"]["okrs"] + stats["totals"]["insights"] + stats["totals"]["suggestions"]
    })
    
    # Convert to DataFrame and format
    if data:
        df = pd.DataFrame(data)
        
        # Format header
        week_start = datetime.fromisoformat(stats["week_start"]).strftime('%Y-%m-%d')
        week_end = datetime.fromisoformat(stats["week_end"]).strftime('%Y-%m-%d')
        header = f"Weekly Storage Statistics ({week_start} to {week_end})"
        
        # Return formatted table
        return f"{header}\n\n{tabulate(df, headers='keys', tablefmt='grid', showindex=False)}"
    else:
        return "No data stored this week."

if __name__ == "__main__":
    # Example usage
    stats = get_weekly_storage_stats()
    print(format_storage_stats_table(stats))
