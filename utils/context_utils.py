from typing import Any, Dict, Optional, Union, List
from decimal import Decimal
from datetime import datetime, timedelta, timezone
import os
import json
import asyncio
import boto3
from boto3.dynamodb.conditions import Key, Attr

from utils.db_utils import get_dynamodb_table
from utils.logging_utils import log_error, measure_time, log_debug
from utils.github_utils import get_github_project_issues, get_github_files_async
from utils.metrics_utils import get_daily_metrics_from_table

def okr_to_markdown(okr: dict) -> str:
    """Convert an OKR to markdown format."""
    markdown = "# OKR Analysis\n\n"

    # Add name and description
    markdown += f"## Name\n{okr.get('name', '')}\n\n"
    markdown += f"## Description\n{okr.get('description', '')}\n\n"

    # Add timestamp if available
    if 'timestamp' in okr:
        timestamp_int = int(okr.get('timestamp', 0))
        markdown += f"## Last Updated\n{datetime.fromtimestamp(timestamp_int/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add metrics output if available
    if 'output' in okr:
        try:
            output_dict = eval(okr['output'])
            markdown += "## Metrics\n"
            markdown += f"- Metric Name: {output_dict.get('Metric', 'N/A')}\n"
            markdown += f"- Description: {output_dict.get('Description', 'N/A')}\n"
            markdown += f"- Date Range: {output_dict.get('start_date', 'N/A')} to {output_dict.get('end_date', 'N/A')}\n"
            if 'values' in output_dict:
                markdown += "- Values:\n"
                for date, value in output_dict['values']:
                    markdown += f"  - {date}: {value}\n"
        except:
            markdown += f"## Raw Output\n{okr.get('output', 'N/A')}\n"

    # Add reach value if available
    if 'reach_value' in okr:
        markdown += f"\n## Reach\n{okr.get('reach_value', 'N/A')}\n"

    return markdown

def insight_to_markdown(insight: dict) -> str:
    """Convert an insight to markdown format."""
    try:
        markdown = "# Insight Analysis\n\n"

        # Add data statement
        markdown += f"## Data Statement\n{insight.get('data_statement', '')}\n\n"

        # Add other sections
        markdown += f"## Problem Statement\n{insight.get('problem_statement', '')}\n\n"
        markdown += f"## Business Objective\n{insight.get('business_objective', '')}\n\n"
        markdown += f"## Hypothesis\n{insight.get('hypothesis', '')}\n\n"

        # Add metrics
        markdown += "## Metrics\n"
        markdown += f"- Frequency: {insight.get('frequency', 'N/A')}\n"
        markdown += f"- Severity: {insight.get('severity', 'N/A')}\n"
        markdown += f"- Severity reasoning: {insight.get('severity_reasoning', 'N/A')}\n"
        markdown += f"- Confidence: {insight.get('confidence', 'N/A')}\n"
        markdown += f"- Confidence reasoning: {insight.get('confidence_reasoning', 'N/A')}\n"

        return markdown
    except Exception as e:
        print(f"Error converting insight to markdown: {e}")
        return f"Error processing insight. Raw data:\n{json.dumps(insight, indent=4)}"

def suggestion_to_markdown(suggestion: Dict[str, Any], timestamp: bool = False) -> str:
    """Convert a suggestion to markdown format."""
    try:
        markdown = []

        # Add header
        if 'Shortened' in suggestion:
            for shortened in suggestion.get('Shortened', []):
                if shortened.get('type') == 'header':
                    markdown.append(f"## {shortened.get('text', '')}\n")

        # Add timestamp if requested
        if timestamp and 'timestamp' in suggestion:
            ts = suggestion['timestamp']
            if isinstance(ts, (int, float, str)):
                try:
                    # Convert to datetime if it's a number
                    if isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                    else:
                        # Try parsing as float first
                        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                    markdown.append(f"**Created:** {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                except:
                    markdown.append(f"**Created:** {ts}")

        # Add tags
        if 'Tags' in suggestion:
            markdown.append("## Tags")
            for tag in suggestion.get('Tags', []):
                markdown.append(f"- **{tag.get('type', '')}:** {tag.get('Value', '')} ({tag.get('Tooltip', '')})")

        # Add expanded content
        if 'Expanded' in suggestion:
            for expanded in suggestion.get('Expanded', []):
                if expanded.get('type') == 'text':
                    markdown.append(f"### {expanded.get('header', '')}\n")
                    markdown.append(expanded.get('text', ''))

        # Add insights
        if 'Insights' in suggestion:
            markdown.append("## Insights")
            for insight in suggestion.get('Insights', []):
                if 'data' in insight:
                    for data_point in insight.get('data', []):
                        if data_point.get('type') == 'Heatmap':
                            markdown.append(f"- **Heatmap (id: {data_point.get('key', '')}, {data_point.get('name', '')}):** [{data_point.get('explanation', '')}]")
                        elif data_point.get('type') == 'Session Recording':
                            markdown.append(f"- **Session Recording (id: {data_point.get('key', '')}, {data_point.get('name', '')}):** [{data_point.get('explanation', '')}]")
                        else:
                            markdown.append(f"- **{data_point.get('type')} (id: {data_point.get('key', '')}, {data_point.get('name', '')}):** [{data_point.get('explanation', '')}]")
                markdown.append(insight.get('text', ''))

        return "\n\n".join(markdown)
    except Exception as e:
        print(f"Error converting suggestion to markdown: {e}")
        return f"Error processing suggestion. Raw data:\n{json.dumps(suggestion, indent=4)}"

# Global cache for prompts
_prompt_cache: Dict[str, List[Dict[str, Any]]] = {}

@measure_time
def get_prompts(refs: Optional[List[str]] = None, max_versions: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get prompts from DynamoDB PromptsTable with version history.
    Returns the first version and the most recent (max_versions-1) versions.
    """
    try:
        table = get_dynamodb_table('PromptsTable')
        
        if refs is None:
            # Scan for all unique refs using ExpressionAttributeNames
            response = table.scan(
                ProjectionExpression='#r',
                ExpressionAttributeNames={'#r': 'ref'}
            )
            refs = list(set(item['ref'] for item in response.get('Items', [])))
            
            # Handle pagination for the scan operation if necessary
            while 'LastEvaluatedKey' in response:
                response = table.scan(
                    ProjectionExpression='#r',
                    ExpressionAttributeNames={'#r': 'ref'},
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                refs.extend(list(set(item['ref'] for item in response.get('Items', []))))
            
            # Remove duplicates
            refs = list(set(refs))
        
        prompts = {}
        for ref in refs:
            # Query for all versions of this ref
            response = table.query(
                KeyConditionExpression='#r = :ref',
                ExpressionAttributeNames={'#r': 'ref'},
                ExpressionAttributeValues={':ref': ref}
            )
            
            if not response['Items']:
                continue
            
            # Handle pagination if needed
            all_versions = response['Items']
            while 'LastEvaluatedKey' in response:
                response = table.query(
                    KeyConditionExpression='#r = :ref',
                    ExpressionAttributeNames={'#r': 'ref'},
                    ExpressionAttributeValues={':ref': ref},
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                all_versions.extend(response['Items'])
                
            # Sort by version (convert to int for proper numerical sorting)
            all_versions.sort(key=lambda x: int(x.get('version', 0)))
            
            # Take the first version and most recent (max_versions - 1) versions
            if max_versions <= 1:
                selected_versions = [all_versions[-1]]  # Just the latest
            else:
                if len(all_versions) <= max_versions:
                    # If we have fewer versions than max_versions, take all of them
                    selected_versions = all_versions
                else:
                    # Take first version and (max_versions-1) most recent
                    selected_versions = [all_versions[0]] + all_versions[-(max_versions-1):]
            
            # Sort again to ensure latest versions are first
            selected_versions.sort(key=lambda x: int(x.get('version', 0)), reverse=True)
            
            prompts[ref] = selected_versions
            _prompt_cache[ref] = selected_versions
            
        log_debug(f"Retrieved prompt history for {len(prompts)} refs")
        return prompts
    except Exception as e:
        log_error(f"Error getting prompts: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {}

@measure_time
def get_data(stream_key: str) -> Dict[str, Any]:
    """
    Get OKRs, insights and suggestions with markdown representations and relationship counts.
    Each OKR includes the number of insights connected.
    Each insight includes the number of suggestions connected.
    Suggestions include design status.
    The 'code' list is a subset of suggestions that include a Code field.
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

        # Get insights from start of week that are connected to an OKR
        insight_response = insight_table.query(
            KeyConditionExpression=Key('streamKey').eq(stream_key) & Key('timestamp').gte(start_of_week_ms)
            # Uncomment the following line to filter only verified insights:
            , FilterExpression=Attr('verified').eq(True)
        )
        insights = [item for item in insight_response.get('Items', []) if 'okr_name' in item]

        # Get suggestions from start of week
        suggestion_response = suggestion_table.query(
            KeyConditionExpression=Key('streamKey').eq(stream_key) & Key('timestamp').gte(start_of_week_s)
            , FilterExpression=Attr('verified').eq(True)
        )
        # Filter suggestions that have an associated InsightConnectionTimestamp
        suggestions = [
            item for item in suggestion_response.get('Items', [])
            if 'InsightConnectionTimestamp' in item
        ]

        processed_data = {
            "okrs": [],
            "insights": [],
            "suggestions": [],
            "code": []
        }

        # Process OKRs: each OKR gets an insight_count field.
        okr_map = {}
        for okr in okrs:
            okr_name = okr.get('name', 'N/A')
            okr_record = {
                "markdown": okr_to_markdown(okr),
                "name": okr_name,
                "insight_count": 0,
                "raw": okr
            }
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
                "suggestion_count": 0,
                "raw": insight
            }
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
                "suggestion_id": suggestion.get("suggestionId", ""),
                "raw": suggestion
            }
            processed_data["suggestions"].append(suggestion_record)
            # Update suggestion count for the associated insight
            if insight_id in insight_map:
                insight_map[insight_id]["suggestion_count"] += 1
            # Add to code list if it includes a Code field
            if has_code:
                processed_data["code"].append(suggestion_record)

        return processed_data
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None

@measure_time
def get_conversation_history_from_s3(stream_key: str, timestamp: float, eval_type: str) -> Optional[str]:
    """
    Fetch conversation history from S3 instead of from DynamoDB.
    
    Args:
        stream_key: The stream key for the evaluation
        timestamp: The timestamp of the evaluation
        eval_type: The type of evaluation (okr, insights, etc.)
        
    Returns:
        The conversation history as a string, or None if not found
    """
    try:
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        # Get bucket name from environment variable
        bucket_name = os.environ.get('CONVERSATION_BUCKET', 'sitewiz-conversations')
        
        # Format the S3 key: conversations/{stream_key}/{timestamp}_{eval_type}.json
        s3_key = f"conversations/{stream_key}/{timestamp}_{eval_type}.json"
        
        log_debug(f"Fetching conversation from S3: {bucket_name}/{s3_key}")
        
        # Try to get the object from S3
        try:
            response = s3.get_object(Bucket=bucket_name, Key=s3_key)
            conversation_data = response['Body'].read().decode('utf-8')
            log_debug(f"Successfully retrieved conversation from S3, size: {len(conversation_data)} bytes")
            return conversation_data
        except s3.exceptions.NoSuchKey:
            # Try alternate format: conversations/{eval_type}/{stream_key}/{timestamp}.json
            alternate_key = f"conversations/{eval_type}/{stream_key}/{timestamp}.json"
            log_debug(f"Key not found, trying alternate format: {bucket_name}/{alternate_key}")
            
            try:
                response = s3.get_object(Bucket=bucket_name, Key=alternate_key)
                conversation_data = response['Body'].read().decode('utf-8')
                log_debug(f"Successfully retrieved conversation from alternate S3 key, size: {len(conversation_data)} bytes")
                return conversation_data
            except s3.exceptions.NoSuchKey:
                log_debug(f"Conversation not found in S3 for {stream_key}/{timestamp}/{eval_type}")
                return None
            
    except Exception as e:
        log_error(f"Error fetching conversation from S3: {str(e)}")
        import traceback
        log_debug(traceback.format_exc())
        return None

@measure_time
def get_context(
    stream_key: str, 
    current_eval_timestamp: Optional[float] = None,
    return_type: str = "string",
    include_github_issues: bool = False,
    include_code_files: bool = True,
    past_eval_count: int = 3  # New parameter to control how many past evaluations to include
) -> Union[str, Dict[str, Any]]:
    """Create context from evaluations, prompts, files, and daily metrics."""
    try:
        # Calculate timestamp for one week ago
        one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        one_week_ago_timestamp = one_week_ago.timestamp()
        
        # First query for evaluations - only use streamKey in the KeyConditionExpression
        # The timestamp attribute is part of the sort key, so we can safely use it in the KeyConditionExpression
        evaluations_table = get_dynamodb_table('EvaluationsTable')
        eval_response = evaluations_table.query(
            KeyConditionExpression='streamKey = :streamKey AND #ts >= :week_ago',
            ExpressionAttributeNames={
                '#ts': 'timestamp'
            },
            ExpressionAttributeValues={
                ':streamKey': stream_key,
                ':week_ago': Decimal(str(one_week_ago_timestamp))
            },
            ScanIndexForward=False,  # Get most recent first
            Limit=past_eval_count + 5  # Fetch more than needed to filter client-side
        )

        evaluations = eval_response.get('Items', [])
        

        # Check if we have any evaluations before continuing
        if not evaluations:
            raise ValueError(f"No evaluations found for stream key: {stream_key}")
        
        # Sort evaluations by timestamp
        evaluations.sort(key=lambda x: float(x.get('timestamp', 0)), reverse=True)
        
        # Get current evaluation
        if current_eval_timestamp:
            current_eval = next(
                (e for e in evaluations if float(e['timestamp']) == current_eval_timestamp),
                evaluations[0]
            )
        else:
            current_eval = evaluations[0]
            
        # Get previous evaluations
        prev_evals = [
            e for e in evaluations 
            if float(e['timestamp']) < float(current_eval['timestamp'])
        ][:past_eval_count]
        
        # Extract prompt refs from current and previous evaluations
        all_prompt_refs = []
        current_prompt_refs = current_eval.get('prompts', [])
        all_prompt_refs.extend([p.get('ref') for p in current_prompt_refs if isinstance(p, dict) and 'ref' in p])
        
        prev_prompts_by_eval = []
        for eval_idx, eval_item in enumerate(prev_evals):
            prompt_refs = []
            for p in eval_item.get('prompts', []):
                if isinstance(p, dict) and 'ref' in p:
                    prompt_refs.append(p)
                    all_prompt_refs.append(p.get('ref'))
            prev_prompts_by_eval.append({
                'eval_index': eval_idx,
                'timestamp': eval_item.get('timestamp'),
                'prompts': prompt_refs
            })
        
        # Get prompts with version history, including those from past evaluations
        prompts_dict = get_prompts(max_versions=10)  # Increased from 5 to 10 to get more versions
        
        # Extract specific prompt contents for the versions used in evaluations
        current_prompt_contents = []
        for p in current_prompt_refs:
            if isinstance(p, dict) and 'ref' in p and 'content' in p:
                current_prompt_contents.append({
                    'ref': p['ref'],
                    'version': p.get('version', 'N/A'),
                    'content': p['content'],
                    'is_object': p.get('is_object', False)
                })
        
        past_prompt_contents = []
        for eval_prompts in prev_prompts_by_eval:
            eval_prompt_contents = []
            for p in eval_prompts.get('prompts', []):
                if isinstance(p, dict) and 'ref' in p and 'content' in p:
                    eval_prompt_contents.append({
                        'ref': p['ref'],
                        'version': p.get('version', 'N/A'),
                        'content': p['content'],
                        'is_object': p.get('is_object', False),
                        'eval_index': eval_prompts['eval_index'],
                        'timestamp': eval_prompts['timestamp']
                    })
            past_prompt_contents.append(eval_prompt_contents)
        
        # Get evaluation type from current evaluation
        eval_type = current_eval.get('type')
        
        # Fetch conversation history from S3 instead of using the direct value
        current_timestamp = float(current_eval['timestamp'])
        conversation_history = get_conversation_history_from_s3(
            stream_key=stream_key, 
            timestamp=current_timestamp, 
            eval_type=eval_type
        ) or current_eval.get('conversation', '')  # Fallback to DynamoDB value if S3 fetch fails
        
        # Also fetch conversation history for previous evaluations from S3
        prev_conversations = []
        for idx, eval_item in enumerate(prev_evals):
            prev_timestamp = float(eval_item['timestamp'])
            prev_conversation = get_conversation_history_from_s3(
                stream_key=stream_key,
                timestamp=prev_timestamp,
                eval_type=eval_item.get('type', eval_type)
            ) or eval_item.get('conversation', '')  # Fallback to DynamoDB value
            prev_conversations.append(prev_conversation)
        
        # Get daily metrics
        metrics_result = get_daily_metrics_from_table(eval_type, days=30, get_prompts=True)
        daily_metrics = []
        historical_prompt_versions = []
        
        # Process daily metrics result
        if metrics_result:
            # Format daily metrics for context
            for date, metrics in metrics_result.get('daily_metrics', {}).items():
                daily_metrics.append({
                    'query_date': date,
                    'type': eval_type,
                    'data': metrics
                })
            
            # Get historical prompt versions if available
            if metrics_result.get('prompt_versions'):
                historical_prompt_versions = metrics_result['prompt_versions']
        
        # Get data for the stream key using the enhanced get_data function
        data = get_data(stream_key)

        # Count items in each data category
        okr_count = len(data.get('okrs', []))
        insight_count = len(data.get('insights', []))
        suggestion_count = len(data.get('suggestions', []))
        code_count = len(data.get('code', []))
        
        github_token = os.getenv('GITHUB_TOKEN')

        # Get GitHub issues if requested
        github_issues = []
        if include_github_issues and github_token:
            github_issues = get_github_project_issues(github_token)
        issue_count = len(github_issues)
        print(f"# of GitHub issues: {issue_count}")
        
        # Get Python files only if requested
        file_contents = []
        if include_code_files and github_token:
            file_contents = asyncio.run(get_github_files_async(github_token))
            file_count = len(file_contents)
            print(f"# of GitHub files: {file_count}")
        else:
            file_count = 0
            file_contents = []
        
        # Add all prompt versions to data statistics
        total_prompt_refs = len(prompts_dict)
        total_prompt_versions = sum(len(versions) for versions in prompts_dict.values())
        
        # Generate data statistics
        data_stats = {
            "evaluations": {
                "current": 1,
                "previous": len(prev_evals)
            },
            "daily_metrics": len(daily_metrics),
            "historical_prompts": len(historical_prompt_versions),
            "all_prompts": total_prompt_refs,  # Add total number of prompt refs
            "all_prompt_versions": total_prompt_versions,  # Add total number of prompt versions
            "okrs": okr_count,
            "insights": insight_count,
            "suggestions": suggestion_count,
            "code": code_count,
            "github_issues": issue_count,
            "code_files": file_count
        }
        
        # Prepare context data with enhanced prompt information
        context_data = {
            "current_eval": {
                "timestamp": datetime.fromtimestamp(float(current_eval['timestamp'])).strftime('%Y-%m-%d %H:%M:%S'),
                "type": current_eval.get('type', 'N/A'),
                "successes": current_eval.get('successes', 0),
                "attempts": current_eval.get('attempts', 0),
                "failure_reasons": current_eval.get('failure_reasons', []),
                "conversation": conversation_history,  # Use S3 conversation
                "prompts_used": current_prompt_refs,
                "prompt_contents": current_prompt_contents,
                "raw": current_eval
            },
            "prev_evals": [{
                "timestamp": datetime.fromtimestamp(float(e['timestamp'])).strftime('%Y-%m-%d %H:%M:%S'),
                "type": e.get('type', 'N/A'),
                "successes": e.get('successes', 0),
                "attempts": e.get('attempts', 0),
                "failure_reasons": e.get('failure_reasons', []),
                "summary": e.get('summary', 'N/A'),
                "prompts_used": e.get('prompts', []),
                "prompt_contents": past_prompt_contents[idx] if idx < len(past_prompt_contents) else [],
                "conversation": prev_conversations[idx] if idx < len(prev_conversations) else "",  # Add conversations
                "raw": e
            } for idx, e in enumerate(prev_evals)],
            "prompts": prompts_dict,
            "data": data,
            "files": file_contents,
            "daily_metrics": daily_metrics,
            "historical_prompt_versions": historical_prompt_versions,
            "data_stats": data_stats
        }
        
        if include_github_issues:
            context_data["github_issues"] = github_issues
        
        if return_type == "dict":
            return context_data
        
        # Create a data statistics section with added prompt stats
        data_stats_str = f"""
Data Statistics:
- Evaluations: {data_stats['evaluations']['current']} current, {data_stats['evaluations']['previous']} previous
- Daily Metrics: {data_stats['daily_metrics']} entries
- Historical Prompts: {data_stats['historical_prompts']} versions
- All Prompts: {data_stats['all_prompts']} refs, {data_stats['all_prompt_versions']} total versions
- OKRs: {data_stats['okrs']}
- Insights: {data_stats['insights']}
- Suggestions: {data_stats['suggestions']}
- Code: {data_stats['code']}
- GitHub Issues: {data_stats['github_issues']}
- Code Files: {data_stats['code_files']}
"""
        print(data_stats_str)
            
        # Build enhanced context string with prompt contents from past evaluations
        context_str = f"""
{data_stats_str}

Daily Metrics (Past Week):
{' '.join(f'''
Date: {metrics['query_date']}
Metrics for Type {metrics['type']}:
- Evaluations: {metrics['data']['evaluations']}
- Successes: {metrics['data']['successes']}
- Success Rate: {(metrics['data']['successes'] / metrics['data']['evaluations'] * 100) if metrics['data']['evaluations'] > 0 else 0:.1f}%
- Quality Metric: {metrics['data']['quality_metric']}
- Turns: {metrics['data']['turns']}
- Attempts: {metrics['data']['attempts']}
''' for metrics in daily_metrics[:7])}  # Limit to most recent 7 days

Historical Prompt Versions:
{' '.join(f'''
Date: {pv['date'] if 'date' in pv else 'N/A'}
Prompt: {pv.get('ref', 'N/A')} (Version {pv.get('version', 'N/A')})
''' for pv in historical_prompt_versions[:10])}  # Limit to 10 most recent versions but show full content

Current Evaluation:
Timestamp: {context_data['current_eval']['timestamp']}
Type: {context_data['current_eval']['type']}
Successes: {context_data['current_eval']['successes']}
Attempts: {context_data['current_eval']['attempts']}
Failure Reasons: {context_data['current_eval']['failure_reasons']}
Conversation History:
{conversation_history}

# ... rest of the format remains similar but with full content ...

Previous Evaluations:
{' '.join(f'''
Evaluation from {e['timestamp']}:
- Type: {e['type']}
- Successes: {e['successes']}
- Attempts: {e['attempts']}
- Failure Reasons: {e['failure_reasons']}
- Summary: {e['summary']}
- Conversation History:
{prev_conversations[idx] if idx < len(prev_conversations) else "No conversation history available"}
''' for idx, e in enumerate(context_data['prev_evals']))}

# ... rest of the context string ...

All Current Prompts and Versions:
{' '.join(f'''
Prompt: {prompt_ref}
{' '.join(f'''
  Version {version.get('version', 'N/A')} ({version.get('updatedAt', 'unknown date')}):
  Content:
  {version.get('content', 'Content not available')}
  ---------------------
''' for version in versions[:3])}  # Limit to latest 3 versions per prompt to avoid context overflow
''' for prompt_ref, versions in list(prompts_dict.items())[:20])}  # Limit to first 20 prompts to avoid context overflow

Current Data:
# ... rest of the existing content (OKRs, Insights, Suggestions, Files) ...
"""

        return context_str
    except Exception as e:
        log_error(f"Error creating context for stream key {stream_key}", e)
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {}