"""Utilities for prompt management."""

from datetime import datetime, timedelta
from decimal import Decimal
import json
import time
import os
import re
from typing import Dict, List, Any, Tuple, Optional, Union
from boto3.dynamodb.conditions import Key, Attr  # Added Key import
from botocore.exceptions import ClientError
import traceback
import asyncio

from .db_utils import get_dynamodb_table
from .logging_utils import log_debug, log_error
from .validation_utils import (
    validate_prompt_format, 
    validate_question_objects_with_documents,
    find_prompt_usage_in_code
)
from .context_utils import calculate_evaluation_score

behavioral_analyst = [
    "behavioral_analyst_system_message"
    "get_session_recording_videos",
    "get_similar_session_recordings"
]

insights_behavioral_analyst = [
    "behavioral_analyst_system_message",
    "get_element"
]

ux_researcher = [
    "ux_researcher_system_message",
    "ux_researcher_description",
    "get_similar_experiments_tool_description"
]

agents = {
    "behavioral_analyst": [
        "behavioral_analyst_system_message",
        "behavioral_analyst_description",
        "get_heatmap_tool_description",
        "get_similar_session_recordings_tool_description",
        "get_session_recording_videos_tool_description",
        "get_top_pages_tool_description",
    ],
    "web_agent": [
        "web_agent_system_message",
        "web_agent_description"
    ],
    "design_agent": [
        "design_agent_system_message",
        "design_agent_description",
        "store_design_tool_description"
    ],
    # "design_user_proxy": [
    #     "design_user_proxy_system_message",
    #     "design_user_proxy_description",
    #     "store_design_tool_description"
    # ],
    "python_analyst": [
        "python_analyst_system_message",
        "python_analyst_description"
    ],
    "okr_python_analyst": [
        "okr_python_analyst_system_message",
        "okr_python_analyst_description"
    ],
    # "okr_research_agent": [
    #     "okr_research_agent_system_message",
    #     "okr_research_agent_description"
    # ],
    # "okr_creator_agent": [
    #     "okr_creator_agent_system_message",
    #     "okr_creator_agent_description"
    # ],
    "okr_store_agent": [
        "okr_store_agent_system_message",
        "okr_store_agent_description",
        "store_okr_tool_description"
    ],
    # "python_analyst_interpreter": [
    #     "python_analyst_interpreter_system_message",
    #     "python_analyst_interpreter_description"
    # ],
    # "okr_python_analyst_interpreter": [
    #     "okr_python_analyst_interpreter_system_message",
    #     "okr_python_analyst_interpreter_description"
    # ],
    "insights_analyst": [
        "insight_analyst_agent_system_message",
        "insight_analyst_agent_description",
        "store_insight_tool_description"
    ],
    "insights_behavioral_analyst": [
        "insights_behavioral_analyst_system_message",
        "insights_behavioral_analyst_description",
        "get_heatmap_tool_description",
        "get_element_tool_description",
        "get_top_pages_tool_description"
    ],
    # "insights_analyst_code": [
    #     "insight_analyst_code_system_message",
    #     "insight_analyst_code_description"
    # ],
    # "insights_user_proxy": [
    #     "insights_user_proxy_system_message",
    #     "insights_user_proxy_description"
    #     "store_insight_tool_description"
    # ],
    # "research_analyst": [
    #     "research_analyst_system_message",
    #     "research_analyst_description"
    # ],
    "ux_researcher": [
        "ux_researcher_system_message",
        "ux_researcher_description",
        "get_screenshot_tool_description",
        "tavily_search_tool_description",
        "get_similar_experiments_tool_description"
    ],
    "suggestions_analyst": [
        "suggestions_analyst_system_message",
        "suggestions_analyst_description",
        "store_suggestion_tool_description",
    ],
    # "suggestions_user_proxy": [
    #     "suggestions_user_proxy_system_message",
    #     "suggestions_user_proxy_description",
    #     "store_suggestion_tool_description",
    # ],
    "website_developer": [
        "website_developer_system_message",
        "website_developer_description",
        "get_website_tool_description",
        "str_replace_editor_tool_description",
        "website_screenshot_tool_description"
    ],
    "website_get_save": [
        "website_get_save_system_message",
        "website_get_save_description",
        "store_website_tool_description"
    ]
}

agent_groups = {
    "okr": [
        # "okr_research_agent",
        # "okr_creator_agent",
        "okr_store_agent",
        "okr_python_analyst",
        # "okr_python_analyst_interpreter",
        "insights_behavioral_analyst",
    ],
    "insights": [
        "insights_analyst",
        "insights_behavioral_analyst",
        # "insights_analyst_code",
        # "insights_user_proxy",
        "python_analyst",
        # "python_analyst_interpreter",
        # "research_analyst",
    ],
    "suggestions": [
        "suggestions_analyst",
        # "suggestions_user_proxy",
        "ux_researcher",
        "behavioral_analyst",
    ],
    "design": [
        "design_agent",
        # "design_user_proxy",
        "web_agent",
    ],
    "code": [
        "website_developer",
        "website_get_save",
    ]
}

AGENT_TOOLS = {}
for group, agent_list in agent_groups.items():
    AGENT_TOOLS[group] = {}
    for agent in agent_list:
        AGENT_TOOLS[group][agent] = []
        for prompt in agents[agent]:
            if prompt.endswith("_tool_description"):
                AGENT_TOOLS[group][agent].append(prompt.removesuffix("_tool_description"))
# for agent, prompts in agents.items():
#     AGENT_TOOLS[agent] = []
#     for prompt in prompts:
#         if prompt.endswith("_tool_description"):
#             AGENT_TOOLS[agent].append(prompt.removesuffix("_tool_description"))

AGENT_GROUPS = {
    "okr": {
        "main": [
            "okr_store_group_instructions",
            "okr_python_group_instructions",
            # "okr_research_agent",
            "insights_behavioral_analyst"
        ],
        "store": {
            "okr_store_group_instructions": [
                "okr_creator_agent",
                "okr_store_agent"
            ],
        },
        "other": {
            "okr_python_group_instructions": [
                "okr_python_analyst",
                # "okr_python_analyst_interpreter"
            ]
        }
    },
    "insights": {
        "main": [
            "insights_analyst_group_instructions",
            "python_group_instructions",
            "insights_behavioral_analyst",
            # "research_analyst"
        ],
        "store": {
            "insights_analyst_group_instructions": [
                "insights_analyst",
                # "insights_analyst_code",
                # "insights_user_proxy"
            ]
        },
        "other": {
            "python_group_instructions": [
                "python_analyst",
                # "python_analyst_interpreter"
            ]
        }
    },
    "suggestions": {
        "main": [
            "suggestions_analyst_group_instructions",
            "ux_researcher",
            "behavioral_analyst"
        ],
        "store": {
            "suggestions_analyst_group_instructions": [
                "suggestions_analyst",
                "suggestions_user_proxy"
            ]
        }
    },
    "design": {
        "main": [
            "design_store_group_instructions",
            "web_agent"
        ],
        "store": {
            "design_store_group_instructions": [
                "design_agent",
                "design_user_proxy"
            ]
        }
    },
    "code": {
        "main": [
            "website_code_store_group_instructions",
            "website_developer"
        ],
        "store": {
            "website_code_store_group_instructions": [
                "website_get_save"
            ]
        }
    }
}

def generate_agent_groups_text(agent_groups, agents):
    """Dynamically generate explanatory text based on AGENT_GROUPS and agents variables."""
    text = "**Agent Group Organization by Task:**\n\n"
    
    # Iterate through all task types (okr, insights, etc.)
    for task_type, task_data in agent_groups.items():
        text += f"## {task_type.upper()} Task\n\n"
        
        # Process main task group for this task type
        if "main" in task_data:
            main_agents = task_data["main"]
            text += "### Main Workflow\n"
            text += f"Primary workflow with {len(main_agents)} components working together:\n"
            text += f"**Components:** {', '.join(main_agents)}\n\n"
            
            # List each main agent with their specific prompts
            for agent_name in main_agents:
                # If it's a group instruction rather than agent
                if (agent_name.endswith("_group_instructions")):
                    text += f"**{agent_name}:**\n"
                    text += f"- Prompt to optimize: `{agent_name}`\n"
                    # If this group is defined in the store or other sections, list its agents
                    for section in ["store", "other"]:
                        if section in task_data and agent_name in task_data[section]:
                            group_agents = task_data[section][agent_name]
                            text += f"- Contains agents: {', '.join(group_agents)}\n"
                # If it's an agent
                elif agent_name in agents:
                    text += f"**{agent_name}:**\n"
                    agent_prompts = agents[agent_name]
                    text += f"- Prompts to optimize: {', '.join([f'`{p}`' for p in agent_prompts])}\n"
                text += "\n"
        
        # Process specific group sections in more detail
        for section_name, section_data in task_data.items():
            if section_name == "main":
                continue  # Already processed above
                
            for group_name, group_agents in section_data.items():
                text += f"### {group_name}\n"
                
                # List all agents in this group with their specific responsibilities
                is_store_group = section_name == "store"
                if is_store_group:
                    # Identify the last agent that handles storing
                    last_agent = group_agents[-1] if group_agents else None
                    text += f"This group is responsible for storing {task_type} data.\n"
                    text += f"**Note:** Only `{last_agent}` should execute the function.\n\n"
                else:
                    text += f"This group handles specific implementation for {task_type}.\n\n"
                
                # List the group instruction prompt
                text += f"**Group Instruction:** `{group_name}`\n\n"
                
                # List each agent in the group with their specific prompts
                text += "**Agents in this group:**\n"
                for agent_name in group_agents:
                    if agent_name in agents:
                        text += f"- **{agent_name}**\n"
                        agent_prompts = agents[agent_name]
                        formatted_prompts = [f"`{p}`" for p in agent_prompts]
                        if formatted_prompts:
                            text += f"  - Prompts to optimize: {', '.join(formatted_prompts)}\n"
                
                # Add the shared task prompts that apply to this group
                text += f"\n**Shared task prompts that apply to this group:**\n"
                task_prompts = [
                    f"`{task_type}_task_context`",
                    f"`{task_type}_task_question`",
                    f"`{task_type}_questions`"
                ]
                text += f"- {', '.join(task_prompts)}\n\n"
        
        text += "---\n\n"
    
    # Add general interaction notes
    text += """
**Important Interaction Notes:**
- Only designated agents should use specific tools (especially store functions)
- The python_analyst is the only agent that can execute code and query databases
- Tool descriptions (ending with _tool_description) control how agents use available tools
- Evaluation questions (ending with _questions) validate outputs and prevent hallucinations - Do not update them, they are there for reference.
- IMPORTANT: If Agents output an empty message, a message saying they need more information, or a message that throws off the chat, that means something is wrong with the system prompt. Please check the prompt and make sure it is correct and remove rules that prevent it from outputting a response or triggering a tool.
- IMPORTANT: The Agent should never wait for the output of another agent to continue the conversation, they must execute the tools they have available regardless
"""
    return text

# Create the dynamic text variable
AGENT_GROUPS_TEXT = generate_agent_groups_text(AGENT_GROUPS, agents)
#concatenate all the prompts
PROMPT_TYPES = {
    "all": [],  # Will be populated with all refs
    "okr": [
        "okr_python_group_instructions",
        "okr_store_group_instructions",
        "okr_task_context",
        "okr_task_question",
        "okr_questions",
    ],
    "insights": [
        "python_group_instructions",
        "insights_analyst_group_instructions",
        "insights_task_context",
        "insights_task_question",
        "insight_questions"
    ],
    "suggestions": [
        "suggestions_analyst_group_instructions",
        "suggestions_task_context",
        "suggestions_task_question",
        "suggestion_questions",
        "data_questions",
    ],
    "design": [
        "design_store_group_instructions",
        "design_task_context",
        "design_task_question",
        "to_be_implemented_questions",
        "already_implemented_questions",
    ],
    "code": [
        "website_code_store_group_instructions",
        "code_task_context",
        "code_task_question",
        "code_questions",
    ]
}

for group, agent_list in agent_groups.items():
    for agent in agent_list:
        for ref in agents[agent]:
            PROMPT_TYPES[group].append(ref)

# Global cache for prompt expected parameters
_prompt_usage_cache = {}

def measure_time(func):
    """Decorator to measure function execution time."""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"⏱️ {func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper

@measure_time
def get_all_prompts() -> List[Dict[str, Any]]:
    """
    Fetch all prompts from DynamoDB PromptsTable, retrieving only the latest version of each prompt reference.
    This significantly improves performance by reducing the amount of data fetched.
    """
    try:
        log_debug("Attempting to get all prompts...")
        table = get_dynamodb_table('PromptsTable')
        
        # First, scan to get all unique refs
        response = table.scan(
            ProjectionExpression='#r',
            ExpressionAttributeNames={
                '#r': 'ref'
            }
        )
        
        # Extract and deduplicate prompt refs
        refs = list(set(item['ref'] for item in response.get('Items', [])))
        log_debug(f"Found {len(refs)} unique prompt references")
        
        # Handle pagination for the scan operation if necessary
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                ProjectionExpression='#r',
                ExpressionAttributeNames={
                    '#r': 'ref'
                },
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            new_refs = list(set(item['ref'] for item in response.get('Items', [])))
            refs.extend(new_refs)
            refs = list(set(refs))  # Deduplicate again
            
        # Now fetch only the latest version for each ref
        latest_prompts = []
        start_time = time.time()
        
        # Use batch operation to reduce network calls
        for i in range(0, len(refs), 25):  # Process in batches of 25 for better performance
            batch_refs = refs[i:i+25]
            log_debug(f"Processing batch of {len(batch_refs)} refs ({i+1}-{i+len(batch_refs)} of {len(refs)})")
            
            # Process each ref in the batch
            batch_results = []
            
            for ref in batch_refs:
                # Query for the latest version of this ref
                response = table.query(
                    KeyConditionExpression='#r = :ref',
                    ExpressionAttributeNames={'#r': 'ref'},
                    ExpressionAttributeValues={':ref': ref},
                    ScanIndexForward=False,  # Sort in descending order (newest first)
                    Limit=1  # Get only the latest version
                )
                
                if response.get('Items'):
                    batch_results.append(response['Items'][0])
            
            latest_prompts.extend(batch_results)
            log_debug(f"Batch processed in {time.time() - start_time:.2f}s - Total prompts: {len(latest_prompts)}")
            start_time = time.time()
        
        log_debug(f"Retrieved {len(latest_prompts)} latest prompt versions")
        return latest_prompts
        
    except Exception as e:
        log_error("Error getting prompts", e)
        print(f"Traceback: {traceback.format_exc()}")
        return []

@measure_time
def get_all_prompt_versions(ref: str) -> List[Dict[str, Any]]:
    """
    Fetch all versions of a specific prompt reference from DynamoDB.
    
    Args:
        ref: The prompt reference ID
        
    Returns:
        List of prompt versions sorted by version number (newest first)
    """
    try:
        log_debug(f"Fetching all versions for prompt ref: {ref}")
        table = get_dynamodb_table('PromptsTable')
        
        # Query for all versions of this ref
        response = table.query(
            KeyConditionExpression='#r = :ref',
            ExpressionAttributeNames={'#r': 'ref'},
            ExpressionAttributeValues={':ref': ref},
            ScanIndexForward=False  # Sort in descending order (newest first)
        )
        
        versions = response.get('Items', [])
        log_debug(f"Found {len(versions)} versions for prompt ref: {ref}")
        
        # Handle pagination if needed
        while 'LastEvaluatedKey' in response:
            response = table.query(
                KeyConditionExpression='#r = :ref',
                ExpressionAttributeNames={'#r': 'ref'},
                ExpressionAttributeValues={':ref': ref},
                ExclusiveStartKey=response['LastEvaluatedKey'],
                ScanIndexForward=False
            )
            versions.extend(response.get('Items', []))
        
        # Get evaluations that used this prompt - enhanced to focus on successful evaluations
        evaluations_table = get_dynamodb_table('EvaluationsTable')
        lookback_date = datetime.now() - timedelta(days=30)  # Look back 30 days
        lookback_timestamp = int(lookback_date.timestamp())
        
        # Scan for relevant evaluations
        scan_response = evaluations_table.scan(
            FilterExpression=Attr('timestamp').gte(Decimal(str(lookback_timestamp))) & 
                           Attr('prompts').exists(),
            ProjectionExpression='streamKey, #ts, #t, successes, attempts, quality_metric, prompts',
            ExpressionAttributeNames={
                '#ts': 'timestamp',
                '#t': 'type'
            }
        )
        
        all_evaluations = scan_response.get('Items', [])
        
        # Handle pagination
        while 'LastEvaluatedKey' in scan_response:
            scan_response = evaluations_table.scan(
                FilterExpression=Attr('timestamp').gte(Decimal(str(lookback_timestamp))) & 
                               Attr('prompts').exists(),
                ProjectionExpression='streamKey, #ts, #t, successes, attempts, quality_metric, prompts',
                ExpressionAttributeNames={
                    '#ts': 'timestamp',
                    '#t': 'type'
                },
                ExclusiveStartKey=scan_response['LastEvaluatedKey']
            )
            all_evaluations.extend(scan_response.get('Items', []))
        
        # Filter for evaluations that used this prompt
        relevant_evaluations = []
        for eval_item in all_evaluations:
            # Extract prompts used in this evaluation
            prompts = eval_item.get('prompts', [])
            for p in prompts:
                if isinstance(p, dict) and p.get('ref') == ref:
                    # Found an evaluation that used this prompt
                    successes = int(eval_item.get('successes', 0))
                    attempts = int(eval_item.get('attempts', 0))
                    quality_metric = float(eval_item.get('quality_metric', 0))
                    
                    # Calculate score
                    if successes == 0 or quality_metric == 0:
                        score = min(10, attempts)
                    else:
                        score = 10 + 10 * quality_metric
                    
                    # Create evaluation record with prompt version info and content if available
                    eval_record = {
                        'evaluation': eval_item,
                        'score': score,
                        'version': p.get('version'),
                        'successes': successes,
                        'timestamp': float(eval_item.get('timestamp', 0))
                    }
                    
                    # Add content if available in the prompt record
                    if 'content' in p:
                        eval_record['content'] = p['content']
                        
                    relevant_evaluations.append(eval_record)
                    break  # Only count each evaluation once
        
        # First, separate successful and unsuccessful evaluations
        successful_evals = [e for e in relevant_evaluations if e['successes'] > 0]
        other_evals = [e for e in relevant_evaluations if e['successes'] == 0]
        
        # Sort successful evaluations by timestamp (most recent first)
        successful_evals.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Sort other evaluations by score (highest first)
        other_evals.sort(key=lambda x: x['score'], reverse=True)
        
        # Combine with successful first, then best scoring
        prioritized_evaluations = successful_evals + other_evals
        
        # Calculate version usage statistics
        version_stats = {}
        for e in prioritized_evaluations:
            version = e.get('version')
            if version not in version_stats:
                version_stats[version] = {
                    'uses': 0,
                    'successful_uses': 0,
                    'total_score': 0,
                    'highest_score': 0,
                    'most_recent_content': None
                }
            
            version_stats[version]['uses'] += 1
            if e['successes'] > 0:
                version_stats[version]['successful_uses'] += 1
            version_stats[version]['total_score'] += e['score']
            version_stats[version]['highest_score'] = max(version_stats[version]['highest_score'], e['score'])
            
            # If this evaluation has content and it's more recent than what we have, use it
            if 'content' in e and (version_stats[version]['most_recent_content'] is None or 
                                   e['timestamp'] > version_stats[version].get('most_recent_timestamp', 0)):
                version_stats[version]['most_recent_content'] = e['content']
                version_stats[version]['most_recent_timestamp'] = e['timestamp']
        
        # Add evaluation usage info to each version
        for version in versions:
            version_num = version.get('version', 0)
            # Find evaluations that used this specific version
            matching_evals = [
                e for e in prioritized_evaluations 
                if e.get('version') == version_num
            ]
            
            # Add evaluation info to version
            if matching_evals:
                version['evaluation_usage'] = [
                    {
                        'timestamp': e['evaluation'].get('timestamp', ''),
                        'type': e['evaluation'].get('type', ''),
                        'score': e['score'],
                        'successes': e['evaluation'].get('successes', 0),
                        'attempts': e['evaluation'].get('attempts', 0),
                        'content_preview': e.get('content', '')[:100] + "..." if len(e.get('content', '')) > 100 else e.get('content', '')
                    }
                    for e in matching_evals[:5]  # Limit to top 5
                ]
                
                # Add version stats
                if version_num in version_stats:
                    version['usage_stats'] = version_stats[version_num]
                    
                    # If we have content from the evaluation but not in the version,
                    # and this is a successful evaluation, use that content
                    if ('content' not in version or not version['content']) and \
                       version_stats[version_num]['most_recent_content'] is not None:
                        version['content'] = version_stats[version_num]['most_recent_content']
                        version['content_source'] = 'evaluation'
            
            # Flag if this version has been used successfully
            if version_num in version_stats:
                version['successful_uses'] = version_stats[version_num]['successful_uses']

        # First sort by successful uses (most first)
        # Then by version (newest first)
        versions.sort(key=lambda x: (-(x.get('successful_uses', 0) > 0), int(x.get('version', 0))), reverse=True)
        return versions
        
    except Exception as e:
        log_error(f"Error getting all versions for prompt {ref}", e)
        log_debug(f"Traceback: {traceback.format_exc()}")
        return []

def get_evaluations_with_prompt(ref: str) -> List[Dict[str, Any]]:
    """
    Get evaluations that used a specific prompt reference.
    
    Args:
        ref: The prompt reference ID
        
    Returns:
        List of evaluations that used this prompt
    """
    try:
        # Get the DynamoDB table
        table = get_dynamodb_table('EvaluationsTable')
        
        # Calculate timestamp for one week ago
        one_week_ago = datetime.now() - timedelta(days=7)
        one_week_ago_timestamp = int(one_week_ago.timestamp())
        
        # Scan for evaluations in the past week
        # Note: In a production system, you might want to use GSIs or other optimization
        response = table.scan(
            FilterExpression=Attr('timestamp').gte(Decimal(str(one_week_ago_timestamp)))
        )
        
        evaluations = response.get('Items', [])
        
        # Handle pagination if needed
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression=Attr('timestamp').gte(Decimal(str(one_week_ago_timestamp))),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            evaluations.extend(response.get('Items', []))
        
        # Filter evaluations that used this prompt
        matching_evaluations = []
        for eval_item in evaluations:
            # Check if any prompt in this evaluation matches the ref
            prompt_refs = eval_item.get('prompts', [])
            if any(
                (isinstance(p, dict) and p.get('ref') == ref) for p in prompt_refs
            ):
                matching_evaluations.append(eval_item)
        
        return matching_evaluations
        
    except Exception as e:
        log_error(f"Error getting evaluations with prompt {ref}", e)
        log_debug(f"Traceback: {traceback.format_exc()}")
        return []

def get_prompt_from_dynamodb(ref: str, substitutions: Dict[str, Any] = None) -> str:
    """
    Get prompt with highest version from DynamoDB PromptsTable by ref.

    Args:
        ref: The reference ID of the prompt to retrieve
        substitutions: Optional dictionary of variables to substitute in the prompt

    Returns:
        The prompt content with substitutions applied if provided
    """
    try:
        table = get_dynamodb_table('PromptsTable')
        # Query the table for all versions of this ref
        response = table.query(
            KeyConditionExpression='#r = :ref',
            ExpressionAttributeNames={'#r': 'ref'},
            ExpressionAttributeValues={':ref': ref},
            ScanIndexForward=False,  # Sort in descending order
            Limit=1  # Only get the most recent version
        )
        
        if not response['Items']:
            print(f"No prompt found for ref: {ref}")
            return ""

        content = response['Items'][0]['content']

        # If substitutions are provided, apply them to the prompt
        if substitutions:
            try:
                content = content.format(**substitutions)
            except KeyError as e:
                error_msg = f"Missing substitution key in prompt {ref}: {e}"
                log_error(error_msg)
                raise ValueError(error_msg)
            except Exception as e:
                error_msg = f"Error applying substitutions to prompt {ref}: {e}"
                log_error(error_msg)
                raise ValueError(error_msg)

        return content
    except Exception as e:
        if not isinstance(e, ValueError):
            print(f"Error getting prompt {ref} from DynamoDB: {e}")
        raise

@measure_time
def update_prompt(prompt_ref: str, content: Any, update_current: bool = False, specific_version: int = None) -> Union[bool, Tuple[bool, str]]:
    """
    Update an existing prompt or create a new version.
    
    Args:
        prompt_ref: The reference ID of the prompt
        content: The new content (string or object)
        update_current: If True, update the current version instead of creating a new one
        specific_version: If provided and update_current is True, update this specific version
        
    Returns:
        Boolean indicating success or tuple of (success, error_message)
    """
    try:
        if prompt_ref.endswith("_questions"):
            return False, "Cannot update evaluation questions directly"
        # Get the DynamoDB table
        table = get_dynamodb_table('PromptsTable')
        
        # Validate prompt parameters first - this will also check the original is_object flag
        is_valid, error_message, details = validate_prompt_parameters(prompt_ref, content)
        if not is_valid:
            return False, error_message
        
        # Get the is_object flag from validation details
        table = get_dynamodb_table('PromptsTable')
        original_response = table.query(
            KeyConditionExpression=Key('ref').eq(prompt_ref) & Key('version').eq(0)
        )
        
        # Default is_object to False if version 0 doesn't exist
        is_object = False
        if original_response.get('Items'):
            is_object = original_response['Items'][0].get('is_object', False)
        
        # Get the latest version of the prompt or a specific version if requested
        if update_current and specific_version is not None:
            # Get the specific version we want to update
            prompt_response = table.query(
                KeyConditionExpression=Key('ref').eq(prompt_ref) & Key('version').eq(specific_version)
            )
            if not prompt_response.get('Items'):
                return False, f"Prompt {prompt_ref} version {specific_version} not found"
            
            prompt_to_update = prompt_response['Items'][0]
            version_to_update = specific_version
        else:
            # Get the latest version as before
            latest_prompt_response = table.query(
                KeyConditionExpression=Key('ref').eq(prompt_ref),
                ScanIndexForward=False,  # Sort in descending order
                Limit=1
            )
            
            if not latest_prompt_response.get('Items'):
                return False, f"Prompt {prompt_ref} not found"
            
            latest_prompt = latest_prompt_response['Items'][0]
            current_version = int(latest_prompt.get('version', 0))
            prompt_to_update = latest_prompt
            
            # If updating current version, use the same version number
            # Otherwise increment for new version
            version_to_update = current_version if update_current else current_version + 1
        
        # Format the content based on the expected type
        if is_object:
            # Content should be stored as a JSON string if it's an object
            if isinstance(content, (dict, list)):
                content_to_store = json.dumps(content)
            else:
                # Already validated as valid JSON string in validate_prompt_parameters
                content_to_store = content
        else:
            try:
                # Already validated as string in validate_prompt_parameters
                content_to_store = content
                
                # Clean up the content (remove code blocks if present)
                content_to_store = content_to_store.strip()
                if content_to_store.startswith("```") and content_to_store.endswith("```"):
                    # Remove first line of content if it starts with ```
                    content_to_store = content_to_store[content_to_store.find("\n") + 1:]
                    # Remove last line of content if it ends with ```
                    content_to_store = content_to_store[:content_to_store.rfind("\n")]
                    content_to_store = content_to_store.strip()
            except Exception as e:
                error_msg = f"Prompt should be in plaintext. Error cleaning up prompt content: {str(e)}"
                return False, error_msg

        # Generate timestamp
        now = int(time.time() * 1000)  # Milliseconds since epoch
        formatted_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create item to put in DynamoDB
        item = {
            'ref': prompt_ref,
            'version': version_to_update,
            'content': content_to_store,
            'is_object': is_object,
            'updatedAt': formatted_date,
            'timestamp': now
        }
        
        # Copy description from existing version if it exists
        if 'description' in prompt_to_update:
            item['description'] = prompt_to_update['description']
            
        # Update or create the prompt version
        table.put_item(Item=item)
        
        action = "Updated" if update_current else "Created new version of"
        log_debug(f"{action} prompt {prompt_ref} version {version_to_update}")
        
        return True, None
    except Exception as e:
        error_msg = f"Error updating prompt: {str(e)}"
        log_error(error_msg)
        log_error(traceback.format_exc())
        return False, error_msg

def get_prompt_expected_parameters(prompt_ref: str) -> Dict[str, Any]:
    """
    Get information about how a prompt is used in code, including expected parameters.
    
    Args:
        prompt_ref: The prompt reference ID
        
    Returns:
        Dictionary with usage information including:
        - parameters: List of parameter names expected by the function call
        - optional_parameters: List of standard optional parameters used
        - file: The file where the prompt is used
        - line: The line number where the prompt is used
        - function_call: The actual function call text
        - found: Whether the prompt reference was found in the code
    """
    global _prompt_usage_cache
    
    # Check cache first
    if prompt_ref in _prompt_usage_cache:
        return _prompt_usage_cache[prompt_ref]
    
    # Define standard optional parameters - extended list
    common_optional_params = [
                    'stream_key', 
                    'context', 
                    'business_context', 
                    'question', 
                    'function_details'
                ]
    
    # Find usages of the prompt in the code using the validation_utils function
    usage = find_prompt_usage_in_code(prompt_ref)
    print(f"Found usage for prompt {prompt_ref}: {usage}")
    
    # If no usages found, return empty info
    if not usage or usage[0] is None:
        result = {
            'parameters': [],
            'optional_parameters': common_optional_params,
            'file': None,
            'line': None,
            'function_call': None,
            'found': False,
            'is_questions': prompt_ref.endswith('_questions')
        }
        _prompt_usage_cache[prompt_ref] = result
        return result
    
    found_ref, found_params = usage
    
    # Calculate optional parameters (intersection of found params and common optional params)
    optional_params = [p for p in found_params if p in common_optional_params]
    
    # All other parameters are considered required
    required_params = [p for p in found_params if p not in optional_params]
    
    # Create and cache the usage info
    usage_info = {
        'parameters': required_params,
        'optional_parameters': optional_params,
        'file': found_ref,
        'line': None,
        'function_call': None,
        'found': True,
        'is_questions': prompt_ref.endswith('_questions')
    }
    
    _prompt_usage_cache[prompt_ref] = usage_info
    return usage_info

async def get_prompt_expected_parameters_async(prompt_ref: str) -> Dict[str, Any]:
    """
    Async version of get_prompt_expected_parameters
    
    Args:
        prompt_ref: The prompt reference ID
        
    Returns:
        Dictionary with usage information
    """
    global _prompt_usage_cache
    
    # Check cache first
    if prompt_ref in _prompt_usage_cache:
        return _prompt_usage_cache[prompt_ref]
    
    # Define standard optional parameters - extended list
    common_optional_params = [
        'stream_key', 
        'context', 
        'business_context', 
        'question', 
        'function_details'
    ]
    
    # Find usages of the prompt in the code using the async validation_utils function
    from .validation_utils import find_prompt_usage_in_code_async
    usage = await find_prompt_usage_in_code_async(prompt_ref)
    print(f"Found async usage for prompt {prompt_ref}: {usage}")
    
    # If no usages found, return empty info
    if not usage or usage[0] is None:
        result = {
            'parameters': [],
            'optional_parameters': common_optional_params,
            'file': None,
            'line': None,
            'function_call': None,
            'found': False,
            'is_questions': prompt_ref.endswith('_questions')
        }
        _prompt_usage_cache[prompt_ref] = result
        return result
    
    found_ref, found_params = usage
    
    # Calculate optional parameters (intersection of found params and common optional params)
    optional_params = [p for p in found_params if p in common_optional_params]
    
    # All other parameters are considered required
    required_params = [p for p in found_params if p not in optional_params]
    
    # Create and cache the usage info
    usage_info = {
        'parameters': required_params,
        'optional_parameters': optional_params,
        'file': found_ref,
        'line': None,
        'function_call': None,
        'found': True,
        'is_questions': prompt_ref.endswith('_questions')
    }
    
    _prompt_usage_cache[prompt_ref] = usage_info
    return usage_info

@measure_time
def validate_prompt_parameters(prompt_ref, content):
    """
    Validate that a prompt string only uses variables that are passed to it.
    For object prompts, validate against the expected schema.
    
    Args:
        prompt_ref: The prompt reference ID
        content: The prompt content to validate
        
    Returns:
        Tuple of (is_valid, error_message, details)
    """
    try:
        # Get version 0 of the prompt to determine the original is_object flag
        table = get_dynamodb_table('PromptsTable')
        original_response = table.query(
            KeyConditionExpression=Key('ref').eq(prompt_ref) & Key('version').eq(0)
        )
        
        # Default is_object to False if version 0 doesn't exist
        original_is_object = False
        if original_response.get('Items'):
            original_is_object = original_response['Items'][0].get('is_object', False)
            
        # Check content type against original is_object flag
        if original_is_object:
            # If original is object, content should be valid JSON
            if isinstance(content, (dict, list)):
                # Content is already a Python object, which is fine
                pass
            elif isinstance(content, str):
                # Try to parse as JSON to ensure it's valid
                try:
                    json.loads(content)
                except json.JSONDecodeError as e:
                    return False, f"Content should be valid JSON for object-type prompt: {str(e)}", {"validation_error": "invalid_json"}
            else:
                return False, f"Invalid content type for object-type prompt: {type(content)}", {"validation_error": "invalid_type"}
        else:
            # For string prompts, ensure content is a string
            if not isinstance(content, str):
                return False, f"Prompt should be a string", {"validation_error": "not_string"}
            
            # Clean up and validate the content
            content_clean = content.strip()
            if content_clean.startswith("{") and content_clean.endswith("}"):
                try:
                    # Check if it's valid JSON which would be wrong for a string prompt
                    json.loads(content_clean)
                    return False, "Prompt should not be a JSON object, it should be a string. Just output the prompt text.", {"validation_error": "wrong_format"}
                except:
                    # If it's not valid JSON, that's fine, it's just a string with braces
                    pass
            if content_clean.startswith("```"):
                return False, "Prompt should not be a code block, it should be a string without starting with '```'. Just output the prompt text.", {"validation_error": "code_block"}

        # First, check for questions prompt with dictionary wrapping issue
        if prompt_ref.endswith("_questions"):
            # Check if content is a dictionary with nested questions
            if isinstance(content, dict):
                # Check for wrappers like "evaluation_questions"
                wrapped_key = None
                for key in content.keys():
                    if key.endswith('_questions'):
                        wrapped_key = key
                        break
                
                if wrapped_key:
                    error_msg = f"Questions must be a direct array, not wrapped in an object with '{wrapped_key}'"
                    log_error(error_msg)
                    return False, error_msg, {"validation_error": error_msg}
            
            # If it's a string, check if it parses to a dictionary with wrapper
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        wrapped_key = None
                        for key in parsed.keys():
                            if key.endswith('_questions'):
                                wrapped_key = key
                                break
                        
                        if wrapped_key:
                            error_msg = f"Questions must be a direct array, not wrapped in an object with '{wrapped_key}'"
                            log_error(error_msg)
                            return False, error_msg, {"validation_error": error_msg}
                except json.JSONDecodeError:
                    # Not valid JSON, handled in the next sections
                    pass
        
        # Check if this is a JSON object that needs schema validation
        is_object = False
        if prompt_ref.endswith("_questions"):
            # For a string, try to parse as JSON
            if isinstance(content, str):
                try:
                    content_obj = json.loads(content)
                    # We expect a direct array for questions
                    if isinstance(content_obj, list):
                        is_object = True
                        content = content_obj  # Use parsed content
                    elif isinstance(content_obj, dict):
                        # Check if there's a key like "evaluation_questions"
                        for k in content_obj.keys():
                            if k.endswith('_questions'):
                                error_msg = f"Questions must be a direct array, not wrapped in object with key '{k}'"
                                log_error(error_msg)
                                return False, error_msg, {"validation_error": error_msg}
                        # If no specific wrapper key found, still not valid format
                        error_msg = "Questions content must be a direct array, not a dictionary"
                        log_error(error_msg)
                        return False, error_msg, {"validation_error": error_msg}
                except json.JSONDecodeError as e:
                    # Not valid JSON
                    error_msg = f"Invalid JSON format: {str(e)}"
                    log_error(error_msg)
                    return False, error_msg, {"validation_error": error_msg}
            # If already an object type, check direct structure
            elif isinstance(content, list):
                is_object = True
            elif isinstance(content, dict):
                # Again check for wrapper keys
                for k in content.keys():
                    if k.endswith('_questions'):
                        error_msg = f"Questions must be a direct array, not wrapped in object with key '{k}'"
                        log_error(error_msg)
                        return False, error_msg, {"validation_error": error_msg}
                error_msg = "Questions content must be a direct array, not a dictionary"
                log_error(error_msg)
                return False, error_msg, {"validation_error": error_msg}
            
            # Now validate questions with document fields
            if is_object:
                try:
                    # Import here to avoid circular imports
                    from .validation_models import QuestionsArray
                    
                    # Update to use Pydantic v2 parsing
                    questions = QuestionsArray(root=content)
                    
                    # Then check if document fields match output/reference fields
                    doc_valid, doc_error, doc_details = validate_question_objects_with_documents(prompt_ref, content)
                    
                    # If validation against document structure failed, return that error
                    if not doc_valid:
                        return doc_valid, doc_error, doc_details
                    
                    # Otherwise, return success with combined details
                    return True, None, {
                        "object_validated": True,
                        "question_count": len(questions),
                        "type": "questions_array",
                        "document_validation": doc_details
                    }
                except Exception as e:
                    import traceback
                    error_msg = f"Invalid questions format: {str(e)}"
                    error_details = traceback.format_exc()
                    log_error(error_msg)
                    log_debug(f"Validation error details: {error_details}")
                    return False, error_msg, {"validation_error": str(e)}
            else:
                error_msg = "Questions content must be a JSON array"
                log_error(error_msg)
                return False, error_msg, {"validation_error": error_msg}
        
        # For string prompts, validate variables
        if not is_object:
            # Find all format variables in the content using regex
            # This updated regex only matches {var} patterns that aren't part of {{var}} or other structures
            # Matches {variable} but not {{variable}} or more complex structures like {var: value}
            format_vars = set(re.findall(r'(?<!\{)\{([a-zA-Z0-9_]+)\}(?!\})', content))
            
            # if not format_vars:
            #     # If no variables found, the prompt is valid
            #     return True, None, {"used_vars": [], "unused_vars": [], "extra_vars": []}
            
            # Get expected parameters
            prompt_usage = get_prompt_expected_parameters(prompt_ref)
            if not prompt_usage['found']:
                # Add standard optional parameters to default assumptions
                standard_optional_params = [
                    'stream_key', 
                    'context', 
                    'business_context', 
                    'question', 
                    'function_details'
                ]
                
                # If we can't find usage, assume all are valid but treat standard ones as optional
                all_expected_params = format_vars
                optional_params = [v for v in format_vars if v in standard_optional_params]
                
                return False, f"Couldn't find usage in code, all variables assumed valid", {
                    "used_vars": list(format_vars),
                    "unused_vars": [],
                    "extra_vars": [],
                    "note": "Couldn't find usage in code, all variables assumed valid"
                }
            
            # Combine required and optional parameters for validation
            all_expected_params = set(prompt_usage['parameters'] + prompt_usage['optional_parameters'])
            
            # Check for mismatch between format variables and expected parameters
            missing_vars = set(prompt_usage['parameters']) - format_vars  # Required variables expected but not in prompt
            extra_vars = format_vars - all_expected_params    # Variables in prompt but not expected at all
            used_vars = format_vars.intersection(all_expected_params)  # Variables properly used
            
            # check if any variable is used more than once and reject the prompt if so
            for var in used_vars:
                if var != "stream_key":
                    if content.count("{" + var + "}") > 1:
                        error_msg = f"Variable {var} is used more than once in prompt. It should be used at most once."
                        log_error(error_msg)
                        return False, error_msg, {
                            "used_vars": list(used_vars),
                            "unused_vars": list(missing_vars),
                            "extra_vars": list(extra_vars),
                            "validation_error": "duplicate_params"
                        }


            # cannot use vars that are not in all_expected_params
            if extra_vars:
                error_msg = f"Extra parameters in prompt that aren't provided: {', '.join(['{'+v+'}' for v in extra_vars])}. Please make sure to use {{{{var}}}} format for text not meant to be substituted like variables"
                log_error(error_msg)
                return False, error_msg, {
                    "used_vars": list(used_vars),
                    "unused_vars": list(missing_vars),
                    "extra_vars": list(extra_vars),
                    "validation_error": "extra_params"
                }

            # Modified validation logic: fail if any required variables are missing
            if missing_vars:
                error_message = f"Missing required parameters in prompt: {', '.join(['{'+v+'}' for v in missing_vars])}"
                log_error(error_message)
                return False, error_message, {
                    "file": prompt_usage['file'],
                    "line": prompt_usage['line'],
                    "used_vars": list(used_vars),
                    "unused_vars": list(missing_vars),
                    "extra_vars": list(extra_vars),
                    "validation_error": "required_params_missing"
                }
            
            if extra_vars:
                error_messages = []
                details = {
                    "file": prompt_usage['file'],
                    "line": prompt_usage['line'],
                    "used_vars": list(used_vars),
                    "unused_vars": list(missing_vars),
                    "extra_vars": list(extra_vars)
                }
                
                extra_list = ", ".join([f"{{{v}}}" for v in extra_vars])
                error_messages.append(f"Extra parameters in prompt that aren't provided: {extra_list}")
                    
                # Include file location in error message
                file_info = f"Error in {prompt_usage['file']}:{prompt_usage['line']}" if prompt_usage['file'] else ""
                error_message = (file_info + "\n" if file_info else "") + "\n" + "\n".join(error_messages)
                
                return False, error_message, details
            
            return True, None, {
                "file": prompt_usage['file'],
                "line": prompt_usage['line'],
                "used_vars": list(used_vars),
                "unused_vars": list(missing_vars),
                "extra_vars": []
            }
    
    except Exception as e:
        error_msg = f"Unexpected error in prompt validation: {str(e)}"
        log_error(error_msg)
        log_debug(f"Validation error trace: {traceback.format_exc()}")
        return False, error_msg, {}

def get_prompt_content_by_ref_and_version(ref: str, version: int) -> Optional[str]:
    """
    Directly retrieve prompt content from DynamoDB by ref and version.
    
    Args:
        ref: The prompt reference ID
        version: The specific version number to retrieve
        
    Returns:
        The prompt content as a string, or None if not found
    """
    try:
        table = get_dynamodb_table('PromptsTable')
        response = table.get_item(
            Key={
                'ref': ref,
                'version': version
            },
            ProjectionExpression='content'
        )
        
        if 'Item' in response and 'content' in response['Item']:
            return response['Item']['content']
        return None
    except Exception as e:
        log_error(f"Error retrieving prompt content for {ref} version {version}: {str(e)}")
        import traceback
        log_debug(f"Traceback: {traceback.format_exc()}")
        return None

# Add a cache for evaluation data to avoid redundant scans
_evaluations_cache = {}

@measure_time
def get_top_prompt_content(
    prompt_ref: str, 
    max_evaluations: int = 5, 
    eval_type: str = None,
    cached_evaluations: List[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Get prompt content and metadata from top-scoring evaluations for a specific prompt reference.
    
    Args:
        prompt_ref: The prompt reference ID
        max_evaluations: Maximum number of top evaluations to include
        eval_type: Optional evaluation type to filter by
        cached_evaluations: Optional list of pre-fetched evaluations to avoid redundant scans
        
    Returns:
        List of dictionaries with prompt content and metadata from top evaluations
    """
    try:
        log_debug(f"Getting top prompt content for ref: {prompt_ref}")
        
        # Use cached evaluations if provided, check global cache otherwise
        cache_key = f"{eval_type or 'all'}_last_7_days"
        all_evaluations = []
        
        if cached_evaluations is not None:
            all_evaluations = cached_evaluations
            log_debug(f"Using provided cached evaluations ({len(all_evaluations)} records)")
        elif cache_key in _evaluations_cache:
            all_evaluations = _evaluations_cache[cache_key]
            log_debug(f"Using global cache for {cache_key} ({len(all_evaluations)} records)")
        else:
            # Get the DynamoDB table
            table = get_dynamodb_table('EvaluationsTable')
            
            # Calculate timestamp for lookback period (30 days)
            lookback_date = datetime.now() - timedelta(days=7)
            lookback_timestamp = int(lookback_date.timestamp())
            
            if eval_type:
                # Use the GSI to directly query by type and timestamp
                log_debug(f"Querying type-timestamp-index for type={eval_type} since {lookback_timestamp}")
                query_response = table.query(
                    IndexName='type-timestamp-index',
                    KeyConditionExpression=Key('type').eq(eval_type) & 
                                         Key('timestamp').gte(Decimal(str(lookback_timestamp))),
                    ProjectionExpression='streamKey, #ts, #t, successes, attempts, prompts',
                    ExpressionAttributeNames={
                        '#ts': 'timestamp',
                        '#t': 'type'
                    }
                )
                
                all_evaluations = query_response.get('Items', [])
                
                # Handle pagination for the query
                while 'LastEvaluatedKey' in query_response:
                    query_response = table.query(
                        IndexName='type-timestamp-index',
                        KeyConditionExpression=Key('type').eq(eval_type) & 
                                            Key('timestamp').gte(Decimal(str(lookback_timestamp))),
                        ProjectionExpression='streamKey, #ts, #t, successes, attempts, prompts',
                        ExpressionAttributeNames={
                            '#ts': 'timestamp',
                            '#t': 'type'
                        },
                        ExclusiveStartKey=query_response['LastEvaluatedKey']
                    )
                    all_evaluations.extend(query_response.get('Items', []))
            else:
                # Fallback to scan for all types (no GSI)
                log_debug(f"Scanning all evaluations since {lookback_timestamp}")
                scan_response = table.scan(
                    FilterExpression=Attr('timestamp').gte(Decimal(str(lookback_timestamp))) & 
                                   Attr('prompts').exists(),
                    ProjectionExpression='streamKey, #ts, #t, successes, attempts, prompts',
                    ExpressionAttributeNames={
                        '#ts': 'timestamp',
                        '#t': 'type'
                    }
                )
                
                all_evaluations = scan_response.get('Items', [])
                
                # Handle pagination for the scan
                while 'LastEvaluatedKey' in scan_response:
                    scan_response = table.scan(
                        FilterExpression=Attr('timestamp').gte(Decimal(str(lookback_timestamp))) & 
                                       Attr('prompts').exists(),
                        ProjectionExpression='streamKey, #ts, #t, successes, attempts, prompts',
                        ExpressionAttributeNames={
                            '#ts': 'timestamp',
                            '#t': 'type'
                        },
                        ExclusiveStartKey=scan_response['LastEvaluatedKey']
                    )
                    all_evaluations.extend(scan_response.get('Items', []))
            
            # Store in cache for future use
            _evaluations_cache[cache_key] = all_evaluations
            log_debug(f"Added {len(all_evaluations)} evaluations to cache with key {cache_key}")
        
        # Filter for evaluations that used the specific prompt ref
        relevant_evaluations = []
        for eval_item in all_evaluations:
            # Extract prompts used in this evaluation
            prompts = eval_item.get('prompts', [])
            for p in prompts:
                if isinstance(p, dict) and p.get('ref') == prompt_ref:
                    # Found an evaluation that used this prompt
                    # Convert Decimal to Python native types for easier handling
                    successes = int(eval_item.get('successes', 0))
                    attempts = int(eval_item.get('attempts', 0))
                    timestamp = float(eval_item.get('timestamp', 0))
                    
                    # Calculate score based on simplified formula: 
                    # If successful, score is 20, otherwise it's min(10, attempts)
                    if successes > 0:
                        score = 20
                    else:
                        score = min(10, attempts)
                    
                    # Record the prompt version from evaluation
                    prompt_version = p.get('version', 'unknown')
                    
                    evaluation_data = {
                        'streamKey': eval_item.get('streamKey', ''),
                        'timestamp': timestamp,
                        'date': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                        'type': eval_item.get('type', ''),
                        'successes': successes,
                        'attempts': attempts,
                        'score': score,
                        'prompt_version': prompt_version,
                        'prompt_content': None  # Will be populated later from the content table
                    }
                    
                    log_debug(f"Found evaluation for {prompt_ref} - Version: {prompt_version}, "
                              f"Score: {score}, Successes: {successes}, Attempts: {attempts}")
                    
                    relevant_evaluations.append(evaluation_data)
                    break  # Only count each evaluation once
        
        log_debug(f"Found {len(relevant_evaluations)} evaluations using prompt ref: {prompt_ref}")
        
        # Sort evaluations by score (highest first)
        relevant_evaluations.sort(key=lambda x: x['score'], reverse=True)
        
        # Take the top evaluations
        top_evaluations = relevant_evaluations[:max_evaluations]
        
        # Fetch content from PromptsTable for all top evaluations
        for eval_data in top_evaluations:
            version = eval_data.get('prompt_version')
            log_debug(f"Fetching content for {prompt_ref} version {version}")
            
            if isinstance(version, (int, str, Decimal)):
                try:
                    # Convert version to integer if it's a Decimal or string
                    if isinstance(version, Decimal):
                        version_int = int(version)
                    elif isinstance(version, str) and version.isdigit():
                        version_int = int(version)
                    else:
                        version_int = int(float(version))
                    
                    # Always fetch from PromptsTable
                    content = get_prompt_content_by_ref_and_version(prompt_ref, version_int)
                    if content:
                        log_debug(f"Successfully fetched content for {prompt_ref} version {version_int}, "
                                  f"content length: {len(content)}")
                        eval_data['prompt_content'] = content
                    else:
                        log_debug(f"No content found for {prompt_ref} version {version_int}")
                except Exception as e:
                    log_error(f"Error converting version '{version}' to integer or fetching content: {str(e)}")
        
        # Log details of evaluations where content is still missing
        for eval_data in top_evaluations:
            if not eval_data.get('prompt_content'):
                log_debug(f"WARNING: Missing content for {prompt_ref} version {eval_data.get('prompt_version')} "
                          f"from {eval_data.get('date')} (score: {eval_data.get('score')})")
        
        log_debug(f"Selected top {len(top_evaluations)} evaluations for prompt ref: {prompt_ref}")
        
        return top_evaluations
        
    except Exception as e:
        log_error(f"Error getting top prompt content for {prompt_ref}: {str(e)}")
        import traceback
        log_debug(f"Traceback: {traceback.format_exc()}")
        return []
