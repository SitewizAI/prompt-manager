import streamlit as st
# Set page config must be the first streamlit command
st.set_page_config(page_title="Prompt Manager", layout="wide")

import boto3
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests
import os
import json
import tiktoken
from decimal import Decimal
from utils import (
    run_completion_with_fallback, 
    SYSTEM_PROMPT, 
    get_github_files, 
    get_file_contents, 
    get_context, 
    get_most_recent_stream_key, 
    get_all_evaluations, 
    get_stream_evaluations, 
    get_evaluation_metrics, 
    get_recent_evaluations,
    measure_time,
    get_daily_metrics_from_table
)
import time
from functools import wraps
import boto3.dynamodb.conditions as conditions
import pandas as pd

load_dotenv()

# Add debug logging
def log_debug(message: str):
    # print(f"DEBUG: {message}")
    # st.write(f"DEBUG: {message}")
    pass

# Add error logging
def log_error(message: str, error: Exception = None):
    error_msg = f"ERROR: {message}"
    if error:
        error_msg += f" - {str(error)}"
    print(error_msg)
    st.error(error_msg)

# Convert UTC timestamp to local time with AM/PM format
def format_timestamp_local(timestamp_float):
    """
    Convert UTC timestamp to local time with AM/PM format.
    
    Args:
        timestamp_float: Unix timestamp as float or Decimal
        
    Returns:
        Formatted datetime string in local timezone with AM/PM
    """
    if isinstance(timestamp_float, Decimal):
        timestamp_float = float(timestamp_float)
    
    # Convert UTC timestamp to datetime object
    utc_time = datetime.fromtimestamp(timestamp_float, tz=timezone.utc)
    
    # Convert to local time (no timezone specified means local)
    local_time = utc_time.astimezone()
    
    # Format with AM/PM
    return local_time.strftime('%Y-%m-%d %I:%M:%S %p')

# AWS credentials
aws_region = os.getenv('AWS_REGION')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')

log_debug(f"AWS Region: {'Set' if aws_region else 'Not Set'}")
log_debug(f"AWS Access Key: {'Set' if aws_access_key else 'Not Set'}")
log_debug(f"AWS Secret Key: {'Set' if aws_secret_key else 'Not Set'}")

def time_function(func):
    """Decorator that times function execution and shows in the UI"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        st.sidebar.text(f"⏱️ {func.__name__}: {elapsed:.2f}s")
        print(f"⏱️ {func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper

# Function to get boto3 resource with credentials
def get_boto3_resource(service_name='dynamodb'):
    return boto3.resource(
        service_name,
        region_name=aws_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

st.title("Prompt Manager")

# Initialize session state for expanders and code loading
if "expanders_open" not in st.session_state:
    st.session_state.expanders_open = True
if "load_code_files" not in st.session_state:
    st.session_state.load_code_files = False
if "evaluations_expanded" not in st.session_state:
    st.session_state.evaluations_expanded = False
if "prompts" not in st.session_state:
    st.session_state.prompts = []

# Add close/open all button
if st.button("Close All" if st.session_state.expanders_open else "Open All"):
    st.session_state.expanders_open = not st.session_state.expanders_open
    st.rerun()

# Add toggle in sidebar
with st.sidebar:
    st.header("Settings")
    if st.toggle("Load Code Files", value=st.session_state.load_code_files):
        st.session_state.load_code_files = True
    else:
        st.session_state.load_code_files = False
    
    # Add timing metrics section
    st.header("Performance Metrics")
    st.text("Function execution times will appear here")

@time_function
def get_all_prompts() -> List[Dict[str, Any]]:
    """
    Fetch all prompts from DynamoDB PromptsTable, retrieving only the latest version of each prompt reference.
    This significantly improves performance by reducing the amount of data fetched.
    """
    try:
        log_debug("Attempting to get all prompts...")
        dynamodb = get_boto3_resource('dynamodb')
        
        # First, scan to get all unique refs
        table = dynamodb.Table('PromptsTable')
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
            log_debug("Handling pagination in ref scan...")
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
            
            # Use parallel processing for each ref in the batch
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
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []

@time_function
def update_prompt(ref: str, version: str, content: str, is_object: bool = False) -> bool:
    """Create a new version of a prompt in DynamoDB PromptsTable."""
    try:
        # Import the utility function instead of directly updating
        from utils import update_prompt as utils_update_prompt
        
        # Log details about the update attempt
        print(f"Attempting to update prompt {ref} (version: {version}, is_object: {is_object})")
        
        # Parse content if it's marked as an object
        result = utils_update_prompt(ref, content)
            
        if result:
            print(f"Successfully updated prompt {ref} to new version")
        else:
            print(f"Failed to update prompt {ref}")
            
        return result
    except Exception as e:
        log_error(f"Error creating new prompt version: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def count_tokens(text: str) -> int:
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))

def convert_decimal(value):
    """Convert Decimal values to float/int for Streamlit metrics."""
    if isinstance(value, Decimal):
        return float(value)
    return value

@time_function
def display_prompt_versions(prompts: List[Dict[str, Any]]):
    """Display prompts with version history in the Streamlit UI."""
    log_debug(f"Displaying {len(prompts)} prompts")
    
    # Organize prompts by ref
    prompts_by_ref = {}
    for prompt in st.session_state.prompts:
        ref = prompt['ref']
        if ref not in prompts_by_ref:
            prompts_by_ref[ref] = []
        prompts_by_ref[ref].append(prompt)
    
    # Sort versions for each ref
    for ref in prompts_by_ref:
        prompts_by_ref[ref].sort(key=lambda x: int(x.get('version', 0)), reverse=True)
    
    # Display prompts
    for ref, versions in prompts_by_ref.items():
        if versions:  # Only show if there are versions
            with st.expander(f"Prompt: {ref}", expanded=st.session_state.expanders_open):
                # Initialize session state for this prompt's historical versions if not exists
                if f"load_history_{ref}" not in st.session_state:
                    st.session_state[f"load_history_{ref}"] = False
                    
                # Add button to load all historical versions
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("Load Previous Versions", key=f"btn_history_{ref}"):
                        st.session_state[f"load_history_{ref}"] = True
                        with st.spinner(f"Loading all versions for {ref}..."):
                            # Load all versions of this prompt
                            all_versions = get_all_prompt_versions(ref)
                            # Update the versions list but keep the first (latest) version from current prompt
                            if all_versions:
                                # Keep the latest version we already have, add older versions from query
                                old_versions = [v for v in all_versions if int(v.get('version', 0)) < int(versions[0].get('version', 0))]
                                if old_versions:
                                    # Replace the versions list with all versions
                                    versions = [versions[0]] + old_versions
                                    # Update the session state prompts for this ref
                                    for idx, p in enumerate(st.session_state.prompts):
                                        if p['ref'] == ref and p['version'] == versions[0]['version']:
                                            # Replace this prompt entry with the full version list
                                            st.session_state.prompts[idx:idx+1] = versions
                                            break
                        st.rerun()
                
                # Display the versions we have
                displayed_versions = versions if st.session_state[f"load_history_{ref}"] else [versions[0]]
                tabs = st.tabs([f"Version {v.get('version', 'N/A')}" for v in displayed_versions])
                
                for tab, version in zip(tabs, displayed_versions):
                    with tab:
                        # Show content
                        content = version.get('content', '')
                        is_object = version.get('is_object', False)
                        
                        if is_object:
                            try:
                                # For object content, display as JSON editor
                                if isinstance(content, str):
                                    try:
                                        content_obj = json.loads(content)
                                        # Format JSON for display
                                        formatted_content = json.dumps(content_obj, indent=2)
                                    except json.JSONDecodeError:
                                        st.error(f"Stored content marked as JSON object but cannot be parsed")
                                        formatted_content = content
                                else:
                                    content_obj = content
                                    formatted_content = json.dumps(content_obj, indent=2)
                                
                                new_content = st.text_area(
                                    "JSON Content (editable)",
                                    formatted_content,
                                    height=400,
                                    key=f"json_content_{ref}_{version.get('version', 'N/A')}"
                                )
                                
                                # Check for changes in the JSON content
                                if new_content != formatted_content:
                                    if st.button("Create New Version", key=f"update_json_{ref}_{version.get('version', 'N/A')}"):
                                        # Always pass is_object=True for JSON content
                                        if update_prompt(ref, version.get('version'), new_content, is_object=True):
                                            st.success("New prompt version created successfully!")
                                            # Clear the session state prompts to force a refresh
                                            st.session_state.prompts = []
                                            st.rerun()
                                        else:
                                            st.error("Failed to create new prompt version")
                            except Exception as e:
                                st.error(f"Error handling JSON content: {str(e)}")
                                st.text_area("Raw Content", content, height=200, disabled=True)
                        else:
                            # For string content, use regular text area
                            new_content = st.text_area(
                                "Content",
                                content,
                                height=200,
                                key=f"content_{ref}_{version.get('version', 'N/A')}"
                            )
                            
                            # Check for changes
                            if new_content != content:
                                if st.button("Create New Version", key=f"update_{ref}_{version.get('version', 'N/A')}"):
                                    # Pass is_object=False for string content
                                    if update_prompt(ref, version.get('version'), new_content, is_object=False):
                                        st.success("New prompt version created successfully!")
                                        # Clear the session state prompts to force a refresh
                                        st.session_state.prompts = []
                                        st.rerun()
                                    else:
                                        st.error("Failed to create new prompt version")
                        
                        # Display additional metadata
                        st.text(f"Last Updated: {version.get('updatedAt', 'N/A')}")
                        st.text(f"Content Type: {'JSON Object' if is_object else 'String'}")
                        if version.get('description'):
                            st.text(f"Description: {version['description']}")
                
                # If we've loaded history, show a button to collapse it again
                if st.session_state[f"load_history_{ref}"]:
                    if st.button("Hide Previous Versions", key=f"btn_hide_history_{ref}"):
                        st.session_state[f"load_history_{ref}"] = False
                        st.rerun()

@time_function
def get_all_prompt_versions(ref: str) -> List[Dict[str, Any]]:
    """
    Fetch all versions of a specific prompt reference.
    Returns them sorted with most recent first.
    """
    try:
        log_debug(f"Fetching all versions for prompt ref: {ref}")
        dynamodb = get_boto3_resource('dynamodb')
        table = dynamodb.Table('PromptsTable')
        
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
        
        # Sort by version (descending)
        versions.sort(key=lambda x: int(x.get('version', 0)), reverse=True)
        return versions
        
    except Exception as e:
        log_error(f"Error getting all versions for prompt {ref}", e)
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []

# Add evaluation type selection above the header
st.header("Evaluation Type")
evaluation_types = [ "okr", "insights","suggestion", "design", "code"]  # Updated to match the schema
selected_eval_type = st.radio(
    "Select evaluation type",
    options=evaluation_types,
    horizontal=True
)
log_debug(f"Selected evaluation type: {selected_eval_type}")

# Add configuration parameters and fetch metrics immediately after type selection
days_to_show = st.sidebar.slider("Days of metrics to show", min_value=1, max_value=90, value=30)
evals_to_show = st.sidebar.slider("Number of recent evaluations to show", min_value=1, max_value=20, value=10)

# Show metrics visualization right after type selection
st.header("Evaluation Metrics")
with st.spinner("Loading metrics..."):
    # Use the direct DateEvaluationsTable query function instead of the legacy function
    start_time = time.time()
    metrics = get_daily_metrics_from_table(eval_type=selected_eval_type, days=days_to_show)
    st.sidebar.text(f"⏱️ Metrics load time: {time.time() - start_time:.2f}s")
    log_debug(f"Loaded metrics: {json.dumps(metrics, default=str)}")

if metrics['total_metrics']:
    log_debug("Rendering metrics visualizations")
    # Display total metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_metrics = metrics['total_metrics']
    total_evals = total_metrics['total_evaluations']
    success_rate = total_metrics.get('success_rate', 0.0)
    total_successes = total_metrics['total_successes']
    avg_turns = total_metrics['total_turns'] / total_evals if total_evals > 0 else 0
    
    col1.metric("Total Evaluations", f"{total_evals:.0f}")
    col2.metric("Success Rate", f"{success_rate:.1f}%")
    col3.metric("Total Successes", f"{total_successes:.0f}")
    col4.metric("Average Turns", f"{avg_turns:.1f}")
    
    # Create DataFrame for visualization
    daily_data = metrics['daily_metrics']
    if daily_data:
        df = pd.DataFrame.from_dict(daily_data, orient='index')
        
        # Sort by date (index)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Display multiple charts in columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Success Rate Chart
            st.subheader("Success Rate")
            st.line_chart(df['success_rate'], height=200)
            
            # Average Turns
            st.subheader("Average Turns")
            df['avg_turns'] = df['turns'] / df['evaluations'].where(df['evaluations'] > 0)
            st.line_chart(df['avg_turns'], height=200)
            
        with col2:
            # Evaluations and Successes
            st.subheader("Daily Activity")
            eval_success_df = df[['evaluations', 'successes']]
            st.line_chart(eval_success_df, height=200)
            
            # Quality Metric
            st.subheader("Quality Metric")
            st.line_chart(df['quality_metric'], height=200)
            
            # Show raw data in expandable section
            with st.expander("Show Raw Data"):
                st.dataframe(df.style.format("{:.2f}"))
                
                # Allow downloading the data
                csv = df.to_csv()
                st.download_button(
                    label="Download metrics as CSV",
                    data=csv,
                    file_name=f'metrics_{selected_eval_type}_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv',
                )
else:
    st.warning(f"No metrics data available for {selected_eval_type}")
    log_debug(f"No metrics found for {selected_eval_type}")

# Load remaining data
with st.spinner("Loading data..."):
    log_debug("Starting data load...")
    start_time = time.time()
    if not st.session_state.prompts:  # Only load if not in session state
        log_debug("Loading prompts into session state...")
        st.session_state.prompts = get_all_prompts()
    prompts = st.session_state.prompts
    log_debug(f"Loaded {len(prompts)} prompts")
    st.sidebar.text(f"⏱️ Prompts load: {time.time() - start_time:.2f}s")
    
    start_time = time.time()
    log_debug("Loading evaluations...")
    recent_evals = get_recent_evaluations(eval_type=selected_eval_type, limit=evals_to_show)
    log_debug(f"Loaded {len(recent_evals)} evaluations")
    st.sidebar.text(f"⏱️ Evals load: {time.time() - start_time:.2f}s")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Prompts", "Recent Evaluations", "Chat Assistant"])

with tab1:
    log_debug("Rendering Prompts tab...")
    start_time = time.time()
    # Sidebar filters
    st.sidebar.header("Filters")

    # Get unique refs for filtering
    all_refs = list(set([p["ref"] for p in prompts]))
    selected_refs = st.sidebar.multiselect(
        "Filter by refs",
        options=all_refs,
    )

    # Search box
    search_term = st.sidebar.text_input("Search content").lower()

    # Filter data based on selections
    filtered_prompts = prompts
    if selected_refs:
        filtered_prompts = [p for p in filtered_prompts if p["ref"] in selected_refs]

    if search_term:
        filtered_prompts = [p for p in filtered_prompts if (
            search_term in p["content"].lower()
        )]

    # Display prompts
    st.header("Prompts")
    display_prompt_versions(filtered_prompts)
    st.sidebar.text(f"⏱️ Render prompts tab: {time.time() - start_time:.2f}s")

# Update recent evaluations display
with tab2:
    log_debug("Rendering Evaluations tab...")
    start_time = time.time()
    st.header("Recent Evaluations")
    
    if not recent_evals:
        st.warning(f"No recent evaluations found for {selected_eval_type}")
    
    for eval in recent_evals:
        # Use our new timestamp formatter function to show local time with AM/PM
        timestamp = format_timestamp_local(float(eval['timestamp']))
        with st.expander(f"Evaluation - {eval.get('type', 'N/A')} - {timestamp}", 
                       expanded=st.session_state.evaluations_expanded):
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Successes", convert_decimal(eval.get('successes', 0)))
            col2.metric("Attempts", convert_decimal(eval.get('attempts', 0)))
            col3.metric("Number of Turns", convert_decimal(eval.get('num_turns', 0)))
            
            # Display question and conversation in separate tabs
            tab_question, tab_convo, tab_failures, tab_prompts = st.tabs(["Question", "Conversation", "Failure Reasons", "Prompts Used"])
            
            with tab_question:
                st.write("### Question")
                st.write(eval.get('question', 'N/A'))
            
            with tab_convo:
                st.write("### Conversation History")
                conversation = eval.get('conversation', '')
                if conversation:
                    st.text_area("Full Conversation", conversation, height=300)
                else:
                    st.info("No conversation history available")
            
            with tab_failures:
                st.write("### Failure Reasons")
                failure_reasons = eval.get('failure_reasons', [])
                if failure_reasons:
                    for reason in failure_reasons:
                        st.error(reason)
                else:
                    st.success("No failures recorded")
            
            # Display prompts used in a dedicated tab
            with tab_prompts:
                st.write("### Prompts Used")
                if eval.get('prompts'):
                    for prompt in eval.get('prompts', []):
                        if isinstance(prompt, dict):
                            st.markdown(f"#### Prompt: `{prompt.get('ref', 'N/A')}` (Version {prompt.get('version', 'N/A')})")
                            try:
                                content = json.loads(prompt.get('content', '{}')) if prompt.get('is_object') else prompt.get('content', '')
                                if isinstance(content, dict):
                                    st.json(content)
                                else:
                                    st.text(content)
                            except:
                                st.text(prompt.get('content', 'Error loading content'))
                            st.divider()
                else:
                    st.info("No prompts used in this evaluation")

            # Display summary if available
            if eval.get('summary'):
                st.write("### Summary")
                st.info(eval['summary'])
    
    st.sidebar.text(f"⏱️ Render evals tab: {time.time() - start_time:.2f}s")

with tab3:
    log_debug("Rendering Chat Assistant tab...")
    start_time = time.time()
    st.header("Chat Assistant")

    # Initialize chat messages state if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Get most recent stream key and all available stream keys
    recent_evals = get_all_evaluations(eval_type=selected_eval_type)
    stream_keys = list(set([eval['streamKey'] for eval in recent_evals]))
    default_stream_key, _ = get_most_recent_stream_key()
    
    # Select stream key with most recent as default
    stream_key = st.selectbox(
        "Select Stream Key",
        options=stream_keys,
        index=stream_keys.index(default_stream_key) if default_stream_key in stream_keys else 0
    ) if stream_keys else st.text_input("Enter Stream Key")

    # Number of past evaluations to include in context
    past_eval_count = st.sidebar.slider("Past evaluations to include", min_value=1, max_value=10, value=3)

    if stream_key:
        # Get evaluations for timestamp selection
        evaluations = get_stream_evaluations(stream_key, eval_type=selected_eval_type)
        if evaluations:
            # Update to use local time with AM/PM format
            eval_options = {
                format_timestamp_local(float(eval['timestamp'])): eval 
                for eval in evaluations
            }
            selected_timestamp = st.selectbox(
                "Select Evaluation Timestamp",
                options=list(eval_options.keys()),
                format_func=lambda x: f"Evaluation from {x}"
            )
            current_eval = eval_options[selected_timestamp]
            current_eval_timestamp = float(current_eval['timestamp'])

            # Display chat messages from session state
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask a question about the data..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get context with code files only if enabled
                context_start = time.time()
                llm_context = get_context(
                    stream_key=stream_key, 
                    current_eval_timestamp=current_eval_timestamp, 
                    return_type="string",
                    include_code_files=st.session_state.load_code_files,
                    past_eval_count=past_eval_count  # Pass the selected number of past evaluations
                )
                st.sidebar.text(f"⏱️ Get context: {time.time() - context_start:.2f}s")
                
                display_context = get_context(
                    stream_key=stream_key, 
                    current_eval_timestamp=current_eval_timestamp, 
                    return_type="dict",
                    include_code_files=st.session_state.load_code_files,
                    past_eval_count=past_eval_count  # Pass the selected number of past evaluations
                )

                # Display data statistics
                if display_context.get('data_stats'):
                    stats = display_context['data_stats']
                    st.subheader("Data Statistics")
                    cols = st.columns(3)
                    
                    # First column - Evaluations & Metrics
                    with cols[0]:
                        st.metric("Evaluations (Current)", stats['evaluations']['current'])
                        st.metric("Evaluations (Previous)", stats['evaluations']['previous'])
                        st.metric("Daily Metrics", stats['daily_metrics'])
                        st.metric("Historical Prompts", stats['historical_prompts'])
                    
                    # Second column - Data
                    with cols[1]:
                        st.metric("OKRs", stats['okrs'])
                        st.metric("Insights", stats['insights'])
                        st.metric("Suggestions", stats['suggestions'])
                        st.metric("Code Samples", stats['code'])
                    
                    # Third column - Code Files & Issues
                    with cols[2]:
                        st.metric("Code Files", stats['code_files'])
                        st.metric("GitHub Issues", stats['github_issues'])

                # Count tokens
                token_count = count_tokens(llm_context)
                st.write(f"Token count: {token_count}")

                try:
                    completion_start = time.time()
                    ai_response = run_completion_with_fallback(
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            *st.session_state.messages[:-1],
                            {"role": "user", "content": f"Context:\n{llm_context}\n\nQuestion: {prompt}"}
                        ]
                    )
                    st.sidebar.text(f"⏱️ AI completion: {time.time() - completion_start:.2f}s")

                    if ai_response:
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        with st.chat_message("assistant"):
                            st.markdown(ai_response)
                    else:
                        st.error("Failed to get AI response")
                except Exception as e:
                    st.error(f"Error getting AI response: {str(e)}")

                # Display current evaluation prompt content if available
                if display_context['current_eval'].get('prompt_contents'):
                    st.subheader("Current Evaluation Prompt Contents")
                    with st.expander("Prompt Contents", expanded=False):
                        for p_content in display_context['current_eval']['prompt_contents']:
                            st.markdown(f"**Prompt**: {p_content.get('ref')} (Version: {p_content.get('version')})")
                            
                            # Handle different content formats
                            try:
                                if p_content.get('is_object'):
                                    if isinstance(p_content['content'], str):
                                        content_obj = json.loads(p_content['content'])
                                        st.json(content_obj)
                                    else:
                                        st.json(p_content['content'])
                                else:
                                    st.code(p_content['content'])
                            except Exception as e:
                                st.text(p_content.get('content', f"Error displaying content: {str(e)}"))
                            st.divider()

                # Display prompt contents from past evaluations
                if display_context['prev_evals']:
                    st.subheader("Past Evaluation Prompt Contents")
                    for idx, eval_data in enumerate(display_context['prev_evals']):
                        if eval_data.get('prompt_contents'):
                            with st.expander(f"Prompts from {eval_data['timestamp']}", expanded=False):
                                for p_content in eval_data['prompt_contents']:
                                    st.markdown(f"**Prompt**: {p_content.get('ref')} (Version: {p_content.get('version')})")
                                    
                                    # Handle different content formats
                                    try:
                                        if p_content.get('is_object'):
                                            if isinstance(p_content['content'], str):
                                                content_obj = json.loads(p_content['content'])
                                                st.json(content_obj)
                                            else:
                                                st.json(p_content['content'])
                                        else:
                                            st.code(p_content['content'])
                                    except Exception as e:
                                        st.text(p_content.get('content', f"Error displaying content: {str(e)}"))
                                    st.divider()

                # Use structured context for display
                current_eval_data = display_context['current_eval']['raw']
                prev_evals = display_context['prev_evals']

                # Display sections using structured data
                st.subheader("Current Evaluation Details")
                with st.expander("Details", expanded=st.session_state.expanders_open):
                    st.write(f"Type: {display_context['current_eval']['type']}")
                    st.write(f"Successes: {display_context['current_eval']['successes']}")
                    st.write(f"Attempts: {display_context['current_eval']['attempts']}")
                    if display_context['current_eval']['failure_reasons']:
                        st.write("Failure Reasons:")
                        for reason in display_context['current_eval']['failure_reasons']:
                            st.error(reason)

                # Display files
                st.subheader("Python Files")
                for file in display_context['files']:
                    with st.expander(f"File: {file['file']['path']}", expanded=st.session_state.expanders_open):
                        st.code(file['content'], language='python')

            # Rest of display sections using same context data
            current_eval_data = current_eval
            prev_evals = [e for e in evaluations if float(e['timestamp']) < current_eval_timestamp][:5]

            # Display previous evaluations
            st.subheader("Previous Evaluations")
            prev_evals = [e for e in evaluations if float(e['timestamp']) < float(current_eval['timestamp'])][:5]
            for eval in prev_evals:
                # Update to use local time with AM/PM format
                timestamp = format_timestamp_local(float(eval['timestamp']))
                with st.expander(f"Evaluation from {timestamp}", expanded=st.session_state.expanders_open):
                    # First show prompts used (add this section)
                    if eval.get('prompts'):
                        st.write("**Prompts Used:**")
                        for prompt in eval.get('prompts', []):
                            if isinstance(prompt, dict):
                                # Display with nice formatting
                                st.markdown(f"- `{prompt.get('ref', 'N/A')}` (Version {prompt.get('version', 'N/A')})")
                                if prompt.get('content'):
                                    with st.expander("Show Prompt Content"):
                                        try:
                                            content = json.loads(prompt.get('content', '{}')) if prompt.get('is_object') else prompt.get('content', '')
                                            if isinstance(content, dict):
                                                st.json(content)
                                            else:
                                                st.text(content)
                                        except:
                                            st.text(prompt.get('content', 'Error loading content'))
                        st.divider()  # Add visual separator
                            
                    # Then show the rest of the evaluation details
                    st.write(f"Type: {eval.get('type', 'N/A')}")
                    st.write(f"Successes: {convert_decimal(eval.get('successes', 0))}")
                    st.write(f"Attempts: {convert_decimal(eval.get('attempts', 0))}")
                    if eval.get('failure_reasons'):
                        st.write("Failure Reasons:")
                        for reason in eval['failure_reasons']:
                            st.error(reason)
                    if eval.get('summary'):
                        st.write("Summary:")
                        st.write(eval['summary'])

        else:
            st.error("No evaluations found for the selected stream key")
    
    st.sidebar.text(f"⏱️ Render chat tab: {time.time() - start_time:.2f}s")
