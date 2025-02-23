import streamlit as st
import boto3
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests
import os
import json
import tiktoken
from decimal import Decimal
from utils import run_completion_with_fallback, SYSTEM_PROMPT, get_github_files, get_file_contents, get_context, get_most_recent_stream_key
import time
from functools import wraps
import boto3.dynamodb.conditions as conditions

load_dotenv()

# Add debug logging
def log_debug(message: str):
    print(f"DEBUG: {message}")
    st.write(f"DEBUG: {message}")

# Add error logging
def log_error(message: str, error: Exception = None):
    error_msg = f"ERROR: {message}"
    if error:
        error_msg += f" - {str(error)}"
    print(error_msg)
    st.error(error_msg)

# AWS credentials
aws_region = os.getenv('AWS_REGION')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')

log_debug(f"AWS Region: {'Set' if aws_region else 'Not Set'}")
log_debug(f"AWS Access Key: {'Set' if aws_access_key else 'Not Set'}")
log_debug(f"AWS Secret Key: {'Set' if aws_secret_key else 'Not Set'}")

# Function to get boto3 resource with credentials
def get_boto3_resource(service_name='dynamodb'):
    return boto3.resource(
        service_name,
        region_name=aws_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

st.set_page_config(page_title="Prompt Manager", layout="wide")
st.title("Prompt Manager")

# Initialize session state for expanders and code loading
if "expanders_open" not in st.session_state:
    st.session_state.expanders_open = True
if "load_code_files" not in st.session_state:
    st.session_state.load_code_files = False
if "evaluations_expanded" not in st.session_state:
    st.session_state.evaluations_expanded = False  # Add this line
if "prompts" not in st.session_state:
    st.session_state.prompts = []

# Add close/open all button
if st.button("Close All" if st.session_state.expanders_open else "Open All"):
    st.session_state.expanders_open = not st.session_state.expanders_open
    st.rerun()

# Add toggle in sidebar
with st.sidebar:
    if st.toggle("Load Code Files", value=st.session_state.load_code_files):
        st.session_state.load_code_files = True
    else:
        st.session_state.load_code_files = False

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
def get_all_prompts() -> List[Dict[str, Any]]:
    """Fetch all prompts from DynamoDB PromptsTable."""
    try:
        log_debug("Attempting to get all prompts...")
        dynamodb = get_boto3_resource('dynamodb')
        table = dynamodb.Table('PromptsTable')

        prompts = []
        response = table.scan()
        log_debug(f"Initial scan response: {json.dumps(response.get('Items', []), default=str)[:200]}...")
        prompts.extend(response.get('Items', []))

        # Handle pagination if there are more items
        while 'LastEvaluatedKey' in response:
            log_debug("Handling pagination...")
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            prompts.extend(response.get('Items', []))

        log_debug(f"Retrieved {len(prompts)} total prompts")
        return prompts
    except Exception as e:
        log_error("Error getting prompts", e)
        return []

def update_prompt(ref: str, version: str, content: str) -> bool:
    """Update a prompt in DynamoDB PromptsTable."""
    try:
        dynamodb = get_boto3_resource('dynamodb')
        table = dynamodb.Table('PromptsTable')

        response = table.update_item(
            Key={
                'ref': ref,
                'version': version
            },
            UpdateExpression='SET content = :content, updatedAt = :timestamp',
            ExpressionAttributeValues={
                ':content': content,
                ':timestamp': datetime.now().isoformat()
            },
            ReturnValues='UPDATED_NEW'
        )
        return True
    except Exception as e:
        print(f"Error updating prompt: {e}")
        return False

@measure_time
def get_all_evaluations(limit_per_stream: int = 10, eval_type: str = None) -> List[Dict[str, Any]]:
    """
    Fetch recent evaluations for all stream keys from DynamoDB EvaluationsTable.

    Args:
        limit_per_stream: Maximum number of items to return per stream key
        eval_type: Optional filter for evaluation type
    """
    try:
        dynamodb = get_boto3_resource('dynamodb')
        table = dynamodb.Table('EvaluationsTable')

        # First get all unique stream keys
        response = table.scan(
            ProjectionExpression='streamKey',
        )
        stream_keys = {item['streamKey'] for item in response.get('Items', [])}
        
        # Handle pagination for stream keys
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                ProjectionExpression='streamKey',
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            stream_keys.update({item['streamKey'] for item in response.get('Items', [])})

        # For each stream key, get recent evaluations
        all_evaluations = []
        for stream_key in stream_keys:
            response = table.query(
                KeyConditionExpression='streamKey = :sk',
                ExpressionAttributeValues={
                    ':sk': stream_key
                },
                ScanIndexForward=False,
                Limit=limit_per_stream
            )
            evaluations = response.get('Items', [])
            
            # Filter by type if specified
            if eval_type:
                evaluations = [e for e in evaluations if e.get('type') == eval_type]
                
            all_evaluations.extend(evaluations)

        return all_evaluations
    except Exception as e:
        print(f"Error getting evaluations: {e}")
        return []

@measure_time
def get_stream_evaluations(stream_key: str, limit: int = 6, eval_type: str = None) -> List[Dict[str, Any]]:
    """
    Fetch recent evaluations for a specific stream key from DynamoDB EvaluationsTable.
    Returns the most recent evaluation and 5 evaluations before it.

    Args:
        stream_key: The stream key to fetch evaluations for
        limit: Maximum number of evaluations to return (default 6 to get current + 5 previous)
        eval_type: Optional filter for evaluation type
    """
    try:
        dynamodb = get_boto3_resource('dynamodb')
        table = dynamodb.Table('EvaluationsTable')

        response = table.query(
            KeyConditionExpression='streamKey = :sk',
            ExpressionAttributeValues={
                ':sk': stream_key
            },
            ScanIndexForward=False,  # Sort in descending order (most recent first)
            Limit=limit
        )

        evaluations = response.get('Items', [])
        
        # Filter by type if specified
        if eval_type:
            evaluations = [e for e in evaluations if e.get('type') == eval_type]
            
        evaluations.sort(key=lambda x: float(x.get('timestamp', 0)), reverse=True)
        return evaluations
    except Exception as e:
        print(f"Error getting evaluations for stream key {stream_key}: {e}")
        return []

def count_tokens(text: str) -> int:
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))

def convert_decimal(value):
    """Convert Decimal values to float/int for Streamlit metrics."""
    if isinstance(value, Decimal):
        return float(value)
    return value

def delete_prompt_version(ref: str, version: str) -> bool:
    """Delete a specific version of a prompt from DynamoDB PromptsTable."""
    try:
        dynamodb = get_boto3_resource('dynamodb')
        table = dynamodb.Table('PromptsTable')
        
        # Delete the specific version
        table.delete_item(
            Key={
                'ref': ref,
                'version': version
            }
        )
        
        # Update session state by removing the deleted version
        if "prompts" in st.session_state:
            st.session_state.prompts = [
                p for p in st.session_state.prompts 
                if not (p['ref'] == ref and p['version'] == version)
            ]
        
        return True
    except Exception as e:
        print(f"Error deleting prompt {ref} version {version}: {e}")
        return False

def display_prompt_versions(prompts: List[Dict[str, Any]]):
    """Display prompts with version history in the Streamlit UI."""
    log_debug(f"Displaying {len(prompts)} prompts")
    # Organize prompts by ref
    prompts_by_ref = {}
    for prompt in st.session_state.prompts:  # Use session state prompts
        ref = prompt['ref']
        if ref not in prompts_by_ref:
            prompts_by_ref[ref] = []
        prompts_by_ref[ref].append(prompt)
    
    # Sort versions for each ref
    for ref in prompts_by_ref:
        prompts_by_ref[ref].sort(key=lambda x: int(x.get('version', 0)), reverse=True)
    
    deleted_items = []  # Track items to delete
    
    # Display prompts
    for ref, versions in prompts_by_ref.items():
        with st.expander(f"Prompt: {ref}", expanded=st.session_state.expanders_open):
            if versions:  # Only show if there are versions
                tabs = st.tabs([f"Version {v.get('version', 'N/A')}" for v in versions])
                for tab, version in zip(tabs, versions):
                    with tab:
                        col1, col2 = st.columns([6, 1])
                        with col2:
                            # Delete button with no rerun
                            button_key = f"delete_{ref}_{version.get('version')}"
                            if button_key not in st.session_state:
                                st.session_state[button_key] = False
                                
                            if st.button("🗑️", key=button_key, type="secondary"):
                                if delete_prompt_version(ref, version.get('version')):
                                    deleted_items.append((ref, version.get('version')))
                                    st.session_state[button_key] = True
                                    
                            # Skip displaying this version if it was deleted
                            if st.session_state[button_key]:
                                continue
                        
                        with col1:
                            # Rest of the existing tab content code
                            content = version.get('content', '')
                            if version.get('is_object', False):
                                try:
                                    content = json.loads(content)
                                    st.json(content)
                                except:
                                    st.text_area("Content", content, height=200, disabled=True)
                            else:
                                new_content = st.text_area(
                                    "Content",
                                    content,
                                    height=200,
                                    key=f"content_{ref}_{version.get('version', 'N/A')}"
                                )
                                if new_content != content:
                                    if st.button("Update", key=f"update_{ref}_{version.get('version', 'N/A')}"):
                                        if update_prompt(ref, new_content):
                                            st.success("Prompt updated successfully!")
                                            st.rerun()
                                        else:
                                            st.error("Failed to update prompt")
                        
                        st.text(f"Last Updated: {version.get('updatedAt', 'N/A')}")
                        if version.get('description'):
                            st.text(f"Description: {version['description']}")

    # Clean up session state after deletions
    if deleted_items:
        st.session_state.prompts = [
            p for p in st.session_state.prompts 
            if not any((p['ref'] == ref and p['version'] == ver) for ref, ver in deleted_items)
        ]

# Add evaluation type selection above the header
st.header("Evaluation Type")
evaluation_types = ["suggestions", "okr", "insights", "code"]
selected_eval_type = st.radio(
    "Select evaluation type",
    options=evaluation_types,
    horizontal=True
)

# Load data
with st.spinner("Loading data..."):
    log_debug("Starting data load...")
    start_time = time.time()
    if not st.session_state.prompts:  # Only load if not in session state
        log_debug("Loading prompts into session state...")
        st.session_state.prompts = get_all_prompts()
    prompts = st.session_state.prompts
    log_debug(f"Loaded {len(prompts)} prompts")
    print(f"⏱️ Loading prompts took {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    log_debug("Loading evaluations...")
    recent_evals = get_all_evaluations(eval_type=selected_eval_type)
    log_debug(f"Loaded {len(recent_evals)} evaluations")
    print(f"⏱️ Loading evaluations took {time.time() - start_time:.2f} seconds")

# Add evaluation score metrics at the top
st.header("Evaluation Scores Overview")
if recent_evals:
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate aggregate metrics
    total_evals = len(recent_evals)
    total_successes = sum(convert_decimal(eval.get('successes', 0)) for eval in recent_evals)
    total_attempts = sum(convert_decimal(eval.get('attempts', 0)) for eval in recent_evals)
    avg_turns = sum(convert_decimal(eval.get('num_turns', 0)) for eval in recent_evals) / total_evals if total_evals > 0 else 0
    success_rate = (total_successes / total_attempts * 100) if total_attempts > 0 else 0
    
    col1.metric("Total Evaluations", total_evals)
    col2.metric("Success Rate", f"{success_rate:.1f}%")
    col3.metric("Total Successes", total_successes)
    col4.metric("Average Turns", f"{avg_turns:.1f}")

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
    print(f"⏱️ Rendering prompts tab took {time.time() - start_time:.2f} seconds")

with tab2:
    log_debug("Rendering Evaluations tab...")
    start_time = time.time()
    st.header("Recent Evaluations")
    
    # Group evaluations by stream key for better organization
    evals_by_stream = {}
    for eval in recent_evals:
        stream_key = eval['streamKey']
        if stream_key not in evals_by_stream:
            evals_by_stream[stream_key] = []
        evals_by_stream[stream_key].append(eval)
    
    for stream_key, stream_evals in evals_by_stream.items():
        st.subheader(f"Stream: {stream_key}")
        
        # Sort evaluations by timestamp
        stream_evals.sort(key=lambda x: float(x['timestamp']), reverse=True)
        
        for eval in stream_evals:
            timestamp = datetime.fromtimestamp(float(eval['timestamp'])).strftime('%Y-%m-%d %H:%M:%S')
            with st.expander(f"Evaluation - {eval.get('type', 'N/A')} - {timestamp}", 
                           expanded=st.session_state.evaluations_expanded):  # Use evaluations_expanded here
                # Create columns for metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Successes", convert_decimal(eval.get('successes', 0)))
                col2.metric("Attempts", convert_decimal(eval.get('attempts', 0)))
                col3.metric("Number of Turns", convert_decimal(eval.get('num_turns', 0)))
                
                st.write(f"Question: {eval.get('question', 'N/A')}")
                
                # Display prompts used with versions
                if eval.get('prompts'):
                    st.subheader("Prompts Used")
                    for prompt in eval.get('prompts', []):
                        if isinstance(prompt, dict):
                            st.write(f"- {prompt.get('ref', 'N/A')} (Version {prompt.get('version', 'N/A')})")
                            try:
                                content = json.loads(prompt.get('content', '{}')) if prompt.get('is_object') else prompt.get('content', '')
                                if isinstance(content, dict):
                                    st.json(content)
                                else:
                                    st.text(content)
                            except:
                                st.text(prompt.get('content', 'Error loading content'))

                if eval.get('failure_reasons'):
                    st.subheader("Failure Reasons")
                    for reason in eval['failure_reasons']:
                        st.error(reason)

                if eval.get('summary'):
                    st.subheader("Summary")
                    st.write(eval['summary'])
                
    print(f"⏱️ Rendering evaluations tab took {time.time() - start_time:.2f} seconds")

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
    recent_evals = get_all_evaluations(limit_per_stream=1, eval_type=selected_eval_type)
    stream_keys = [eval['streamKey'] for eval in recent_evals]
    default_stream_key = get_most_recent_stream_key()
    
    # Select stream key with most recent as default
    stream_key = st.selectbox(
        "Select Stream Key",
        options=stream_keys,
        index=stream_keys.index(default_stream_key) if default_stream_key in stream_keys else 0
    ) if stream_keys else st.text_input("Enter Stream Key")

    if stream_key:
        # Get evaluations for timestamp selection
        evaluations = get_stream_evaluations(stream_key, eval_type=selected_eval_type)
        if evaluations:
            eval_options = {
                datetime.fromtimestamp(float(eval['timestamp'])).strftime('%Y-%m-%d %H:%M:%S'): eval 
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
                llm_context = get_context(
                    stream_key=stream_key, 
                    current_eval_timestamp=current_eval_timestamp, 
                    return_type="string",
                    include_code_files=st.session_state.load_code_files
                )
                
                display_context = get_context(
                    stream_key=stream_key, 
                    current_eval_timestamp=current_eval_timestamp, 
                    return_type="dict",
                    include_code_files=st.session_state.load_code_files
                )

                # Count tokens
                token_count = count_tokens(llm_context)
                st.write(f"Token count: {token_count}")

                try:
                    ai_response = run_completion_with_fallback(
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            *st.session_state.messages[:-1],
                            {"role": "user", "content": f"Context:\n{llm_context}\n\nQuestion: {prompt}"}
                        ]
                    )

                    if ai_response:
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        with st.chat_message("assistant"):
                            st.markdown(ai_response)
                    else:
                        st.error("Failed to get AI response")
                except Exception as e:
                    st.error(f"Error getting AI response: {str(e)}")

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

                # ...rest of display code...

            # Rest of display sections using same context data
            current_eval_data = current_eval
            prev_evals = [e for e in evaluations if float(e['timestamp']) < current_eval_timestamp][:5]

            # Display previous evaluations
            st.subheader("Previous Evaluations")
            prev_evals = [e for e in evaluations if float(e['timestamp']) < float(current_eval['timestamp'])][:5]
            for eval in prev_evals:
                timestamp = datetime.fromtimestamp(float(eval['timestamp'])).strftime('%Y-%m-%d %H:%M:%S')
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

            # Rest of the display sections remain unchanged
            # st.subheader("Current Evaluation Details")
            # with st.expander("Conversation History", expanded=st.session_state.expanders_open):
            #     if 'conversation' in current_eval:
            #         st.markdown(current_eval['conversation'])
            #     else:
            #         st.write("No conversation history available")

            # Rest of the display sections remain unchanged
            # st.subheader("Current Prompts")
            # for prompt in filtered_prompts:
            #     with st.expander(f"Prompt: {prompt['ref']}", expanded=st.session_state.expanders_open):
            #         st.code(prompt['content'])

            # Display Python files
            # if st.session_state.load_code_files:
            #     st.subheader("Python Files Content")
            #     for file_info, content in zip(python_files, file_contents):
            #         with st.expander(f"File: {file_info['path']}", expanded=st.session_state.expanders_open):
            #             st.code(content, language='python')

        else:
            st.error("No evaluations found for the selected stream key")
    print(f"⏱️ Rendering chat assistant tab took {time.time() - start_time:.2f} seconds")
