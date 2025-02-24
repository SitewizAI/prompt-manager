import streamlit as st
# Set page config must be the first streamlit command
st.set_page_config(page_title="Prompt Manager", layout="wide")

import boto3
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests
import os
import json
import tiktoken
from decimal import Decimal
from utils import run_completion_with_fallback, SYSTEM_PROMPT, get_github_files, get_file_contents, get_context, get_most_recent_stream_key, get_all_evaluations, get_stream_evaluations, get_evaluation_metrics, get_recent_evaluations
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

def count_tokens(text: str) -> int:
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))

def convert_decimal(value):
    """Convert Decimal values to float/int for Streamlit metrics."""
    if isinstance(value, Decimal):
        return float(value)
    return value

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
                tabs = st.tabs([f"Version {v.get('version', 'N/A')}" for v in versions])
                for tab, version in zip(tabs, versions):
                    with tab:
                        # Show content
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
                                    # Fixed: Add version to update_prompt call
                                    if update_prompt(ref, version.get('version'), new_content):
                                        st.success("Prompt updated successfully!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to update prompt")
                        
                        st.text(f"Last Updated: {version.get('updatedAt', 'N/A')}")
                        if version.get('description'):
                            st.text(f"Description: {version['description']}")

# Add evaluation type selection above the header
st.header("Evaluation Type")
evaluation_types = ["suggestions", "okr", "insights", "code"]
selected_eval_type = st.radio(
    "Select evaluation type",
    options=evaluation_types,
    horizontal=True
)
log_debug(f"Selected evaluation type: {selected_eval_type}")

# Add configuration parameters and fetch metrics immediately after type selection
days_to_show = st.sidebar.slider("Days of metrics to show", min_value=1, max_value=90, value=30)
evals_to_show = st.sidebar.slider("Number of recent evaluations to show", min_value=1, max_value=20, value=10)  # Changed default to 10

# Show metrics visualization right after type selection
st.header("Evaluation Metrics")
with st.spinner("Loading metrics..."):
    metrics = get_evaluation_metrics(days=days_to_show, eval_type=selected_eval_type)
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
    
    col1.metric("Total Evaluations", total_evals)
    col2.metric("Success Rate", f"{success_rate:.1f}%")
    col3.metric("Total Successes", total_successes)
    col4.metric("Average Turns", f"{avg_turns:.1f}")
    
    # Create DataFrame for visualization
    daily_data = metrics['daily_metrics']
    if daily_data:
        df = pd.DataFrame.from_dict(daily_data, orient='index')
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
    print(f"⏱️ Loading prompts took {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    log_debug("Loading evaluations...")
    recent_evals = get_all_evaluations(eval_type=selected_eval_type)
    log_debug(f"Loaded {len(recent_evals)} evaluations")
    print(f"⏱️ Loading evaluations took {time.time() - start_time:.2f} seconds")

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

# Update recent evaluations display
with tab2:
    log_debug("Rendering Evaluations tab...")
    start_time = time.time()
    st.header("Recent Evaluations")
    recent_evals = get_recent_evaluations(eval_type=selected_eval_type, limit=evals_to_show)  # Using the slider value
    log_debug(f"Retrieved {len(recent_evals)} recent evaluations")
    
    if not recent_evals:
        st.warning(f"No recent evaluations found for {selected_eval_type}")
    
    for eval in recent_evals:
        timestamp = datetime.fromtimestamp(float(eval['timestamp'])).strftime('%Y-%m-%d %H:%M:%S')
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
    recent_evals = get_all_evaluations(eval_type=selected_eval_type)
    # stream_keys = [eval['streamKey'] for eval in recent_evals]
    stream_keys = list(set([eval['streamKey'] for eval in recent_evals]))
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
