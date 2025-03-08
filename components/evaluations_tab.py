"""Component for the Evaluations tab in the app."""

import streamlit as st
import json
import time
from decimal import Decimal
from typing import List, Dict, Any
from utils import (
    get_evaluation_by_timestamp,
    log_debug,
    convert_decimal,
    get_conversation_history,
)

def format_timestamp_local(timestamp_float):
    """
    Convert UTC timestamp to local time with AM/PM format.
    
    Args:
        timestamp_float: Unix timestamp as float or Decimal
        
    Returns:
        Formatted datetime string in local timezone with AM/PM
    """
    from datetime import datetime, timezone
    
    if isinstance(timestamp_float, Decimal):
        timestamp_float = float(timestamp_float)
    
    # Convert UTC timestamp to datetime object
    utc_time = datetime.fromtimestamp(timestamp_float, tz=timezone.utc)
    
    # Convert to local time (no timezone specified means local)
    local_time = utc_time.astimezone()
    
    # Format with AM/PM
    return local_time.strftime('%Y-%m-%d %I:%M:%S %p')

def render_evaluations_tab(recent_evals: List[Dict[str, Any]], selected_eval_type: str):
    """Render the recent evaluations tab."""
    log_debug("Rendering Evaluations tab...")
    start_time = time.time()
    
    # Create header row with reload button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("Recent Evaluations")
    with col2:
        # Add a reload button that will clear the cached evaluations
        if st.button("🔄 Reload Evaluations", key="btn_reload_evals", type="primary"):
            # Clear the stored evaluation conversations to force a refresh
            st.session_state.evaluation_conversations = {}
            # Clear any other cached evaluation data
            if "recent_evaluations" in st.session_state:
                del st.session_state.recent_evaluations
            st.rerun()
    
    # Initialize session state for storing fetched conversation histories
    if "evaluation_conversations" not in st.session_state:
        st.session_state.evaluation_conversations = {}
    
    if not recent_evals:
        st.warning(f"No recent evaluations found for {selected_eval_type}")
    
    # Counter for auto-fetching first 3 evaluations' conversations
    eval_counter = 0
    
    for eval in recent_evals:
        # Use our new timestamp formatter function to show local time with AM/PM
        timestamp = format_timestamp_local(float(eval['timestamp']))
        eval_timestamp = float(eval['timestamp'])
        eval_streamkey = eval['streamKey']
        eval_key = f"{eval_streamkey}_{eval_timestamp}_{eval.get('type', 'unknown')}"
        
        # Create unique expander key
        expander_key = f"expander_{eval_key}"
        is_expanded = st.session_state.get(expander_key, st.session_state.evaluations_expanded)
        
        with st.expander(f"Evaluation - {eval.get('type', 'N/A')} - {timestamp}", 
                       expanded=is_expanded):
            
            # When expander is opened, fetch the full evaluation if not already in session state
            if is_expanded and eval_key not in st.session_state.evaluation_conversations:
                with st.spinner("Loading full evaluation data..."):
                    full_eval = get_evaluation_by_timestamp(
                        stream_key=eval_streamkey,
                        timestamp=eval_timestamp,
                        eval_type=eval.get('type')
                    )
                    
                    if full_eval:
                        # Store in session state for future reference
                        st.session_state.evaluation_conversations[eval_key] = full_eval
                        # Use full evaluation data for this display
                        display_eval = full_eval
                    else:
                        # If fetch failed, use the summary data we already have
                        display_eval = eval
            elif eval_key in st.session_state.evaluation_conversations:
                # Use cached full evaluation
                display_eval = st.session_state.evaluation_conversations[eval_key]
            else:
                # Use summary data if not expanded
                display_eval = eval
            
            # Store current expander state in session state
            st.session_state[expander_key] = True
            
            # Create conversation key for this evaluation
            conversation_key = f"conversation_{eval_key}"
            
            # Automatically fetch conversation history for first 3 evaluations if not already present
            if eval_counter < 3 and conversation_key not in st.session_state:
                with st.spinner(f"Auto-fetching conversation for evaluation {eval_counter + 1}/3..."):
                    fetched_conversation = get_conversation_history(
                        stream_key=eval_streamkey,
                        timestamp=eval_timestamp,
                        eval_type=eval.get('type')
                    )
                    
                    if fetched_conversation:
                        st.session_state[conversation_key] = fetched_conversation
                    else:
                        st.session_state[conversation_key] = "Conversation history not found"
            
            # Render evaluation content
            render_evaluation_content(display_eval, eval_key, eval_streamkey, eval_timestamp)
            
            # Increment counter
            eval_counter += 1
    
    st.sidebar.text(f"⏱️ Render evals tab: {time.time() - start_time:.2f}s")

def render_evaluation_content(display_eval: Dict[str, Any], eval_key: str, 
                             stream_key: str, timestamp: float):
    """Render the content of a single evaluation."""
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Successes", convert_decimal(display_eval.get('successes', 0)))
    col2.metric("Attempts", convert_decimal(display_eval.get('attempts', 0)))
    col3.metric("Number of Turns", convert_decimal(display_eval.get('num_turns', 0)))
    
    # Display tabs in new order: Failure Reasons first, then Conversation, etc.
    tab_failures, tab_convo, tab_question, tab_prompts = st.tabs([
        "Failure Reasons", "Conversation", "Question", "Prompts Used"
    ])
    
    with tab_failures:
        st.write("### Failure Reasons")
        failure_reasons = display_eval.get('failure_reasons', [])
        if failure_reasons:
            for reason in failure_reasons:
                st.error(reason)
        else:
            st.success("No failures recorded")
    
    with tab_convo:
        st.write("### Conversation History")
        conversation = display_eval.get('conversation', '')
        
        # Initialize conversation session state key if not exists
        conversation_key = f"conversation_{eval_key}"
        if conversation_key not in st.session_state:
            st.session_state[conversation_key] = conversation
            
        if st.session_state[conversation_key]:
            # Try to parse the conversation as a list of messages
            print(st.session_state[conversation_key])
            try:
                # Try to parse as JSON
                try:
                    messages = json.loads(st.session_state[conversation_key])["conversation"]
                except json.JSONDecodeError:
                    # Not JSON, treat as a raw string and split by double newlines
                    # This is a fallback for unstructured conversation data
                    raw_text = st.session_state[conversation_key]
                    segments = raw_text.split("\n\n")
                    messages = [{"message": segment} for segment in segments if segment.strip()]
            except Exception as e:
                st.error(f"Failed to parse conversation history: {str(e)}")
                st.text_area("Raw Conversation", st.session_state[conversation_key], height=300)
                messages = []
                
            # Display messages in a more structured format if we have them
            if messages:
                st.write("#### Message History")
                
                for i, msg in enumerate(messages):
                    # Check if message is a dict or string
                    if isinstance(msg, dict):
                        message_content = msg.get("message", "")
                        role = msg.get("role", "")
                        agent = msg.get("agent", "")
                        source = msg.get("source", "")
                    else:
                        # Handle case where message might be a simple string
                        message_content = str(msg)
                        role = ""
                        agent = ""
                        source = ""
                    
                    # Create a header showing the role/agent/source
                    header = ""
                    if role:
                        header += f"Role: {role} | "
                    if agent:
                        header += f"Agent: {agent} | "
                    if source:
                        header += f"Source: {source} | "
                    header += f"Message #{i+1}"
                    
                    # Create a container for each message
                    message_container = st.container()
                    with message_container:
                        # Display message header
                        st.markdown(f"**{header}**")
                        
                        # Display full message content in a scrollable text area
                        # Calculate height based on message length, but with min/max constraints
                        message_height = min(max(100, min(50 + len(message_content) // 10, 300)), 400)
                        st.text_area("", message_content, height=message_height,
                                   key=f"msg_content_{eval_key}_{i}", disabled=True)
                        
                        # Show metadata if available
                        if isinstance(msg, dict):
                            meta = {k: v for k, v in msg.items() 
                                  if k not in ["message", "role", "agent", "source"]}
                            if meta:
                                st.write("**Message Metadata:**")
                                for k, v in meta.items():
                                    st.write(f"- **{k}**: {v}")
                        
                        # Add a divider between messages
                        st.divider()
            else:
                # Fallback to raw display
                st.text_area("Full Conversation", st.session_state[conversation_key], height=300)
        else:
            # Add a button to fetch complete conversation history
            if st.button("Get Conversation History", key=f"btn_get_conv_{eval_key}"):
                with st.spinner("Fetching conversation history..."):
                    # Fetch conversation history from DynamoDB
                    fetched_conversation = get_conversation_history(
                        stream_key=stream_key,
                        timestamp=timestamp,
                        eval_type=display_eval.get('type')
                    )
                    
                    if fetched_conversation:
                        st.session_state[conversation_key] = fetched_conversation
                        st.rerun()
                    else:
                        st.warning("Conversation history not found")
            else:
                st.info("No conversation history available. Click the button to fetch it.")
    
    with tab_question:
        st.write("### Question")
        st.write(display_eval.get('question', 'N/A'))

    with tab_prompts:
        st.write("### Prompts Used")
        if display_eval.get('prompts'):
            for prompt in display_eval.get('prompts', []):
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
        if display_eval.get('summary'):
            st.write("### Summary")
            st.info(display_eval['summary'])