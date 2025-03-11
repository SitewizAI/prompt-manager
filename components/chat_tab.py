"""Component for the Chat Assistant tab in the app."""

import streamlit as st
import json
import time
from utils import (
    run_completion_with_fallback,
    SYSTEM_PROMPT,
    get_context,
    get_most_recent_stream_key,
    get_all_evaluations,
    get_stream_evaluations,
    count_tokens,
    log_debug
)

def render_chat_tab(selected_eval_type: str):
    """Render the chat assistant tab."""
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
        render_chat_interface(stream_key, selected_eval_type, past_eval_count)
    else:
        st.error("No stream key available. Please make sure evaluations exist.")
        
    st.sidebar.text(f"⏱️ Render chat tab: {time.time() - start_time:.2f}s")

def render_chat_interface(stream_key: str, eval_type: str, past_eval_count: int):
    """Render the chat interface with context for a selected stream key."""
    # Get evaluations for timestamp selection
    evaluations = get_stream_evaluations(stream_key, eval_type=eval_type)
    if not evaluations:
        st.error("No evaluations found for the selected stream key")
        return
        
    # Format timestamp for display
    from datetime import datetime
    format_timestamp = lambda ts: datetime.fromtimestamp(float(ts)).strftime('%Y-%m-%d %H:%M:%S')
    
    # Create options dictionary with formatted timestamps
    eval_options = {
        format_timestamp(float(eval['timestamp'])): eval 
        for eval in evaluations
    }
    
    # Select evaluation timestamp
    selected_timestamp = st.selectbox(
        "Select Evaluation Timestamp",
        options=list(eval_options.keys()),
        format_func=lambda x: f"Evaluation from {x}"
    )
    
    current_eval = eval_options[selected_timestamp]
    current_eval_timestamp = float(current_eval['timestamp'])

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process new chat input
    if prompt := st.chat_input("Ask a question about the data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get context with code files based on settings
        with st.spinner("Loading context data..."):
            context_start = time.time()
            llm_context = get_context(
                stream_key=stream_key, 
                current_eval_timestamp=current_eval_timestamp, 
                return_type="string",
                include_code_files=st.session_state.load_code_files,
                past_eval_count=past_eval_count
            )
            st.sidebar.text(f"⏱️ Get context: {time.time() - context_start:.2f}s")
            
            display_context = get_context(
                stream_key=stream_key, 
                current_eval_timestamp=current_eval_timestamp, 
                return_type="dict",
                include_code_files=st.session_state.load_code_files,
                past_eval_count=past_eval_count
            )

        # Display data statistics
        render_data_statistics(display_context)
        
        # Count tokens
        token_count = count_tokens(llm_context)
        st.write(f"Token count: {token_count}")

        try:
            # Get AI response
            with st.spinner("Generating response..."):
                completion_start = time.time()
                ai_response = run_completion_with_fallback(
                    messages=[
                        {"role": "system", "content": "You are a helpful prompt optimization assistant"},
                        *st.session_state.messages[:-1],
                        {"role": "user", "content": f"Context:\n{llm_context}\n\nQuestion: {prompt}\n\nYou may use this system prompt to help with the response:\n{SYSTEM_PROMPT}"}
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

        # Display evaluation details
        render_evaluation_details(display_context)

def render_data_statistics(display_context: dict):
    """Render statistics about the available data."""
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

def render_evaluation_details(display_context: dict):
    """Render details about the current evaluation and its prompts."""
    if not display_context:
        return
        
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

    # Display details about current evaluation
    st.subheader("Current Evaluation Details")
    with st.expander("Details", expanded=st.session_state.expanders_open):
        st.write(f"Type: {display_context['current_eval']['type']}")
        st.write(f"Successes: {display_context['current_eval']['successes']}")
        st.write(f"Attempts: {display_context['current_eval']['attempts']}")
        if display_context['current_eval']['failure_reasons']:
            st.write("Failure Reasons:")
            for reason in display_context['current_eval']['failure_reasons']:
                st.error(reason)

    # Display Python files
    st.subheader("Python Files")
    for file in display_context['files']:
        with st.expander(f"File: {file['file']['path']}", expanded=st.session_state.expanders_open):
            st.code(file['content'], language='python')
