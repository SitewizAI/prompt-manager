import streamlit as st
# Set page config must be the first streamlit command
st.set_page_config(page_title="Prompt Manager", layout="wide")

from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
import time
import pandas as pd

from utils import (
    get_all_prompts,
    get_daily_metrics_from_table,
    get_recent_evaluations,
    measure_time,
    log_debug, 
    log_error
)

from components import (
    render_prompts_tab, 
    render_evaluations_tab, 
    render_chat_tab
)

load_dotenv()

# Initialize session state for expanders and code loading
if "expanders_open" not in st.session_state:
    st.session_state.expanders_open = True
if "load_code_files" not in st.session_state:
    st.session_state.load_code_files = False
if "evaluations_expanded" not in st.session_state:
    st.session_state.evaluations_expanded = False
if "prompts" not in st.session_state:
    st.session_state.prompts = []
if "prompt_validation" not in st.session_state:
    st.session_state.prompt_validation = False

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
    
    if st.toggle("Enable Prompt Validation", value=st.session_state.prompt_validation):
        st.session_state.prompt_validation = True
    else:
        st.session_state.prompt_validation = False
    
    # Add timing metrics section
    st.header("Performance Metrics")
    st.text("Function execution times will appear here")

st.title("Prompt Manager")

# Add evaluation type selection above the header
st.header("Evaluation Type")
evaluation_types = ["okr", "insights", "suggestion", "design", "code"]
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
    log_debug(f"Loaded metrics: {str(metrics)[:200]}...")

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
    render_prompts_tab(prompts)

with tab2:
    render_evaluations_tab(recent_evals, selected_eval_type)

with tab3:
    render_chat_tab(selected_eval_type)
