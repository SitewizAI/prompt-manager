"""Component for the Prompts tab in the app."""

import streamlit as st
from typing import List, Dict, Any
import json
import time
from utils import (
    validate_prompt_parameters, 
    get_all_prompt_versions, 
    update_prompt,
    log_debug
)

def render_prompts_tab(prompts: List[Dict[str, Any]]):
    """Render the prompts tab with prompt versions, editing, and validation."""
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
                    
                # Add buttons to load history and revert to original version
                col1, col2, col3 = st.columns([2, 1, 1])
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
                
                with col3:
                    if st.button("Revert to Original", key=f"btn_revert_{ref}"):
                        with st.spinner(f"Reverting {ref} to original version..."):
                            # Get all versions to find the original (version 0)
                            all_versions = get_all_prompt_versions(ref)
                            # Find version 0 (the original version)
                            original_version = next((v for v in all_versions if int(v.get('version', 0)) == 0), None)
                            
                            if original_version:
                                # Get the content and is_object flag from original version
                                original_content = original_version.get('content', '')
                                is_object = original_version.get('is_object', False)
                                
                                # Update prompt with original content
                                if update_prompt(ref, original_content):
                                    st.success(f"Successfully reverted {ref} to original version!")
                                    # Clear the session state prompts to force a refresh
                                    st.session_state.prompts = []
                                    st.rerun()
                                else:
                                    st.error(f"Failed to revert {ref} to original version.")
                            else:
                                st.error(f"Could not find original version for {ref}")
                
                # Display the versions we have
                displayed_versions = versions if st.session_state[f"load_history_{ref}"] else [versions[0]]
                tabs = st.tabs([f"Version {v.get('version', 'N/A')}" for v in displayed_versions])
                
                for tab, version in zip(tabs, displayed_versions):
                    with tab:
                        render_prompt_version_editor(ref, version)
                
                # If we've loaded history, show a button to collapse it again
                if st.session_state[f"load_history_{ref}"]:
                    if st.button("Hide Previous Versions", key=f"btn_hide_history_{ref}"):
                        st.session_state[f"load_history_{ref}"] = False
                        st.rerun()

def render_prompt_version_editor(ref: str, version: Dict[str, Any]):
    """Render an editor for a single prompt version with validation."""
    # Show content
    content = version.get('content', '')
    is_object = version.get('is_object', False)
    
    # Run validation if enabled - for both string and object prompts
    if st.session_state.prompt_validation:
        with st.spinner(f"Validating prompt {ref}..."):
            is_valid, error_message, details = validate_prompt_parameters(ref, content)
            
            # Display validation results
            if not is_valid:
                st.error(f"Validation error: {error_message}")
                if details:
                    if "validation_error" in details:
                        st.error(f"Schema validation error: {details['validation_error']}")
                    if details.get("extra_vars"):
                        st.warning(f"Extra variables: {', '.join(details['extra_vars'])}")
                    if details.get("unused_vars"):
                        st.warning(f"Missing required variables: {', '.join(details['unused_vars'])}")
                    # Add specific document validation error display
                    if details.get("missing_output_fields"):
                        st.warning(f"Missing output fields in document structure: {', '.join(details['missing_output_fields'])}")
                    if details.get("missing_reference_fields"):
                        st.warning(f"Missing reference fields in document structure: {', '.join(details['missing_reference_fields'])}")
            else:
                if details.get("object_validated"):
                    success_message = f"JSON schema validation passed! {details.get('question_count', 0)} questions validated."
                    if details.get("document_validation"):
                        doc_details = details.get("document_validation", {})
                        document_fields = doc_details.get("document_fields", [])
                        output_fields = doc_details.get("output_fields", [])
                        reference_fields = doc_details.get("reference_fields", [])
                        success_message += f"\nAll output and reference fields found in documents."
                        st.success(success_message)
                        # Show detailed field information
                        with st.expander("Show Document Fields"):
                            st.markdown("**Document Fields:**")
                            for field in document_fields:
                                st.markdown(f"- `{field}`")
                            st.markdown("\n**Output Fields Used:**")
                            for field in output_fields:
                                st.markdown(f"- `{field}`")
                            st.markdown("\n**Reference Fields Used:**")
                            for field in reference_fields:
                                st.markdown(f"- `{field}`")
                    else:
                        st.success(success_message)
                else:
                    st.success("Prompt validation passed!")
                    if details.get("used_vars"):
                        st.info(f"Used variables: {', '.join(details['used_vars'])}")
    
    # Render editor based on prompt type
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
            
            # Always display the button and check for changes when clicked
            if st.button("Create New Version", key=f"update_json_{ref}_{version.get('version', 'N/A')}"):
                if new_content != formatted_content:
                    # Always pass is_object=True for JSON content
                    if update_prompt(ref, new_content):
                        st.success("New prompt version created successfully!")
                        # Clear the session state prompts to force a refresh
                        st.session_state.prompts = []
                        st.rerun()
                    else:
                        st.error("Failed to create new prompt version")
                else:
                    st.info("No changes detected. The content is the same.")
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
        
        # Always display the button and check for changes when clicked
        if st.button("Create New Version", key=f"update_{ref}_{version.get('version', 'N/A')}"):
            if new_content != content:
                # Pass is_object=False for string content
                if update_prompt(ref, new_content):
                    st.success("New prompt version created successfully!")
                    # Clear the session state prompts to force a refresh
                    st.session_state.prompts = []
                    st.rerun()
                else:
                    st.error("Failed to create new prompt version")
            else:
                st.info("No changes detected. The content is the same.")
    
    # Display additional metadata
    st.text(f"Last Updated: {version.get('updatedAt', 'N/A')}")
    st.text(f"Content Type: {'JSON Object' if is_object else 'String'}")
    if version.get('description'):
        st.text(f"Description: {version['description']}")
