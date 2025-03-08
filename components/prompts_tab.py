"""Component for the Prompts tab in the app."""

import streamlit as st
from typing import List, Dict, Any, Set, Tuple
import json
import time
import re
from datetime import datetime, timedelta
from utils import (
    validate_prompt_parameters, 
    get_all_prompt_versions,
    get_prompts_by_date,
    revert_all_prompts_to_date,
    update_prompt,
    log_debug,
    PROMPT_TYPES
)

def analyze_prompt_references(prompts: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Analyze all prompts to find references between them.

    Args:
        prompts: List of prompt dictionaries

    Returns:
        Dictionary with two keys:
        - 'uses': Dict mapping each prompt ref to a list of prompt refs it uses
        - 'used_by': Dict mapping each prompt ref to a list of prompt refs that use it
    """
    log_debug("Analyzing prompt references...")

    # Initialize result dictionaries
    uses = {}  # prompt_ref -> list of refs it uses
    used_by = {}  # prompt_ref -> list of refs that use it

    # Initialize for all prompts
    all_refs = set(p["ref"] for p in prompts)
    for ref in all_refs:
        uses[ref] = []
        used_by[ref] = []

    # Analyze each prompt for references to other prompts
    for prompt in prompts:
        ref = prompt["ref"]
        content = prompt.get("content", "")

        # Skip if content is not a string (e.g., JSON object)
        if not isinstance(content, str):
            continue

        # Look for explicit references to other prompts in the content
        # Pattern to match prompt references like {prompt:other_prompt_ref}
        prompt_ref_pattern = r'\{prompt:([a-zA-Z0-9_-]+)\}'
        referenced_prompts = re.findall(prompt_ref_pattern, content)

        # Also look for variables that might be references to other prompts
        # This pattern matches variable names that end with "_prompt" or "_template"
        var_pattern = r'\{([a-zA-Z0-9_]+(prompt|template))\}'
        var_refs = re.findall(var_pattern, content)

        # Add explicit references to the 'uses' dictionary
        for referenced_ref in referenced_prompts:
            if referenced_ref in all_refs and referenced_ref not in uses[ref]:
                uses[ref].append(referenced_ref)

                # Also update the 'used_by' dictionary
                if ref not in used_by[referenced_ref]:
                    used_by[referenced_ref].append(ref)

        # For variable references, check if any prompt ref matches the variable name
        for var_ref, _ in var_refs:
            # Check if there's a prompt with this name
            if var_ref in all_refs and var_ref not in uses[ref]:
                uses[ref].append(var_ref)

                # Also update the 'used_by' dictionary
                if ref not in used_by[var_ref]:
                    used_by[var_ref].append(ref)

    return {"uses": uses, "used_by": used_by}

def render_prompts_tab(prompts: List[Dict[str, Any]]):
    """Render the prompts tab with prompt versions, editing, and validation."""
    log_debug("Rendering Prompts tab...")
    start_time = time.time()
    
    # Analyze prompt references
    if "prompt_references" not in st.session_state:
        st.session_state.prompt_references = analyze_prompt_references(prompts)

    # Initialize session state for date-based version viewing
    if "view_by_date" not in st.session_state:
        st.session_state.view_by_date = False
    if "selected_date" not in st.session_state:
        st.session_state.selected_date = datetime.now().strftime("%Y-%m-%d")
    if "prompts_by_date" not in st.session_state:
        st.session_state.prompts_by_date = {}

    # Add a section at the top for date-based version viewing
    st.header("Prompt Versions by Date")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Date picker for selecting a specific date
        selected_date = st.date_input(
            "Select date to view prompt versions",
            value=datetime.strptime(st.session_state.selected_date, "%Y-%m-%d").date(),
            max_value=datetime.now().date()
        )
        st.session_state.selected_date = selected_date.strftime("%Y-%m-%d")

    with col2:
        # Button to view prompts as of the selected date
        if st.button("Show Prompts by Date"):
            with st.spinner(f"Loading prompts as of {st.session_state.selected_date}..."):
                st.session_state.prompts_by_date = get_prompts_by_date(st.session_state.selected_date)
                st.session_state.view_by_date = True

    with col3:
        # Button to revert all prompts to the selected date
        if st.button("Revert All to This Date"):
            with st.spinner(f"Reverting all prompts to {st.session_state.selected_date}..."):
                success, successful_refs, failed_refs = revert_all_prompts_to_date(st.session_state.selected_date)

                if success:
                    st.success(f"Successfully reverted {len(successful_refs)} prompts to {st.session_state.selected_date}")
                    if failed_refs:
                        st.warning(f"Failed to revert {len(failed_refs)} prompts")
                        with st.expander("Show failed prompts"):
                            for ref in failed_refs:
                                st.write(f"- {ref}")

                    # Clear the session state prompts and references to force a refresh
                    st.session_state.prompts = []
                    if "prompt_references" in st.session_state:
                        del st.session_state.prompt_references
                    st.session_state.view_by_date = False
                    st.rerun()
                else:
                    st.error(f"Failed to revert prompts to {st.session_state.selected_date}")
                    if failed_refs:
                        with st.expander("Show errors"):
                            for ref in failed_refs:
                                st.write(f"- {ref}")

    # Button to return to current versions if viewing by date
    if st.session_state.view_by_date:
        if st.button("Return to Current Versions"):
            st.session_state.view_by_date = False
            st.session_state.prompts_by_date = {}
            st.rerun()

        # Display prompts as of the selected date
        st.subheader(f"Prompts as of {st.session_state.selected_date}")

        if not st.session_state.prompts_by_date:
            st.info(f"No prompts found as of {st.session_state.selected_date}")
        else:
            # Convert the dictionary to a list for display
            date_prompts = list(st.session_state.prompts_by_date.values())
            display_prompt_versions(date_prompts, read_only=True)

        # Don't show the regular prompts view when viewing by date
        st.sidebar.text(f"⏱️ Render prompts tab: {time.time() - start_time:.2f}s")
        return

    # Sidebar filters
    st.sidebar.header("Filters")

    # Populate the "all" category with all unique refs
    all_refs = list(set([p["ref"] for p in prompts]))
    PROMPT_TYPES["all"] = all_refs

    # Prompt type selection
    prompt_type_options = list(PROMPT_TYPES.keys())
    selected_prompt_type = st.sidebar.selectbox(
        "Select prompt type",
        options=prompt_type_options,
        index=0  # Default to "all"
    )

    # Get refs for the selected prompt type
    type_specific_refs = PROMPT_TYPES[selected_prompt_type]

    # If "all" is selected, allow further filtering by specific refs
    if selected_prompt_type == "all":
        selected_refs = st.sidebar.multiselect(
            "Filter by refs",
            options=all_refs,
        )
    else:
        # Show the refs for the selected type (informational only)
        st.sidebar.write("Prompt refs for this type:")
        for ref in type_specific_refs:
            st.sidebar.write(f"- {ref}")
        selected_refs = type_specific_refs

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
    st.header(f"Prompts - {selected_prompt_type.capitalize()}")
    display_prompt_versions(filtered_prompts)
    st.sidebar.text(f"⏱️ Render prompts tab: {time.time() - start_time:.2f}s")

def display_prompt_versions(prompts: List[Dict[str, Any]], read_only: bool = False):
    """
    Display prompts with version history in the Streamlit UI.

    Args:
        prompts: List of prompt dictionaries to display
        read_only: If True, display prompts in read-only mode (for date-based viewing)
    """
    log_debug(f"Displaying {len(prompts)} prompts")
    
    # Organize prompts by ref
    prompts_by_ref = {}

    # If in read-only mode, use the provided prompts directly
    if read_only:
        for prompt in prompts:
            ref = prompt['ref']
            if ref not in prompts_by_ref:
                prompts_by_ref[ref] = []
            prompts_by_ref[ref].append(prompt)
    else:
        # In normal mode, use prompts from session state
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

                # Display prompt references (prompts that use this one and prompts used by this one)
                if "prompt_references" in st.session_state and not read_only:
                    references = st.session_state.prompt_references

                    # Prompts that use this prompt
                    used_by = references.get("used_by", {}).get(ref, [])
                    if used_by:
                        st.markdown("##### Used by:")
                        used_by_links = []
                        for using_ref in used_by:
                            used_by_links.append(f"[{using_ref}](#prompt-{using_ref})")
                        st.markdown(", ".join(used_by_links))

                    # Prompts that this prompt uses
                    uses = references.get("uses", {}).get(ref, [])
                    if uses:
                        st.markdown("##### Uses:")
                        uses_links = []
                        for used_ref in uses:
                            uses_links.append(f"[{used_ref}](#prompt-{used_ref})")
                        st.markdown(", ".join(uses_links))

                    # Add a separator if we displayed any references
                    if used_by or uses:
                        st.markdown("---")
                
                # Only show action buttons in normal mode (not read-only)
                if not read_only:
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
                                        # Clear the session state prompts and references to force a refresh
                                        st.session_state.prompts = []
                                        if "prompt_references" in st.session_state:
                                            del st.session_state.prompt_references
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to revert {ref} to original version.")
                                else:
                                    st.error(f"Could not find original version for {ref}")
                
                # Display the versions we have
                if read_only:
                    # In read-only mode, just show the single version
                    displayed_versions = [versions[0]]
                else:
                    # In normal mode, show all loaded versions if history is loaded
                    displayed_versions = versions if st.session_state[f"load_history_{ref}"] else [versions[0]]

                tabs = st.tabs([f"Version {v.get('version', 'N/A')}" for v in displayed_versions])
                
                for tab, version in zip(tabs, displayed_versions):
                    with tab:
                        render_prompt_version_editor(ref, version, read_only=read_only)
                
                # If we've loaded history, show a button to collapse it again
                if not read_only and st.session_state[f"load_history_{ref}"]:
                    if st.button("Hide Previous Versions", key=f"btn_hide_history_{ref}"):
                        st.session_state[f"load_history_{ref}"] = False
                        st.rerun()

def render_prompt_version_editor(ref: str, version: Dict[str, Any], read_only: bool = False):
    """
    Render an editor for a single prompt version with validation.

    Args:
        ref: The prompt reference ID
        version: The prompt version dictionary
        read_only: If True, display in read-only mode (for date-based viewing)
    """
    # Add an HTML anchor for this prompt so links can navigate to it
    st.markdown(f'<div id="prompt-{ref}"></div>', unsafe_allow_html=True)

    # Show content
    content = version.get('content', '')
    is_object = version.get('is_object', False)
    
    # Extract variables from the prompt content for analysis
    variables = []
    if isinstance(content, str) and not is_object:
        # Find all format variables in the content using regex
        # This regex matches {var} patterns that aren't part of {{var}} or other structures
        variables = re.findall(r'(?<!\{)\{([a-zA-Z0-9_]+)\}(?!\})', content)

        if variables:
            st.markdown("##### Variables used in this prompt:")
            st.markdown(", ".join([f"`{var}`" for var in variables]))

    # Run validation if enabled and not in read-only mode
    if st.session_state.prompt_validation and not read_only:
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
                        
                        # Replace expander with checkbox to avoid nesting expanders
                        show_fields = st.checkbox("Show Document Fields", 
                                                key=f"show_fields_{ref}_{version.get('version', 'N/A')}")
                        if show_fields:
                            st.markdown("**Document Fields:**")
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                for field in document_fields[:len(document_fields)//2 + len(document_fields)%2]:
                                    st.markdown(f"- `{field}`")
                            with col2:
                                for field in document_fields[len(document_fields)//2 + len(document_fields)%2:]:
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
            
            if read_only:
                # In read-only mode, display as disabled text area
                st.text_area(
                    "JSON Content (read-only)",
                    formatted_content,
                    height=400,
                    key=f"json_content_ro_{ref}_{version.get('version', 'N/A')}",
                    disabled=True
                )
            else:
                # In normal mode, display as editable text area
                new_content = st.text_area(
                    "JSON Content (editable)",
                    formatted_content,
                    height=400,
                    key=f"json_content_{ref}_{version.get('version', 'N/A')}"
                )
            
            # Only display the button in normal mode
            if not read_only and st.button("Create New Version", key=f"update_json_{ref}_{version.get('version', 'N/A')}"):
                if new_content != formatted_content:
                    # Check return type - could be bool or tuple of (bool, str)
                    update_result = update_prompt(ref, new_content)
                    
                    # Handle both return types (boolean or tuple)
                    if isinstance(update_result, tuple):
                        success, error_msg = update_result
                    else:
                        success, error_msg = update_result, None
                    
                    if success:
                        st.success("New prompt version created successfully!")
                        # Clear the session state prompts and references to force a refresh
                        st.session_state.prompts = []
                        if "prompt_references" in st.session_state:
                            del st.session_state.prompt_references
                        st.rerun()
                    else:
                        # Show detailed error if available
                        if error_msg:
                            st.error(f"Failed to create new prompt version: {error_msg}")
                        else:
                            st.error("Failed to create new prompt version")
                else:
                    st.info("No changes detected. The content is the same.")
        except Exception as e:
            st.error(f"Error handling JSON content: {str(e)}")
            st.text_area("Raw Content", content, height=200, disabled=True)
    else:
        # For string content
        if read_only:
            # In read-only mode, display as disabled text area
            st.text_area(
                "Content (read-only)",
                content,
                height=200,
                key=f"content_ro_{ref}_{version.get('version', 'N/A')}",
                disabled=True
            )
        else:
            # In normal mode, use regular text area
            new_content = st.text_area(
                "Content",
                content,
                height=200,
                key=f"content_{ref}_{version.get('version', 'N/A')}"
            )
        
        # Only display the button in normal mode
        if not read_only and st.button("Create New Version", key=f"update_{ref}_{version.get('version', 'N/A')}"):
            if new_content != content:
                # Check return type - could be bool or tuple of (bool, str)
                update_result = update_prompt(ref, new_content)
                
                # Handle both return types (boolean or tuple)
                if isinstance(update_result, tuple):
                    success, error_msg = update_result
                else:
                    success, error_msg = update_result, None
                
                if success:
                    st.success("New prompt version created successfully!")
                    # Clear the session state prompts and references to force a refresh
                    st.session_state.prompts = []
                    if "prompt_references" in st.session_state:
                        del st.session_state.prompt_references
                    st.rerun()
                else:
                    # Show detailed error if available
                    if error_msg:
                        st.error(f"Failed to create new prompt version: {error_msg}")
                    else:
                        st.error("Failed to create new prompt version")
            else:
                st.info("No changes detected. The content is the same.")
    
    # Display additional metadata
    st.text(f"Last Updated: {version.get('updatedAt', 'N/A')}")
    st.text(f"Content Type: {'JSON Object' if is_object else 'String'}")
    if version.get('description'):
        st.text(f"Description: {version['description']}")
