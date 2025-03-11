"""Component for the Prompts tab in the app."""

import streamlit as st
from typing import List, Dict, Any, Set, Tuple
import json
import time
import re
from utils import (
    validate_prompt_parameters, 
    get_all_prompt_versions, 
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

    # Initialize prompt type in session state if not present
    if "selected_prompt_type" not in st.session_state:
        st.session_state.selected_prompt_type = "all"
    
    # Initialize version 0 only filter
    if "show_version_0_only" not in st.session_state:
        st.session_state.show_version_0_only = False
        
    # Display header with refresh button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.header("Prompts")
    with col2:
        if st.button("🔄 Refresh", type="primary", help="Reload all prompts from the database"):
            # Clear prompts and references in session state to force a refresh
            if "prompts" in st.session_state:
                st.session_state.prompts = []
            if "prompt_references" in st.session_state:
                del st.session_state.prompt_references
            st.success("Refreshing prompts...")
            st.rerun()
    
    # Create buttons for prompt type filtering at the top of the page
    st.write("Filter by prompt type:")
    
    # Create a horizontal layout for buttons
    cols = st.columns(len(PROMPT_TYPES))
    
    # Create a button for each prompt type
    for i, prompt_type in enumerate(PROMPT_TYPES.keys()):
        with cols[i]:
            # Check if this is the currently selected type to apply styling
            is_selected = st.session_state.selected_prompt_type == prompt_type
            
            # Create a styled button with different appearance when selected
            button_label = prompt_type.capitalize()
            if st.button(
                button_label,
                type="primary" if is_selected else "secondary",
                key=f"btn_type_{prompt_type}"
            ):
                st.session_state.selected_prompt_type = prompt_type
                # Clear any specific ref selections when changing type
                if "selected_refs" in st.session_state:
                    st.session_state.selected_refs = []
                st.rerun()

    # Add Version 0 Only filter button
    col1, col2 = st.columns([4, 2])
    with col1:
        st.markdown("---")
    with col2:
        version_filter_btn_label = "🔍 Show Original (V0) Prompts Only" if not st.session_state.show_version_0_only else "🔍 Show All Versions"
        if st.button(version_filter_btn_label, type="primary" if st.session_state.show_version_0_only else "secondary"):
            st.session_state.show_version_0_only = not st.session_state.show_version_0_only
            st.rerun()
    
    # Add a separator after the buttons
    st.markdown("---")

    # Sidebar filters
    st.sidebar.header("Filters")

    # Populate the "all" category with all unique refs
    all_refs = list(set([p["ref"] for p in prompts]))
    PROMPT_TYPES["all"] = all_refs

    # Get the current selected prompt type from session state
    selected_prompt_type = st.session_state.selected_prompt_type
    
    # Show the current prompt type in the sidebar as well (informational)
    st.sidebar.selectbox(
        "Current prompt type",
        options=list(PROMPT_TYPES.keys()),
        index=list(PROMPT_TYPES.keys()).index(selected_prompt_type),
        key="sidebar_prompt_type",
        on_change=lambda: setattr(st.session_state, "selected_prompt_type", st.session_state.sidebar_prompt_type)
    )

    # Get refs for the selected prompt type
    type_specific_refs = PROMPT_TYPES[selected_prompt_type]

    # If "all" is selected, allow further filtering by specific refs
    if selected_prompt_type == "all":
        # Initialize selected_refs in session state if not present
        if "selected_refs" not in st.session_state:
            st.session_state.selected_refs = []
            
        st.session_state.selected_refs = st.sidebar.multiselect(
            "Filter by refs",
            options=all_refs,
            default=st.session_state.selected_refs
        )
        selected_refs = st.session_state.selected_refs
    else:
        # Show the refs for the selected type (informational only)
        st.sidebar.write("Prompt refs for this type:")
        for ref in type_specific_refs:
            st.sidebar.write(f"- {ref}")
        selected_refs = type_specific_refs

    # Search box
    search_term = st.sidebar.text_input("Search content").lower()

    # Filter data based on selections - FIXED FILTERING LOGIC
    filtered_prompts = prompts
    
    # First, filter by prompt type/refs
    if selected_prompt_type != "all" or selected_refs:
        # If specific refs are selected or a type other than "all" is chosen
        # Only show prompts with refs in the selected_refs list
        filtered_prompts = [p for p in filtered_prompts if p["ref"] in selected_refs]
    
    # Then apply search term filter if provided
    if search_term:
        filtered_prompts = [p for p in filtered_prompts if (
            isinstance(p.get("content", ""), str) and search_term in p["content"].lower()
        )]
    
    # Don't filter by version here anymore - we'll handle this in display_prompt_versions
    # Remove the version filter code and just pass in the show_version_0_only flag
    
    # Display prompt count and update the header with the type
    version_filter_text = " (Version 0 only)" if st.session_state.show_version_0_only else ""
    st.write(f"Showing prompts{version_filter_text} of type: **{selected_prompt_type.capitalize()}**")
    
    if st.session_state.show_version_0_only:
        st.info("Displaying only original (version 0) prompts. Click 'Show All Versions' to see all prompt versions.")

    # Pass the filter flag to display_prompt_versions instead of filtering here
    display_prompt_versions(filtered_prompts, show_version_0_only=st.session_state.show_version_0_only)
    st.sidebar.text(f"⏱️ Render prompts tab: {time.time() - start_time:.2f}s")

def display_prompt_versions(prompts: List[Dict[str, Any]], show_version_0_only: bool = False):
    """Display prompts with version history in the Streamlit UI."""
    log_debug(f"Displaying {len(prompts)} prompts")
    
    # Group prompts by ref
    prompts_by_ref = {}
    for prompt in prompts:
        ref = prompt['ref']
        if ref not in prompts_by_ref:
            prompts_by_ref[ref] = []
        prompts_by_ref[ref].append(prompt)
    
    # Process each ref group and count displayed prompts
    displayed_count = 0
    
    # If we're showing version 0 only, prepare to fetch version 0 for refs that need it
    v0_prompts = {}
    if show_version_0_only:
        refs_needing_v0 = []
        for ref, versions in prompts_by_ref.items():
            # First check if version 0 is already in our data
            existing_v0 = next((v for v in versions if int(v.get('version', 0)) == 0), None)
            if existing_v0:
                v0_prompts[ref] = existing_v0
            else:
                refs_needing_v0.append(ref)
        
        # Fetch version 0 for refs that need it
        if refs_needing_v0:
            with st.spinner(f"Fetching original versions for {len(refs_needing_v0)} prompts..."):
                for ref in refs_needing_v0:
                    # Fetch all versions and find v0
                    all_versions = get_all_prompt_versions(ref)
                    version_0 = next((v for v in all_versions if int(v.get('version', 0)) == 0), None)
                    if version_0:
                        v0_prompts[ref] = version_0
    
    # Sort versions for each ref
    for ref in prompts_by_ref:
        prompts_by_ref[ref].sort(key=lambda x: int(x.get('version', 0)), reverse=True)
    
    # Display prompts
    for ref, versions in prompts_by_ref.items():
        if versions:  # Only show if there are versions
            displayed_count += 1
            with st.expander(f"Prompt: {ref}", expanded=st.session_state.expanders_open):
                # Initialize session state for this prompt's historical versions if not exists
                if f"load_history_{ref}" not in st.session_state:
                    st.session_state[f"load_history_{ref}"] = False

                # Display prompt references (prompts that use this one and prompts used by this one)
                if "prompt_references" in st.session_state:
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
                    
                # Add buttons to load history, refresh prompt, and revert to original version
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                
                # Refresh button - fetches just this prompt from DB
                with col1:
                    if st.button("🔄 Refresh Prompt", key=f"btn_refresh_{ref}", help=f"Reload latest version of {ref} from database"):
                        with st.spinner(f"Refreshing prompt {ref}..."):
                            # Remove this prompt from session state prompts to force a refresh
                            if "prompts" in st.session_state:
                                # Filter out prompts with this ref
                                st.session_state.prompts = [p for p in st.session_state.prompts if p['ref'] != ref]
                            # Also update prompt references
                            if "prompt_references" in st.session_state:
                                del st.session_state.prompt_references
                            st.success(f"Refreshed prompt {ref}")
                            st.rerun()
                
                # Load Previous Versions button
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
                
                # Revert to Original button
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
                
                # Determine which versions to display based on settings
                if st.session_state[f"load_history_{ref}"]:
                    # If user explicitly requested history, show all versions
                    displayed_versions = versions
                elif show_version_0_only:
                    # If showing v0 only and we found a v0 version
                    if ref in v0_prompts:
                        displayed_versions = [v0_prompts[ref]]
                    else:
                        # Fall back to latest version if v0 not found
                        log_debug(f"No version 0 found for {ref}, using latest version")
                        displayed_versions = [versions[0]]
                else:
                    # Default: show latest version
                    displayed_versions = [versions[0]]
                
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
    # Add an HTML anchor for this prompt so links can navigate to it
    st.markdown(f'<div id="prompt-{ref}"></div>', unsafe_allow_html=True)

    # Show content
    content = version.get('content', '')
    is_object = version.get('is_object', False)
    current_version = version.get('version', 'N/A')
    
    # Extract the numeric version for use in the update function
    try:
        numeric_version = int(current_version)
    except (ValueError, TypeError):
        numeric_version = None
    
    # Extract variables from the prompt content for analysis
    variables = []
    if isinstance(content, str) and not is_object:
        # Find all format variables in the content using regex
        # This regex matches {var} patterns that aren't part of {{var}} or other structures
        variables = re.findall(r'(?<!\{)\{([a-zA-Z0-9_]+)\}(?!\})', content)

        if variables:
            st.markdown("##### Variables used in this prompt:")
            st.markdown(", ".join([f"`{var}`" for var in variables]))

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
            
            new_content = st.text_area(
                "JSON Content (editable)",
                formatted_content,
                height=400,
                key=f"json_content_{ref}_{current_version}"
            )
            
            # Button layout with columns
            col1, col2 = st.columns(2)
            with col1:
                # Create New Version button
                if st.button("Create New Version", key=f"update_json_{ref}_{current_version}"):
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
            
            with col2:
                # Update Current Version button
                if st.button("Update This Version", key=f"update_current_json_{ref}_{current_version}"):
                    if new_content != formatted_content:
                        # Pass update_current=True and specific_version to update the current version
                        update_result = update_prompt(ref, new_content, update_current=True, specific_version=numeric_version)
                        
                        # Handle both return types (boolean or tuple)
                        if isinstance(update_result, tuple):
                            success, error_msg = update_result
                        else:
                            success, error_msg = update_result, None
                        
                        if success:
                            st.success(f"Version {current_version} updated successfully!")
                            # Clear the session state prompts and references to force a refresh
                            st.session_state.prompts = []
                            if "prompt_references" in st.session_state:
                                del st.session_state.prompt_references
                            st.rerun()
                        else:
                            # Show detailed error if available
                            if error_msg:
                                st.error(f"Failed to update version {current_version}: {error_msg}")
                            else:
                                st.error(f"Failed to update version {current_version}")
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
            key=f"content_{ref}_{current_version}"
        )
        
        # Button layout with columns
        col1, col2 = st.columns(2)
        with col1:
            # Create New Version button
            if st.button("Create New Version", key=f"update_{ref}_{current_version}"):
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
        
        with col2:
            # Update Current Version button
            if st.button("Update This Version", key=f"update_current_{ref}_{current_version}"):
                if new_content != content:
                    # Pass update_current=True and specific_version to update the current version
                    update_result = update_prompt(ref, new_content, update_current=True, specific_version=numeric_version)
                    
                    # Handle both return types (boolean or tuple)
                    if isinstance(update_result, tuple):
                        success, error_msg = update_result
                    else:
                        success, error_msg = update_result, None
                    
                    if success:
                        st.success(f"Version {current_version} updated successfully!")
                        # Clear the session state prompts and references to force a refresh
                        st.session_state.prompts = []
                        if "prompt_references" in st.session_state:
                            del st.session_state.prompt_references
                        st.rerun()
                    else:
                        # Show detailed error if available
                        if error_msg:
                            st.error(f"Failed to update version {current_version}: {error_msg}")
                        else:
                            st.error(f"Failed to update version {current_version}")
                else:
                    st.info("No changes detected. The content is the same.")
    
    # Display additional metadata
    st.text(f"Last Updated: {version.get('updatedAt', 'N/A')}")
    st.text(f"Content Type: {'JSON Object' if is_object else 'String'}")
    if version.get('description'):
        st.text(f"Description: {version['description']}")
