"""Utilities for handling conversation data formats."""

from typing import List, Dict, Any, Union
import json
from .logging_utils import log_debug, log_error

def extract_conversation(data: Union[str, Dict, List]) -> List[Dict[str, Any]]:
    """
    Extract conversation messages from various data formats.
    
    Args:
        data: Conversation data in various formats (string JSON, dict, or list)
        
    Returns:
        List of message dictionaries
    """
    try:
        # If input is a string, try to parse it as JSON
        if isinstance(data, str):
            try:
                data = json.loads(data)
                log_debug(f"Successfully parsed JSON string to {type(data)}")
            except json.JSONDecodeError as e:
                log_error(f"Failed to parse conversation JSON: {e}")
                return []
        
        # If we have a dict with a 'conversation' key, extract that
        if isinstance(data, dict):
            if 'conversation' in data:
                conversation = data['conversation']
                log_debug(f"Extracted 'conversation' key from dict, got: {type(conversation)}")
            else:
                # Check if this is a single message object
                if 'message' in data or 'content' in data:
                    log_debug("Found single message object, converting to list")
                    conversation = [data]
                else:
                    log_debug(f"No 'conversation' key found in dict, using full dict: {list(data.keys())[:5]}")
                    conversation = data
        else:
            conversation = data
            
        # At this point, conversation should be a list
        if not isinstance(conversation, list):
            log_error(f"Conversation data is not a list: {type(conversation)}")
            # Try to convert non-list to a list containing the item
            if conversation is not None:
                log_debug("Attempting to convert non-list to list")
                conversation = [conversation]
            else:
                return []
            
        # Ensure each item in the list is a dictionary with at least a 'message' or 'content' key
        valid_messages = []
        for i, item in enumerate(conversation):
            if isinstance(item, dict):
                if 'message' in item or 'content' in item:
                    valid_messages.append(item)
                else:
                    log_debug(f"Skipping item {i} - missing 'message' or 'content' key: {list(item.keys())[:5] if item else None}")
            else:
                # If it's a string, try to wrap it as a message
                if isinstance(item, str):
                    log_debug(f"Converting string item to message object at index {i}")
                    valid_messages.append({"message": item, "role": "unknown"})
                else:
                    log_debug(f"Skipping invalid conversation item at index {i}: {type(item)}")
        
        log_debug(f"Successfully extracted {len(valid_messages)} valid messages")
        return valid_messages
    
    except Exception as e:
        log_error(f"Error extracting conversation: {e}")
        import traceback
        log_debug(traceback.format_exc())
        return []

def format_conversation(messages: List[Dict[str, Any]]) -> str:
    """
    Format a conversation for display.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Formatted string representation of the conversation
    """
    formatted = []
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('message', msg.get('content', ''))
        formatted.append(f"**{role.capitalize()}**: {content}")
        
    return "\n\n".join(formatted)
