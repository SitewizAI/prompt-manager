# This is a hypothetical file - modify the actual file where the error occurs

from .conversation_utils import extract_conversation, format_conversation
from .logging_utils import log_debug, log_error

def display_conversation(conversation_data):
    """
    Display a conversation after properly parsing it.
    
    Args:
        conversation_data: Conversation data in various formats
    
    Returns:
        Formatted conversation string
    """
    try:
        log_debug(f"Attempting to display conversation of type: {type(conversation_data)}")
        
        # First extract the conversation messages using our utility
        messages = extract_conversation(conversation_data)
        
        if not messages:
            log_debug("No valid messages found in conversation data")
            # Try one more approach - if it's a string and not JSON, display it directly
            if isinstance(conversation_data, str) and not conversation_data.strip().startswith('{'):
                return conversation_data
            return "No valid conversation messages found."
        
        # Format the messages for display
        formatted = format_conversation(messages)
        log_debug(f"Successfully formatted {len(messages)} messages")
        return formatted
        
    except Exception as e:
        log_error(f"Error displaying conversation: {e}")
        import traceback
        log_debug(traceback.format_exc())
        return f"Failed to display conversation: {str(e)}"
