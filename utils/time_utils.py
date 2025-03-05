"""Utilities for working with timestamps and time zones."""

import datetime
import pytz
from typing import Union, Optional

def convert_to_est(timestamp: Union[datetime.datetime, str, int, float]) -> datetime.datetime:
    """
    Convert a timestamp to Eastern Standard Time (EST/EDT).
    
    Args:
        timestamp: Input timestamp, can be a datetime object, string ISO format, 
                  or epoch timestamp (float/int)
    
    Returns:
        The timestamp converted to EST timezone
    """
    eastern_tz = pytz.timezone('US/Eastern')
    
    # Handle different input types
    if isinstance(timestamp, datetime.datetime):
        # If the timestamp has a timezone, convert it to EST
        if timestamp.tzinfo is not None:
            return timestamp.astimezone(eastern_tz)
        # If no timezone (naive), assume it's UTC and convert to EST
        utc_time = pytz.utc.localize(timestamp)
        return utc_time.astimezone(eastern_tz)
    
    elif isinstance(timestamp, (int, float)):
        # Assume epoch timestamp is in UTC
        utc_time = datetime.datetime.fromtimestamp(timestamp, pytz.utc)
        return utc_time.astimezone(eastern_tz)
    
    elif isinstance(timestamp, str):
        # Try to parse ISO format string
        try:
            # Parse with timezone if present
            dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = pytz.utc.localize(dt)
            return dt.astimezone(eastern_tz)
        except (ValueError, TypeError):
            # Fall back to assuming it's UTC if parsing fails
            try:
                dt = datetime.datetime.fromisoformat(timestamp)
                utc_time = pytz.utc.localize(dt)
                return utc_time.astimezone(eastern_tz)
            except (ValueError, TypeError):
                raise ValueError(f"Could not parse timestamp: {timestamp}")
    
    raise TypeError(f"Unsupported timestamp type: {type(timestamp)}")

def format_est_time(timestamp: Union[datetime.datetime, str, int, float], 
                   format_str: str = "%Y-%m-%d %I:%M:%S %p %Z") -> str:
    """
    Convert a timestamp to EST and format it as a string.
    
    Args:
        timestamp: Input timestamp to convert
        format_str: Optional string format (defaults to "YYYY-MM-DD HH:MM:SS AM/PM TZ")
    
    Returns:
        Formatted timestamp string in EST timezone
    """
    est_time = convert_to_est(timestamp)
    return est_time.strftime(format_str)

def get_current_est_time() -> datetime.datetime:
    """
    Get the current time in EST timezone.
    
    Returns:
        Current datetime in EST timezone
    """
    eastern_tz = pytz.timezone('US/Eastern')
    return datetime.datetime.now(eastern_tz)

def get_current_est_time_formatted(format_str: str = "%Y-%m-%d %I:%M:%S %p %Z") -> str:
    """
    Get the current time in EST timezone as a formatted string.
    
    Args:
        format_str: Optional string format (defaults to "YYYY-MM-DD HH:MM:SS AM/PM TZ")
    
    Returns:
        Current time formatted string in EST timezone
    """
    return get_current_est_time().strftime(format_str)
