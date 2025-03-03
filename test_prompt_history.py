"""Test script for prompt history functionality."""

import json
from utils import (
    get_prompt_historical_performance, 
    get_all_prompt_versions,
    get_daily_metrics_from_table,
    log_debug
)

def test_prompt_history():
    """Test function to verify prompt history retrieval."""
    # Test prompt ref and eval type
    prompt_ref = "okr_questions"
    eval_type = "okr"
    
    # Get historical performance for the prompt
    log_debug(f"Getting historical performance for prompt: {prompt_ref}")
    history = get_prompt_historical_performance(prompt_ref, eval_type, days=7)
    
    print("\n=========== PROMPT HISTORICAL PERFORMANCE ===========")
    print(history)
    
    # Get all versions of the prompt
    log_debug(f"Getting all versions for prompt: {prompt_ref}")
    versions = get_all_prompt_versions(prompt_ref)
    
    print("\n=========== PROMPT VERSION HISTORY ===========")
    print(f"Found {len(versions)} versions")
    
    # Show version details
    for version in versions:
        print(f"\nVersion: {version.get('version')}")
        print(f"Updated: {version.get('updatedAt')}")
        print(f"Is Object: {version.get('is_object')}")
    
    # Get daily metrics
    log_debug(f"Getting daily metrics for: {eval_type}")
    metrics = get_daily_metrics_from_table(eval_type=eval_type, days=7, get_prompts=True)
    
    print("\n=========== DAILY METRICS ===========")
    print(f"Total evaluations: {metrics['total_metrics'].get('total_evaluations')}")
    print(f"Total successes: {metrics['total_metrics'].get('total_successes')}")
    print(f"Success rate: {metrics['total_metrics'].get('success_rate'):.1f}%")
    
    if metrics.get('prompt_versions'):
        print(f"\nFound {len(metrics['prompt_versions'])} stored prompt versions")
        prompt_versions = [v for v in metrics['prompt_versions'] if v.get('ref') == prompt_ref]
        print(f"Of which {len(prompt_versions)} are for {prompt_ref}")
        
        for version in prompt_versions:
            print(f"Date: {version.get('date')}, Version: {version.get('version')}")

if __name__ == "__main__":
    test_prompt_history()
