"""
Utils package for Prompt Manager.
Provides utilities for prompt management, validation, DynamoDB interaction, etc.
"""

# Import and re-export key functions to maintain backward compatibility
from .logging_utils import log_debug, log_error, ToolMessageTracker, measure_time
from .prompt_utils import (
    get_prompt_from_dynamodb, 
    update_prompt, 
    get_all_prompts,
    get_all_prompt_versions,
    get_prompt_expected_parameters,
    validate_prompt_parameters,
    get_available_prompt_dates,
    get_prompt_versions_by_date,
    revert_prompts_to_date,
    PROMPT_TYPES,
)
from .completion_utils import (
    run_completion_with_fallback,
    initialize_vertex_ai,
    SYSTEM_PROMPT,
    PROMPT_INSTRUCTIONS,
    count_tokens
)
from .db_utils import (
    get_boto3_resource,
    get_boto3_client, 
    get_api_key,
    get_dynamodb_table,
    convert_decimal,
    parallel_dynamodb_query,
)
from .metrics_utils import (
    get_evaluation_metrics,
    get_daily_metrics_from_table,
    get_most_recent_stream_key,
    get_all_evaluations,
    get_stream_evaluations,
    get_recent_evaluations,
    get_evaluation_by_timestamp,
    get_conversation_history
)
from .context_utils import (
    get_context,
    get_data,
    okr_to_markdown,
    insight_to_markdown,
    suggestion_to_markdown,
    get_prompts
)
from .github_utils import (
    get_github_files,
    get_file_contents,
    get_github_project_issues,
    get_github_files_async,
    create_github_issue_with_project,
    fetch_and_cache_code_files,
)
from .validation_utils import (
    validate_prompt_format,
    validate_question_objects_with_documents,
    get_document_structure
)
