import os
import json
import pytest
from unittest.mock import patch, MagicMock
from lambda_function import (
    lambda_handler,
    analyze_issue_with_llm,
    get_github_issues,
    create_github_issue,
    AnalysisResponse,
    GithubIssue,
    PromptChange
)

@pytest.fixture
def mock_github_response():
    return {
        "id": 1,
        "html_url": "https://github.com/test/repo/issues/1",
        "title": "Test Issue",
        "body": "Test Body",
        "number": 1
    }

@pytest.fixture
def mock_llm_response():
    return {
        "github_issues": [
            {
                "title": "Test Issue",
                "body": "Test Body",
                "labels": ["fix-me"]
            }
        ],
        "prompt_changes": [
            {
                "ref": "test-prompt",
                "version": "1.0",
                "content": "Updated content",
                "reason": "Test reason"
            }
        ]
    }

def test_analyze_issue_with_llm(mock_llm_response):
    with patch("lambda_function.run_completion_with_fallback", return_value=mock_llm_response):
        result = analyze_issue_with_llm("test issue", "test context")
        assert isinstance(result, AnalysisResponse)
        assert len(result.github_issues) == 1
        assert len(result.prompt_changes) == 1
        assert result.github_issues[0].title == "Test Issue"
        assert result.prompt_changes[0].ref == "test-prompt"

def test_get_github_issues(mock_github_response):
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [mock_github_response],
            headers={}
        )
        result = get_github_issues("test-token", "test/repo")
        assert len(result) == 1
        assert result[0]["html_url"] == mock_github_response["html_url"]

def test_create_github_issue(mock_github_response):
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=201,
            json=lambda: mock_github_response
        )
        result = create_github_issue(
            "test-token",
            "test/repo",
            "Test Issue",
            "Test Body",
            ["fix-me"]
        )
        assert result["html_url"] == mock_github_response["html_url"]

def test_lambda_handler_success(mock_github_response, mock_llm_response):
    event = {
        "issue_content": "test issue",
        "github_token": "test-token",
        "repo": "test/repo",
        "context": "test context"
    }

    with patch("lambda_function.get_github_issues", return_value=[mock_github_response]), \
         patch("lambda_function.analyze_issue_with_llm", return_value=AnalysisResponse(**mock_llm_response)), \
         patch("lambda_function.create_github_issue", return_value=mock_github_response):

        result = lambda_handler(event, None)
        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert "created_issues" in body
        assert "prompt_changes" in body
        assert len(body["created_issues"]) == 1
        assert len(body["prompt_changes"]) == 1

def test_lambda_handler_missing_params():
    event = {
        "issue_content": "test issue",
        # Missing required params
    }
    result = lambda_handler(event, None)
    assert result["statusCode"] == 400
    assert "error" in json.loads(result["body"])

def test_lambda_handler_error():
    event = {
        "issue_content": "test issue",
        "github_token": "test-token",
        "repo": "test/repo"
    }
    with patch("lambda_function.get_github_issues", side_effect=Exception("Test error")):
        result = lambda_handler(event, None)
        assert result["statusCode"] == 500
        assert "error" in json.loads(result["body"])
