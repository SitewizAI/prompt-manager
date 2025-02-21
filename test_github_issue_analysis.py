import pytest
from unittest.mock import patch, MagicMock
import json

def test_github_issue_analysis():
    # Test data
    context = """
    Current OKRs:
    Improve website performance by 20%

    Recent Insights:
    Page load time increased by 5 seconds

    Recent Suggestions:
    Optimize image loading

    Code Suggestions:
    Implement lazy loading for images
    """

    prompt = "The website is loading very slowly, especially the images. Can you help?"

    # Mock response data
    mock_response_data = {
        "should_create_issue": True,
        "issue": {
            "title": "Performance Issue: Slow Image Loading",
            "description": "Website performance is degraded due to slow image loading times. Current page load time has increased by 5 seconds.",
            "labels": ["performance", "optimization"],
            "priority": "high"
        }
    }

    # Create mock response object
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(mock_response_data)

    # Mock the litellm.completion function
    with patch('litellm.completion', return_value=mock_response):
        from app import litellm

        # Test the issue analysis
        issue_response = litellm.completion(
            model="litellm_proxy/gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI that analyzes user questions and determines if they should be GitHub issues. If yes, provide a structured response with issue details."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnalyze if this should be a GitHub issue and if so, provide details in JSON format with title, description, labels, and priority fields."}
            ],
            response_format={
                "type": "object",
                "properties": {
                    "should_create_issue": {"type": "boolean"},
                    "issue": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "labels": {"type": "array", "items": {"type": "string"}},
                            "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                        }
                    }
                }
            }
        )

        issue_data = json.loads(issue_response.choices[0].message.content)

        # Assertions
        assert isinstance(issue_data, dict)
        assert "should_create_issue" in issue_data
        assert isinstance(issue_data["should_create_issue"], bool)
        assert issue_data["should_create_issue"] is True

        assert "issue" in issue_data
        issue = issue_data["issue"]
        assert "title" in issue
        assert "description" in issue
        assert "labels" in issue
        assert "priority" in issue
        assert isinstance(issue["labels"], list)
        assert issue["priority"] in ["low", "medium", "high"]

        # Verify specific mock data
        assert issue["title"] == "Performance Issue: Slow Image Loading"
        assert "slow image loading" in issue["description"].lower()
        assert "performance" in issue["labels"]
        assert issue["priority"] == "high"
