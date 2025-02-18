import pytest
from unittest.mock import MagicMock, patch
from app import get_recent_evals

@patch('app.client')
def test_get_recent_evals(mock_client):
    # Mock data
    mock_calls = []
    for i in range(5):
        mock_call = MagicMock()
        mock_call.output = {
            "scores": {
                "failure_reasons": [f"Error {i}"],
                "attempts": i + 1,
                "successes": i,
                "num_turns": i + 2
            }
        }
        mock_call.inputs = {
            "example": {
                "options": {"type": "test_type"},
                "stream_key": f"stream_{i}"
            }
        }
        mock_calls.append(mock_call)

    # Configure mock
    mock_client.get_calls.return_value = mock_calls

    # Test getting recent evaluations
    evals = get_recent_evals(5)

    # Check that we get a list
    assert isinstance(evals, list)

    # Check that we don't get more than 5 evaluations
    assert len(evals) <= 5

    # Check the structure of each evaluation
    for i, eval in enumerate(evals):
        assert isinstance(eval, dict)
        assert "failure_reasons" in eval
        assert "type" in eval
        assert "stream_key" in eval
        assert "attempts" in eval
        assert "successes" in eval
        assert "num_turns" in eval

        # Check that failure_reasons is a list
        assert isinstance(eval["failure_reasons"], list)
        assert eval["failure_reasons"] == [f"Error {i}"]

        # Check numeric fields are integers
        assert isinstance(eval["attempts"], int)
        assert isinstance(eval["successes"], int)
        assert isinstance(eval["num_turns"], int)
        assert eval["attempts"] == i + 1
        assert eval["successes"] == i
        assert eval["num_turns"] == i + 2

        # Check string fields
        assert eval["type"] == "test_type"
        assert eval["stream_key"] == f"stream_{i}"
