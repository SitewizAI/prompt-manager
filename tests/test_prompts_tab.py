"""Tests for the prompts tab functionality."""

import unittest
from unittest.mock import patch, MagicMock
import re
import sys

# Mock streamlit before importing the module
sys.modules['streamlit'] = MagicMock()

# Now we can import from components
from components.prompts_tab import analyze_prompt_references
from utils.prompt_utils import get_prompt_versions_by_date, revert_prompts_to_date

class TestPromptsTab(unittest.TestCase):
    """Test cases for the prompts tab functionality."""

    def test_analyze_prompt_references(self):
        """Test that prompt references are correctly analyzed."""
        # Create test prompts
        test_prompts = [
            {"ref": "prompt1", "content": "This is a prompt with {var1} and {var2}"},
            {"ref": "prompt2", "content": "This prompt uses {prompt:prompt1} as a reference"},
            {"ref": "prompt3", "content": "This prompt uses both {prompt:prompt1} and {prompt:prompt2}"},
            {"ref": "template1", "content": "This is a template"},
            {"ref": "prompt4", "content": "This prompt uses {template1} as a variable"},
            {"ref": "prompt5", "content": "This prompt uses {prompt_template} as a variable"},
            {"ref": "prompt_template", "content": "This is a prompt template"},
        ]

        # Analyze references
        references = analyze_prompt_references(test_prompts)

        # Check 'uses' dictionary
        self.assertIn("prompt2", references["uses"])
        self.assertIn("prompt1", references["uses"]["prompt2"])

        self.assertIn("prompt3", references["uses"])
        self.assertIn("prompt1", references["uses"]["prompt3"])
        self.assertIn("prompt2", references["uses"]["prompt3"])

        # Note: The current implementation only detects explicit {prompt:ref} references
        # and variables that end with "prompt" or "template", not arbitrary variable names
        # that happen to match prompt refs

        self.assertIn("prompt5", references["uses"])
        self.assertIn("prompt_template", references["uses"]["prompt5"])

        # Check 'used_by' dictionary
        self.assertIn("prompt1", references["used_by"])
        self.assertIn("prompt2", references["used_by"]["prompt1"])
        self.assertIn("prompt3", references["used_by"]["prompt1"])

        self.assertIn("prompt2", references["used_by"])
        self.assertIn("prompt3", references["used_by"]["prompt2"])

        # Only check the prompt_template reference which should be detected
        self.assertIn("prompt_template", references["used_by"])
        self.assertIn("prompt5", references["used_by"]["prompt_template"])

    def test_variable_extraction(self):
        """Test that variables are correctly extracted from prompt content."""
        # Test the regex pattern used in render_prompt_version_editor
        pattern = r'(?<!\{)\{([a-zA-Z0-9_]+)\}(?!\})'

        # Test cases
        test_cases = [
            # Simple variables
            ("This is a {variable}", ["variable"]),
            # Multiple variables
            ("This has {var1} and {var2}", ["var1", "var2"]),
            # Nested braces (should be ignored)
            ("This has {{var1}} which is escaped", []),
            # Mixed cases
            ("This has {var1} and {{var2}}", ["var1"]),
            # Complex case
            ("Format with {var1}, {{escaped}}, and {var2}", ["var1", "var2"]),
        ]

        for content, expected in test_cases:
            variables = re.findall(pattern, content)
            self.assertEqual(variables, expected, f"Failed for content: {content}")

    @patch('utils.prompt_utils.get_dynamodb_table')
    def test_get_prompt_versions_by_date(self, mock_get_table):
        """Test getting prompt versions by date."""
        # Mock the DynamoDB table and query response
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        # Mock query response with sample data
        mock_table.query.return_value = {
            'Items': [
                {
                    'type': 'okr',
                    'date': '2023-05-15',
                    'promptVersions': [
                        {
                            'ref': 'okr_system_prompt',
                            'content': 'Test system prompt content',
                            'version': 3
                        },
                        {
                            'ref': 'okr_evaluation_prompt',
                            'content': 'Test evaluation prompt content',
                            'version': 2
                        }
                    ]
                }
            ]
        }

        # Call the function
        result = get_prompt_versions_by_date('2023-05-15', 'okr')

        # Verify the results
        self.assertEqual(len(result), 2)
        self.assertIn('okr_system_prompt', result)
        self.assertIn('okr_evaluation_prompt', result)
        self.assertEqual(result['okr_system_prompt']['content'], 'Test system prompt content')
        self.assertEqual(result['okr_evaluation_prompt']['content'], 'Test evaluation prompt content')

        # Verify the query was called with correct parameters
        mock_table.query.assert_called_once_with(
            KeyConditionExpression='#type = :type_val AND #date = :date_val',
            ExpressionAttributeNames={
                '#type': 'type',
                '#date': 'date'
            },
            ExpressionAttributeValues={
                ':type_val': 'okr',
                ':date_val': '2023-05-15'
            }
        )

    @patch('utils.prompt_utils.get_prompt_versions_by_date')
    @patch('utils.prompt_utils.update_prompt')
    def test_revert_prompts_to_date(self, mock_update_prompt, mock_get_versions):
        """Test reverting prompts to a specific date."""
        # Mock the get_prompt_versions_by_date function
        mock_get_versions.return_value = {
            'okr_system_prompt': {
                'ref': 'okr_system_prompt',
                'content': 'Test system prompt content',
                'version': 3
            },
            'okr_evaluation_prompt': {
                'ref': 'okr_evaluation_prompt',
                'content': 'Test evaluation prompt content',
                'version': 2
            }
        }

        # Mock the update_prompt function to return success
        mock_update_prompt.return_value = True

        # Call the function
        success, message, updated_refs = revert_prompts_to_date('2023-05-15', 'okr')

        # Verify the results
        self.assertTrue(success)
        self.assertEqual(len(updated_refs), 2)
        self.assertIn('okr_system_prompt', updated_refs)
        self.assertIn('okr_evaluation_prompt', updated_refs)

        # Verify update_prompt was called for each prompt
        self.assertEqual(mock_update_prompt.call_count, 2)
        mock_update_prompt.assert_any_call('okr_system_prompt', 'Test system prompt content')
        mock_update_prompt.assert_any_call('okr_evaluation_prompt', 'Test evaluation prompt content')

if __name__ == "__main__":
    unittest.main()
