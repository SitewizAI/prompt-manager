"""Tests for the prompts tab functionality."""

import unittest
from unittest.mock import patch, MagicMock, call
import re
import sys
from datetime import datetime

# Mock streamlit before importing the module
sys.modules['streamlit'] = MagicMock()

# Now we can import from components
from components.prompts_tab import analyze_prompt_references
from utils.prompt_utils import get_prompts_by_date, revert_all_prompts_to_date

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
    @patch('utils.prompt_utils.get_all_prompt_versions')
    def test_get_prompts_by_date(self, mock_get_all_versions, mock_get_table):
        """Test that prompts can be retrieved by date."""
        # Mock the DynamoDB table and scan response
        mock_table = MagicMock()
        mock_get_table.return_value = mock_table

        # Mock scan response with refs
        mock_table.scan.return_value = {
            'Items': [
                {'ref': 'prompt1'},
                {'ref': 'prompt2'},
                {'ref': 'prompt3'}
            ]
        }

        # Mock versions for each prompt
        test_date = "2023-01-15"
        target_date = f"{test_date}T23:59:59.999999"

        # Create test versions with different dates
        prompt1_versions = [
            {'ref': 'prompt1', 'version': 2, 'updatedAt': '2023-01-20T12:00:00', 'content': 'v2 content'},
            {'ref': 'prompt1', 'version': 1, 'updatedAt': '2023-01-10T12:00:00', 'content': 'v1 content'},
            {'ref': 'prompt1', 'version': 0, 'updatedAt': '2023-01-01T12:00:00', 'content': 'v0 content'}
        ]

        prompt2_versions = [
            {'ref': 'prompt2', 'version': 3, 'updatedAt': '2023-01-25T12:00:00', 'content': 'v3 content'},
            {'ref': 'prompt2', 'version': 2, 'updatedAt': '2023-01-18T12:00:00', 'content': 'v2 content'},
            {'ref': 'prompt2', 'version': 1, 'updatedAt': '2023-01-05T12:00:00', 'content': 'v1 content'}
        ]

        prompt3_versions = [
            {'ref': 'prompt3', 'version': 1, 'updatedAt': '2023-01-12T12:00:00', 'content': 'v1 content'},
            {'ref': 'prompt3', 'version': 0, 'updatedAt': '2022-12-01T12:00:00', 'content': 'v0 content'}
        ]

        # Configure mock to return different versions for each prompt
        def get_versions_side_effect(ref):
            if ref == 'prompt1':
                return prompt1_versions
            elif ref == 'prompt2':
                return prompt2_versions
            elif ref == 'prompt3':
                return prompt3_versions
            return []

        mock_get_all_versions.side_effect = get_versions_side_effect

        # Call the function
        result = get_prompts_by_date(test_date)

        # Verify the results
        self.assertEqual(len(result), 3)

        # For prompt1, should get version 1 (from 2023-01-10)
        self.assertEqual(result['prompt1']['version'], 1)
        self.assertEqual(result['prompt1']['updatedAt'], '2023-01-10T12:00:00')

        # For prompt2, should get version 1 (from 2023-01-05)
        self.assertEqual(result['prompt2']['version'], 1)
        self.assertEqual(result['prompt2']['updatedAt'], '2023-01-05T12:00:00')

        # For prompt3, should get version 1 (from 2023-01-12)
        self.assertEqual(result['prompt3']['version'], 1)
        self.assertEqual(result['prompt3']['updatedAt'], '2023-01-12T12:00:00')

        # Verify the correct calls were made
        mock_get_table.assert_called_once_with('PromptsTable')
        mock_table.scan.assert_called_once()
        self.assertEqual(mock_get_all_versions.call_count, 3)
        mock_get_all_versions.assert_has_calls([
            call('prompt1'),
            call('prompt2'),
            call('prompt3')
        ], any_order=True)

    @patch('utils.prompt_utils.get_prompts_by_date')
    @patch('utils.prompt_utils.update_prompt')
    def test_revert_all_prompts_to_date(self, mock_update_prompt, mock_get_prompts_by_date):
        """Test that all prompts can be reverted to a specific date."""
        # Mock the prompts by date
        test_date = "2023-01-15"
        mock_get_prompts_by_date.return_value = {
            'prompt1': {'ref': 'prompt1', 'content': 'content1', 'is_object': False},
            'prompt2': {'ref': 'prompt2', 'content': '{"key": "value"}', 'is_object': True},
            'prompt3': {'ref': 'prompt3', 'content': 'content3', 'is_object': False}
        }

        # Mock update_prompt to succeed for prompt1 and prompt3, fail for prompt2
        def update_side_effect(ref, content):
            if ref == 'prompt2':
                return False, "Error updating prompt2"
            return True, None

        mock_update_prompt.side_effect = update_side_effect

        # Call the function
        success, successful_refs, failed_refs = revert_all_prompts_to_date(test_date)

        # Verify the results
        self.assertTrue(success)
        self.assertEqual(len(successful_refs), 2)
        self.assertEqual(len(failed_refs), 1)
        self.assertIn('prompt1', successful_refs)
        self.assertIn('prompt3', successful_refs)
        self.assertIn('prompt2', failed_refs[0])

        # Verify the correct calls were made
        mock_get_prompts_by_date.assert_called_once_with(test_date)
        self.assertEqual(mock_update_prompt.call_count, 3)
        mock_update_prompt.assert_has_calls([
            call('prompt1', 'content1'),
            call('prompt2', {'key': 'value'}),
            call('prompt3', 'content3')
        ], any_order=True)

if __name__ == "__main__":
    unittest.main()
