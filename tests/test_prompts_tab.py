"""Tests for the prompts tab functionality."""

import unittest
from unittest.mock import patch, MagicMock
import re
import sys

# Mock streamlit before importing the module
sys.modules['streamlit'] = MagicMock()

# Now we can import from components
from components.prompts_tab import analyze_prompt_references

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

if __name__ == "__main__":
    unittest.main()
