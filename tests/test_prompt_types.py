"""Test for prompt types functionality."""

import unittest
import sys
import os

# Add the parent directory to the path so we can import the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.prompt_utils import PROMPT_TYPES

class TestPromptTypes(unittest.TestCase):
    """Test the prompt types functionality."""

    def test_prompt_types_structure(self):
        """Test that the prompt types dictionary has the expected structure."""
        # Check that all expected types are present
        expected_types = ["all", "okr", "insights", "suggestion", "design", "code"]
        for type_name in expected_types:
            self.assertIn(type_name, PROMPT_TYPES, f"Expected type '{type_name}' not found in PROMPT_TYPES")

        # Check that each type (except 'all') has a non-empty list of refs
        for type_name, refs in PROMPT_TYPES.items():
            if type_name != "all":  # 'all' can be empty initially
                self.assertIsInstance(refs, list, f"Expected refs for '{type_name}' to be a list")
                self.assertTrue(len(refs) > 0, f"Expected refs for '{type_name}' to be non-empty")

                # Check that each ref follows the expected naming pattern
                for ref in refs:
                    self.assertTrue(
                        ref.startswith(type_name) or ref.endswith(type_name),
                        f"Expected ref '{ref}' to start or end with '{type_name}'"
                    )

if __name__ == "__main__":
    unittest.main()
