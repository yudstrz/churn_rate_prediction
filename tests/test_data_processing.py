import unittest

class TestDataProcessingLogic(unittest.TestCase):
    def test_check_missing_columns(self):
        """Test missing column detection logic."""
        features = ["feat_a", "feat_b", "feat_c"]
        
        # Test case: all columns present
        class MockDF:
            columns = ["feat_a", "feat_b", "feat_c", "extra_col"]
            
        missing = set(features) - set(MockDF.columns)
        self.assertEqual(len(missing), 0)

        # Test case: missing columns
        class MockDFIncomplete:
            columns = ["feat_a"]
            
        missing = set(features) - set(MockDFIncomplete.columns)
        self.assertEqual(missing, {"feat_b", "feat_c"})

    def test_preprocessing_categorical_mapping(self):
        """Test yes/no binary encoding mapping logic."""
        mapping = {"yes": 1, "no": 0}
        self.assertEqual(mapping.get("yes"), 1)
        self.assertEqual(mapping.get("no"), 0)
        self.assertIsNone(mapping.get("maybe"))

if __name__ == "__main__":
    unittest.main()
