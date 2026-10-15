import unittest
import os
from pathlib import Path
from src.config import BASE_DIR, MODEL_PATH, SCALER_PATH, FEATURES_PATH, SAMPLE_DATA_PATH, CHURN_THRESHOLD

class TestConfig(unittest.TestCase):
    def test_paths_exist(self):
        """Ensure critical project artifact files exist."""
        self.assertTrue(os.path.exists(MODEL_PATH), f"Model path not found: {MODEL_PATH}")
        self.assertTrue(os.path.exists(SCALER_PATH), f"Scaler path not found: {SCALER_PATH}")
        self.assertTrue(os.path.exists(FEATURES_PATH), f"Features path not found: {FEATURES_PATH}")
        self.assertTrue(os.path.exists(SAMPLE_DATA_PATH), f"Sample data not found: {SAMPLE_DATA_PATH}")

    def test_threshold_bounds(self):
        """Ensure churn threshold is a valid probability between 0 and 1."""
        self.assertIsInstance(CHURN_THRESHOLD, float)
        self.assertGreaterEqual(CHURN_THRESHOLD, 0.0)
        self.assertLessEqual(CHURN_THRESHOLD, 1.0)

if __name__ == "__main__":
    unittest.main()
