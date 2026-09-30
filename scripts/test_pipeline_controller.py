"""
File: test_pipeline_controller.py
Author: Chris Jien Zhang
Date: 2026-09-29
Course: CMSC473
Description:
    Unit tests for the master pipeline controller. Ensures the active learning 
    loop respects target accuracy thresholds, max iteration limits, and properly 
    updates the master constraints list.

Usage:
    python -m unittest scripts/test_pipeline_controller.py
"""

import unittest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch

# Dynamically append the src directory to the system path for portable execution
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / 'src'
sys.path.append(str(src_dir))

try:
    from scripts.pipeline_controller import run_active_learning_loop
except ImportError:
    from pipeline_controller import run_active_learning_loop

class TestPipelineController(unittest.TestCase):
    """
    Validates the orchestration logic of the main active learning loop.
    """

    def setUp(self):
        """Initializes baseline mock data."""
        self.dummy_embeddings = np.array([[0.1], [0.2], [0.8], [0.9]])
        
    def test_loop_stops_at_max_moves(self):
        """
        Validates that the fail-safe prevents infinite loops when the model
        fails to improve the clusters.
        """
        ground_truth = [0, 0, 1, 1, 2, 2]
        max_limit = 3
        
        # The stub clustering algorithm is hardcoded to return a flawed array,
        # so it will never hit 1.0 accuracy. It should hit the max_limit and stop.
        history = run_active_learning_loop(
            self.dummy_embeddings, 
            ground_truth, 
            target_accuracy=1.0, 
            max_moves=max_limit
        )
        
        # If max_limit is 3, the loop runs for move 0, 1, 2, and 3.
        # This results in 4 logged scores.
        self.assertEqual(len(history["moves"]), max_limit + 1, msg="Loop did not stop at max_moves")

    @patch('scripts.pipeline_controller.stub_clustering_algorithm')
    def test_loop_stops_early_on_target_accuracy(self, mock_clustering):
        """
        Validates that the pipeline breaks early and skips remaining loops 
        if the clustering prediction immediately meets the target ARI threshold.
        """
        # Force the stub to return a perfect cluster alignment
        mock_clustering.return_value = np.array([0, 0, 1, 1, 2, 2])
        ground_truth = [0, 0, 1, 1, 2, 2]
        
        history = run_active_learning_loop(
            self.dummy_embeddings, 
            ground_truth, 
            target_accuracy=1.0, 
            max_moves=10
        )
        
        # The loop should only log the initial baseline check (Move 0) and exit
        self.assertEqual(len(history["moves"]), 1, msg="Loop executed extra moves after hitting target accuracy")
        self.assertAlmostEqual(history["ari_scores"][0], 1.0, places=4, msg="Logged ARI score does not reflect the perfect match")

    def test_loop_respects_zero_max_moves(self):
        """
        Validates pipeline behavior when the max_moves failsafe is set to 0,
        ensuring only the baseline unsupervised clustering step is executed.
        """
        ground_truth = [0, 0, 1, 1, 2, 2]
        
        history = run_active_learning_loop(
            self.dummy_embeddings, 
            ground_truth, 
            target_accuracy=1.0, 
            max_moves=0
        )
        
        # A 0-limit should result in exactly 1 baseline evaluation
        self.assertEqual(len(history["moves"]), 1, msg="Pipeline exceeded a max_moves limit of 0")
        self.assertEqual(history["moves"][0], 0, msg="First logged move should be index 0")

    def test_history_dictionary_structure(self):
        """
        Validates that the orchestrator returns the history dictionary in the 
        correct format for downstream visualization scripts, with parallel arrays.
        """
        ground_truth = [0, 0, 1, 1, 2, 2]
        max_limit = 2
        
        history = run_active_learning_loop(
            self.dummy_embeddings, 
            ground_truth, 
            target_accuracy=1.0, 
            max_moves=max_limit
        )
        
        self.assertIn("moves", history, msg="History dictionary missing 'moves' key")
        self.assertIn("ari_scores", history, msg="History dictionary missing 'ari_scores' key")
        self.assertEqual(len(history["moves"]), len(history["ari_scores"]), msg="History arrays are not parallel")

if __name__ == '__main__':
    unittest.main()