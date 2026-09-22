"""
File: test_simulator.py
Author: Chris Jien Zhang
Date: 2026-09-22
Course: CMSC473
Description:
    Comprehensive unit test suite for the SyntheticUser class. 
    Validates cluster evaluation accuracy, constraint generation logic, 
    and metric tracking. Designed to be portable and executable on any OS.

Usage:
    Run directly from the terminal at the repository root:
    python -m unittest test_simulator.py
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Dynamically append the src directory to the system path for portable execution
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / 'src'
sys.path.append(str(src_dir))

# Attempt import; will fail gracefully if the path is incorrect
try:
    from simulation_suite.simulator import SyntheticUser
except ImportError:
    # Fallback for flat directory structures during initial testing
    from simulator import SyntheticUser

class TestSyntheticUser(unittest.TestCase):
    """
    Unit tests ensuring the simulation engine correctly evaluates clusters
    and generates valid constraints under various data conditions.
    """

    def setUp(self):
        """
        Initializes baseline mock data before every test.
        """
        self.perfect_labels = [0, 0, 1, 1, 2, 2]
        self.user = SyntheticUser(self.perfect_labels)

    def test_evaluate_clusters_perfect_match(self):
        """
        Validates that identical cluster assignments yield a perfect ARI score of 1.0.
        """
        predicted = [0, 0, 1, 1, 2, 2]
        ari = self.user.evaluate_clusters(predicted)
        self.assertAlmostEqual(ari, 1.0, places=4, msg="Perfect match should yield ARI of 1.0")

    def test_evaluate_clusters_shifted_labels(self):
        """
        Validates that ARI relies on groupings, not specific integer values.
        """
        predicted_shifted = [2, 2, 0, 0, 1, 1]
        ari = self.user.evaluate_clusters(predicted_shifted)
        self.assertAlmostEqual(ari, 1.0, places=4, msg="Shifted labels with identical groupings should yield ARI of 1.0")

    def test_evaluate_clusters_poor_match(self):
        """
        Validates that completely incorrect groupings lower the ARI score.
        """
        predicted_poor = [0, 1, 0, 1, 0, 1]
        ari = self.user.evaluate_clusters(predicted_poor)
        self.assertLess(ari, 0.5, msg="Poor clustering should yield a low ARI score")

    def test_generate_next_move_finds_error(self):
        """
        Validates the engine successfully locates a separated item and generates a constraint.
        """
        # The item at index 1 belongs in cluster 0, but was placed in cluster 1
        flawed_prediction = [0, 1, 2, 2, 3, 3]
        move = self.user.generate_next_move(flawed_prediction)
        
        self.assertIsNotNone(move, msg="Engine failed to find an existing error")
        self.assertIsInstance(move, tuple, msg="Constraint must be returned as a tuple")
        self.assertEqual(len(move), 2, msg="Constraint tuple must contain exactly two indices")
        self.assertEqual(move, (0, 1), msg="Engine did not return the expected constraint indices")

    def test_generate_next_move_perfect_clusters(self):
        """
        Validates the engine returns None when no further corrections are needed.
        """
        move = self.user.generate_next_move(self.perfect_labels)
        self.assertIsNone(move, msg="Perfect clusters should not generate a constraint")

    def test_history_logging(self):
        """
        Validates that moves and ARI scores are logged correctly for graph generation.
        """
        flawed_prediction = [0, 1, 2, 2, 3, 3]
        
        self.user.evaluate_clusters(flawed_prediction)
        self.user.generate_next_move(flawed_prediction)
        
        self.assertEqual(len(self.user.history["moves"]), 1)
        self.assertEqual(len(self.user.history["ari_scores"]), 1)
        self.assertEqual(self.user.history["moves"][0], 0)

if __name__ == '__main__':
    unittest.main()