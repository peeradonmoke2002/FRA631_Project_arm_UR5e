#!/usr/bin/env python3
"""
test_move2object.py

Unit tests for the Move2Object class functions that do not require hardware.
We test the transformation and reordering functionalities.
"""

import unittest
import numpy as np
from classrobot.point3d import Point3D
from FRA631_Project_Dual_arm_UR5_planing.moveobject import Move2Object

class DummyMove2Object(Move2Object):
    """
    A dummy subclass of Move2Object that bypasses the hardware-dependent
    initializations. This allows testing of pure functions.
    """
    def __init__(self):
        # Do not initialize hardware connections.
        # Instead, set only the attributes required for testing.
        self.overlap_threshold = 0.05
        self.Test_RPY = [-1, 0, 1]
        # Use a dummy transformation matrix – identity matrix for testing.
        self.best_matrix = np.eye(4)
        # Other attributes may be set to dummy values if needed.
    
    def load_matrix(self):
        """Override load_matrix to return an identity matrix."""
        return np.eye(4)
    
    # Override hardware-dependent functions to do nothing.
    def robot_moveL(self, pose, speed):
        pass

    def move_home(self):
        pass

    def stop_all(self):
        pass


class TestMove2Object(unittest.TestCase):
    def setUp(self):
        # Create a dummy instance that bypasses hardware initialization.
        self.move_object = DummyMove2Object()

    def test_transform_marker_points_identity(self):
        """
        Test transform_marker_points by applying the identity matrix,
        so that the output should equal the input.
        """
        # Create dummy markers with known coordinates.
        markers = [
            {"id": 1, "point": Point3D(0.1, 0.2, 0.3)},
            {"id": 2, "point": Point3D(0.4, 0.5, 0.6)}
        ]
        identity = np.eye(4)
        transformed = self.move_object.transform_marker_points(markers, identity)
        self.assertEqual(len(transformed), len(markers))
        for orig, trans in zip(markers, transformed):
            self.assertEqual(orig["id"], trans["id"])
            self.assertAlmostEqual(orig["point"].x, trans["point"].x, places=6)
            self.assertAlmostEqual(orig["point"].y, trans["point"].y, places=6)
            self.assertAlmostEqual(orig["point"].z, trans["point"].z, places=6)

    def test_reorder_markers_for_overlap(self):
        """
        Test reorder_markers_for_overlap by creating a list of markers where two
        markers are close enough (overlapped) and one is far. The test expects that
        the two overlapped markers are returned first (sorted by increasing y) and then the non-overlapped.
        """
        # Create three markers.
        # Marker 1: located at (0.1, 0.2, 0.3)
        # Marker 2: located very close to Marker 1 at (0.12, 0.21, 0.31)
        # Marker 3: located far from markers 1 & 2 at (1, 1, 1)
        markers = [
            {"id": 1, "point": Point3D(0.1, 0.2, 0.3)},
            {"id": 2, "point": Point3D(0.12, 0.21, 0.31)},
            {"id": 3, "point": Point3D(1, 1, 1)}
        ]
        # Call reorder_markers_for_overlap (which uses self.overlap_threshold=0.05)
        ordered = self.move_object.reorder_markers_for_overlap(markers)
        # We expect that markers 1 and 2 are considered overlapped.
        # Within the overlapped group, markers are sorted by y (ascending):
        # Marker 1 (y = 0.2) should appear before Marker 2 (y = 0.21), then Marker 3.
        expected_order = [1, 2, 3]
        actual_order = [marker["id"] for marker in ordered]
        self.assertEqual(actual_order, expected_order)


if __name__ == "__main__":
    unittest.main()
