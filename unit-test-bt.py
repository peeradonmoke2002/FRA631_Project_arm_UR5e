# test_move2object_bt.py

import unittest
import numpy as np
from bt import (
    Move2Object,
    build_behavior_tree,
    Point3D
)

# Create a fake version of Move2Object to override hardware calls.
class FakeMove2Object(Move2Object):
    def __init__(self, stacked=True):
        # Do not call super().__init__() because we want to avoid hardware initialization.
        self.calls = []  # Will record method call history.
        self.best_matrix = np.eye(4)
        self.empty_pos = [1.0, 1.0, 1.0]
        self.Test_RPY = [0, 0, 0]
        self.stacked = stacked

    def move_home(self):
        self.calls.append("move_home")

    def cam_relasense(self):
        # Return simulated marker data.
        # For "stacked" scenario, one marker with z < 3.6.
        if self.stacked:
            return [
                {'id': 1, 'point': Point3D(0, 0, 3.0)},  # This marker is "stacked"
                {'id': 2, 'point': Point3D(1, 1, 4.0)}
            ]
        else:
            # No markers with z below 3.6.
            return [
                {'id': 1, 'point': Point3D(0, 0, 4.0)},
                {'id': 2, 'point': Point3D(1, 1, 4.5)}
            ]

    def transform_marker_points(self, marker_points, transformation_matrix):
        # For testing purposes, assume markers are already in target coordinates.
        return marker_points

    def check_box_is_stack(self, transformed_points):
        # Return True if any marker point's z is below 3.6.
        if self.stacked:
            return any(marker['point'].z < 3.6 for marker in transformed_points)
        return False

    def pick_box(self, marker):
        self.calls.append(f"pick_box {marker['id']}")

    def empyty_pose(self):
        # Return a dummy empty position.
        return [1.0, 1.0, 1.0]

    def place_box_at(self, point):
        # Record the call with the point's coordinates.
        self.calls.append(f"place_box_at {point.x},{point.y},{point.z}")

    def sort_pick_and_place(self):
        self.calls.append("sort_pick_and_place")


class TestBehaviorTree(unittest.TestCase):
    def test_stacked_box(self):
        """
        In this test, we simulate a stacked box scenario.
        A marker with id=1 has z=3.0 (< 3.6). The expected behavior is:
          1. move_home is called.
          2. Markers are detected.
          3. Check-box-is-stacked condition is true.
          4. pick_box is called with marker id 1.
          5. place_box_at is called with the empty pose.
          6. The sort-pick-and-place action is NOT called.
        """
        fake_obj = FakeMove2Object(stacked=True)
        bt_tree = build_behavior_tree(fake_obj)
        blackboard = {}
        status = bt_tree.tick(blackboard)

        self.assertEqual(status, "SUCCESS")
        # Verify move_home was called.
        self.assertIn("move_home", fake_obj.calls)
        # Verify that pick_box is called with marker id 1.
        self.assertIn("pick_box 1", fake_obj.calls)
        # Verify that place_box_at is called with the expected empty position: 1.0,1.0,1.0
        self.assertIn("place_box_at 1.0,1.0,1.0", fake_obj.calls)
        # Ensure sort_pick_and_place is not called.
        self.assertNotIn("sort_pick_and_place", fake_obj.calls)

    def test_no_stacked_box(self):
        """
        In this test, we simulate a scenario with no stacked box.
        All markers have a z value >= 3.6.
        The expected behavior is:
          1. move_home is called.
          2. Markers are detected.
          3. Check-box-is-stacked condition is false.
          4. The sort_pick_and_place action is executed.
        """
        fake_obj = FakeMove2Object(stacked=False)
        bt_tree = build_behavior_tree(fake_obj)
        blackboard = {}
        status = bt_tree.tick(blackboard)

        self.assertEqual(status, "SUCCESS")
        # Verify move_home was called.
        self.assertIn("move_home", fake_obj.calls)
        # Verify that sort_pick_and_place is called.
        self.assertIn("sort_pick_and_place", fake_obj.calls)
        # Ensure that neither pick_box nor place_box_at are called.
        combined_calls = " ".join(fake_obj.calls)
        self.assertNotIn("pick_box", combined_calls)
        self.assertNotIn("place_box_at", combined_calls)


if __name__ == "__main__":
    unittest.main()
