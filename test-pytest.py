import pytest
from moveobjectsim import Move2Object
from classrobot.point3d import Point3D

@pytest.fixture
def sample_points():
    # mix of real boxes (ID<100) and one “empty” marker (ID=101)
    return [
        {'id': 13,  'point': Point3D(x=0.057,  y=-0.069, z=1.455)},
        {'id':  1,  'point': Point3D(x=0.348,  y= 0.068, z=1.445)},
        {'id':  2,  'point': Point3D(x=0.435,  y=-0.048, z=1.455)},
        {'id':101,  'point': Point3D(x=0.036,  y=-0.442, z=1.574)},  # should be ignored
        {'id':  3,  'point': Point3D(x=-0.152, y=-0.062, z=1.455)},
        {'id': 11,  'point': Point3D(x=-0.344, y=-0.089, z=1.505)},
        {'id': 14,  'point': Point3D(x=0.236,  y=-0.049, z=1.494)},
    ]

def test_detect_overlaps_all_real_boxes_grouped(sample_points):
    mover = Move2Object()
    groups = mover.detect_overlaps(sample_points)
    # Should only find one group of all real-box IDs, since all |Δy|<0.3
    assert len(groups) == 1

    group_ids = sorted(m['id'] for m in groups[0])
    assert group_ids == [1, 2, 3, 11, 13, 14]

def test_detect_overlaps_ignores_empty_markers(sample_points):
    mover = Move2Object()
    groups = mover.detect_overlaps(sample_points)
    # Ensure ID 101 never appears in any group
    for group in groups:
        assert 101 not in [m['id'] for m in group]