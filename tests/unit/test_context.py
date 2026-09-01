import numpy as np

from src.environment.context import ContextClassifier
from src.environment.obstacles import Worker


def test_open_context():

    grid = np.zeros(
        (12, 12),
        dtype=np.int8,
    )

    classifier = ContextClassifier(
        static_grid=grid
    )

    result = classifier.classify(
        (6, 6)
    )

    assert result["context"] == "open"
    assert result["obstacle_density"] == 0.0
    assert result["nearest_static_distance"] is None


def test_nearest_static_obstacle():

    grid = np.zeros(
        (12, 12),
        dtype=np.int8,
    )

    grid[6, 7] = 1

    classifier = ContextClassifier(
        static_grid=grid
    )

    result = classifier.classify(
        (6, 6)
    )

    assert result["nearest_static_distance"] == 1


def test_nearest_dynamic_obstacle():

    grid = np.zeros(
        (12, 12),
        dtype=np.int8,
    )

    worker = Worker(
        position=(6, 8),
        movement_pattern="predefined_patrol",
        patrol_route=[(6, 8)],
        active=True,
    )

    classifier = ContextClassifier(
        static_grid=grid,
        workers=[worker],
    )

    result = classifier.classify(
        (6, 6)
    )

    assert result["nearest_dynamic_distance"] == 2


def test_inactive_dynamic_obstacle_ignored():

    grid = np.zeros(
        (12, 12),
        dtype=np.int8,
    )

    worker = Worker(
        position=(6, 8),
        movement_pattern="predefined_patrol",
        patrol_route=[(6, 8)],
        active=False,
    )

    classifier = ContextClassifier(
        static_grid=grid,
        workers=[worker],
    )

    result = classifier.classify(
        (6, 6)
    )

    assert result["nearest_dynamic_distance"] is None


def test_available_directions_center():

    grid = np.zeros(
        (12, 12),
        dtype=np.int8,
    )

    classifier = ContextClassifier(
        static_grid=grid
    )

    result = classifier.classify(
        (6, 6)
    )

    assert result["available_directions"] == 8


def test_available_directions_corner():

    grid = np.zeros(
        (12, 12),
        dtype=np.int8,
    )

    classifier = ContextClassifier(
        static_grid=grid
    )

    result = classifier.classify(
        (0, 0)
    )

    assert result["available_directions"] == 3


def test_dense_context():

    grid = np.ones(
        (12, 12),
        dtype=np.int8,
    )

    grid[6, 6] = 0

    classifier = ContextClassifier(
        static_grid=grid
    )

    result = classifier.classify(
        (6, 6)
    )

    assert result["context"] == "dense"


def test_aisle_structure():

    grid = np.zeros(
        (12, 12),
        dtype=np.int8,
    )

    # Shelves on both sides of the robot.
    grid[5, 6] = 1
    grid[7, 6] = 1

    classifier = ContextClassifier(
        static_grid=grid
    )

    result = classifier.classify(
        (6, 6)
    )

    assert result["aisle_structure"] is True
    assert result["context"] == "aisle"


def test_classify_returns_expected_fields():

    grid = np.zeros(
        (12, 12),
        dtype=np.int8,
    )

    classifier = ContextClassifier(
        static_grid=grid
    )

    result = classifier.classify(
        (6, 6)
    )

    expected_fields = {
        "context",
        "obstacle_density",
        "nearest_static_distance",
        "nearest_dynamic_distance",
        "available_directions",
        "aisle_structure",
        "dynamic_obstacle_count",
        "dynamic_density",
        "risk_level",
    }

    assert set(result.keys()) == expected_fields


def test_classify_includes_risk():

    grid = np.zeros(
        (12, 12),
        dtype=np.int8,
    )

    classifier = ContextClassifier(
        static_grid=grid
    )

    result = classifier.classify(
        (5, 5)
    )

    assert "risk_level" in result

    assert result["risk_level"] in [
        "low",
        "medium",
        "high",
    ]


def test_no_dynamic_obstacles_has_low_risk():

    grid = np.zeros(
        (12, 12),
        dtype=np.int8,
    )

    classifier = ContextClassifier(
        static_grid=grid
    )

    result = classifier.classify(
        (5, 5)
    )

    assert result["risk_level"] == "low"


def test_near_dynamic_obstacle_has_high_risk():

    grid = np.zeros(
        (12, 12),
        dtype=np.int8,
    )

    worker = Worker(
        position=(5, 6),
        movement_pattern="predefined_patrol",
        patrol_route=[(5, 6)],
        active=True,
    )

    classifier = ContextClassifier(
        static_grid=grid,
        workers=[worker],
    )

    result = classifier.classify(
        (5, 5)
    )

    assert result["risk_level"] == "high"


def test_dynamic_obstacle_count():

    grid = np.zeros(
        (12, 12),
        dtype=np.int8,
    )

    worker = Worker(
        position=(5, 6),
        movement_pattern="predefined_patrol",
        patrol_route=[(5, 6)],
        active=True,
    )

    classifier = ContextClassifier(
        static_grid=grid,
        workers=[worker],
    )

    result = classifier.classify(
        (5, 5)
    )

    assert result["dynamic_obstacle_count"] == 1


def test_inactive_dynamic_obstacle_not_counted():

    grid = np.zeros(
        (12, 12),
        dtype=np.int8,
    )

    worker = Worker(
        position=(5, 6),
        movement_pattern="predefined_patrol",
        patrol_route=[(5, 6)],
        active=False,
    )

    classifier = ContextClassifier(
        static_grid=grid,
        workers=[worker],
    )

    result = classifier.classify(
        (5, 5)
    )

    assert result["dynamic_obstacle_count"] == 0