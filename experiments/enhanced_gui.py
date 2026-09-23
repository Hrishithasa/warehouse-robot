"""
Enhanced GUI for Hybrid DDQN-A* Warehouse Robot Navigation.

Stage 11: live performance analytics with dynamic obstacles in all layouts.

Place this file at:
    experiments/enhanced_gui.py

Run from the project root:
    python experiments/enhanced_gui.py

The GUI reuses the existing:
- WarehouseEnv
- StateAugmenter
- AStarPlanner
- DynamicReplanner
- WaypointManager
- DDQNAgent
- HybridAgent

No navigation logic is duplicated here; the GUI is a presentation/control layer.
"""

import os
import sys
from collections import deque

import numpy as np
import torch

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# Make project root importable when launched from experiments/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment.warehouse_env import WarehouseEnv
from src.environment.obstacles import Worker, DynamicRobot
from src.environment.state_augmentation import StateAugmenter
from src.environment.constants import ACTIONS, NUM_ACTIONS
from src.astar.astar_planner import AStarPlanner
from src.astar.replanner import DynamicReplanner
from src.hybrid.waypoint_manager import WaypointManager
from src.ddqn.new_agent import DDQNAgent
from src.ddqn.network import ACTIONS as DDQN_ACTIONS
from src.hybrid.hybrid_agent import HybridAgent
from src.gui.warehouse_canvas import WarehouseCanvas
from src.gui.decision_panel import DecisionPanel


MODEL_PATH = os.path.join(
    PROJECT_ROOT, "experiments", "checkpoints", "hybrid_ddqn_ep2500.pth"
)



class MetricCard(QFrame):
    def __init__(self, title, value="--", accent="#38BDF8", parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)

        title_label = QLabel(title.upper())
        title_label.setObjectName("metricTitle")

        self.value_label = QLabel(str(value))
        self.value_label.setObjectName("metricValue")
        self.value_label.setStyleSheet(f"color: {accent};")

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value):
        self.value_label.setText(str(value))


class HybridModePanel(QFrame):
    """Current Hybrid DDQN-A* mode, displayed outside the maze."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modePanel")
        self.setMinimumHeight(58)
        self.setMaximumHeight(64)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        self.indicator = QLabel()
        self.indicator.setObjectName("modeIndicator")
        self.indicator.setFixedSize(9, 9)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)

        self.title_label = QLabel("READY")
        self.title_label.setObjectName("modeTitle")
        self.detail_label = QLabel("Waiting for navigation")
        self.detail_label.setObjectName("modeDetail")
        self.detail_label.setWordWrap(False)

        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.detail_label)
        layout.addWidget(self.indicator)
        layout.addLayout(text_layout, 1)

        self.set_mode("READY", "Waiting for navigation", "#64748B")

    def set_mode(self, title, detail, color):
        self.title_label.setText(title)
        self.detail_label.setText(detail)
        self.indicator.setStyleSheet(
            f"QLabel#modeIndicator {{ background: {color}; border-radius: 4px; }}"
        )
        self.setStyleSheet(
            f"QFrame#modePanel {{ background: #0F172A; border: 1px solid {color}; border-radius: 8px; }}"
        )


class CanvasLegend(QFrame):
    """Compact legend displayed below the warehouse canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("canvasLegend")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 9, 10, 9)
        outer.setSpacing(6)

        title = QLabel("LEGEND")
        title.setObjectName("legendTitle")
        outer.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)

        items = [
            ("Static shelf", "#64748B", "square"),
            ("Robot", "#22D3EE", "square"),
            ("Worker", "#F97316", "circle"),
            ("Dynamic robot", "#A78BFA", "diamond"),
            ("Start", "#94A3B8", "circle"),
            ("Goal", "#4ADE80", "circle"),
            ("Waypoint", "#A78BFA", "square"),
            ("A* route", "#38BDF8", "line"),
            ("Trajectory", "#FBBF24", "dash"),
        ]

        for index, (text, color, symbol_type) in enumerate(items):
            row = index % 5
            col = index // 5
            grid.addWidget(
                self.item_widget(text, color, symbol_type),
                row,
                col,
            )

        outer.addLayout(grid, 1)

    def item_widget(self, text, color, symbol_type):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        symbol = QLabel()
        if symbol_type == "line":
            symbol.setText("━━━━")
        elif symbol_type == "dash":
            symbol.setText("- - -")
        elif symbol_type == "diamond":
            symbol.setText("◆")
        elif symbol_type == "square":
            symbol.setText("■")
        else:
            symbol.setText("●")

        symbol.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: 700; min-width: 24px;"
        )
        label = QLabel(text)
        label.setObjectName("legendItem")
        layout.addWidget(symbol)
        layout.addWidget(label)
        return row


class PerformancePanel(QFrame):
    """Compact live performance analytics for the current episode."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("performancePanel")
        self.steps = 0
        self.total_reward = 0.0
        self.avg_reward = 0.0
        self.collisions = 0
        self.replans = 0
        self.recoveries = 0
        self.goal_distance = 0.0
        self.reward_history = []

        layout = QGridLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(4)

        self.labels = {}
        fields = [
            ("Steps", "steps"),
            ("Total Reward", "total_reward"),
            ("Avg Reward", "avg_reward"),
            ("Collisions", "collisions"),
            ("Replans", "replans"),
            ("Recoveries", "recoveries"),
            ("Goal Distance", "goal_distance"),
        ]

        for i, (title, key) in enumerate(fields):
            row = i // 2
            col = (i % 2) * 2
            title_label = QLabel(title)
            title_label.setObjectName("perfTitle")
            value_label = QLabel("--")
            value_label.setObjectName("perfValue")
            value_label.setAlignment(Qt.AlignRight)
            layout.addWidget(title_label, row, col)
            layout.addWidget(value_label, row, col + 1)
            self.labels[key] = value_label

        # Tiny live reward trend across the bottom.
        self.trend = RewardTrendWidget()
        layout.addWidget(self.trend, 4, 0, 1, 4)

    def update_metrics(self, controller, distance):
        self.steps = int(controller.env._step_count) if controller.env else 0
        self.total_reward = float(controller.total_reward)
        history = list(controller.reward_history)
        self.avg_reward = (sum(history) / len(history)) if history else 0.0
        self.collisions = int(controller.collisions)
        self.replans = int(controller.replans)
        hybrid = controller.hybrid
        self.recoveries = (
            int(hybrid.loop_recoveries + hybrid.progress_recoveries)
            if hybrid is not None else 0
        )
        self.goal_distance = float(distance)

        values = {
            "steps": str(self.steps),
            "total_reward": f"{self.total_reward:+.2f}",
            "avg_reward": f"{self.avg_reward:+.2f}",
            "collisions": str(self.collisions),
            "replans": str(self.replans),
            "recoveries": str(self.recoveries),
            "goal_distance": f"{self.goal_distance:.2f}",
        }
        for key, value in values.items():
            self.labels[key].setText(value)

        self.trend.set_values(history)

    def clear(self):
        for label in self.labels.values():
            label.setText("--")
        self.trend.set_values([])


class RewardTrendWidget(QWidget):
    """Small presentation-friendly reward trend chart."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = []
        self.setMinimumHeight(28)

    def set_values(self, values):
        self.values = list(values)[-40:]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0B1220"))

        if len(self.values) < 2:
            painter.setPen(QColor("#475569"))
            painter.drawText(self.rect(), Qt.AlignCenter, "REWARD TREND")
            painter.end()
            return

        left, top = 4, 4
        width = max(1, self.width() - 8)
        height = max(1, self.height() - 8)
        lo = min(self.values)
        hi = max(self.values)
        span = hi - lo if hi != lo else 1.0

        pen = QPen(QColor("#38BDF8"), 1.5)
        painter.setPen(pen)
        points = []
        for i, value in enumerate(self.values):
            x = left + (width * i / (len(self.values) - 1))
            y = top + height - ((value - lo) / span) * height
            points.append((x, y))

        for a, b in zip(points, points[1:]):
            painter.drawLine(a[0], a[1], b[0], b[1])

        painter.setPen(QColor("#64748B"))
        painter.drawText(5, self.height() - 2, "REWARD TREND")
        painter.end()


class SimulationController:
    """Owns the existing project objects and exposes GUI-friendly state."""

    def __init__(self):
        self.env = None
        self.state_builder = None
        self.agent = None
        self.hybrid = None
        self.info = {}
        self.obs = None

        self.path = []
        self.trajectory = []
        self.reward_history = deque(maxlen=100)
        self.distance_history = deque(maxlen=100)

        self.total_reward = 0.0
        self.replans = 0
        self.collisions = 0
        self.done = False
        self.started = False
        self.last_action = None
        self.last_waypoint = None
        self.last_status = "READY"
        self.last_state = None
        self.q_values = None
        self.valid_actions = []
        self.last_path = []

        # Stage 9: A* path analytics
        self.astar_old_path_length = 0
        self.astar_replan_position = None
        self.astar_path_status = "NOT PLANNED"
        self.scenario_mode = "open"

    def initialize(self, config):
        self.scenario_mode = config
        # Dynamic obstacles are part of every normal warehouse environment.
        # The dropdown selects the static layout only: open / aisle / dense.
        self.env = WarehouseEnv(config=config)
        self.state_builder = StateAugmenter(
            self.env._static_grid,
            local_radius=2,
            max_dynamic_obstacles=4,
        )

        self.agent = DDQNAgent(state_dim=51, num_actions=8)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Checkpoint not found:\n{MODEL_PATH}"
            )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.agent.device,
        )
        self.agent.online_network.load_state_dict(checkpoint)
        self.agent.target_network.load_state_dict(checkpoint)
        self.agent.target_network.eval()
        self.agent.epsilon = 0.0

        self.hybrid = HybridAgent(
            self.agent,
            WaypointManager(),
            DynamicReplanner(AStarPlanner(self.env._static_grid)),
            AStarPlanner(self.env._static_grid),
        )

        self.obs, self.info = self.env.reset()

        # Stage 11: dynamic obstacles are always active alongside shelves.
        self._setup_dynamic_obstacles()

        self.hybrid.waypoint_manager.set_path([])
        self.hybrid.reset_navigation_memory()

        self.path = []
        self.trajectory = [self.env.robot_pos]
        self.reward_history.clear()
        self.distance_history.clear()

        self.total_reward = 0.0
        self.replans = 0
        self.collisions = 0
        self.done = False
        self.started = True
        self.last_action = None
        self.last_waypoint = None
        self.last_status = "READY"
        self.last_state = None
        self.q_values = None

        # Stage 9: reset A* analytics
        self.astar_old_path_length = 0
        self.astar_replan_position = None
        self.astar_path_status = "NOT PLANNED"

    def _candidate_dynamic_routes(self, length, exclude=None):
        """Return many possible free patrol routes across the warehouse.

        Routes are collected from both horizontal and vertical free-cell
        segments.  They are later shuffled so dynamic obstacles do not
        always appear in the same part of the warehouse.
        """
        exclude = set(exclude or set())
        grid = self.env._static_grid
        rows, cols = grid.shape
        candidates = []

        # Horizontal candidates.
        for r in range(rows):
            for c in range(cols - length + 1):
                route = [(r, c + k) for k in range(length)]
                if all(
                    pos not in exclude
                    and not self.env._is_shelf(pos[0], pos[1])
                    for pos in route
                ):
                    candidates.append(route)

        # Vertical candidates.
        for r in range(rows - length + 1):
            for c in range(cols):
                route = [(r + k, c) for k in range(length)]
                if all(
                    pos not in exclude
                    and not self.env._is_shelf(pos[0], pos[1])
                    for pos in route
                ):
                    candidates.append(route)

        return candidates

    def _choose_dynamic_route(self, length, exclude=None, avoid_routes=None):
        """Choose a varied patrol route for a dynamic obstacle."""
        candidates = self._candidate_dynamic_routes(
            length, exclude=exclude
        )
        if not candidates:
            return []

        avoid_routes = avoid_routes or []
        used_cells = set()
        for route in avoid_routes:
            used_cells.update(route)

        # Prefer routes separated from already-selected obstacles so that
        # the dynamic objects are visibly distributed around the warehouse.
        def separation_score(route):
            if not used_cells:
                return 0
            return min(
                abs(r - ur) + abs(c - uc)
                for r, c in route
                for ur, uc in used_cells
            )

        # Randomize candidates first, then retain routes with good spatial
        # separation.  The environment RNG changes the locations on reset.
        rng = getattr(self.env, "_rng", None)
        if rng is not None:
            rng.shuffle(candidates)

        if used_cells:
            candidates.sort(key=separation_score, reverse=True)
            pool = candidates[:max(1, min(12, len(candidates)))]
            if rng is not None:
                return pool[int(rng.integers(0, len(pool)))]
            return pool[0]

        pool = candidates[:max(1, min(20, len(candidates)))]
        if rng is not None:
            return pool[int(rng.integers(0, len(pool)))]
        return pool[0]

    def _setup_dynamic_obstacles(self):
        """Attach moving obstacles at varied locations in every layout."""
        exclude = {self.env.robot_pos, self.env.goal_pos}
        selected_routes = []

        # Worker: random free patrol route.
        worker_route = self._choose_dynamic_route(
            4,
            exclude=exclude,
            avoid_routes=selected_routes,
        )
        worker = None
        if worker_route:
            worker = Worker(
                position=worker_route[0],
                movement_pattern="predefined_patrol",
                speed=1.0,
                patrol_route=worker_route,
                pause_steps=0,
            )
            selected_routes.append(worker_route)
            exclude.update(worker_route)

        # Dynamic robot: deliberately choose another spatially separated
        # route, so the two moving obstacles are not stacked together.
        robot_route = self._choose_dynamic_route(
            3,
            exclude=exclude,
            avoid_routes=selected_routes,
        )
        dynamic_robot = None
        if robot_route:
            dynamic_robot = DynamicRobot(
                position=robot_route[0],
                movement_pattern="path",
                speed=1.0,
                movement_path=robot_route,
                allow_reverse=True,
                grid_size=(self.env.grid_size, self.env.grid_size),
            )

        workers = [worker] if worker is not None else []
        dynamic_robots = [dynamic_robot] if dynamic_robot is not None else []
        self.env.set_dynamic_obstacles(
            workers=workers,
            dynamic_robots=dynamic_robots,
        )

        context_info = self.env.context_classifier.classify(self.env.robot_pos)
        self.info.update({
            "context": context_info["context"],
            "risk_level": context_info["risk_level"],
            "dynamic_obstacle_count": len(workers) + len(dynamic_robots),
        })

    def _update_dynamic_obstacles(self):
        """Move all active dynamic obstacles after each robot action."""
        obstacles = self.env.workers + self.env.dynamic_robots
        occupied = {self.env.robot_pos, self.env.goal_pos}

        for obstacle in obstacles:
            if not getattr(obstacle, "active", True):
                continue

            if hasattr(obstacle, "update"):
                try:
                    obstacle.update(occupied_positions=list(occupied))
                except TypeError:
                    obstacle.update()

            occupied.add(obstacle.position)

    def step(self):
        if not self.started or self.done:
            return None

        current_pos = self.env.robot_pos
        goal_pos = self.env.goal_pos

        dynamic_obstacles = self.env.workers + self.env.dynamic_robots
        blocked_cells = {
            obj.position
            for obj in dynamic_obstacles
            if getattr(obj, "active", True)
        }

        old_path = list(self.hybrid.waypoint_manager.path)

        waypoint = self.hybrid.update_route_and_waypoint(
            current_pos,
            goal_pos,
            blocked_cells,
        )

        new_path = list(self.hybrid.waypoint_manager.path)

        if old_path and new_path != old_path:
            self.replans += 1
            self.last_status = "A* REPLANNING"

            # Stage 9: capture the route transition for analytics.
            self.astar_old_path_length = len(old_path)
            self.astar_replan_position = current_pos
            self.astar_path_status = (
                f"REPLANNED {len(old_path)}→{len(new_path)}"
            )
        else:
            self.last_status = "NAVIGATING"
            if new_path:
                self.astar_path_status = "CLEAR"
            else:
                self.astar_path_status = "NO PATH"

        self.path = new_path
        self.last_waypoint = waypoint
        self.env.set_astar_path(new_path)

        state = self.state_builder.build_state(
            robot_position=current_pos,
            goal_position=goal_pos,
            workers=self.env.workers,
            dynamic_robots=self.env.dynamic_robots,
            waypoint=waypoint,
            context=self.info.get("context", "open"),
            risk_level=self.info.get("risk_level", "low"),
        )

        self.last_state = np.asarray(state, dtype=np.float32)
        self.q_values = self.agent.get_q_values(self.last_state)

        valid_actions = self.env.get_valid_actions()
        self.valid_actions = list(valid_actions)

        action = self.hybrid.get_action(
            state,
            epsilon=0.0,
            current_pos=current_pos,
            waypoint=waypoint,
            blocked_cells=blocked_cells,
            valid_actions=valid_actions,
        )
        self.last_action = action

        previous = self.env.robot_pos
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        self.obs = next_obs
        self.info = info
        self.done = terminated or truncated

        # Move configured dynamic obstacles after the robot action so the
        # next decision observes their new positions and can trigger replanning.
        self._update_dynamic_obstacles()
        self.info["dynamic_obstacle_count"] = len(
            [
                obj for obj in self.env.workers + self.env.dynamic_robots
                if getattr(obj, "active", True)
            ]
        )

        self.total_reward += float(reward)
        self.reward_history.append(float(reward))

        if info.get("collided", False):
            self.collisions += 1
            self.last_status = "COLLISION"
        elif terminated:
            self.last_status = "GOAL REACHED"
        elif truncated:
            self.last_status = "MAX STEPS"
        elif self.last_status != "A* REPLANNING":
            self.last_status = "NAVIGATING"

        self.trajectory.append(self.env.robot_pos)

        # Euclidean distance to goal for live analytics
        r, c = self.env.robot_pos
        gr, gc = self.env.goal_pos
        distance = float(np.hypot(gr - r, gc - c))
        self.distance_history.append(distance)

        return {
            "previous_position": previous,
            "position": self.env.robot_pos,
            "reward": float(reward),
            "action": action,
            "info": info,
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.controller = SimulationController()
        self.show_astar = True
        self.running = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.simulation_tick)

        self.setWindowTitle("Hybrid DDQN-A* | Warehouse Navigation")
        self.resize(1550, 920)
        self.setMinimumSize(1280, 760)

        self.build_ui()
        self.apply_theme()
        self.set_controls_enabled(False)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)

        # Header
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("HYBRID WAREHOUSE NAVIGATION")
        title.setObjectName("mainTitle")
        subtitle = QLabel(
            "Double DQN  •  A* Path Planning  •  Dynamic Replanning  •  Adaptive Reward"
        )
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.system_status = QLabel("● READY")
        self.system_status.setObjectName("systemStatus")
        self.system_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        header.addLayout(title_box)
        header.addStretch()
        header.addWidget(self.system_status)
        root.addLayout(header)

        # Top metrics
        metrics = QHBoxLayout()
        self.step_card = MetricCard("Step", "0")
        self.reward_card = MetricCard("Total Reward", "0.00", "#4ADE80")
        self.replan_card = MetricCard("A* Replans", "0", "#A78BFA")
        self.collision_card = MetricCard("Collisions", "0", "#FB7185")
        self.distance_card = MetricCard("Goal Distance", "--", "#FACC15")
        self.status_card = MetricCard("Navigation", "READY", "#38BDF8")
        self.recovery_card = MetricCard("Loop Recoveries", "0", "#FBBF24")
        self.progress_recovery_card = MetricCard(
            "Progress Recoveries",
            "0",
            "#A78BFA",
        )
        
        for card in (
            self.step_card,
            self.reward_card,
            self.replan_card,
            self.collision_card,
            self.distance_card,
            self.status_card,
            self.recovery_card,
            self.progress_recovery_card,
        ):
            metrics.addWidget(card)

        root.addLayout(metrics)

        # Main content
        content = QHBoxLayout()
        content.setSpacing(12)

        # Left control panel
        controls = QFrame()
        controls.setObjectName("panel")
        controls.setFixedWidth(225)
        cv = QVBoxLayout(controls)
        cv.setContentsMargins(16, 16, 16, 16)
        cv.setSpacing(10)

        cv.addWidget(self.section_label("SIMULATION"))

        cv.addWidget(QLabel("Warehouse layout"))
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["open", "aisle", "dense"])
        cv.addWidget(self.layout_combo)

        scenario_hint = QLabel(
            "Dynamic workers and robots are active in every warehouse layout."
        )
        scenario_hint.setObjectName("smallMuted")
        scenario_hint.setWordWrap(True)
        cv.addWidget(scenario_hint)

        self.start_btn = QPushButton("▶  START / LOAD")
        self.start_btn.clicked.connect(self.start_simulation)
        cv.addWidget(self.start_btn)

        row1 = QHBoxLayout()
        self.pause_btn = QPushButton("Ⅱ  PAUSE")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.step_btn = QPushButton("⏭  STEP")
        self.step_btn.clicked.connect(self.manual_step)
        row1.addWidget(self.pause_btn)
        row1.addWidget(self.step_btn)
        cv.addLayout(row1)

        self.reset_btn = QPushButton("↻  RESET")
        self.reset_btn.clicked.connect(self.reset_simulation)
        cv.addWidget(self.reset_btn)

        cv.addSpacing(8)
        cv.addWidget(self.section_label("SIMULATION SPEED"))

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 20)
        self.speed_slider.setValue(8)
        self.speed_slider.valueChanged.connect(self.update_speed)
        cv.addWidget(self.speed_slider)

        self.speed_label = QLabel("8 steps/sec")
        self.speed_label.setAlignment(Qt.AlignCenter)
        cv.addWidget(self.speed_label)

        cv.addSpacing(8)
        cv.addWidget(self.section_label("VISUALIZATION"))

        self.path_btn = QPushButton("✓  A* PATH")
        self.path_btn.setCheckable(True)
        self.path_btn.setChecked(True)
        self.path_btn.clicked.connect(self.toggle_path)
        cv.addWidget(self.path_btn)

        cv.addStretch()

        # Center simulation
        sim_panel = QFrame()
        sim_panel.setObjectName("panel")
        sv = QVBoxLayout(sim_panel)
        sv.setContentsMargins(10, 10, 10, 10)

        sim_header = QHBoxLayout()
        sim_title = QLabel("WAREHOUSE SIMULATION")
        sim_title.setObjectName("panelTitle")
        self.scenario_label = QLabel("Scenario: --")
        self.scenario_label.setObjectName("smallMuted")
        sim_header.addWidget(sim_title)
        sim_header.addStretch()
        sim_header.addWidget(self.scenario_label)
        sv.addLayout(sim_header)

        # Hybrid navigation mode is kept outside the maze so it never
        # covers the A* route, trajectory, robot, or action mask.
        self.mode_panel = HybridModePanel()
        sv.addWidget(self.mode_panel)

        # Warehouse visualization
        self.canvas = WarehouseCanvas(self.controller)
        sv.addWidget(self.canvas, 1)

        # The legend is intentionally NOT placed below/inside the maze.
        # It is placed in the right analysis column so it never covers
        # the robot, route, trajectory, or action indicators.

        # Right analysis panel
        # Kept intentionally compact: the top metric cards already show
        # episode-level reward, collisions, distance and recovery counts.
        right = QFrame()
        right.setObjectName("panel")
        # Wider analysis column keeps decision values and labels readable.
        right.setMinimumWidth(455)
        right.setMaximumWidth(500)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(12, 12, 12, 12)
        rv.setSpacing(5)

        # --------------------------------------------------------------
        # Compact analysis cards: Robot Status + A* Planner side-by-side
        # --------------------------------------------------------------
        analysis_top = QHBoxLayout()
        analysis_top.setSpacing(8)

        status_box = QFrame()
        status_box.setObjectName("analysisSubPanel")
        status_layout = QVBoxLayout(status_box)
        status_layout.setContentsMargins(10, 8, 10, 8)
        status_layout.setSpacing(3)
        status_layout.addWidget(self.section_label("ROBOT STATUS"))

        self.position_label = self.data_row(status_layout, "Position", "--")
        self.goal_label = self.data_row(status_layout, "Goal", "--")
        self.waypoint_label = self.data_row(status_layout, "Waypoint", "--")
        self.action_label = self.data_row(status_layout, "DDQN Action", "--")
        self.context_label = self.data_row(status_layout, "Context", "--")
        self.risk_label = self.data_row(status_layout, "Risk", "--")
        self.dynamic_label = self.data_row(
            status_layout, "Dynamic Obstacles", "--"
        )

        planner_box = QFrame()
        planner_box.setObjectName("analysisSubPanel")
        planner_layout = QVBoxLayout(planner_box)
        planner_layout.setContentsMargins(10, 8, 10, 8)
        planner_layout.setSpacing(3)
        planner_layout.addWidget(self.section_label("A* PLANNER"))

        self.path_length_label = self.data_row(
            planner_layout, "Path Length", "--"
        )
        self.remaining_path_label = self.data_row(
            planner_layout, "Remaining", "--"
        )
        self.path_status_label = self.data_row(
            planner_layout, "Path Status", "--"
        )
        self.replan_label = self.data_row(
            planner_layout, "Replans / Point", "--"
        )
        self.last_reward_label = self.data_row(
            planner_layout, "Last Reward", "--"
        )

        analysis_top.addWidget(status_box, 1)
        analysis_top.addWidget(planner_box, 1)
        rv.addLayout(analysis_top)

        # --------------------------------------------------------------
        # Hybrid decision analysis
        # --------------------------------------------------------------
        rv.addWidget(self.section_label("HYBRID DECISION"))

        self.decision_panel = DecisionPanel()
        self.decision_panel.setMinimumHeight(180)
        self.decision_panel.setMaximumHeight(205)
        rv.addWidget(self.decision_panel)

        # --------------------------------------------------------------
        # Event log
        # --------------------------------------------------------------
        rv.addWidget(self.section_label("EVENT LOG"))

        self.event_log = QLabel(
            "System initialized.\\nWaiting for simulation..."
        )
        self.event_log.setObjectName("eventLog")
        self.event_log.setWordWrap(True)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.event_log)
        scroll.setObjectName("eventScroll")
        rv.addWidget(scroll, 1)

        # --------------------------------------------------------------
        # Legend — right sidebar, never over the maze
        # --------------------------------------------------------------
        self.canvas_legend = CanvasLegend()
        self.canvas_legend.setMinimumHeight(142)
        self.canvas_legend.setMaximumHeight(155)
        rv.addWidget(self.canvas_legend)

        # Keep the old analytics objects alive because refresh_ui updates
        # them, but do not place them in the visible right column. Their
        # information is already represented by the top metric cards and
        # the compact Last Reward row.
        self.performance_panel = PerformancePanel()
        self.reward_value = QLabel("0.00")
        self.reward_bar = QProgressBar()
        self.reward_bar.setRange(-100, 100)
        self.reward_bar.setValue(0)

        content.addWidget(controls)
        content.addWidget(sim_panel, 1)
        content.addWidget(right)
        root.addLayout(content, 1)

    def section_label(self, text):
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

    def data_row(self, layout, title, value):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        left = QLabel(title)
        left.setObjectName("dataTitle")
        right = QLabel(value)
        right.setObjectName("dataValue")
        right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        right.setMinimumWidth(105)
        row.addWidget(left, 1)
        row.addWidget(right)
        layout.addLayout(row)
        return right

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0B1220;
                color: #E2E8F0;
                font-family: "Segoe UI";
            }

            QFrame#panel {
                background: #111827;
                border: 1px solid #243244;
                border-radius: 12px;
            }

            QFrame#metricCard {
                background: #111827;
                border: 1px solid #243244;
                border-radius: 10px;
            }

            QFrame#analysisSubPanel {
                background: #0F172A;
                border: 1px solid #243244;
                border-radius: 9px;
            }

            QLabel#mainTitle {
                color: #F8FAFC;
                font-size: 27px;
                font-weight: 700;
            }

            QLabel#subtitle {
                color: #94A3B8;
                font-size: 14px;
            }

            QLabel#systemStatus {
                color: #4ADE80;
                font-size: 14px;
                font-weight: 700;
                padding: 8px 12px;
            }

            QLabel#metricTitle {
                color: #64748B;
                font-size: 12px;
                font-weight: 700;
            }

            QLabel#metricValue {
                font-size: 22px;
                font-weight: 700;
            }

            QFrame#performancePanel {
                background: #0F172A;
                border: 1px solid #243244;
                border-radius: 8px;
            }

            QLabel#perfTitle {
                color: #64748B;
                font-size: 10px;
                font-weight: 700;
            }

            QLabel#perfValue {
                color: #E2E8F0;
                font-size: 12px;
                font-weight: 700;
            }

            QLabel#sectionLabel {
                color: #64748B;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
            }

            QLabel#panelTitle {
                color: #F8FAFC;
                font-size: 15px;
                font-weight: 700;
            }

            QLabel#smallMuted {
                color: #64748B;
                font-size: 12px;
            }

            QLabel#dataTitle {
                color: #94A3B8;
                font-size: 12px;
            }

            QLabel#dataValue {
                color: #E2E8F0;
                font-size: 12px;
                font-weight: 600;
            }

            QLabel#bigValue {
                color: #4ADE80;
                font-size: 27px;
                font-weight: 700;
            }

            QLabel#eventLog {
                color: #CBD5E1;
                background: #0F172A;
                padding: 9px;
                font-size: 11px;
            }

            QComboBox, QPushButton {
                background: #172033;
                border: 1px solid #334155;
                border-radius: 7px;
                padding: 9px;
                font-size: 12px;
                color: #E2E8F0;
            }

            QComboBox:hover, QPushButton:hover {
                border-color: #38BDF8;
            }

            QPushButton:checked {
                background: #164E63;
                border-color: #22D3EE;
            }

            QFrame#modePanel {
                background: #0F172A;
                border: 1px solid #243244;
                border-radius: 8px;
            }

            QLabel#modeTitle {
                color: #F8FAFC;
                font-size: 12px;
                font-weight: 700;
            }

            QLabel#modeDetail {
                color: #94A3B8;
                font-size: 11px;
                font-weight: 500;
            }

            QFrame#decisionPanel {
                background: #0F172A;
                border: 1px solid #243244;
                border-radius: 8px;
            }

            QLabel#decisionAction {
                color: #94A3B8;
                font-size: 11px;
                font-weight: 600;
            }

            QLabel#qValue {
                color: #94A3B8;
                font-size: 11px;
            }

            QLabel#decisionSubTitle {
                color: #64748B;
                font-size: 10px;
                font-weight: 700;
            }

            QLabel#selectedAction {
                color: #22D3EE;
                font-size: 21px;
                font-weight: 700;
            }
            
            QLabel#hybridDecisionValue {
                color: #22D3EE;
                font-size: 14px;
                font-weight: 700;
            }

            QLabel#decisionMode {
                color: #A78BFA;
                background: #17152A;
                border: 1px solid #3B2F68;
                border-radius: 6px;
                padding: 5px;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#selectedQ {
                color: #E2E8F0;
                font-size: 12px;
                font-weight: 600;
            }

            QLabel#epsilonLabel {
                color: #64748B;
                font-size: 10px;
            }

            QFrame#decisionDivider {
                color: #243244;
            }

            QProgressBar#qBar {
                background: #172033;
                border: none;
                border-radius: 3px;
                height: 7px;
            }

            QProgressBar#qBar::chunk {
                background: #38BDF8;
                border-radius: 3px;
            }

            QProgressBar {
                background: #172033;
                border: none;
                border-radius: 4px;
                height: 8px;
            }

            QProgressBar::chunk {
                background: #4ADE80;
                border-radius: 4px;
            }

            QSlider::groove:horizontal {
                background: #334155;
                height: 4px;
            }

            QSlider::handle:horizontal {
                background: #38BDF8;
                width: 12px;
                margin: -5px 0;
                border-radius: 6px;
            }

            QScrollArea {
                border: none;
            }
            QFrame#canvasLegend {
                background: #0F172A;
                border: 1px solid #38BDF8;
                border-radius: 9px;
            }

            QLabel#legendTitle {
                color: #F8FAFC;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 1px;
            }

            QLabel#legendItem {
                color: #CBD5E1;
                font-size: 11px;
                font-weight: 500;
            }

            QFrame#performancePanel {
                background: #0F172A;
                border: 1px solid #243244;
                border-radius: 8px;
            }

            QLabel#perfTitle {
                color: #64748B;
                font-size: 9px;
            }

            QLabel#perfValue {
                color: #E2E8F0;
                font-size: 10px;
                font-weight: 700;
            }
        """)

    def set_controls_enabled(self, enabled):
        self.pause_btn.setEnabled(enabled)
        self.step_btn.setEnabled(enabled)
        self.reset_btn.setEnabled(enabled)
        self.path_btn.setEnabled(enabled)

    def start_simulation(self):
        try:
            self.controller.initialize(self.layout_combo.currentText())
        except Exception as exc:
            self.log_event(f"ERROR: {exc}")
            self.system_status.setText("● CHECKPOINT ERROR")
            return

        self.set_controls_enabled(True)
        self.running = True
        self.timer.start(max(20, int(1000 / self.speed_slider.value())))
        self.scenario_label.setText(
            f"Scenario: {self.layout_combo.currentText().upper()}"
        )
        self.system_status.setText("● RUNNING")
        self.log_event(
            f"Simulation started: {self.layout_combo.currentText().upper()} scenario."
        )
        self.refresh_ui()

    def toggle_pause(self):
        if self.running:
            self.running = False
            self.timer.stop()
            self.pause_btn.setText("▶  RESUME")
            self.system_status.setText("● PAUSED")
            self.log_event("Simulation paused.")
        else:
            self.running = True
            self.timer.start(max(20, int(1000 / self.speed_slider.value())))
            self.pause_btn.setText("Ⅱ  PAUSE")
            self.system_status.setText("● RUNNING")
            self.log_event("Simulation resumed.")

    def manual_step(self):
        if self.controller.started and not self.controller.done:
            result = self.controller.step()
            self.handle_step_event(result)
            self.refresh_ui()

    def simulation_tick(self):
        if not self.running:
            return

        result = self.controller.step()
        self.handle_step_event(result)
        self.refresh_ui()

        if self.controller.done:
            self.running = False
            self.timer.stop()
            self.pause_btn.setText("Ⅱ  PAUSE")
            self.system_status.setText("● " + self.controller.last_status)

    def handle_step_event(self, result):
        if not result:
            return

        info = result["info"]
        action = result["action"]
        reward = result["reward"]

        hybrid = self.controller.hybrid
        step = info.get("step_count", 0)

        if hybrid.last_loop_detected:
            self.log_event(
                f"Step {step}: LOOP detected → A* recovery."
            )

        elif hybrid.last_progress_stall:
            self.log_event(
                f"Step {step}: NO PROGRESS → A* recovery."
            )

        elif (
            hybrid.last_astar_action is not None
            and action == hybrid.last_astar_action
        ):
            self.log_event(
                f"Step {step}: DDQN + A* guidance → "
                f"{DDQN_ACTIONS[action]}"
            )

        elif (
            hybrid.last_action_masked
            and len(self.controller.valid_actions)
            < self.controller.agent.num_actions
        ):
            self.log_event(
                f"Step {step}: collision-aware action mask "
                f"({len(self.controller.valid_actions)}/"
                f"{self.controller.agent.num_actions} valid)."
            )

        elif self.controller.last_status == "A* REPLANNING":
            self.log_event(
                f"Step {step}: dynamic/path condition changed "
                f"→ A* replanning."
            )

        elif info.get("collided"):
            self.log_event(
                f"Step {step}: collision-aware movement blocked."
            )

        elif self.controller.done and (
            result["position"] == self.controller.env.goal_pos
        ):
            self.log_event(
                f"Step {step}: GOAL REACHED."
            )

        else:
            self.log_event(
                f"Step {step}: DDQN → "
                f"{DDQN_ACTIONS[action]}, "
                f"reward {reward:+.2f}"
            )

    def _dynamic_obstacle_nearby(self):
        """Return True when an active dynamic obstacle is within 2 cells."""
        env = self.controller.env
        if env is None or env.robot_pos is None:
            return False

        rr, rc = env.robot_pos
        for obj in list(env.workers) + list(env.dynamic_robots):
            if not getattr(obj, "active", True):
                continue
            orow, ocol = obj.position
            if max(abs(rr - orow), abs(rc - ocol)) <= 2:
                return True
        return False

    def update_mode_panel(self):
        """Update the external mode panel without drawing over the maze."""
        c = self.controller
        hybrid = c.hybrid

        if hybrid is None:
            self.mode_panel.set_mode("READY", "Waiting for navigation", "#64748B")
            return

        action = c.last_action
        action_name = (
            DDQN_ACTIONS[action]
            if action is not None and 0 <= action < len(DDQN_ACTIONS)
            else "--"
        )
        valid_text = (
            f"{len(c.valid_actions)}/{len(ACTIONS)} valid"
            if c.valid_actions is not None
            else "mask --"
        )

        if hybrid.last_loop_detected:
            title, color = "LOOP RECOVERY", "#FB7185"
            detail = f"A* recovery  •  action {action_name}  •  {valid_text}"
        elif hybrid.last_progress_stall:
            title, color = "PROGRESS RECOVERY", "#FBBF24"
            detail = f"A* recovery  •  action {action_name}  •  {valid_text}"
        elif c.last_status == "A* REPLANNING":
            title, color = "A* REPLANNING", "#A78BFA"
            detail = f"old path → new path  •  action {action_name}  •  {valid_text}"
        elif self._dynamic_obstacle_nearby():
            title, color = "DYNAMIC OBSTACLE NEARBY", "#FB7185"
            detail = f"DDQN + A* guidance  •  action {action_name}  •  {valid_text}"
        elif hybrid.last_astar_action is not None:
            title, color = "NORMAL HYBRID CONTROL", "#22D3EE"
            detail = f"DDQN + A* guidance  •  action {action_name}  •  {valid_text}"
        else:
            title, color = "DDQN CONTROL", "#4ADE80"
            detail = f"learned action  •  action {action_name}  •  {valid_text}"

        self.mode_panel.set_mode(title, detail, color)

    def reset_simulation(self):
        self.running = False
        self.timer.stop()
        self.pause_btn.setText("Ⅱ  PAUSE")

        if self.controller.started:
            try:
                self.controller.initialize(self.layout_combo.currentText())
                self.log_event("Simulation reset with the selected scenario configuration.")
            except Exception as exc:
                self.log_event(f"ERROR: {exc}")

        self.system_status.setText("● READY")
        self.refresh_ui()

    def update_speed(self, value):
        self.speed_label.setText(f"{value} steps/sec")
        if self.running:
            self.timer.start(max(20, int(1000 / value)))

    def toggle_path(self, checked):
        self.show_astar = checked
        self.path_btn.setText("✓  A* PATH" if checked else "○  A* PATH")
        self.canvas.update()

    def log_event(self, message):
        existing = self.event_log.text().splitlines()
        existing.append(message)
        existing = existing[-12:]
        self.event_log.setText("\n".join(existing))

    def refresh_ui(self):
        c = self.controller
        env = c.env

        if not c.started or env is None:
            self.decision_panel.clear()
            self.performance_panel.clear()
            self.mode_panel.set_mode("READY", "Waiting for navigation", "#64748B")
            self.canvas.update()
            return

        step = env._step_count
        goal = env.goal_pos
        pos = env.robot_pos

        gr, gc = goal
        r, col = pos
        distance = float(np.hypot(gr - r, gc - col))

        self.step_card.set_value(step)
        self.reward_card.set_value(f"{c.total_reward:+.2f}")
        self.replan_card.set_value(c.replans)
        self.collision_card.set_value(c.collisions)
        self.distance_card.set_value(f"{distance:.2f}")
        self.status_card.set_value(c.last_status)
        self.recovery_card.set_value(
            c.hybrid.loop_recoveries if c.hybrid is not None else 0
        )
        self.progress_recovery_card.set_value(
            c.hybrid.progress_recoveries
            if c.hybrid is not None
            else 0
        )

        self.position_label.setText(str(pos))
        self.goal_label.setText(str(goal))
        self.waypoint_label.setText(
            str(c.last_waypoint) if c.last_waypoint is not None else "--"
        )
        self.action_label.setText(
            DDQN_ACTIONS[c.last_action] if c.last_action is not None and 0 <= c.last_action < len(DDQN_ACTIONS) else "--"
            if c.last_action is not None
            else "--"
        )
        self.context_label.setText(c.info.get("context", "--"))
        self.risk_label.setText(c.info.get("risk_level", "--"))
        self.dynamic_label.setText(
            str(c.info.get("dynamic_obstacle_count", 0))
        )

        path_length = len(c.path)
        self.path_length_label.setText(str(path_length))

        # Remaining route = nodes still ahead of the robot on the current A* path.
        remaining = path_length
        if c.path and pos in c.path:
            try:
                current_index = c.path.index(pos)
                remaining = max(0, path_length - current_index - 1)
            except ValueError:
                remaining = path_length
        self.remaining_path_label.setText(str(remaining))

        self.path_status_label.setText(c.astar_path_status)
        replan_point = (
            str(c.astar_replan_position)
            if c.astar_replan_position is not None
            else "--"
        )
        self.replan_label.setText(f"{c.replans} / {replan_point}")

        self.decision_panel.update_values(
            c.q_values,
            selected_index=c.last_action,
            epsilon=c.agent.epsilon if c.agent is not None else None,
            astar_action=(
                c.hybrid.last_astar_action
                if c.hybrid is not None
                else None
            ),
            recovery_reason=(
                c.hybrid.last_recovery_reason
                if c.hybrid is not None
                else None
            ),
            valid_actions=c.valid_actions,
            action_masked=(
                c.hybrid.last_action_masked
                if c.hybrid is not None
                else False
            ),
            status=c.last_status,
        )

        self.performance_panel.update_metrics(c, distance)
        self.update_mode_panel()

        last_reward_text = (
            f"{c.reward_history[-1]:+.2f}" if c.reward_history else "0.00"
        )
        self.reward_value.setText(last_reward_text)
        if hasattr(self, "last_reward_label"):
            self.last_reward_label.setText(last_reward_text)

        if c.reward_history:
            reward = c.reward_history[-1]
            bar_value = max(-100, min(100, int(reward * 20)))
            self.reward_bar.setValue(bar_value)

        self.canvas.update()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
