"""
Warehouse visualization widget for the Enhanced Hybrid DDQN-A* GUI.

Target project location:
    src/gui/warehouse_canvas.py
"""

from PySide6.QtCore import Qt, QRectF, QLineF
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import QWidget

from src.environment.constants import ACTIONS, ACTION_NAMES


class WarehouseCanvas(QWidget):
    """
    Renders the current WarehouseEnv state.

    The canvas is presentation-only:
    navigation decisions remain inside WarehouseEnv / HybridAgent.
    """

    def __init__(self, controller, parent=None):
        super().__init__(parent)

        self.controller = controller
        self.setMinimumSize(560, 560)

        # Stage 7: A* replanning visualization state.
        # Presentation-only; it does not affect navigation.
        self._last_seen_path = []
        self._replan_old_path = []
        self._replan_position = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        env = self.controller.env

        # --------------------------------------------------------------
        # Empty / uninitialized state
        # --------------------------------------------------------------

        if env is None or env.robot_pos is None:
            painter.fillRect(
                self.rect(),
                QColor("#0B1220"),
            )

            painter.setPen(
                QColor("#94A3B8")
            )

            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Press  START  to initialize the simulation",
            )

            painter.end()
            return

        grid = env._static_grid

        rows = len(grid)
        cols = len(grid[0])

        # --------------------------------------------------------------
        # Canvas geometry
        # --------------------------------------------------------------

        margin = 18

        available_w = self.width() - 2 * margin
        available_h = self.height() - 2 * margin

        cell = min(
            available_w / cols,
            available_h / rows,
        )

        board_w = cell * cols
        board_h = cell * rows

        ox = (self.width() - board_w) / 2
        oy = (self.height() - board_h) / 2

        painter.fillRect(
            self.rect(),
            QColor("#0B1220"),
        )

        # --------------------------------------------------------------
        # Helper: grid position -> screen position
        # --------------------------------------------------------------

        def center(pos):
            r, c = pos

            return (
                ox + c * cell + cell / 2,
                oy + r * cell + cell / 2,
            )

        # --------------------------------------------------------------
        # 1. Warehouse floor and static shelves
        # --------------------------------------------------------------

        for r in range(rows):
            for c in range(cols):

                x = ox + c * cell
                y = oy + r * cell

                rect = QRectF(
                    x,
                    y,
                    cell,
                    cell,
                )

                if grid[r][c] == 1:
                    painter.setBrush(
                        QBrush(
                            QColor("#334155")
                        )
                    )

                else:
                    floor = (
                        "#172033"
                        if (r + c) % 2 == 0
                        else "#1B263A"
                    )

                    painter.setBrush(
                        QBrush(
                            QColor(floor)
                        )
                    )

                painter.setPen(
                    QPen(
                        QColor("#273449"),
                        1,
                    )
                )

                painter.drawRect(rect)

                # Shelf inner block
                if grid[r][c] == 1:

                    inner = rect.adjusted(
                        cell * 0.12,
                        cell * 0.12,
                        -cell * 0.12,
                        -cell * 0.12,
                    )

                    painter.setBrush(
                        QBrush(
                            QColor("#475569")
                        )
                    )

                    painter.setPen(Qt.NoPen)

                    painter.drawRoundedRect(
                        inner,
                        4,
                        4,
                    )

        # --------------------------------------------------------------
        # 2. A* route + Stage 7 replanning visualization
        # --------------------------------------------------------------

        path = list(
            getattr(
                self.controller,
                "path",
                [],
            ) or []
        )

        show_astar = getattr(
            self.controller,
            "show_astar",
            True,
        )

        last_status = getattr(
            self.controller,
            "last_status",
            "",
        )

        # A new/reset simulation clears the visualization history.
        if not path:
            self._last_seen_path = []
            self._replan_old_path = []
            self._replan_position = None

        # When the controller reports a replan and the route changed,
        # preserve the previous route for comparison.
        elif (
            last_status == "A* REPLANNING"
            and self._last_seen_path
            and path != self._last_seen_path
        ):
            self._replan_old_path = list(
                self._last_seen_path
            )
            self._replan_position = getattr(
                env,
                "robot_pos",
                None,
            )

        if path:
            self._last_seen_path = list(path)

        # --------------------------------------------------------------
        # 2A. Previous A* route — faded/dashed
        # --------------------------------------------------------------
        if (
            show_astar
            and self._replan_old_path
            and len(self._replan_old_path) >= 2
        ):
            old_points = [
                center(p)
                for p in self._replan_old_path
            ]

            old_pen = QPen(
                QColor("#64748B"),
                max(
                    2.0,
                    cell * 0.055,
                ),
            )
            old_pen.setStyle(Qt.DashLine)
            old_pen.setCapStyle(Qt.RoundCap)
            old_pen.setJoinStyle(Qt.RoundJoin)

            painter.setPen(old_pen)

            for p1, p2 in zip(
                old_points,
                old_points[1:],
            ):
                painter.drawLine(
                    QLineF(
                        p1[0],
                        p1[1],
                        p2[0],
                        p2[1],
                    )
                )

        # --------------------------------------------------------------
        # 2B. Current A* route — bright/new
        # --------------------------------------------------------------
        if (
            show_astar
            and len(path) >= 2
        ):
            points = [
                center(p)
                for p in path
            ]

            pen = QPen(
                QColor("#38BDF8"),
                max(
                    2.0,
                    cell * 0.075,
                ),
            )
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)

            painter.setPen(pen)

            for p1, p2 in zip(
                points,
                points[1:],
            ):
                painter.drawLine(
                    QLineF(
                        p1[0],
                        p1[1],
                        p2[0],
                        p2[1],
                    )
                )

            # A* waypoint markers
            painter.setPen(Qt.NoPen)
            painter.setBrush(
                QBrush(
                    QColor("#67E8F9")
                )
            )

            radius = max(
                2.0,
                cell * 0.07,
            )

            for point in points[1:-1]:
                painter.drawEllipse(
                    QRectF(
                        point[0] - radius,
                        point[1] - radius,
                        radius * 2,
                        radius * 2,
                    )
                )

        # --------------------------------------------------------------
        # 2C. Replanning point
        # --------------------------------------------------------------
        if (
            show_astar
            and self._replan_position is not None
        ):
            px, py = center(
                self._replan_position
            )

            painter.setPen(
                QPen(
                    QColor("#A78BFA"),
                    max(
                        2.0,
                        cell * 0.045,
                    ),
                )
            )
            painter.setBrush(Qt.NoBrush)

            painter.drawEllipse(
                QRectF(
                    px - cell * 0.28,
                    py - cell * 0.28,
                    cell * 0.56,
                    cell * 0.56,
                )
            )

            painter.setPen(Qt.NoPen)
            painter.setBrush(
                QBrush(
                    QColor("#C4B5FD")
                )
            )

            painter.drawEllipse(
                QRectF(
                    px - cell * 0.09,
                    py - cell * 0.09,
                    cell * 0.18,
                    cell * 0.18,
                )
            )

            painter.setPen(
                QColor("#DDD6FE")
            )

            painter.drawText(
                QRectF(
                    px + cell * 0.30,
                    py - cell * 0.20,
                    cell * 1.6,
                    cell * 0.35,
                ),
                Qt.AlignLeft | Qt.AlignVCenter,
                "REPLAN",
            )

        # --------------------------------------------------------------
        # 3. Actual robot trajectory
        # --------------------------------------------------------------

        trajectory = getattr(
            self.controller,
            "trajectory",
            [],
        )

        if len(trajectory) >= 2:

            points = [
                center(p)
                for p in trajectory
            ]

            pen = QPen(
                QColor("#FBBF24"),
                max(
                    2.0,
                    cell * 0.035,
                ),
            )

            pen.setStyle(
                Qt.DashLine
            )

            pen.setCapStyle(
                Qt.RoundCap
            )

            painter.setPen(pen)

            for p1, p2 in zip(
                points,
                points[1:],
            ):

                painter.drawLine(
                    QLineF(
                        p1[0],
                        p1[1],
                        p2[0],
                        p2[1],
                    )
                )

        # --------------------------------------------------------------
        # 4. Start position
        # --------------------------------------------------------------

        if trajectory:

            start_pos = trajectory[0]

            sx, sy = center(
                start_pos
            )

            painter.setPen(
                QPen(
                    QColor("#94A3B8"),
                    max(
                        1.5,
                        cell * 0.035,
                    ),
                )
            )

            painter.setBrush(
                Qt.NoBrush
            )

            painter.drawEllipse(
                QRectF(
                    sx - cell * 0.18,
                    sy - cell * 0.18,
                    cell * 0.36,
                    cell * 0.36,
                )
            )

            painter.setPen(
                QColor("#CBD5E1")
            )

            painter.drawText(
                QRectF(
                    sx + cell * 0.25,
                    sy - cell * 0.30,
                    cell * 1.2,
                    cell * 0.4,
                ),
                Qt.AlignLeft
                | Qt.AlignVCenter,
                "START",
            )

        # --------------------------------------------------------------
        # 5. Goal
        # --------------------------------------------------------------

        if env.goal_pos is not None:

            gx, gy = center(
                env.goal_pos
            )

            painter.setPen(
                QPen(
                    QColor("#4ADE80"),
                    max(
                        2.0,
                        cell * 0.07,
                    ),
                )
            )

            painter.setBrush(
                Qt.NoBrush
            )

            painter.drawEllipse(
                QRectF(
                    gx - cell * 0.28,
                    gy - cell * 0.28,
                    cell * 0.56,
                    cell * 0.56,
                )
            )

            painter.setPen(
                QColor("#4ADE80")
            )

            painter.drawText(
                QRectF(
                    gx + cell * 0.28,
                    gy - cell * 0.30,
                    cell * 1.0,
                    cell * 0.4,
                ),
                Qt.AlignLeft
                | Qt.AlignVCenter,
                "GOAL",
            )

            painter.setPen(Qt.NoPen)

            painter.setBrush(
                QBrush(
                    QColor("#4ADE80")
                )
            )

            painter.drawEllipse(
                QRectF(
                    gx - cell * 0.10,
                    gy - cell * 0.10,
                    cell * 0.20,
                    cell * 0.20,
                )
            )

        # --------------------------------------------------------------
        # --------------------------------------------------------------
        # 6. Dynamic workers
        # --------------------------------------------------------------

        # Stage 6:
        # Visually detect dynamic obstacles that are close to the robot.
        # This does not change navigation logic; it only improves
        # situational awareness in the GUI.
        nearby_dynamic = False
        robot_position = getattr(
            env,
            "robot_pos",
            None,
        )

        for worker in env.workers:
            if not getattr(
                worker,
                "active",
                True,
            ):
                continue

            wx, wy = center(
                worker.position
            )

            is_nearby = False

            if robot_position is not None:
                rr, rc = robot_position
                wr, wc = worker.position

                # Chebyshev distance gives the number of 8-connected
                # grid steps between the robot and the worker.
                distance = max(
                    abs(rr - wr),
                    abs(rc - wc),
                )

                is_nearby = distance <= 2

                if is_nearby:
                    nearby_dynamic = True

            # Warning ring for a nearby worker
            if is_nearby:
                painter.setPen(
                    QPen(
                        QColor("#FB7185"),
                        max(
                            2.0,
                            cell * 0.035,
                        ),
                    )
                )
                painter.setBrush(Qt.NoBrush)

                painter.drawEllipse(
                    QRectF(
                        wx - cell * 0.38,
                        wy - cell * 0.38,
                        cell * 0.76,
                        cell * 0.76,
                    )
                )

            # Worker outer ring
            painter.setPen(
                QPen(
                    QColor(
                        "#FB7185"
                        if is_nearby
                        else "#FDBA74"
                    ),
                    max(
                        1.5,
                        cell * 0.035,
                    ),
                )
            )

            painter.setBrush(
                QBrush(
                    QColor("#F97316")
                )
            )

            painter.drawEllipse(
                QRectF(
                    wx - cell * 0.22,
                    wy - cell * 0.22,
                    cell * 0.44,
                    cell * 0.44,
                )
            )

            # Worker helmet
            painter.setPen(Qt.NoPen)

            painter.setBrush(
                QBrush(
                    QColor("#FACC15")
                )
            )

            painter.drawEllipse(
                QRectF(
                    wx - cell * 0.11,
                    wy - cell * 0.11,
                    cell * 0.22,
                    cell * 0.22,
                )
            )

            # Nearby label
            if is_nearby:
                painter.setPen(
                    QColor("#FB7185")
                )

                painter.drawText(
                    QRectF(
                        wx + cell * 0.25,
                        wy - cell * 0.18,
                        cell * 0.95,
                        cell * 0.30,
                    ),
                    Qt.AlignLeft
                    | Qt.AlignVCenter,
                    "NEARBY",
                )

        # --------------------------------------------------------------
        # 7. Dynamic robots
        # --------------------------------------------------------------

        for dynamic_robot in env.dynamic_robots:
            if not getattr(
                dynamic_robot,
                "active",
                True,
            ):
                continue

            rx, ry = center(
                dynamic_robot.position
            )

            is_nearby = False

            if robot_position is not None:
                rr, rc = robot_position
                dr, dc = dynamic_robot.position

                distance = max(
                    abs(rr - dr),
                    abs(rc - dc),
                )

                is_nearby = distance <= 2

                if is_nearby:
                    nearby_dynamic = True

            # Warning ring for a nearby dynamic robot
            if is_nearby:
                painter.setPen(
                    QPen(
                        QColor("#FB7185"),
                        max(
                            2.0,
                            cell * 0.035,
                        ),
                    )
                )
                painter.setBrush(Qt.NoBrush)

                painter.drawEllipse(
                    QRectF(
                        rx - cell * 0.38,
                        ry - cell * 0.38,
                        cell * 0.76,
                        cell * 0.76,
                    )
                )

            # Dynamic robot body
            painter.setPen(
                QPen(
                    QColor(
                        "#FB7185"
                        if is_nearby
                        else "#C4B5FD"
                    ),
                    max(
                        1.5,
                        cell * 0.035,
                    ),
                )
            )

            painter.setBrush(
                QBrush(
                    QColor("#7C3AED")
                )
            )

            painter.drawRoundedRect(
                QRectF(
                    rx - cell * 0.22,
                    ry - cell * 0.22,
                    cell * 0.44,
                    cell * 0.44,
                ),
                5,
                5,
            )

            # Nearby label
            if is_nearby:
                painter.setPen(
                    QColor("#FB7185")
                )

                painter.drawText(
                    QRectF(
                        rx + cell * 0.25,
                        ry - cell * 0.18,
                        cell * 0.95,
                        cell * 0.30,
                    ),
                    Qt.AlignLeft
                    | Qt.AlignVCenter,
                    "NEARBY",
                )

        # 8. Collision-aware action mask
        # --------------------------------------------------------------

        if env.robot_pos is not None:

            valid_actions = getattr(
                self.controller,
                "valid_actions",
                None,
            )

            if valid_actions is not None:

                rx_mask, ry_mask = center(
                    env.robot_pos
                )

                # Draw the eight possible action directions
                # around the robot.
                for action_index, (dr, dc) in ACTIONS.items():

                    distance = cell * 0.43

                    tx = (
                        rx_mask
                        + dc * distance
                    )

                    ty = (
                        ry_mask
                        + dr * distance
                    )

                    is_valid = (
                        action_index
                        in valid_actions
                    )

                    if is_valid:

                        # Valid action indicator
                        painter.setPen(
                            QPen(
                                QColor("#4ADE80"),
                                max(
                                    1.5,
                                    cell * 0.025,
                                ),
                            )
                        )

                        painter.setBrush(
                            QBrush(
                                QColor(
                                    74,
                                    222,
                                    128,
                                    80,
                                )
                            )
                        )

                        painter.drawEllipse(
                            QRectF(
                                tx - cell * 0.055,
                                ty - cell * 0.055,
                                cell * 0.11,
                                cell * 0.11,
                            )
                        )

                    else:

                        # Blocked action indicator
                        painter.setPen(
                            QPen(
                                QColor("#FB7185"),
                                max(
                                    1.5,
                                    cell * 0.025,
                                ),
                            )
                        )

                        painter.setBrush(Qt.NoBrush)

                        cross_size = cell * 0.07

                        painter.drawLine(
                            QLineF(
                                tx - cross_size,
                                ty - cross_size,
                                tx + cross_size,
                                ty + cross_size,
                            )
                        )

                        painter.drawLine(
                            QLineF(
                                tx - cross_size,
                                ty + cross_size,
                                tx + cross_size,
                                ty - cross_size,
                            )
                        )

                    # Action name
                    if 0 <= action_index < len(ACTION_NAMES):
                        action_name = ACTION_NAMES[action_index]
                    else:
                        action_name = str(action_index)

                    painter.setPen(
                        QColor(
                            "#86EFAC" if is_valid else "#FDA4AF"
                        )
                    )

                    painter.drawText(
                        QRectF(
                            tx - cell * 0.16,
                            ty + cell * 0.08,
                            cell * 0.32,
                            cell * 0.22,
                        ),
                        Qt.AlignCenter,
                        action_name,
                    )

        # --------------------------------------------------------------
        # 9. Robot
        # --------------------------------------------------------------

        if env.robot_pos is not None:

            rx, ry = center(
                env.robot_pos
            )

            # Robot outer/status ring
            painter.setPen(
                QPen(
                    QColor("#22D3EE"),
                    max(
                        2.0,
                        cell * 0.055,
                    ),
                )
            )

            painter.setBrush(
                QBrush(
                    QColor("#0E7490")
                )
            )

            painter.drawRoundedRect(
                QRectF(
                    rx - cell * 0.29,
                    ry - cell * 0.29,
                    cell * 0.58,
                    cell * 0.58,
                ),
                7,
                7,
            )

            # Robot center
            painter.setPen(Qt.NoPen)

            painter.setBrush(
                QBrush(
                    QColor("#67E8F9")
                )
            )

            painter.drawEllipse(
                QRectF(
                    rx - cell * 0.07,
                    ry - cell * 0.07,
                    cell * 0.14,
                    cell * 0.14,
                )
            )

            # ----------------------------------------------------------
            # Current movement direction
            # ----------------------------------------------------------

            last_action = getattr(
                self.controller,
                "last_action",
                None,
            )

            if last_action is not None:

                move = ACTIONS.get(
                    last_action
                )

                if move is not None:

                    dr, dc = move

                    if dr != 0 or dc != 0:

                        length = cell * 0.30

                        ex = (
                            rx
                            + dc * length
                        )

                        ey = (
                            ry
                            + dr * length
                        )

                        # Direction line
                        painter.setPen(
                            QPen(
                                QColor("#FFFFFF"),
                                max(
                                    2.0,
                                    cell * 0.045,
                                ),
                            )
                        )

                        painter.drawLine(
                            QLineF(
                                rx,
                                ry,
                                ex,
                                ey,
                            )
                        )

                        # Direction label
                        if (
                            0
                            <= last_action
                            < len(ACTION_NAMES)
                        ):
                            action_name = (
                                ACTION_NAMES[
                                    last_action
                                ]
                            )
                        else:
                            action_name = str(
                                last_action
                            )

                        painter.setPen(
                            QColor("#FFFFFF")
                        )

                        painter.drawText(
                            QRectF(
                                ex + cell * 0.08,
                                ey - cell * 0.18,
                                cell * 1.0,
                                cell * 0.35,
                            ),
                            Qt.AlignLeft
                            | Qt.AlignVCenter,
                            action_name,
                        )

        # --------------------------------------------------------------
        # 10. Current waypoint highlight
        # --------------------------------------------------------------

        waypoint = getattr(
            self.controller,
            "last_waypoint",
            None,
        )

        if (
            waypoint is not None
            and waypoint != env.robot_pos
        ):

            wx, wy = center(
                waypoint
            )

            painter.setPen(
                QPen(
                    QColor("#A78BFA"),
                    max(
                        2.0,
                        cell * 0.045,
                    ),
                )
            )

            painter.setBrush(
                Qt.NoBrush
            )

            painter.drawRect(
                QRectF(
                    wx - cell * 0.31,
                    wy - cell * 0.31,
                    cell * 0.62,
                    cell * 0.62,
                )
            )

            painter.setPen(
                QColor("#C4B5FD")
            )

            painter.drawText(
                QRectF(
                    wx + cell * 0.30,
                    wy - cell * 0.30,
                    cell * 1.4,
                    cell * 0.4,
                ),
                Qt.AlignLeft
                | Qt.AlignVCenter,
                "WAYPOINT",
            )

        # --------------------------------------------------------------
        # 11. Live hybrid navigation status
        # --------------------------------------------------------------

        hybrid = getattr(
            self.controller,
            "hybrid",
            None,
        )

        if hybrid is not None:

            # ----------------------------------------------------------
            # Current action
            # ----------------------------------------------------------

            action = getattr(
                self.controller,
                "last_action",
                None,
            )

            if (
                action is not None
                and 0 <= action < len(ACTION_NAMES)
            ):
                action_name = ACTION_NAMES[action]
            elif action is not None:
                action_name = str(action)
            else:
                action_name = "--"

            # ----------------------------------------------------------
            # Determine current hybrid state
            # ----------------------------------------------------------

            valid_actions = getattr(
                self.controller,
                "valid_actions",
                None,
            )

            if valid_actions is not None:
                valid_text = (
                    f"{len(valid_actions)}/{len(ACTIONS)} VALID"
                )
            else:
                valid_text = "MASK --"

            if hybrid.last_loop_detected:

                status_title = "LOOP RECOVERY"

                status_detail = (
                    f"A* RECOVERY  •  ACTION {action_name}  •  "
                    f"{valid_text}"
                )

                status_color = "#FB7185"

            elif hybrid.last_progress_stall:

                status_title = "PROGRESS RECOVERY"

                status_detail = (
                    f"A* RECOVERY  •  ACTION {action_name}  •  "
                    f"{valid_text}"
                )

                status_color = "#FBBF24"

            elif getattr(
                self.controller,
                "last_status",
                "",
            ) == "A* REPLANNING":

                status_title = "A* REPLANNING"

                status_detail = (
                    f"OLD PATH → NEW PATH  •  ACTION {action_name}  •  "
                    f"{valid_text}"
                )

                status_color = "#A78BFA"

            elif nearby_dynamic:

                status_title = "DYNAMIC OBSTACLE NEARBY"

                status_detail = (
                    f"DDQN + A* GUIDANCE  •  ACTION {action_name}  •  "
                    f"{valid_text}"
                )

                status_color = "#FB7185"

            elif hybrid.last_astar_action is not None:

                status_title = "NORMAL HYBRID CONTROL"

                status_detail = (
                    f"DDQN + A* GUIDANCE  •  ACTION {action_name}  •  "
                    f"{valid_text}"
                )

                status_color = "#22D3EE"

            else:

                status_title = "DDQN CONTROL"

                status_detail = (
                    f"LEARNED ACTION  •  ACTION {action_name}  •  "
                    f"{valid_text}"
                )

                status_color = "#4ADE80"

            # ----------------------------------------------------------
            # Overlay dimensions
            # ----------------------------------------------------------

            overlay_width = min(
                280,
                max(
                    220,
                    int(board_w * 0.42),
                ),
            )

            overlay_height = 58

            overlay_x = ox + 8

            overlay_y = (
                oy
                + board_h
                - overlay_height
                - 8
            )

            # ----------------------------------------------------------
            # Overlay background
            # ----------------------------------------------------------

            painter.setBrush(
                QBrush(
                    QColor(
                        15,
                        23,
                        42,
                        235,
                    )
                )
            )

            painter.setPen(
                QPen(
                    QColor(status_color),
                    1.5,
                )
            )

            painter.drawRoundedRect(
                QRectF(
                    overlay_x,
                    overlay_y,
                    overlay_width,
                    overlay_height,
                ),
                8,
                8,
            )

            # ----------------------------------------------------------
            # Status indicator
            # ----------------------------------------------------------

            painter.setBrush(
                QBrush(
                    QColor(status_color)
                )
            )

            painter.setPen(Qt.NoPen)

            painter.drawEllipse(
                QRectF(
                    overlay_x + 12,
                    overlay_y + 14,
                    9,
                    9,
                )
            )

            # ----------------------------------------------------------
            # Main status
            # ----------------------------------------------------------

            painter.setPen(
                QColor("#F8FAFC")
            )

            painter.drawText(
                QRectF(
                    overlay_x + 28,
                    overlay_y + 7,
                    overlay_width - 38,
                    20,
                ),
                Qt.AlignLeft
                | Qt.AlignVCenter,
                status_title,
            )

            # ----------------------------------------------------------
            # Secondary status
            # ----------------------------------------------------------

            painter.setPen(
                QColor("#94A3B8")
            )

            painter.drawText(
                QRectF(
                    overlay_x + 28,
                    overlay_y + 29,
                    overlay_width - 38,
                    18,
                ),
                Qt.AlignLeft
                | Qt.AlignVCenter,
                status_detail,
            )

        # --------------------------------------------------------------
        # Stage 7 route legend
        # --------------------------------------------------------------
        if (
            show_astar
            and self._replan_old_path
        ):
            legend_x = ox + 10
            legend_y = oy + 10

            painter.setPen(
                QColor("#CBD5E1")
            )
            painter.drawText(
                QRectF(
                    legend_x,
                    legend_y,
                    210,
                    18,
                ),
                Qt.AlignLeft | Qt.AlignVCenter,
                "— — OLD A* PATH",
            )

            painter.setPen(
                QColor("#38BDF8")
            )
            painter.drawText(
                QRectF(
                    legend_x,
                    legend_y + 20,
                    210,
                    18,
                ),
                Qt.AlignLeft | Qt.AlignVCenter,
                "━━ NEW A* PATH",
            )

        # --------------------------------------------------------------
        # Finish painting
        # --------------------------------------------------------------

        painter.end()