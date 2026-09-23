"""
Compact DDQN + A* decision-analysis panel.

Target project location:
    src/gui/decision_panel.py

The panel is presentation-only. It receives values from MainWindow and
does not change navigation decisions.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
)


class DecisionPanel(QFrame):
    ACTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setObjectName("decisionPanel")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(4)

        title = QLabel("HYBRID DECISION")
        title.setObjectName("decisionSubTitle")
        root.addWidget(title)

        # Four columns × two rows keeps all eight actions readable.
        q_grid = QGridLayout()
        q_grid.setContentsMargins(0, 0, 0, 0)
        q_grid.setHorizontalSpacing(9)
        q_grid.setVerticalSpacing(5)

        self.rows = []

        for i, action_name in enumerate(self.ACTION_NAMES):
            row_index = i // 4
            column = i % 4

            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)

            name = QLabel(action_name)
            name.setObjectName("decisionAction")
            name.setFixedWidth(22)

            bar = QProgressBar()
            bar.setObjectName("qBar")
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(7)

            value = QLabel("--")
            value.setObjectName("qValue")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setMinimumWidth(48)

            row.addWidget(name)
            row.addWidget(bar, 1)
            row.addWidget(value)

            q_grid.addLayout(row, row_index, column)
            self.rows.append((name, bar, value))

        root.addLayout(q_grid)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setObjectName("decisionDivider")
        root.addWidget(divider)

        self.ddqn_label = QLabel("DDQN: --")
        self.ddqn_label.setObjectName("decisionInfo")

        self.astar_label = QLabel("A*: --")
        self.astar_label.setObjectName("decisionInfo")

        self.final_label = QLabel("FINAL: --")
        self.final_label.setObjectName("hybridDecisionValue")

        self.valid_label = QLabel("Valid: --")
        self.valid_label.setObjectName("decisionInfo")

        self.reason_label = QLabel("Decision: --")
        self.reason_label.setObjectName("decisionInfo")
        self.reason_label.setWordWrap(True)

        bottom = QGridLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setHorizontalSpacing(8)
        bottom.setVerticalSpacing(2)

        bottom.addWidget(self.ddqn_label, 0, 0)
        bottom.addWidget(self.astar_label, 1, 0)
        bottom.addWidget(self.final_label, 2, 0)
        bottom.addWidget(self.valid_label, 3, 0)
        bottom.addWidget(self.reason_label, 4, 0)

        self.selected_q = QLabel("Q = --")
        self.selected_q.setObjectName("selectedQ")
        self.selected_q.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.epsilon_label = QLabel("ε = --")
        self.epsilon_label.setObjectName("epsilonLabel")
        self.epsilon_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        right_info = QVBoxLayout()
        right_info.setContentsMargins(0, 0, 0, 0)
        right_info.setSpacing(2)
        right_info.addWidget(self.selected_q)
        right_info.addWidget(self.epsilon_label)

        bottom.addLayout(right_info, 0, 1, 5, 1)
        root.addLayout(bottom)

        self.clear()

    @staticmethod
    def _format_q(value):
        if value is None:
            return "--"
        return f"{float(value):+.1f}"

    def _set_bar(self, bar, value, selected=False):
        if value is None:
            bar.setValue(0)
            return

        # Presentation-only normalization. It does not affect the agent.
        q = float(value)
        normalized = int(max(0, min(100, 50 + q * 1.5)))
        bar.setValue(normalized)

        if selected:
            bar.setStyleSheet(
                """
                QProgressBar#qBar {
                    background: #172033;
                    border: none;
                    border-radius: 3px;
                    height: 7px;
                }
                QProgressBar#qBar::chunk {
                    background: #22D3EE;
                    border-radius: 3px;
                }
                """
            )
        else:
            bar.setStyleSheet(
                """
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
                """
            )

    def update_values(
        self,
        q_values,
        selected_index=None,
        epsilon=None,
        astar_action=None,
        recovery_reason=None,
        valid_actions=None,
        action_masked=False,
        status=None,
    ):
        values = list(q_values) if q_values is not None else []

        selected_q = None

        for i, (name, bar, value_label) in enumerate(self.rows):
            q = values[i] if i < len(values) else None
            is_selected = i == selected_index

            name.setStyleSheet(
                "color: #22D3EE; font-weight: 700;"
                if is_selected
                else "color: #94A3B8; font-weight: 600;"
            )

            value_label.setText(self._format_q(q))
            value_label.setStyleSheet(
                "color: #E2E8F0; font-weight: 700;"
                if is_selected
                else "color: #94A3B8;"
            )

            self._set_bar(bar, q, selected=is_selected)

            if is_selected:
                selected_q = q

        def action_text(action):
            if action is None:
                return "--"
            if 0 <= int(action) < len(self.ACTION_NAMES):
                return self.ACTION_NAMES[int(action)]
            return str(action)

        self.ddqn_label.setText(f"DDQN: {action_text(selected_index)}")
        self.astar_label.setText(f"A*: {action_text(astar_action)}")
        self.final_label.setText(f"FINAL: {action_text(selected_index)}")

        if valid_actions is None:
            self.valid_label.setText("Valid: --")
        else:
            self.valid_label.setText(
                f"Valid: {len(valid_actions)}/{len(self.ACTION_NAMES)}"
            )

        if recovery_reason:
            reason = str(recovery_reason).replace("_", " ")
            self.reason_label.setText(
                f"Decision: A* recovery: {reason}."
            )
        elif action_masked:
            self.reason_label.setText(
                "Decision: collision-aware action mask."
            )
        elif astar_action is not None:
            self.reason_label.setText(
                "Decision: DDQN + A* guidance."
            )
        elif status:
            self.reason_label.setText(
                f"Decision: {str(status).lower()}."
            )
        else:
            self.reason_label.setText("Decision: --")

        self.selected_q.setText(
            f"Q = {self._format_q(selected_q)}"
        )

        if epsilon is None:
            self.epsilon_label.setText("ε = --")
        else:
            self.epsilon_label.setText(
                f"ε = {float(epsilon):.3f}"
            )

    def clear(self):
        for name, bar, value_label in self.rows:
            name.setStyleSheet(
                "color: #94A3B8; font-weight: 600;"
            )
            value_label.setText("--")
            value_label.setStyleSheet("color: #94A3B8;")
            bar.setValue(0)
            self._set_bar(bar, None)

        self.ddqn_label.setText("DDQN: --")
        self.astar_label.setText("A*: --")
        self.final_label.setText("FINAL: --")
        self.valid_label.setText("Valid: --")
        self.reason_label.setText("Decision: --")
        self.selected_q.setText("Q = --")
        self.epsilon_label.setText("ε = --")