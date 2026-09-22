"""
Compact DDQN + A* hybrid decision-analysis panel.

Target project location:
    src/gui/decision_panel.py
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
    """Displays DDQN Q-values and the final hybrid decision compactly."""

    ACTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("decisionPanel")

        root = QVBoxLayout(self)
        root.setContentsMargins(9, 7, 9, 7)
        root.setSpacing(2)

        title = QLabel("HYBRID DECISION")
        title.setObjectName("sectionLabel")
        root.addWidget(title)

        # 4 columns x 2 rows keeps all eight Q-values visible without
        # consuming the vertical space needed by the decision explanation.
        self.rows = []
        q_grid = QGridLayout()
        q_grid.setContentsMargins(0, 0, 0, 0)
        q_grid.setHorizontalSpacing(7)
        q_grid.setVerticalSpacing(2)

        for i, action_name in enumerate(self.ACTION_NAMES):
            column = i % 4
            row_index = i // 4

            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(3)

            name = QLabel(action_name)
            name.setFixedWidth(20)
            name.setObjectName("decisionAction")

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(5)
            bar.setObjectName("qBar")

            value = QLabel("--")
            value.setFixedWidth(34)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setObjectName("qValue")

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
        self.astar_label = QLabel("A*: --")
        self.final_label = QLabel("FINAL: --")
        self.valid_label = QLabel("Valid: --")
        self.reason_label = QLabel("Decision: --")

        for label in (
            self.ddqn_label,
            self.astar_label,
            self.final_label,
            self.valid_label,
        ):
            label.setObjectName("decisionInfo")
            root.addWidget(label)

        self.final_label.setObjectName("decisionFinal")

        self.reason_label.setWordWrap(True)
        self.reason_label.setObjectName("decisionReason")
        root.addWidget(self.reason_label)

        self.selected_q = QLabel("Q = --")
        self.selected_q.setObjectName("selectedQ")
        self.selected_q.setAlignment(Qt.AlignCenter)
        root.addWidget(self.selected_q)

        self.epsilon_label = QLabel("ε = --")
        self.epsilon_label.setObjectName("epsilonLabel")
        self.epsilon_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.epsilon_label)

        self.clear()

    def clear(self):
        for name, bar, value in self.rows:
            bar.setValue(0)
            value.setText("--")
            name.setStyleSheet("color: #94A3B8;")
        self.ddqn_label.setText("DDQN: --")
        self.astar_label.setText("A*: --")
        self.final_label.setText("FINAL: --")
        self.valid_label.setText("Valid: --")
        self.reason_label.setText("Decision: --")
        self.selected_q.setText("Q = --")
        self.epsilon_label.setText("ε = --")

    def _name(self, index):
        return (
            self.ACTION_NAMES[index]
            if index is not None and 0 <= index < len(self.ACTION_NAMES)
            else "--"
        )

    def _build_reason(
        self,
        final_index,
        ddqn_index,
        astar_action,
        recovery_reason,
        action_masked,
        status,
    ):
        final = self._name(final_index)
        ddqn = self._name(ddqn_index)
        astar = self._name(astar_action)

        if recovery_reason == "LOOP":
            return "A* recovery: loop detected."
        if recovery_reason == "NO_PROGRESS":
            return "A* recovery: progress stalled."
        if status == "A* REPLANNING":
            return "A* replanning: previous route blocked or invalid."
        if action_masked and final != "--" and final != ddqn:
            return f"Unsafe DDQN action masked; final: {final}."
        if astar != "--" and final == astar and ddqn != astar:
            return f"A* aligned DDQN: {ddqn} → {final}."
        if astar != "--" and final == astar:
            return f"DDQN and A* agree: {final}."
        if final != "--":
            return f"DDQN selected {final}."
        return "Waiting for a navigation decision."

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
        if q_values is None:
            self.clear()
            return

        values = [float(v) for v in q_values]
        if not values:
            self.clear()
            return

        lo, hi = min(values), max(values)
        if hi == lo:
            normalized = [50] * len(values)
        else:
            normalized = [
                int(10 + 90 * ((v - lo) / (hi - lo)))
                for v in values
            ]

        for i, (_, bar, value_label) in enumerate(self.rows):
            if i < len(values):
                bar.setValue(normalized[i])
                value_label.setText(f"{values[i]:+.2f}")
                value_label.setStyleSheet(
                    "color: #22D3EE; font-weight: 700;"
                    if selected_index == i
                    else "color: #94A3B8;"
                )
            else:
                bar.setValue(0)
                value_label.setText("--")

        if selected_index is not None and 0 <= selected_index < len(values):
            self.rows[selected_index][0].setStyleSheet(
                "color: #22D3EE; font-weight: 700;"
            )
            self.selected_q.setText(
                f"Q = {values[selected_index]:+.3f}"
            )

        ddqn_index = max(range(len(values)), key=lambda i: values[i])
        self.ddqn_label.setText(f"DDQN: {self._name(ddqn_index)}")
        self.astar_label.setText(f"A*: {self._name(astar_action)}")
        self.final_label.setText(f"FINAL: {self._name(selected_index)}")

        if valid_actions is None:
            self.valid_label.setText("Valid: --")
        else:
            self.valid_label.setText(f"Valid: {len(valid_actions)}/8")

        self.reason_label.setText(
            "Decision: "
            + self._build_reason(
                selected_index,
                ddqn_index,
                astar_action,
                recovery_reason,
                action_masked,
                status,
            )
        )

        if epsilon is not None:
            self.epsilon_label.setText(f"ε = {float(epsilon):.3f}")
