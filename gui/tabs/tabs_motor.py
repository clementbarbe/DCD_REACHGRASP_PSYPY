# tabs_motorplanning.py
"""
PyQt6 tab for MotorPlanning — Grasp/Touch × Hand/Tool.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QHBoxLayout,
    QLabel, QComboBox, QPushButton,
)


class MotorPlanningTab(QWidget):

    def __init__(self, parent_menu):
        super().__init__()
        self.parent_menu = parent_menu
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ── SÉQUENCE ────────────────────────────────────────────
        seq_group = QGroupBox("Séquence de runs")
        seq_layout = QHBoxLayout()

        seq_layout.addWidget(QLabel("Design :"))
        self.combo_sequence = QComboBox()
        self.combo_sequence.addItems([
            "8 runs (hand,tool alternés)",
            "4 runs (hand,tool alternés)",
            "2 runs (hand puis tool)",
        ])
        seq_layout.addWidget(self.combo_sequence)
        seq_layout.addStretch()

        seq_group.setLayout(seq_layout)
        layout.addWidget(seq_group)

        # ── LANCEMENT ───────────────────────────────────────────
        launch_group = QGroupBox("Lancement")
        launch_layout = QVBoxLayout()

        btn = QPushButton("Lancer Motor Planning")
        btn.clicked.connect(self.run_task)
        launch_layout.addWidget(btn)

        launch_group.setLayout(launch_layout)
        layout.addWidget(launch_group)

        layout.addStretch()

    # ── helpers ─────────────────────────────────────────────────

    _SEQUENCE_MAP = {
        "8 runs (hand,tool alternés)": ["hand", "tool"] * 4,
        "4 runs (hand,tool alternés)": ["hand", "tool"] * 2,
        "2 runs (hand puis tool)":     ["hand", "tool"],
    }

    def get_common(self):
        seq_label = self.combo_sequence.currentText()
        return {
            "tache":        "MotorPlanning",
            "run_sequence": self._SEQUENCE_MAP[seq_label],
        }

    def run_task(self):
        self.parent_menu.run_experiment(self.get_common())