# tabs_handrepresentation.py

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox,
                              QHBoxLayout, QLabel, QComboBox, QPushButton)


class HandRepresentationTab(QWidget):

    def __init__(self, parent_menu):
        super().__init__()
        self.parent_menu = parent_menu
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ── MAIN ────────────────────────────────────────────────
        hand_group = QGroupBox("Main testée")
        hand_layout = QHBoxLayout()

        hand_layout.addWidget(QLabel("Main :"))
        self.combo_hand = QComboBox()
        self.combo_hand.addItems(["droite", "gauche"])
        hand_layout.addWidget(self.combo_hand)
        hand_layout.addStretch()

        hand_group.setLayout(hand_layout)
        layout.addWidget(hand_group)

        # ── BLOCK ───────────────────────────────────────────────
        block_group = QGroupBox("Block (100 trials)")
        block_layout = QHBoxLayout()

        block_layout.addWidget(QLabel("Block :"))
        self.combo_block = QComboBox()
        self.combo_block.addItems(["Block 1 Pre", "Block 2 Pre", "Block 3 Post"])
        block_layout.addWidget(self.combo_block)
        block_layout.addStretch()

        block_group.setLayout(block_layout)
        layout.addWidget(block_group)

        # ── LANCEMENT ───────────────────────────────────────────
        launch_group = QGroupBox("Lancement")
        launch_layout = QVBoxLayout()

        btn = QPushButton("Lancer Hand Representation")
        btn.clicked.connect(self.run_task)
        launch_layout.addWidget(btn)

        launch_group.setLayout(launch_layout)
        layout.addWidget(launch_group)

        layout.addStretch()

    # ── helpers ─────────────────────────────────────────────────

    def get_common(self):
        block_map = {"Block 1 Pre": 1, "Block 2 Pre": 2, "Block 3 Post": 3}
        block_label = self.combo_block.currentText()
        return {
            "tache":          "HandRepresentation",
            "hand":           self.combo_hand.currentText(),   # 'droite' ou 'gauche'
            "block_label":    block_label,
            "block_number":   block_map[block_label],
            "n_blocks":       1,
            "trial_duration": 4.0,
            "camera_index":   0,
        }

    def run_task(self):
        self.parent_menu.run_experiment(self.get_common())