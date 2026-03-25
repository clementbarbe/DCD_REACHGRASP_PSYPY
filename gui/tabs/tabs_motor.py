# tabs_motorplanning.py
"""
PyQt6 control panel for MotorPlanning — Grasp/Touch × Hand/Tool.
Lance UN run à la fois avec un numéro de run et un effecteur.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QFrame,
    QComboBox, QMessageBox,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _h_separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def _label(text: str) -> QLabel:
    return QLabel(text)


def _spin_int(value: int, lo: int, hi: int, suffix: str = "") -> QSpinBox:
    sb = QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(value)
    if suffix:
        sb.setSuffix(suffix)
    sb.setMinimumWidth(80)
    return sb


def _spin_float(
    value: float, lo: float, hi: float,
    step: float = 0.1, decimals: int = 1, suffix: str = ""
) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(value)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    if suffix:
        sb.setSuffix(suffix)
    sb.setMinimumWidth(80)
    return sb


# ═════════════════════════════════════════════════════════════════════════════

class MotorPlanningTab(QWidget):

    def __init__(self, parent_menu):
        super().__init__()
        self.parent_menu = parent_menu
        self._init_ui()

    # ─────────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────────

    def _init_ui(self):
        root = QVBoxLayout()
        root.setSpacing(10)
        self.setLayout(root)

        root.addWidget(self._build_run_group())
        root.addWidget(self._build_trial_timing_group())
        root.addWidget(self._build_sound_group())
        root.addWidget(self._build_launch_group())

        # ── Estimation ──
        self.lbl_estimate = QLabel("")
        self.lbl_estimate.setStyleSheet("color: #2196F3; font-weight: bold;")
        self.lbl_estimate.setWordWrap(True)
        root.addWidget(self.lbl_estimate)

        root.addStretch()
        self._update_time_estimate()

    # ── Run Selection ────────────────────────────────────────────────────

    def _build_run_group(self) -> QGroupBox:
        grp = QGroupBox("🏃 Identification du Run")
        grid = QGridLayout()
        grid.setColumnStretch(2, 1)
        row = 0

        grid.addWidget(_label("Run n° :"), row, 0)
        self.spin_run = _spin_int(1, 1, 8)
        grid.addWidget(self.spin_run, row, 1)
        grid.addWidget(_label("(1-8, 4 par effecteur)"), row, 2)
        row += 1

        grid.addWidget(_label("Effecteur :"), row, 0)
        self.combo_effector = QComboBox()
        self.combo_effector.addItems(["hand", "tool"])
        self.combo_effector.setMinimumWidth(100)
        grid.addWidget(self.combo_effector, row, 1)
        grid.addWidget(_label("Main / Outil"), row, 2)
        row += 1

        grid.addWidget(_label("Essais / condition :"), row, 0)
        self.spin_trials_per_cond = _spin_int(20, 5, 50)
        self.spin_trials_per_cond.valueChanged.connect(
            self._update_time_estimate
        )
        grid.addWidget(self.spin_trials_per_cond, row, 1)
        grid.addWidget(_label("(× 2 conditions = essais/run)"), row, 2)

        grp.setLayout(grid)
        return grp

    # ── Trial Timing ─────────────────────────────────────────────────────

    def _build_trial_timing_group(self) -> QGroupBox:
        grp = QGroupBox("⏱ Timing des essais")
        grid = QGridLayout()
        grid.setColumnStretch(2, 1)
        row = 0

        grid.addWidget(_label("Preview (s) :"), row, 0)
        self.spin_preview = _spin_float(2.0, 0.5, 10.0, 0.5, 1, " s")
        self.spin_preview.valueChanged.connect(self._update_time_estimate)
        grid.addWidget(self.spin_preview, row, 1)
        row += 1

        grid.addWidget(_label("Cue duration (s) :"), row, 0)
        self.spin_cue_dur = _spin_float(0.5, 0.1, 2.0, 0.1, 1, " s")
        grid.addWidget(self.spin_cue_dur, row, 1)
        row += 1

        grid.addWidget(_label("Plan duration (s) :"), row, 0)
        self.spin_plan = _spin_float(5.5, 2.0, 15.0, 0.5, 1, " s")
        self.spin_plan.valueChanged.connect(self._update_time_estimate)
        grid.addWidget(self.spin_plan, row, 1)
        row += 1

        grid.addWidget(_label("Plan jitter (± s) :"), row, 0)
        self.spin_plan_jitter = _spin_float(0.5, 0.0, 3.0, 0.1, 1, " s")
        grid.addWidget(self.spin_plan_jitter, row, 1)
        row += 1

        grid.addWidget(_label("Go beep duration (s) :"), row, 0)
        self.spin_go_dur = _spin_float(0.5, 0.1, 2.0, 0.1, 1, " s")
        grid.addWidget(self.spin_go_dur, row, 1)
        row += 1

        grid.addWidget(_label("Execute duration (s) :"), row, 0)
        self.spin_execute = _spin_float(2.0, 0.5, 10.0, 0.5, 1, " s")
        self.spin_execute.valueChanged.connect(self._update_time_estimate)
        grid.addWidget(self.spin_execute, row, 1)
        row += 1

        grid.addWidget(_label("ITI (s) :"), row, 0)
        self.spin_iti = _spin_float(8.0, 2.0, 20.0, 1.0, 1, " s")
        self.spin_iti.valueChanged.connect(self._update_time_estimate)
        grid.addWidget(self.spin_iti, row, 1)
        row += 1

        grid.addWidget(_h_separator(), row, 0, 1, 3)
        row += 1

        grid.addWidget(_label("Baseline initiale (s) :"), row, 0)
        self.spin_baseline_init = _spin_float(10.0, 0.0, 60.0, 1.0, 1, " s")
        self.spin_baseline_init.valueChanged.connect(self._update_time_estimate)
        grid.addWidget(self.spin_baseline_init, row, 1)
        row += 1

        grid.addWidget(_label("Baseline finale (s) :"), row, 0)
        self.spin_baseline_final = _spin_float(10.0, 0.0, 60.0, 1.0, 1, " s")
        self.spin_baseline_final.valueChanged.connect(self._update_time_estimate)
        grid.addWidget(self.spin_baseline_final, row, 1)

        grp.setLayout(grid)
        return grp

    # ── Son ──────────────────────────────────────────────────────────────

    def _build_sound_group(self) -> QGroupBox:
        grp = QGroupBox("🔊 Stimuli Auditifs")
        grid = QGridLayout()
        grid.setColumnStretch(2, 1)
        row = 0

        grid.addWidget(_label("Go beep freq (Hz) :"), row, 0)
        self.spin_beep_freq = _spin_float(
            1000.0, 200.0, 5000.0, 100.0, 0, " Hz"
        )
        grid.addWidget(self.spin_beep_freq, row, 1)
        row += 1

        info = QLabel(
            "ℹ  Fichiers WAV attendus dans sounds/ : grasp.wav, touch.wav\n"
            "   Si absents, des tons purs de remplacement sont utilisés."
        )
        info.setStyleSheet("color: gray; font-style: italic;")
        grid.addWidget(info, row, 0, 1, 3)

        grp.setLayout(grid)
        return grp

    # ── Launch ───────────────────────────────────────────────────────────

    def _build_launch_group(self) -> QGroupBox:
        grp = QGroupBox("🚀 Lancement")
        layout = QVBoxLayout()

        btn = QPushButton("🧠  Lancer Motor Planning")
        btn.setMinimumHeight(48)
        btn.setStyleSheet(
            "QPushButton { background-color: #FF5722; color: white; "
            "font-weight: bold; border-radius: 4px; padding: 6px 16px; "
            "font-size: 14px; }"
            "QPushButton:hover { background-color: #E64A19; }"
        )
        btn.clicked.connect(self.run_task)
        layout.addWidget(btn)

        grp.setLayout(layout)
        return grp

    # ─────────────────────────────────────────────────────────────────────
    # TIME ESTIMATION
    # ─────────────────────────────────────────────────────────────────────

    def _update_time_estimate(self):
        n_per_cond = self.spin_trials_per_cond.value()
        n_trials   = 2 * n_per_cond

        preview  = self.spin_preview.value()
        plan     = self.spin_plan.value()
        execute  = self.spin_execute.value()
        iti      = self.spin_iti.value()
        bl_init  = self.spin_baseline_init.value()
        bl_final = self.spin_baseline_final.value()

        trial_dur = preview + plan + execute + iti
        total_s   = bl_init + n_trials * trial_dur + bl_final

        self.lbl_estimate.setText(
            f"📊  {n_trials} essais ({n_per_cond} grasp + {n_per_cond} touch)  │  "
            f"~{trial_dur:.1f} s/essai  │  "
            f"Durée run : ~{total_s / 60:.1f} min ({total_s:.0f} s)"
        )

    # ─────────────────────────────────────────────────────────────────────
    # PARAMETER ASSEMBLY
    # ─────────────────────────────────────────────────────────────────────

    def _get_params(self) -> dict:
        return {
            "tache":                     "MotorPlanning",
            "effector":                  self.combo_effector.currentText(),
            "run_number":                self.spin_run.value(),
            "n_trials_per_condition":    self.spin_trials_per_cond.value(),
            "preview_duration":          self.spin_preview.value(),
            "cue_duration":              self.spin_cue_dur.value(),
            "plan_duration":             self.spin_plan.value(),
            "plan_jitter":               self.spin_plan_jitter.value(),
            "go_duration":               self.spin_go_dur.value(),
            "execute_duration":          self.spin_execute.value(),
            "iti_duration":              self.spin_iti.value(),
            "initial_baseline":          self.spin_baseline_init.value(),
            "final_baseline":            self.spin_baseline_final.value(),
            "go_beep_freq":              self.spin_beep_freq.value(),
        }

    def _confirm_launch(self) -> bool:
        params = self._get_params()
        n_trials = 2 * params["n_trials_per_condition"]
        trial_dur = (
            params["preview_duration"] + params["plan_duration"]
            + params["execute_duration"] + params["iti_duration"]
        )
        total_s = (
            params["initial_baseline"]
            + n_trials * trial_dur
            + params["final_baseline"]
        )

        reply = QMessageBox.question(
            self,
            "Confirmer le lancement",
            f"Motor Planning — Run {params['run_number']:02d}\n"
            f"Effecteur : {params['effector']}\n"
            f"{n_trials} essais ({params['n_trials_per_condition']} grasp "
            f"+ {params['n_trials_per_condition']} touch)\n"
            f"Durée estimée : {total_s / 60:.1f} min\n\n"
            f"Lancer ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    # ─────────────────────────────────────────────────────────────────────
    # LAUNCH
    # ─────────────────────────────────────────────────────────────────────

    def run_task(self):
        if not self._confirm_launch():
            return
        params = self._get_params()
        self.parent_menu.run_experiment(params)