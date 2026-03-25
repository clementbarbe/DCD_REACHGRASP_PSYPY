from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QTabWidget, QLineEdit, QCheckBox, QLabel,
                            QSpinBox, QGroupBox, QMessageBox, QComboBox)
from PyQt6.QtGui import QFont
import sys

# Direct imports for task tabs
from gui.tabs.tabs_motor import MotorPlanningTab
from gui.tabs.tabs_hand import HandRepresentationTab
from utils.utils import is_valid_name
from utils.logger import get_logger

logger = get_logger()

class ExperimentMenu(QMainWindow):
    def __init__(self, last_config=None):
        super().__init__()
        self.setWindowTitle("Configuration Expérimentale")
        
        # --- POLICE GLOBALE TAILLE 12 (S'applique à TOUT, y compris Hardware) ---
        self.global_font = QFont("Segoe UI", 12)
        self.setFont(self.global_font)
        
        # Fenêtre redimensionnée pour le confort visuel
        self.setFixedSize(1400, 1200)
        
        self.hardware_present = False 
        self.eyelink_present = False  
        self.final_config = None

        self.check_hardware_availability()

        self.default_config = {
            'nom': '', 'session': '01', 'enregistrer': True, 
            'fullscr': True, 'screenid': 1, 'monitor' : 'temp_monitor', 
            'colorspace' : 'rgb', 'parport_actif': False, 
            'eyetracker_actif':False, 'mode': 'fmri'
        }

        if last_config:
            self.default_config.update(last_config)
            try:
                current_sess = int(self.default_config['session'])
                self.default_config['session'] = f"{current_sess + 1:02d}"
            except ValueError: pass

        self.initUI()

    def check_hardware_availability(self):
        # Logique de détection (Identique à l'original)
        try:
            from hardware.parport import ParPort
            test_port = ParPort(address=0x378)
            self.hardware_present = not test_port.dummy_mode
        except: self.hardware_present = False

        try:
            import pylink
            self.eyelink_present = True
        except: self.eyelink_present = False

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        main_widget.setLayout(main_layout)
        
        self.create_general_section(main_layout)
        self.create_task_tabs(main_layout)

    def create_general_section(self, parent_layout):
        group = QGroupBox("Configuration Générale")
        layout = QHBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 20, 15, 15)

        # -- Champs Participant & Display --
        layout.addWidget(QLabel("ID Participant:"))
        self.txt_name = QLineEdit()
        self.txt_name.setFixedWidth(180)
        self.txt_name.setText(self.default_config.get('nom', ''))
        layout.addWidget(self.txt_name)
        
        layout.addWidget(QLabel("Session:"))
        self.spin_session = QSpinBox()
        self.spin_session.setRange(1, 20)
        self.spin_session.setFixedWidth(100)
        try: self.spin_session.setValue(int(self.default_config.get('session', 1)))
        except: self.spin_session.setValue(1)
        layout.addWidget(self.spin_session)

        layout.addWidget(QLabel("Écran:"))
        self.screenid = QSpinBox()
        self.screenid.setRange(1, len(QApplication.screens()))
        self.screenid.setFixedWidth(100)
        saved_screen = self.default_config.get('screenid', 1)
        self.screenid.setValue(saved_screen + 1)
        layout.addWidget(self.screenid)
        
        layout.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["fmri", "PC"])
        self.combo_mode.setCurrentText(self.default_config.get('mode', 'fmri'))
        layout.addWidget(self.combo_mode)

        self.chk_save = QCheckBox("Enregistrer")
        self.chk_save.setChecked(self.default_config.get('enregistrer', True))
        layout.addWidget(self.chk_save)

        ACTIVE_STYLE = "color: #2e7d32; font-size: 16px;" 
        INACTIVE_STYLE = "color: #757575; font-size: 16px;" 

        self.chk_parport = QCheckBox("Port Parallèle")
        self.chk_eyetracker = QCheckBox("Eye Tracker")

        for chk, present, key in [(self.chk_parport, self.hardware_present, 'parport_actif'), 
                                  (self.chk_eyetracker, self.eyelink_present, 'eyetracker_actif')]:
            
            lbl_sep = QLabel("|")
            # On ajuste aussi la taille du séparateur pour qu'il suive
            lbl_sep.setStyleSheet("color: #bdbdbd; font-size: 14px;") 
            layout.addWidget(lbl_sep)
            
            chk.setChecked(present and self.default_config.get(key, False))
            chk.setEnabled(present)
            chk.setStyleSheet(ACTIVE_STYLE if present else INACTIVE_STYLE)
            layout.addWidget(chk)

        layout.addStretch()
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_task_tabs(self, parent_layout):
        self.tabs = QTabWidget()
        self.tabs.addTab(HandRepresentationTab(self), "Hand Representation")
        self.tabs.addTab(MotorPlanningTab(self), "Motor Planning")
        

        parent_layout.addWidget(self.tabs)

    def validate_config(self):
        nom = self.txt_name.text().strip()
        if not is_valid_name(nom):
            QMessageBox.warning(self, "Erreur", "ID Participant invalide.")
            return None
        
        config = self.default_config.copy()
        config.update({
            'nom': nom,
            'session': f"{self.spin_session.value():02d}",
            'enregistrer': self.chk_save.isChecked(),
            'screenid': self.screenid.value() - 1,
            'mode': self.combo_mode.currentText(),
            'parport_actif': self.chk_parport.isChecked(),
            'eyetracker_actif': self.chk_eyetracker.isChecked()
        })
        return config

    def run_experiment(self, task_params):
        general_config = self.validate_config()
        if not general_config: return
        self.final_config = {**general_config, **task_params}
        self.close()
        QApplication.instance().quit()

    def get_config(self):
        return self.final_config

    def closeEvent(self, event):
        event.accept()

def show_qt_menu(last_config=None):
    app = QApplication.instance() or QApplication(sys.argv)
    menu = ExperimentMenu(last_config)
    menu.show()
    app.exec()
    return menu.get_config()