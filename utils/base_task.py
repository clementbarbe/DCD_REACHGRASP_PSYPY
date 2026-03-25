"""
BaseTask - Classe mère pour les tâches fMRI/Comportementales
------------------------------------------------------------
Gère l'initialisation commune, le hardware, les chemins et les
fonctions de timing standard (Trigger, Resting State).

"""

import os
import csv
from datetime import datetime

from psychopy import visual, core
from psychopy.hardware.keyboard import Keyboard
from utils.logger import get_logger
from utils.hardware_manager import setup_hardware


class BaseTask:
    """
    Classe mère de toute tâche cognitive.

    API Clavier (3 méthodes) :
        flush_keyboard()  → vide le buffer
        wait_keys()       → bloquant
        get_keys()        → non-bloquant (polling)
    """

    QUIT_KEYS = ['escape', 'q']

    def __init__(self, win, nom, session, task_name, folder_name,
                 eyetracker_actif=False, parport_actif=False,
                 enregistrer=True, et_prefix='TSK'):
        self.win = win
        self.nom = str(nom)
        self.session = str(session)
        self.task_name = task_name
        self.et_prefix = et_prefix

        self.eyetracker_actif = eyetracker_actif
        self.parport_actif = parport_actif
        self.enregistrer = enregistrer

        self.logger = get_logger()
        self.task_clock = core.Clock()
        self.codes = {}

        self._incremental_path = None
        self._incremental_header_written = False

        self._init_paths(folder_name)
        self._init_hardware()
        self._init_keyboard()
        self._init_common_stimuli()

    # =================================================================
    #  QUIT — P0-FIX : signature unique et cohérente
    # =================================================================

    def should_quit(self, force_quit=False):
        """
        Vérifie si on doit quitter l'expérience.

        Args:
            force_quit (bool): Si True, quitte immédiatement sans checker le clavier.

        Returns:
            True si quit détecté (en pratique, core.quit() est appelé avant le return).
        """
        quit_detected = force_quit

        if not quit_detected:
            keys = self.kb.getKeys(
                keyList=['escape', 'q', 'a'],
                waitRelease=False,
                clear=True
            )
            if keys:
                quit_detected = True

        if quit_detected:
            self.logger.warn("Quit signal detected. Cleaning up...")

            # P0-FIX: sauvegarde d'urgence avant de quitter
            self._emergency_save()

            if self.win:
                try:
                    self.win.close()
                except Exception:
                    pass
            core.quit()
            return True

        return False

    def _emergency_save(self):
        """P0-FIX: Sauvegarde d'urgence des données en mémoire avant quit."""
        data = getattr(self, 'global_records', [])
        if data and self.enregistrer:
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_name = self.task_name.replace(' ', '')
                fname = f"{self.nom}_{safe_name}_EMERGENCY_{timestamp}.csv"
                path = os.path.join(self.data_dir, fname)
                all_keys = sorted(set().union(*(d.keys() for d in data)))
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=all_keys)
                    writer.writeheader()
                    writer.writerows(data)
                self.logger.warn(f"Emergency save: {path}")
            except Exception as e:
                self.logger.err(f"Emergency save FAILED: {e}")

    # =================================================================
    #  CLAVIER — 3 MÉTHODES
    # =================================================================

    def _init_keyboard(self):
        self.kb = Keyboard(clock=self.task_clock)
        self.logger.log("Keyboard init (psychtoolbox backend).")

    def flush_keyboard(self):
        """Vide le buffer."""
        self.kb.clearEvents()

    def _build_key_list(self, key_list):
        """
        Injecte TOUJOURS les quit keys dans la liste d'écoute.
        Retourne None si key_list est None (= toutes les touches).
        """
        if key_list is None:
            return None
        # set() pour éviter les doublons si escape est déjà dedans
        return list(set(key_list + self.QUIT_KEYS))

    def _filter_and_check_quit(self, keys, original_key_list):
        """
        1. Vérifie si une quit key a été pressée → force quit
        2. Filtre les quit keys hors du retour pour l'appelant
        
        Args:
            keys: Liste de Key objects retournée par kb
            original_key_list: La key_list demandée par l'appelant (avant injection)
        
        Returns:
            list[Key] contenant uniquement les touches demandées par l'appelant
        """
        if not keys:
            return keys

        # 1. Quit check
        for k in keys:
            if k.name in self.QUIT_KEYS:
                self.logger.warn(f"Quit key pressed: {k.name}")
                self.should_quit(force_quit=True)
                return []  # Ne sera jamais atteint (core.quit above), sécurité

        # 2. Filtrer : ne retourner que ce que l'appelant a demandé
        if original_key_list is not None:
            keys = [k for k in keys if k.name in original_key_list]

        return keys

    def wait_keys(self, key_list=None, max_wait=float('inf')):
        """
        Attente bloquante d'un appui clavier.
        Les quit keys (escape, q) sont TOUJOURS écoutées.

        Args:
            key_list: Touches acceptées. None = toutes.
            max_wait: Timeout en secondes.

        Returns:
            list[Key] ou None si timeout.
        """
        self.flush_keyboard()

        effective_list = self._build_key_list(key_list)

        keys = self.kb.waitKeys(
            keyList=effective_list,
            maxWait=max_wait,
            waitRelease=False,
            clear=True
        )

        if not keys:
            return None

        keys = self._filter_and_check_quit(keys, key_list)
        return keys if keys else None

    def get_keys(self, key_list=None):
        """
        Lecture non-bloquante du buffer clavier.
        Les quit keys (escape, q) sont TOUJOURS écoutées.

        Args:
            key_list: Filtre touches. None = toutes.

        Returns:
            list[Key] (peut être vide).
        """
        effective_list = self._build_key_list(key_list)

        keys = self.kb.getKeys(
            keyList=effective_list,
            waitRelease=False,
            clear=True
        )

        return self._filter_and_check_quit(keys, key_list)

    # =================================================================
    #  CHEMINS
    # =================================================================

    def _init_paths(self, folder_name):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(base_dir) in ('tasks', 'utils'):
            self.root_dir = os.path.dirname(base_dir)
        else:
            self.root_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )

        self.data_dir = os.path.join(self.root_dir, 'data', folder_name)
        self.img_dir = os.path.join(self.root_dir, 'image')

        if self.enregistrer and not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir, exist_ok=True)
                self.logger.log(f"Dossier créé : {self.data_dir}")
            except OSError as e:
                self.logger.err(f"Erreur dossier : {e}")

    # =================================================================
    #  HARDWARE
    # =================================================================

    def _init_hardware(self):
        self.logger.log(f"Hardware init : {self.task_name}")
        try:
            self.ParPort, self.EyeTracker = setup_hardware(
                self.parport_actif, self.eyetracker_actif, self.win
            )
            if self.eyetracker_actif:
                short_nom = self.nom[:3] if len(self.nom) >= 3 else self.nom
                et_filename = f"{self.et_prefix}_{short_nom}{self.session}"
                if len(et_filename) > 8:
                    et_filename = et_filename[:8]
                    self.logger.warn(f"Nom EDF tronqué : {et_filename}")
                self.EyeTracker.initialize(file_name=et_filename)
        except Exception as e:
            self.logger.err(f"Hardware Error: {e}")
            raise

    # =================================================================
    #  STIMULI COMMUNS
    # =================================================================

    def _init_common_stimuli(self):
        self.fixation = visual.TextStim(
            self.win, text='+', height=0.05, color='white'
        )
        self.instr_stim = visual.TextStim(
            self.win, text='', height=0.06, color='white', wrapWidth=1.5
        )

    # =================================================================
    #  PROTOCOLE STANDARD
    # =================================================================

    def show_instructions(self, text_override=None):
        msg = text_override or (
            f"Bienvenue : {self.task_name}\n\n"
            "Appuyez sur une touche pour continuer."
        )
        self.instr_stim.text = msg
        self.instr_stim.draw()
        self.win.flip()
        core.wait(0.5)
        self.wait_keys()

    def wait_for_trigger(self, trigger_key='t'):
        self.instr_stim.text = f"Attente trigger IRM [{trigger_key}]..."
        self.instr_stim.draw()
        self.win.flip()
        self.logger.log("Waiting for trigger...")

        self.flush_keyboard()
        self.kb.waitKeys(keyList=[trigger_key], waitRelease=False, clear=True)

        self.task_clock.reset()

        if self.eyetracker_actif:
            self.EyeTracker.start_recording()
            self.EyeTracker.send_message(f"START_{self.task_name.upper()}")

        self.logger.ok(f"Trigger reçu")

    def show_resting_state(self, duration_s=10.0,
                           code_start_key='rest_start',
                           code_end_key='rest_end'):
        self.logger.log(f"Rest : {duration_s}s")

        if self.eyetracker_actif:
            self.EyeTracker.send_message("REST_START")

        self.fixation.draw()
        self.win.flip()
        core.wait(duration_s)

        if code_end_key:
            if self.eyetracker_actif:
                self.EyeTracker.send_message("REST_END")

    # =================================================================
    #  SAUVEGARDE — P0-FIX : incrémentale + finale
    # =================================================================

    def _init_incremental_file(self, suffix=""):
        """
        P0-FIX: Crée le fichier CSV incrémental au début de la session.
        Appelé une fois, avant le premier trial.
        """
        if not self.enregistrer:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = self.task_name.replace(' ', '')
        fname = f"{self.nom}_{safe_name}{suffix}_{timestamp}_incremental.csv"
        self._incremental_path = os.path.join(self.data_dir, fname)
        self._incremental_header_written = False
        self.logger.log(f"Incremental file ready: {self._incremental_path}")

    def save_trial_incremental(self, trial_record):
        """
        P0-FIX: Écrit UN trial immédiatement sur disque (append).
        Appelé à la fin de chaque trial dans run_trial().

        Args:
            trial_record (dict): Données du trial à sauvegarder.
        """
        if not self.enregistrer or not trial_record:
            return
        if self._incremental_path is None:
            self._init_incremental_file()

        try:
            file_exists = os.path.exists(self._incremental_path)
            with open(self._incremental_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(trial_record.keys()))

                # Écrire le header seulement la première fois
                if not self._incremental_header_written or not file_exists:
                    writer.writeheader()
                    self._incremental_header_written = True

                writer.writerow(trial_record)
        except Exception as e:
            self.logger.err(f"Incremental save error: {e}")

    def save_data(self, data_list=None, filename_suffix=""):
        """
        Sauvegarde finale complète (appelée dans finally).
        Le fichier incrémental sert de backup ; celui-ci est le fichier propre.
        """
        if data_list is None:
            data_list = getattr(self, 'global_records', [])

        if not self.enregistrer or not data_list:
            self.logger.warn("Rien à sauvegarder.")
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = self.task_name.replace(' ', '')
        fname = f"{self.nom}_{safe_name}{filename_suffix}_{timestamp}.csv"
        path = os.path.join(self.data_dir, fname)

        try:
            all_keys = sorted(set().union(*(d.keys() for d in data_list)))
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                writer.writerows(data_list)
            self.logger.ok(f"Saved: {path}")
        except Exception as e:
            self.logger.err(f"SAVE ERROR: {e}")
            try:
                with open(path + '.bak', 'w') as f:
                    f.write(str(data_list))
            except Exception:
                pass

        return path

    # =================================================================
    #  INTERFACE
    # =================================================================

    def run(self):
        raise NotImplementedError("Implémenter run() dans la sous-classe.")