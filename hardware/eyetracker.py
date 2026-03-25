import pylink
import os
import time

class EyeTracker:
    def __init__(self, sample_rate=1000, dummy_mode=False):
        self.dummy_mode = dummy_mode
        self.el = None
        self.filename = "TEST.EDF"
        self.active = False
        self.sample_rate = sample_rate
        
    def initialize(self, file_name="TEST.EDF"):
        """
        Initialise la connexion avec le Eyelink et ouvre le fichier.
        file_name : Doit faire 8 caractères max (sans l'extension).
        """
        # Gestion de la longueur du nom de fichier (Max 8 char pour DOS/Eyelink)
        base_name = os.path.splitext(file_name)[0]
        if len(base_name) > 8:
            print(f"ATTENTION: Nom de fichier Eyelink trop long ({base_name}). Tronqué à 8 char.")
            base_name = base_name[:8]
        self.filename = base_name + ".EDF"

        if self.dummy_mode:
            print("EyeLink: Mode Dummy activé.")
            self.el = pylink.EyeLink(None)
            self.active = True
            return

        try:
            # Connexion IP par défaut du Eyelink
            self.el = pylink.EyeLink("100.1.1.1")
            self.active = True
            print(f"EyeLink: Connecté (Version {self.el.getTrackerVersion()})")
        except RuntimeError:
            print("EyeLink: Erreur de connexion. Passage en mode Dummy.")
            self.el = pylink.EyeLink(None)
            self.dummy_mode = True
            self.active = False

        # Configuration de base
        self.el.sendCommand(f"sample_rate = {self.sample_rate}")
        
        # Ouverture du fichier sur le Eyelink (Disque dur du Host PC)
        self.el.openDataFile(self.filename)
        
        # Configuration des données à enregistrer (Events + Samples)
        # AREA, GAZE, GAZERES, HREF, PUPIL, STATUS, INPUT
        self.el.sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT")
        self.el.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,HREF,AREA,GAZERES,STATUS,INPUT")
        self.el.sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT")
        self.el.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT")

    def start_recording(self):
        """Démarre l'enregistrement des échantillons et événements"""
        if self.el:
            # Arguments: file_samples, file_events, link_samples, link_events
            self.el.startRecording(1, 1, 1, 1)
            # Attendre un peu que le mode s'active
            time.sleep(0.1)

    def stop_recording(self):
        """Arrête l'enregistrement"""
        if self.el:
            self.el.stopRecording()

    def send_message(self, msg):
        """Envoie un marqueur (trigger) dans le fichier EDF"""
        if self.el:
            self.el.sendMessage(msg)

    def close_and_transfer_data(self, local_folder="data"):
        """
        Ferme le fichier sur le tracker et le télécharge sur le PC local.
        """
        if self.el:
            self.el.closeDataFile()
            
            # Création du dossier local si inexistant
            if not os.path.exists(local_folder):
                os.makedirs(local_folder)
                
            local_path = os.path.join(local_folder, self.filename)
            
            print(f"EyeLink: Transfert de {self.filename} vers {local_path}...")
            try:
                # receiveDataFile(nom_distant, nom_local)
                self.el.receiveDataFile(self.filename, local_path)
                print("EyeLink: Transfert terminé avec succès.")
            except Exception as e:
                print(f"EyeLink: Erreur lors du transfert : {e}")
            
            self.el.close()