from psychopy import parallel, core
import time

class DummyParPort:
    def __init__(self, *args, **kwargs):
        pass 

    def send_trigger(self, code, duration=0.0001):
        pass 

    def reset(self):
        pass

# --- CLASSE PRINCIPALE (CONNEXION PHYSIQUE) ---
class ParPort:
    def __init__(self, address=0x378):
        """
        Initialise le port parallèle.
        """
        self.address = address
        self.port = None
        self.dummy_mode = False

        try:
            parallel.setPortAddress(address)
            self.port = parallel.ParallelPort(address)
            self.port.setData(0)
        except Exception as e:
            self.dummy_mode = True

    import time

    def send_trigger(self, code, duration=0.0005):
        """
        Envoie un trigger sur le port parallèle avec une durée sub-ms.
        duration : durée du pulse en secondes (ex: 0.0005 = 0.5 ms)
        """

        if self.dummy_mode:
            return

        try:
            start = time.perf_counter()

            # front montant
            self.port.setData(int(code))

            # busy wait haute précision
            while (time.perf_counter() - start) < duration:
                pass

            # remise à 0
            self.port.setData(0)

        except Exception as e:
            print(f"Erreur envoi trigger {code}: {e}")

    def reset(self):
        """Force la remise à zéro des pins"""
        if not self.dummy_mode and self.port:
            self.port.setData(0)