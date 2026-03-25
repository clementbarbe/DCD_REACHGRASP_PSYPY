import sys
import inspect
from datetime import datetime
from pathlib import Path

# --- COULEURS ANSI ---
class Colors:
    RESET   = '\033[0m'
    RED     = '\033[91m'  # Rouge brillant pour les erreurs
    GREEN   = '\033[92m'  # Vert brillant pour OK
    YELLOW  = '\033[93m'  # Jaune pour Warnings
    BLUE    = '\033[94m'  # Bleu
    GREY    = '\033[37m'  # Gris/Blanc pour info standard

class Logger:
    def __init__(self):
        # On repère le dossier racine du projet pour afficher des chemins propres
        self.root_dir = Path(__file__).parent.parent

    def _get_context(self):
        """Récupère qui appelle le logger (Fichier + Fonction)"""
        try:
            # On remonte la pile d'appel : 
            # 0: _get_context, 1: _print, 2: log/ok/warn, 3: TON CODE
            stack = inspect.stack()[3]
            path = Path(stack.filename)
            
            # Essayer de rendre le chemin relatif (ex: tasks/doorreward.py)
            try:
                module_name = str(path.relative_to(self.root_dir)).replace('.py', '').replace('/', '.').replace('\\', '.')
            except ValueError:
                module_name = path.stem # Si hors projet, juste le nom du fichier
            
            func_name = stack.function
            if func_name == '<module>':
                return module_name
            return f"{module_name}.{func_name}"
            
        except Exception:
            return "unknown"

    def _print(self, msg, color=Colors.RESET):
        """Formatage et affichage"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        context = self._get_context()
        
        # Format : [HH:MM:SS - module.fonction] Message
        print(f"{color}[{timestamp} - {context}] {msg}{Colors.RESET}")

    # --- MÉTHODES PUBLIQUES (Celles que tu utilises) ---

    def log(self, msg):
        """Message standard (Info)"""
        self._print(msg, Colors.GREY)

    def ok(self, msg):
        """Succès (Vert)"""
        self._print(msg, Colors.GREEN)

    def warn(self, msg):
        """Attention (Jaune)"""
        self._print(msg, Colors.YELLOW)

    def err(self, msg):
        """Erreur (Rouge)"""
        self._print(msg, Colors.RED)


# --- INSTANTIATION UNIQUE ---
# On crée une seule instance ici. 
_logger_instance = Logger()

def get_logger():
    """Retourne toujours la même instance du logger"""
    return _logger_instance