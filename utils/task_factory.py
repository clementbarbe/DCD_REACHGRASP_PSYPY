# task_factory.py
from tasks.motor_planning import MotorPlanning


def create_task(config, win):

    task_name = config["tache"]

    # ═════════════════════════════════════════════════════════════════════
    # MotorPlanning
    # ═════════════════════════════════════════════════════════════════════
    if task_name == "MotorPlanning":

        return MotorPlanning(
            win=win,
            nom=config["nom"],
            session=config["session"],
            mode=config.get("mode", "emg"),
            enregistrer=config.get("enregistrer", True),
            parport_actif=config.get("parport_actif", False),

            # ── séquence de runs (depuis le menu) ──
            run_sequence=config.get("run_sequence", ["hand", "tool"] * 4),
        )

    else:
        print(f"Tâche inconnue : {task_name}")
        return None