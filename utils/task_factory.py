# task_factory.py
from tasks.motor_planning import MotorPlanning
from tasks.hand_representation import HandRepresentationTask


def create_task(config, win):

    base_kwargs = {
        "win": win,
        "nom": config["nom"],
        "session": config["session"],
        "enregistrer": config["enregistrer"],
    }

    task_name = config["tache"]

    # ═════════════════════════════════════════════════════════════════════
    # MotorPlanning
    # ═════════════════════════════════════════════════════════════════════
    if task_name == "MotorPlanning":

        return MotorPlanning(
            win=win,
            nom=config["nom"],
            session=config["session"],
            mode=config["mode"],
            enregistrer=config["enregistrer"],
            parport_actif=config["parport_actif"],

            # ── run identification ──
            effector=config.get("effector", "hand"),
            run_number=config.get("run_number", 1),

            # ── trial design ──
            n_trials_per_condition=config.get("n_trials_per_condition", 20),

            # ── timing ──
            preview_duration=config.get("preview_duration", 2.0),
            cue_duration=config.get("cue_duration", 0.5),
            plan_duration=config.get("plan_duration", 5.5),
            plan_jitter=config.get("plan_jitter", 0.5),
            go_duration=config.get("go_duration", 0.5),
            execute_duration=config.get("execute_duration", 2.0),
            iti_duration=config.get("iti_duration", 8.0),

            # ── baselines ──
            initial_baseline=config.get("initial_baseline", 10.0),
            final_baseline=config.get("final_baseline", 10.0),

            # ── audio ──
            go_beep_freq=config.get("go_beep_freq", 1000.0),
        )

    # ═════════════════════════════════════════════════════════════════════
    # HandRepresentation
    # ═════════════════════════════════════════════════════════════════════
    elif task_name == "HandRepresentation":

        return HandRepresentationTask(
            **base_kwargs,
            n_blocks=config.get("n_blocks", 1),
            trial_duration=config.get("trial_duration", 4.0),
            camera_index=config.get("camera_index", 0),
            handedness=config.get("handedness", "droitier"),
            block_label=config.get("block_label", "Block 1 Pre"),
            block_number=config.get("block_number", 1),
        )

    else:
        print(f"Tâche inconnue : {task_name}")
        return None