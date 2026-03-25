import sys
import signal
from PyQt6.QtWidgets import QApplication
from gui.menu import ExperimentMenu
from utils.logger import get_logger

signal.signal(signal.SIGINT, signal.SIG_DFL)


def show_menu_and_get_config(app, last_config=None):
    menu = ExperimentMenu(last_config)
    menu.show()
    app.exec()
    config = menu.get_config()
    menu.deleteLater()
    app.processEvents()
    return config


def run_task_logic(config):
    logger = get_logger()
    from psychopy import visual, core, logging
    from utils.task_factory import create_task

    logging.console.setLevel(logging.ERROR)

    win = visual.Window(
        fullscr=config.get('fullscr', True),
        color='black',
        units='norm',
        screen=config.get('screenid', 0),
        checkTiming=False,
        waitBlanking=True
    )
    win.mouseVisible = False

    task = create_task(config, win)

    if not task:
        logger.err(f"Factory Error: Could not create task '{config.get('tache')}'")
        win.close()
        return

    try:
        win.flip()
        core.wait(0.5)
        task.run()
    except Exception as e:
        logger.err(f"Runtime Error during task execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        win.close()


def main():
    logger = get_logger()
    app = QApplication(sys.argv)
    last_config = None

    while True:
        config = show_menu_and_get_config(app, last_config)

        if not config:
            logger.log("Sortie demandée par l'utilisateur.")
            break

        try:
            logger.log(f"Lancement de la tâche : {config.get('tache', 'Unknown')}...")
            run_task_logic(config)
            last_config = config
        except Exception as e:
            logger.err(f"Erreur fatale dans la boucle principale : {e}")
            pass

    logger.log("Application shutdown.")
    app.quit()
    sys.exit(0)


if __name__ == '__main__':
    main()