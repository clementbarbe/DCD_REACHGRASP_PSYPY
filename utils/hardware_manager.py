import sys
from utils.logger import get_logger

logger = get_logger()


# =============================================================================
# 1. FAIL-SAFE DUMMY CLASSES
# =============================================================================

class SafeDummyParPort:
    def __init__(self):
        pass

    def send_trigger(self, code, duration=0.03):
        pass

    def reset(self):
        pass


class SafeDummyEyeTracker:
    def __init__(self, sample_rate=1000, dummy_mode=True):
        pass

    def initialize(self, file_name="TEST.EDF"):
        logger.log(f"[Dummy ET] Virtual file defined: {file_name}")

    def send_message(self, msg):
        pass

    def start_recording(self):
        logger.log("[Dummy ET] Start Recording (Simulation)")

    def stop_recording(self):
        pass

    def close_and_transfer_data(self, local_folder="data"):
        logger.log(f"[Dummy ET] Data transfer simulation to {local_folder}")


# =============================================================================
# 2. SECURE IMPORTS
# =============================================================================

try:
    from hardware.parport import ParPort
    ParPortAvailable = True
except (ImportError, OSError):
    ParPort = SafeDummyParPort
    ParPortAvailable = False

try:
    from hardware.eyetracker import EyeTracker
    EyeTrackerAvailable = True
except (ImportError, OSError):
    EyeTracker = SafeDummyEyeTracker
    EyeTrackerAvailable = False


# =============================================================================
# 3. FACTORY FUNCTION
# =============================================================================

def setup_hardware(parport_actif=False, eyetracker_actif=False, window=None):
    lpt = None
    if parport_actif:
        if ParPortAvailable:
            try:
                lpt = ParPort(address=0x378)
                logger.ok("LPT: Parallel Port connected successfully.")
            except Exception as e:
                logger.err(f"LPT: Init failed ({e}). Reverting to Dummy.")
                lpt = SafeDummyParPort()
        else:
            logger.log("LPT: Active in config but drivers missing. Using Dummy.")
            lpt = SafeDummyParPort()
    else:
        lpt = SafeDummyParPort()

    et = None
    if eyetracker_actif:
        if EyeTrackerAvailable:
            try:
                et = EyeTracker(dummy_mode=False)
                if not getattr(et, 'dummy_mode', False):
                    logger.ok("EyeTracker: Connected and Ready.")
                else:
                    logger.warn("EyeTracker: Driver loaded but device not found.")
            except Exception as e:
                logger.err(f"EyeTracker: Init failed ({e}). Reverting to Dummy.")
                et = SafeDummyEyeTracker()
        else:
            logger.log("EyeTracker: Active in config but drivers missing. Using Dummy.")
            et = SafeDummyEyeTracker()
    else:
        et = SafeDummyEyeTracker()

    return lpt, et