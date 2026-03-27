# parport.py
"""Parallel port interface for EMG/EEG triggers."""

import time

from psychopy import parallel


class DummyParPort:
    """No-op parallel port for testing without hardware."""

    def send_trigger(self, code: int, duration: float = 0.0005) -> None:
        pass

    def reset(self) -> None:
        pass


class ParPort:
    """
    Parallel port trigger interface.

    Sends TTL pulses with sub-ms precision using busy-wait.
    Falls back to dummy mode silently if hardware is unavailable.

    Parameters
    ----------
    address : int
        Port address (default 0x378 for LPT1).
    """

    def __init__(self, address: int = 0x378) -> None:
        self.address = address
        self.port = None
        self.dummy_mode = False

        try:
            parallel.setPortAddress(address)
            self.port = parallel.ParallelPort(address)
            self.port.setData(0)
        except Exception:
            self.dummy_mode = True

    def send_trigger(self, code: int, duration: float = 0.005) -> None:
        """
        Send a TTL pulse on the parallel port.

        Parameters
        ----------
        code : int
            Trigger value (1–255).
        duration : float
            Pulse width in seconds (default 0.5 ms).
            Uses busy-wait for sub-ms precision.
        """
        if self.dummy_mode:
            return

        try:
            start = time.perf_counter()
            self.port.setData(int(code))
            while (time.perf_counter() - start) < duration:
                pass
            self.port.setData(0)
        except Exception as e:
            print(f"Trigger error (code={code}): {e}")

    def reset(self) -> None:
        """Force all pins to zero."""
        if not self.dummy_mode and self.port:
            self.port.setData(0)