# motor_planning.py
"""
Motor Planning Protocol — Grasp / Touch × Hand / Tool
======================================================
EMG co-registration — Sequential multi-run — PTB audio scheduling.

AUDIO : PsychoPy Sound + psychtoolbox (PTB) — PRE-SCHEDULED
────────────────────────────────────────────────────────────
Problème : snd.play() immédiat → jitter 5-20 ms (buffer fill time).
Solution : snd.play(when=T_ptb) appelé ~100 ms AVANT l'onset cible.
  - PTB pré-remplit le buffer audio vers le DAC
  - Le son démarre à l'instant T_ptb avec un jitter < 0.1 ms
  - Le trigger port parallèle est envoyé au même moment par busy-wait

Deux horloges coexistent :
  - task_clock (PsychoPy core.Clock) : temps relatif du run (reset à 0)
  - ptb.GetSecs() : horloge absolue du driver audio
  - Mapping : target_ptb = ptb.GetSecs() + (target_task - task_clock.now)

Validation oscilloscope :
  - Canal 1 : front trigger port parallèle (~µs)
  - Canal 2 : onset audio (sortie jack/DAC)
  - Attendu : < 1 ms de décalage avec PTB scheduling

ARCHITECTURE : PRE-COMPUTED TIMELINE (reconstruite par run)
───────────────────────────────────────────────────────────
1. Avant chaque run, TOUS les événements sont pré-calculés et triés.
2. ESPACE → clock reset → exécution séquentielle.
3. Les sons sont pré-schedulés 100 ms avant leur onset.
4. Zéro calcul pendant l'acquisition.

Structure d'un essai :
    Preview (fixation 2s) → Cue audio ("Grasp"/"Touch") → Plan (5.5s ± jitter)
    → Go beep → Execute (2s) → ITI (8s)

Pseudo-randomisation :
    Jamais plus de 3 essais consécutifs de la même condition.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# PsychoPy audio prefs — DOIT précéder tout import psychopy.sound
# ═══════════════════════════════════════════════════════════════════════════
# En haut du fichier motor_planning.py, REMPLACER le bloc prefs :
from psychopy import prefs
import sys

if sys.platform == 'win32':
    # Windows : PTB WASAPI → timing sub-ms
    prefs.hardware['audioLib'] = ['ptb', 'sounddevice']
    prefs.hardware['audioLatencyMode'] = 3
elif sys.platform == 'darwin':
    # macOS : PTB CoreAudio
    prefs.hardware['audioLib'] = ['ptb', 'sounddevice']
    prefs.hardware['audioLatencyMode'] = 3
else:
    # Linux : sounddevice (PTB ALSA trop instable)
    prefs.hardware['audioLib'] = ['sounddevice', 'ptb']
    prefs.hardware['audioLatencyMode'] = 0

import csv
import gc
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from psychopy import core, visual
from psychopy.sound import Sound

from utils.base_task import BaseTask

# ═══════════════════════════════════════════════════════════════════════════
# psychtoolbox — pour le scheduling audio sub-ms
# ═══════════════════════════════════════════════════════════════════════════
try:
    import psychtoolbox as ptb
    _HAS_PTB = True
except ImportError:
    ptb = None
    _HAS_PTB = False

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

CONDITIONS: List[str] = ["grasp", "touch"]
EFFECTORS:  List[str] = ["hand", "tool"]

# Codes port parallèle — marqueurs EMG
DEFAULT_EVENT_CODES: Dict[str, int] = {
    "run_start":      100,
    "run_end":        200,
    "trial_start":      1,
    "preview_start":    2,
    "cue_grasp":       10,
    "cue_touch":       20,
    "go_grasp":        30,
    "go_touch":        40,
    "execute_start":   50,
    "iti_start":       60,
}

# Priorité de tri — même onset → ordre visuel < parport < son < marqueur
_ACTION_PRIORITY: Dict[str, int] = {
    "visual_fixation":     0,
    "visual_instruction":  0,
    "parport_event":       1,
    "sound_cue":           2,
    "sound_go":            2,
    "marker":              3,
}


# ═════════════════════════════════════════════════════════════════════════════

class MotorPlanning(BaseTask):
    """
    Sequential multi-run motor planning task with PTB audio scheduling.

    Parameters
    ----------
    run_sequence : list[str] | None
        Séquence d'effecteurs, ex: ['hand','tool','hand','tool',…].
        Défaut : alternance × 4 = 8 runs.
    sound_preload_s : float
        Temps de pré-chargement audio avant onset (défaut 0.100 s).
        PTB remplit le buffer DAC pendant ce délai, puis déclenche
        le son à l'instant exact.
    """

    def __init__(
        self,
        win: visual.Window,
        nom: str,
        session: str = "01",
        mode: str = "emg",
        # ── Séquence de runs ──
        run_sequence: Optional[List[str]] = None,
        # ── Nombre d'essais ──
        n_trials_per_condition: int = 20,
        max_consecutive: int = 3,
        # ── Timing (secondes) ──
        preview_duration: float = 2.0,
        cue_duration: float = 0.5,
        plan_duration: float = 5.5,
        plan_jitter: float = 0.5,
        go_duration: float = 0.5,
        execute_duration: float = 2.0,
        iti_duration: float = 8.0,
        initial_baseline: float = 10.0,
        final_baseline: float = 10.0,
        # ── Audio scheduling ──
        sound_preload_s: float = 0.100,
        # ── Sons ──
        sound_dir: str = "sounds",
        grasp_sound_file: str = "grasp.wav",
        touch_sound_file: str = "touch.wav",
        go_beep_freq: float = 1000.0,
        go_beep_volume: float = 0.7,
        # ── Misc ──
        enregistrer: bool = True,
        eyetracker_actif: bool = False,
        parport_actif: bool = True,
        event_codes: Optional[Dict[str, int]] = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            win=win,
            nom=nom,
            session=session,
            task_name="Motor_Planning",
            folder_name="motor_planning",
            eyetracker_actif=eyetracker_actif,
            parport_actif=parport_actif,
            enregistrer=enregistrer,
            et_prefix="MP",
        )

        # ── Mode ──
        self.mode: str = mode.lower()

        # ── Séquence de runs ──
        if run_sequence is None:
            run_sequence = ["hand", "tool"] * 4
        self.run_sequence: List[str] = [e.lower() for e in run_sequence]
        self.n_runs: int = len(self.run_sequence)
        if self.n_runs == 0:
            raise ValueError("run_sequence ne peut pas être vide.")

        # ── Design ──
        self.n_trials_per_condition: int = n_trials_per_condition
        self.n_trials_total: int = len(CONDITIONS) * n_trials_per_condition
        self.max_consecutive: int = max_consecutive

        # ── Timing ──
        self.preview_duration: float = max(0.5, preview_duration)
        self.cue_duration: float     = cue_duration
        self.plan_duration: float    = plan_duration
        self.plan_jitter: float      = plan_jitter
        self.go_duration: float      = go_duration
        self.execute_duration: float = execute_duration
        self.iti_duration: float     = iti_duration
        self.initial_baseline: float = initial_baseline
        self.final_baseline: float   = final_baseline

        # ── Audio scheduling ──
        self.sound_preload_s: float = sound_preload_s

        # ── Sons ──
        self.sound_dir: str          = sound_dir
        self.grasp_sound_file: str   = grasp_sound_file
        self.touch_sound_file: str   = touch_sound_file
        self.go_beep_freq: float     = go_beep_freq
        self.go_beep_volume: float   = go_beep_volume

        # ── Codes triggers ──
        self.event_codes: Dict[str, int] = (
            event_codes if event_codes else dict(DEFAULT_EVENT_CODES)
        )

        # ── État runtime ──
        self.global_records: List[Dict[str, Any]] = []
        self.current_run_records: List[Dict[str, Any]] = []
        self.timeline: List[Dict[str, Any]] = []
        self.current_run: int = 0
        self.current_effector: str = ""

        # ── PsychoPy Sound objects ──
        self._sounds: Dict[str, Sound] = {}

        # ── Init chain ──
        self._detect_display_scaling()
        self._measure_frame_rate()
        self._setup_visual_stimuli()
        self._setup_sounds()

        # ── Log PTB status ──
        if _HAS_PTB:
            self.logger.ok(
                f"psychtoolbox DISPONIBLE — audio scheduling activé "
                f"(preload = {self.sound_preload_s * 1000:.0f} ms)"
            )
        else:
            self.logger.warn(
                "psychtoolbox NON DISPONIBLE — "
                "audio timing dégradé (play immédiat). "
                "Installez : pip install psychtoolbox"
            )

        # ── Durée estimée ──
        trial_dur = (
            self.preview_duration + self.plan_duration
            + self.execute_duration + self.iti_duration
        )
        run_dur = (
            self.initial_baseline
            + self.n_trials_total * trial_dur
            + self.final_baseline
        )
        session_dur = run_dur * self.n_runs

        self.logger.ok(
            f"MotorPlanning ready | {self.n_runs} runs | "
            f"seq={self.run_sequence} | "
            f"{self.n_trials_total} trials/run | "
            f"max {self.max_consecutive} consécutifs | "
            f"{self.frame_rate:.1f} Hz | "
            f"~{run_dur / 60:.1f} min/run | "
            f"~{session_dur / 60:.0f} min total"
        )

    # ═════════════════════════════════════════════════════════════════════
    #  INIT HELPERS
    # ═════════════════════════════════════════════════════════════════════

    def _detect_display_scaling(self) -> None:
        self.pixel_scale = 2.0 if self.win.size[1] > 1200 else 1.0

    def _measure_frame_rate(self) -> None:
        measured = self.win.getActualFrameRate(
            nIdentical=10, nMaxFrames=100, threshold=1,
        )
        self.frame_rate = measured if measured else 60.0
        self.frame_duration_s  = 1.0 / self.frame_rate
        self.frame_tolerance_s = 0.75 / self.frame_rate
        self.logger.log(
            f"Frame rate: {self.frame_rate:.1f} Hz → "
            f"{self.frame_duration_s * 1000:.2f} ms/frame"
        )

    # ═════════════════════════════════════════════════════════════════════
    #  VISUAL STIMULI
    # ═════════════════════════════════════════════════════════════════════

    def _setup_visual_stimuli(self) -> None:
        self.cue_stim = visual.TextStim(
            self.win, text="", height=0.06, color="white",
            pos=(0.0, 0.0), wrapWidth=1.5, font="Arial", bold=False,
        )

    # ═════════════════════════════════════════════════════════════════════
    #  SOUND SETUP — PsychoPy Sound (PTB / sounddevice)
    # ═════════════════════════════════════════════════════════════════════

    def _setup_sounds(self) -> None:
        """
        Pré-charge tous les stimuli audio.
        Tente PTB, fallback sur sounddevice si échec.
        """
        import sys

        # ── Détection backend réel ──
        self._ptb_available = False
        try:
            from psychopy.sound.backend_ptb import SoundPTB
            self._ptb_available = _HAS_PTB and sys.platform in ('win32', 'darwin')
            self.logger.log(
                f"Audio backend: PTB={'OUI' if self._ptb_available else 'NON'} | "
                f"OS={sys.platform}"
            )
        except ImportError:
            self.logger.log("Audio backend: sounddevice (PTB non importable)")

        base = Path(self.root_dir) / self.sound_dir

        # ── Chargement avec fallback ──
        def _safe_load(value, name, fallback_freq=400, fallback_dur=0.5):
            """Charge un son, fallback sur ton pur si échec."""
            try:
                snd = Sound(value=value, name=name)
                self.logger.ok(f"Loaded: {name} ({value})")
                return snd
            except Exception as e:
                self.logger.warn(
                    f"Échec chargement {name} ({value}): {e}"
                )
                try:
                    snd = Sound(
                        value=fallback_freq, secs=fallback_dur,
                        volume=self.go_beep_volume, name=f"{name}_fallback",
                    )
                    self.logger.warn(
                        f"Fallback ton {fallback_freq} Hz pour {name}"
                    )
                    return snd
                except Exception as e2:
                    self.logger.err(f"Fallback AUSSI échoué pour {name}: {e2}")
                    return None

        # ── Cue Grasp ──
        grasp_path = base / self.grasp_sound_file
        if grasp_path.exists():
            self._sounds["grasp"] = _safe_load(
                str(grasp_path), "grasp_cue", 400, self.cue_duration,
            )
        else:
            self._sounds["grasp"] = _safe_load(
                400, "grasp_tone", 400, self.cue_duration,
            )

        # ── Cue Touch ──
        touch_path = base / self.touch_sound_file
        if touch_path.exists():
            self._sounds["touch"] = _safe_load(
                str(touch_path), "touch_cue", 600, self.cue_duration,
            )
        else:
            self._sounds["touch"] = _safe_load(
                600, "touch_tone", 600, self.cue_duration,
            )

        # ── Go beep ──
        self._sounds["go"] = _safe_load(
            self.go_beep_freq, "go_beep",
            self.go_beep_freq, self.go_duration,
        )

        # ── Vérifier qu'on a au moins les sons critiques ──
        missing = [k for k, v in self._sounds.items() if v is None]
        if missing:
            raise RuntimeError(
                f"Sons critiques non chargés : {missing}. "
                f"Vérifiez votre configuration audio."
            )

        # ── Warmup ──
        try:
            warmup = Sound(value=100, secs=0.01, volume=0.0, name="warmup")
            warmup.play()
            core.wait(0.05)
            warmup.stop()
            self.logger.ok("Audio driver primed.")
        except Exception as e:
            self.logger.warn(f"Audio warmup failed (non bloquant): {e}")

    # ═════════════════════════════════════════════════════════════════════
    #  PSEUDO-RANDOMISATION (max N consécutifs)
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _check_max_consecutive(seq: List[str], max_c: int) -> bool:
        """Vérifie la contrainte de répétition maximale."""
        if not seq:
            return True
        count = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                count += 1
                if count > max_c:
                    return False
            else:
                count = 1
        return True

    @staticmethod
    def _pseudo_random_max_consecutive(
        items: List[str],
        reps: int,
        max_c: int = 3,
        max_attempts: int = 10000,
    ) -> List[str]:
        """
        Séquence pseudo-aléatoire : jamais plus de max_c essais
        consécutifs de la même condition.

        Phase 1 : shuffle-and-check (rapide, stochastique).
        Phase 2 : construction incrémentale (déterministe, garanti).
        """
        pool = [it for it in items for _ in range(reps)]
        n = len(pool)

        # Phase 1 : shuffle-and-check
        for _ in range(max_attempts):
            random.shuffle(pool)
            count = 1
            valid = True
            for i in range(1, n):
                if pool[i] == pool[i - 1]:
                    count += 1
                    if count > max_c:
                        valid = False
                        break
                else:
                    count = 1
            if valid:
                return list(pool)

        # Phase 2 : construction incrémentale (fallback garanti)
        seq: List[str] = []
        remaining = {it: reps for it in items}
        for _ in range(n):
            available = []
            for it in items:
                if remaining[it] <= 0:
                    continue
                if (
                    len(seq) >= max_c
                    and all(s == it for s in seq[-max_c:])
                ):
                    continue
                available.append(it)
            if not available:
                available = [it for it in items if remaining[it] > 0]
            chosen = random.choice(available)
            seq.append(chosen)
            remaining[chosen] -= 1
        return seq

    def _build_trial_order(self) -> List[str]:
        """Construit l'ordre pseudo-aléatoire pour le run courant."""
        order = self._pseudo_random_max_consecutive(
            CONDITIONS, self.n_trials_per_condition,
            max_c=self.max_consecutive,
        )
        assert self._check_max_consecutive(order, self.max_consecutive), (
            f"Contrainte max {self.max_consecutive} consécutifs violée !"
        )
        assert len(order) == self.n_trials_total, (
            f"Nombre d'essais incorrect : {len(order)} ≠ {self.n_trials_total}"
        )
        return order

    # ═════════════════════════════════════════════════════════════════════
    #  TIMELINE CONSTRUCTION
    # ═════════════════════════════════════════════════════════════════════

    def _add_event(
        self, onset_s: float, action: str, **kwargs: Any,
    ) -> None:
        event: Dict[str, Any] = {
            "onset_s":   round(onset_s, 6),
            "action":    action,
            "_priority": _ACTION_PRIORITY.get(action, 9),
        }
        event.update(kwargs)
        self.timeline.append(event)

    def _build_trial_events(
        self, t: float, trial_idx: int, condition: str,
    ) -> float:
        """
        Ajoute tous les événements pour UN essai commençant à t.
        Returns : onset du prochain essai.
        """
        trial_start_t = t
        jitter = random.uniform(-self.plan_jitter, self.plan_jitter)
        plan_dur = max(2.0, self.plan_duration + jitter)

        # ── TRIAL START ──
        self._add_event(
            t, "marker",
            label="trial_start",
            trial_index=trial_idx,
            condition=condition,
            effector=self.current_effector,
            plan_duration_s=round(plan_dur, 4),
            jitter_s=round(jitter, 4),
        )
        if self.parport_actif:
            self._add_event(
                t, "parport_event",
                label="trial_start_trigger",
                trial_index=trial_idx,
                condition=condition,
                pin_code=self.event_codes.get("trial_start", 1),
            )

        # ── PREVIEW (fixation) ──
        self._add_event(
            t, "visual_fixation",
            label="preview_start",
            trial_index=trial_idx,
            condition=condition,
        )
        if self.parport_actif:
            self._add_event(
                t + 0.001, "parport_event",
                label="preview_trigger",
                trial_index=trial_idx,
                pin_code=self.event_codes.get("preview_start", 2),
            )

        # ── CUE AUDIO → Plan phase ──
        t_cue = trial_start_t + self.preview_duration
        cue_code_key = f"cue_{condition}"
        self._add_event(
            t_cue, "sound_cue",
            label=f"cue_{condition}",
            trial_index=trial_idx,
            condition=condition,
            sound_id=condition,
            pin_code=self.event_codes.get(cue_code_key, 10),
        )

        # ── GO BEEP → Execute phase ──
        t_go = t_cue + plan_dur
        go_code_key = f"go_{condition}"
        self._add_event(
            t_go, "sound_go",
            label="go_beep",
            trial_index=trial_idx,
            condition=condition,
            sound_id="go",
            pin_code=self.event_codes.get(go_code_key, 30),
        )
        self._add_event(
            t_go + 0.001, "marker",
            label="execute_start",
            trial_index=trial_idx,
            condition=condition,
        )
        if self.parport_actif:
            self._add_event(
                t_go + 0.002, "parport_event",
                label="execute_trigger",
                trial_index=trial_idx,
                pin_code=self.event_codes.get("execute_start", 50),
            )

        # ── ITI ──
        t_iti = t_go + self.execute_duration
        self._add_event(
            t_iti, "marker",
            label="iti_start",
            trial_index=trial_idx,
            condition=condition,
        )
        if self.parport_actif:
            self._add_event(
                t_iti, "parport_event",
                label="iti_trigger",
                trial_index=trial_idx,
                pin_code=self.event_codes.get("iti_start", 60),
            )
        self._add_event(
            t_iti + 0.003, "visual_fixation",
            label="iti_fixation",
            trial_index=trial_idx,
        )

        # ── TRIAL END ──
        t_end = t_iti + self.iti_duration
        self._add_event(
            t_end, "marker",
            label="trial_end",
            trial_index=trial_idx,
            condition=condition,
            trial_duration_s=round(t_end - trial_start_t, 4),
        )
        return t_end

    def _build_full_timeline(self) -> None:
        """Construit la timeline pré-calculée pour le run courant."""
        self.timeline.clear()
        trial_order = self._build_trial_order()
        t = 0.0

        # ── Run start ──
        self._add_event(
            t, "marker",
            label="run_start",
            effector=self.current_effector,
            run_number=self.current_run,
            n_trials=len(trial_order),
            trial_order=str(trial_order),
        )
        if self.parport_actif:
            self._add_event(
                t + 0.001, "parport_event",
                label="run_start_trigger",
                pin_code=self.event_codes.get("run_start", 100),
            )

        # ── Baseline initiale ──
        self._add_event(t, "visual_fixation", label="baseline_start")
        t += self.initial_baseline

        # ── Essais ──
        for trial_idx, condition in enumerate(trial_order, start=1):
            t = self._build_trial_events(t, trial_idx, condition)

        # ── Baseline finale ──
        self._add_event(t, "visual_fixation", label="final_baseline_start")
        t += self.final_baseline

        # ── Run end ──
        self._add_event(t, "marker", label="run_end")
        if self.parport_actif:
            self._add_event(
                t, "parport_event",
                label="run_end_trigger",
                pin_code=self.event_codes.get("run_end", 200),
            )

        # ── Tri stable ──
        self.timeline.sort(key=lambda e: (e["onset_s"], e["_priority"]))
        for i, evt in enumerate(self.timeline):
            evt["event_index"] = i

        self._validate_timeline()

        total_dur = self.timeline[-1]["onset_s"] if self.timeline else 0
        n_sound = sum(
            1 for e in self.timeline if e["action"].startswith("sound_")
        )
        self.logger.log(
            f"Run {self.current_run} ({self.current_effector}) : "
            f"{len(self.timeline)} events ({n_sound} sons) | "
            f"~{total_dur:.1f} s ({total_dur / 60:.1f} min)"
        )

    def _validate_timeline(self) -> None:
        """Vérifie l'absence de collisions flip/son et la marge de preload."""
        visual_onsets = set()
        sound_onsets  = set()
        for evt in self.timeline:
            if evt["action"].startswith("visual_"):
                visual_onsets.add(evt["onset_s"])
            elif evt["action"].startswith("sound_"):
                sound_onsets.add(evt["onset_s"])

        # Collision flip / son
        collisions = visual_onsets & sound_onsets
        if collisions:
            self.logger.err(
                f"COLLISION flip/son : {len(collisions)} détectée(s) !"
            )
        else:
            self.logger.ok("Timeline : aucune collision flip/son.")

        # Vérifier que le preload ne chevauche pas un autre son
        sound_times = sorted(sound_onsets)
        for i in range(1, len(sound_times)):
            gap = sound_times[i] - sound_times[i - 1]
            if gap < self.sound_preload_s + 0.010:
                self.logger.warn(
                    f"Sons trop rapprochés : {gap:.3f} s "
                    f"(preload={self.sound_preload_s:.3f} s) "
                    f"à t={sound_times[i]:.3f} s"
                )

    def _save_planned_timeline(self) -> None:
        if not self.enregistrer or not self.timeline:
            return
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = (
            f"{self.nom}_Motor_Planning"
            f"_{self.current_effector}_run{self.current_run:02d}"
            f"_{timestamp}_planned.csv"
        )
        path = os.path.join(self.data_dir, fname)
        try:
            all_keys = sorted(
                set().union(*(e.keys() for e in self.timeline))
                - {"_priority"}
            )
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f, fieldnames=all_keys, extrasaction='ignore',
                )
                writer.writeheader()
                for evt in self.timeline:
                    writer.writerow(
                        {k: v for k, v in evt.items() if k != "_priority"}
                    )
            self.logger.ok(f"Planned timeline → {path}")
        except Exception as e:
            self.logger.err(f"Échec sauvegarde planned : {e}")

    # ═════════════════════════════════════════════════════════════════════
    #  TIMELINE EXECUTION
    # ═════════════════════════════════════════════════════════════════════

    def _wait_until(
        self, target_s: float, high_precision: bool = False,
    ) -> None:
        """
        Attend que task_clock atteigne target_s.

        high_precision : sleep → busy-wait les 2 dernières ms.
        """
        remaining = target_s - self.task_clock.getTime()
        if remaining <= 0:
            return
        if high_precision:
            if remaining > 0.003:
                core.wait(remaining - 0.002, hogCPUperiod=0.0)
            while self.task_clock.getTime() < target_s:
                pass
        else:
            core.wait(remaining, hogCPUperiod=0.0)

    # ── COEUR : exécution d'un événement sonore avec pre-scheduling ──────

    def _execute_sound_event(self, event: Dict[str, Any]) -> float:
        """
        Exécution d'un son en DEUX PHASES pour timing sub-ms :

        Phase 1 (onset - preload_s) :
            → snd.play(when=target_ptb)
            PTB pré-remplit le buffer DAC. Le son ne démarre PAS
            encore, il est schedulé pour target_ptb.

        Phase 2 (onset) :
            → busy-wait exact → trigger port parallèle
            Le trigger et le son arrivent au même instant (< 1 ms).

        Fallback (sans PTB) :
            → busy-wait → trigger → snd.play() immédiat
            Jitter ~5-20 ms (acceptable pour debug, pas pour collecte).
        """
        target_onset = event["onset_s"]
        sound_id = (
            "go" if event["action"] == "sound_go"
            else event.get("sound_id", "grasp")
        )
        snd = self._sounds.get(sound_id)

        if snd is None:
            self.logger.warn(f"Son introuvable : {sound_id}")
            self._wait_until(target_onset, high_precision=True)
            if self.parport_actif:
                self.ParPort.send_trigger(event.get("pin_code", 0))
            return self.task_clock.getTime()

        # ══════════════════════════════════════════════════════════════
        # CHEMIN PTB : pre-scheduling (sub-ms jitter)
        # ══════════════════════════════════════════════════════════════
        if self._ptb_available:
            # Phase 1 : attendre jusqu'à (onset - preload), puis scheduler
            preload_t = max(0.0, target_onset - self.sound_preload_s)
            self._wait_until(preload_t, high_precision=False)

            # Mapper task_clock → PTB clock
            now_task = self.task_clock.getTime()
            now_ptb  = ptb.GetSecs()
            delta    = target_onset - now_task

            if delta < 0.005:
                # Trop tard pour scheduler — play immédiat + trigger
                self.logger.warn(
                    f"Preload trop tard (delta={delta * 1000:.1f} ms) "
                    f"T{event.get('trial_index', '?')} "
                    f"{event.get('label', '?')} — play immédiat"
                )
                self._wait_until(target_onset, high_precision=True)
                if self.parport_actif:
                    self.ParPort.send_trigger(event.get("pin_code", 0))
                snd.stop()
                snd.play()
            else:
                # Scheduler le son au temps PTB exact
                target_ptb = now_ptb + delta
                snd.stop()
                snd.play(when=target_ptb)

                # Phase 2 : busy-wait → trigger synchronisé
                self._wait_until(target_onset, high_precision=True)
                if self.parport_actif:
                    self.ParPort.send_trigger(event.get("pin_code", 0))

        # ══════════════════════════════════════════════════════════
        # CHEMIN FALLBACK : play immédiat (Linux / sans PTB)
        # ══════════════════════════════════════════════════════════
        else:
            self._wait_until(target_onset, high_precision=True)
            if self.parport_actif:
                self.ParPort.send_trigger(event.get("pin_code", 0))
            snd.stop()
            snd.play()

        return self.task_clock.getTime()

    # ── Dispatch non-son ─────────────────────────────────────────────────

    def _dispatch_event(self, event: Dict[str, Any]) -> float:
        """Exécute un événement NON-sonore (visuel, parport, marqueur)."""
        action = event["action"]

        if action == "visual_fixation":
            self.fixation.draw()
            self.win.flip()
            return self.task_clock.getTime()

        if action == "visual_instruction":
            self.cue_stim.text = event.get("instruction_text", "")
            self.cue_stim.draw()
            self.fixation.draw()
            self.win.flip()
            return self.task_clock.getTime()

        if action == "parport_event":
            if self.parport_actif:
                self.ParPort.send_trigger(event.get("pin_code", 0))
            return self.task_clock.getTime()

        # marker
        return self.task_clock.getTime()

    # ── Record ───────────────────────────────────────────────────────────

    def _build_execution_record(
        self, event: Dict[str, Any], actual_time_s: float,
    ) -> Dict[str, Any]:
        error_ms = (actual_time_s - event["onset_s"]) * 1000.0
        return {
            "participant":         self.nom,
            "session":             self.session,
            "effector":            self.current_effector,
            "run_number":          self.current_run,
            "event_index":         event.get("event_index", ""),
            "action":              event["action"],
            "label":               event.get("label", ""),
            "onset_planned_s":     event["onset_s"],
            "onset_actual_s":      round(actual_time_s, 6),
            "scheduling_error_ms": round(error_ms, 3),
            "trial_index":         event.get("trial_index", ""),
            "condition":           event.get("condition", ""),
            "sound_id":            event.get("sound_id", ""),
            "pin_code":            event.get("pin_code", ""),
            "plan_duration_s":     event.get("plan_duration_s", ""),
            "jitter_s":            event.get("jitter_s", ""),
            "trial_duration_s":    event.get("trial_duration_s", ""),
        }

    # ── Boucle d'exécution principale ────────────────────────────────────

    def _execute_timeline(self) -> None:
        """
        Parcourt la timeline pré-calculée.

        Sons : routés vers _execute_sound_event (2 phases, PTB scheduling).
        Autres : routés vers _dispatch_event (standard).
        """
        n_events = len(self.timeline)
        self.logger.log(
            f"Exécution run {self.current_run} "
            f"({self.current_effector}) : {n_events} événements …"
        )

        gc.disable()

        try:
            for i, event in enumerate(self.timeline):
                is_sound = event["action"].startswith("sound_")

                # ── Dispatch ──
                if is_sound:
                    actual_t = self._execute_sound_event(event)
                else:
                    self.should_quit()
                    self._wait_until(event["onset_s"])
                    actual_t = self._dispatch_event(event)

                # ── Enregistrement ──
                record = self._build_execution_record(event, actual_t)
                self.current_run_records.append(record)
                self.global_records.append(record)
                self.save_trial_incremental(record)

                # ── Eye tracker ──
                if self.eyetracker_actif:
                    label = event.get("label", event["action"])
                    self.EyeTracker.send_message(
                        f"R{self.current_run:02d}_"
                        f"E{i:04d}_{label.upper()}"
                    )

                # ── Alerte timing pour les sons ──
                if is_sound:
                    err_ms = record["scheduling_error_ms"]
                    if abs(err_ms) > 2.0:
                        self.logger.warn(
                            f"TIMING E{i} "
                            f"T{event.get('trial_index', '?')} "
                            f"({event.get('label', '?')}): "
                            f"{err_ms:+.2f} ms"
                        )

                # ── Log fin de trial ──
                if event.get("label") == "trial_end":
                    t_idx = event.get("trial_index", "?")
                    cond  = event.get("condition", "?")
                    self.logger.log(
                        f"  Trial {t_idx}/{self.n_trials_total} "
                        f"({cond}) [t={actual_t:.2f} s]"
                    )

        finally:
            gc.enable()
            gc.collect()

        self.logger.ok(
            f"Run {self.current_run} ({self.current_effector}) terminé."
        )
        self._log_timing_summary()

    def _log_timing_summary(self) -> None:
        """Résumé timing des événements sonores du run courant."""
        sound_records = [
            r for r in self.current_run_records
            if r["action"].startswith("sound_")
            and isinstance(r["scheduling_error_ms"], (int, float))
        ]
        if not sound_records:
            return
        errors    = [abs(r["scheduling_error_ms"]) for r in sound_records]
        mean_err  = sum(errors) / len(errors)
        max_err   = max(errors)
        n_over_05 = sum(1 for e in errors if e > 0.5)
        n_over_1  = sum(1 for e in errors if e > 1.0)
        n_over_2  = sum(1 for e in errors if e > 2.0)
        self.logger.log(
            f"Run {self.current_run} timing sons : "
            f"{len(sound_records)} events | "
            f"mean |err|={mean_err:.3f} ms | "
            f"max |err|={max_err:.3f} ms | "
            f">0.5ms:{n_over_05} | >1ms:{n_over_1} | >2ms:{n_over_2}"
        )

    # ═════════════════════════════════════════════════════════════════════
    #  SAUVEGARDE
    # ═════════════════════════════════════════════════════════════════════

    def _save_run_data(self) -> Optional[str]:
        if not self.enregistrer or not self.current_run_records:
            return None
        return self.save_data(
            data_list=self.current_run_records,
            filename_suffix=(
                f"_{self.current_effector}_run{self.current_run:02d}"
            ),
        )

    # ═════════════════════════════════════════════════════════════════════
    #  UI — INSTRUCTIONS & PAUSES
    # ═════════════════════════════════════════════════════════════════════

    def _show_session_instructions(self) -> None:
        trial_dur = (
            self.preview_duration + self.plan_duration
            + self.execute_duration + self.iti_duration
        )
        run_dur_min = (
            self.initial_baseline
            + self.n_trials_total * trial_dur
            + self.final_baseline
        ) / 60.0
        total_min = run_dur_min * self.n_runs

        ptb_status = "OUI (sub-ms)" if _HAS_PTB else "NON (dégradé)"

        txt = (
            "══════════════════════════════════\n"
            "     PLANIFICATION MOTRICE\n"
            "══════════════════════════════════\n\n"
            f"Participant : {self.nom}\n"
            f"Session : {self.session}\n"
            f"Nombre de runs : {self.n_runs}\n"
            f"Audio PTB scheduling : {ptb_status}\n"
            f"Durée estimée : ~{total_min:.0f} min\n\n"
            "À chaque essai :\n"
            "  1. Fixez la croix\n"
            "  2. Écoutez l'instruction\n"
            "     • « Grasp » → Saisissez l'objet\n"
            "     • « Touch » → Touchez l'objet\n"
            "  3. Attendez le BIP pour exécuter\n"
            "  4. Revenez en position de départ\n\n"
            "  ► ESPACE pour continuer"
        )
        self.instr_stim.text = txt
        self.instr_stim.draw()
        self.win.flip()
        core.wait(0.5)
        self.flush_keyboard()
        self.wait_keys(key_list=["space"])

    def _show_run_instructions(self) -> None:
        eff_label = "la MAIN" if self.current_effector == "hand" else "l'OUTIL"
        txt = (
            f"─── Run {self.current_run} / {self.n_runs} ───\n\n"
            f"Effecteur : {eff_label}\n"
            f"{self.n_trials_total} essais "
            f"({self.n_trials_per_condition} grasp + "
            f"{self.n_trials_per_condition} touch)\n\n"
            "Préparez-vous.\n\n"
            "  ► ESPACE quand prêt"
        )
        self.instr_stim.text = txt
        self.instr_stim.draw()
        self.win.flip()
        core.wait(0.5)
        self.flush_keyboard()
        self.wait_keys(key_list=["space"])

    def _wait_for_run_start(self) -> None:
        """Reset clock → t=0 pour le run. Enregistre l'offset PTB."""
        self.task_clock.reset()

        if self.eyetracker_actif:
            self.EyeTracker.start_recording()
            self.EyeTracker.send_message(
                f"START_RUN{self.current_run:02d}"
                f"_{self.current_effector.upper()}"
            )

        self.logger.ok(
            f"Run {self.current_run} ({self.current_effector}) "
            f"démarré — clock reset"
        )

    def _show_inter_run_pause(self) -> None:
        next_eff = self.run_sequence[self.current_run]  # 0-indexed
        next_label = "la MAIN" if next_eff == "hand" else "l'OUTIL"
        txt = (
            f"═══ Run {self.current_run}/{self.n_runs} terminé ═══\n\n"
            f"Prochain : run {self.current_run + 1}/{self.n_runs}\n"
            f"Effecteur : {next_label}\n\n"
            "Prenez une pause.\n\n"
            "  ► ESPACE quand prêt"
        )
        self.instr_stim.text = txt
        self.instr_stim.draw()
        self.win.flip()
        core.wait(1.0)
        self.flush_keyboard()
        self.wait_keys(key_list=["space"])

    def _show_session_end(self) -> None:
        txt = (
            "══════════════════════════════════\n"
            "      SESSION TERMINÉE\n"
            "══════════════════════════════════\n\n"
            f"Les {self.n_runs} runs sont complétés.\n\n"
            "Merci pour votre participation !"
        )
        self.instr_stim.text = txt
        self.instr_stim.draw()
        self.win.flip()
        core.wait(5.0)

    # ═════════════════════════════════════════════════════════════════════
    #  MAIN ENTRY — SEQUENTIAL MULTI-RUN
    # ═════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        """
        Exécute tous les runs en séquence.

        1. Instructions session → ESPACE
        2. Pour chaque run :
           a. Build timeline + sauvegarde planned
           b. Instructions run → ESPACE
           c. Clock reset → exécution timeline
           d. Sauvegarde données run
           e. Pause inter-run → ESPACE (sauf dernier)
        3. Sauvegarde combinée + écran fin
        """
        finished = False

        try:
            self._show_session_instructions()

            for run_idx, effector in enumerate(self.run_sequence):
                self.current_run = run_idx + 1
                self.current_effector = effector
                self.current_run_records = []

                # ── Construire timeline ──
                self._build_full_timeline()
                self._save_planned_timeline()

                # ── Fichier incrémental ──
                self._init_incremental_file(
                    suffix=f"_{effector}_run{self.current_run:02d}"
                )

                # ── Instructions run ──
                self._show_run_instructions()

                # ── Démarrage ──
                self._wait_for_run_start()

                # ── Exécution ──
                self._execute_timeline()

                # ── Sauvegarde run ──
                saved_path = self._save_run_data()

                # ── QC optionnel ──
                if saved_path and os.path.exists(saved_path):
                    try:
                        from tasks.qc.qc_motor_planning import (
                            qc_motor_planning,
                        )
                        qc_motor_planning(saved_path)
                    except ImportError:
                        pass
                    except Exception as qc_exc:
                        self.logger.warn(f"QC échoué : {qc_exc}")

                # ── Eye tracker fin de run ──
                if self.eyetracker_actif:
                    self.EyeTracker.send_message(
                        f"END_RUN{self.current_run:02d}"
                    )

                self.logger.ok(
                    f"Run {self.current_run}/{self.n_runs} "
                    f"({effector}) complété."
                )

                # ── Pause inter-run ──
                if run_idx < len(self.run_sequence) - 1:
                    self._show_inter_run_pause()

            finished = True
            self.logger.ok(
                f"Session complète : {self.n_runs} runs terminés."
            )

        except (KeyboardInterrupt, SystemExit):
            self.logger.warn("Interruption manuelle.")

        except Exception as exc:
            self.logger.err(f"ERREUR CRITIQUE : {exc}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            for snd in self._sounds.values():
                try:
                    snd.stop()
                except Exception:
                    pass

            if self.eyetracker_actif:
                self.EyeTracker.stop_recording()
                self.EyeTracker.send_message("END_SESSION")
                self.EyeTracker.close_and_transfer_data(self.data_dir)

            if self.global_records:
                self.save_data(
                    data_list=self.global_records,
                    filename_suffix="_all_runs",
                )

            if finished:
                self._show_session_end()