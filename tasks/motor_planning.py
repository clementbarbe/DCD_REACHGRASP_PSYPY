# motor_planning.py
"""
Motor Planning Protocol — Grasp / Touch × Hand / Tool
======================================================
EMG co-registration — Sequential multi-run — PTB audio scheduling.

Platforms : Windows, macOS only.

AUDIO : PsychoPy Sound + psychtoolbox (PTB) — PRE-SCHEDULED
────────────────────────────────────────────────────────────
Problème : snd.play() immédiat → jitter 5-20 ms (buffer fill time).
Solution : snd.play(when=T_ptb) appelé ~100 ms AVANT l'onset cible.
  - PTB pré-remplit le buffer audio vers le DAC
  - Le son démarre à T_ptb avec jitter < 0.1 ms
  - Le trigger port parallèle suit par busy-wait sur horloge PTB

Deux horloges coexistent :
  - task_clock (PsychoPy core.Clock) : temps relatif du run (reset à 0)
  - ptb.GetSecs() : horloge absolue du driver audio
  - Mapping : target_ptb = now_ptb + (target_task − now_task)
  - Retour : actual_task = now_task + (actual_ptb − now_ptb)

Paramètre audio_hw_delay_s :
  - Compense la latence analogique (DAC → haut-parleur) non gérée par PTB.
  - Défaut 0.0 s.  À calibrer à l'oscilloscope (typ. 0-8 ms).
  - Le trigger est décalé de +hw_delay pour s'aligner sur l'onset réel.
  - scheduling_error_ms est corrigé de ce délai → mesure le jitter pur.

ARCHITECTURE : PRE-COMPUTED TIMELINE (reconstruite par run)
───────────────────────────────────────────────────────────
1. Avant chaque run, TOUS les événements sont pré-calculés et triés.
2. ESPACE → clock reset → exécution séquentielle.
3. Les sons sont pré-schedulés 100 ms avant leur onset.
4. Zéro calcul pendant l'acquisition.

Structure d'un essai :
    Preview (fixation 2 s) → Cue audio ("Grasp"/"Touch") → Plan (5.5 s ± jitter)
    → Go beep → Execute (2 s) → ITI (8 s)

Triggers port parallèle par essai (4 max) :
    trial_start (30) → cue (11/12) → go (21/22) → iti (40)
    Espacement minimal garanti > 2 s par design.

Pseudo-randomisation :
    Jamais plus de N essais consécutifs de la même condition.
    Seed logué dans chaque fichier pour reproductibilité.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════
# Audio prefs — DOIT précéder tout import psychopy.sound
# ═══════════════════════════════════════════════════════════════════════
from psychopy import prefs
import sys

# FIXED: retrait support Linux — fail-fast explicite
if sys.platform not in ("win32", "darwin"):
    raise RuntimeError(
        f"Plateforme non supportée : {sys.platform}. "
        "Seuls Windows et macOS sont pris en charge."
    )

prefs.hardware["audioLib"] = ["ptb", "sounddevice"]
prefs.hardware["audioLatencyMode"] = 4 if sys.platform == "win32" else 3

import csv
import gc
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from psychopy import core, visual
from psychopy.sound import Sound

from utils.base_task import BaseTask

# ═══════════════════════════════════════════════════════════════════════
# psychtoolbox — scheduling audio sub-ms
# ═══════════════════════════════════════════════════════════════════════
try:
    import psychtoolbox as ptb

    _HAS_PTB = True
except ImportError:
    ptb = None
    _HAS_PTB = False

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

CONDITIONS: List[str] = ["grasp", "touch"]
EFFECTORS: List[str] = ["hand", "tool"]

# FIXED: codes uniques + aucun doublon — discrimination EMG possible
DEFAULT_EVENT_CODES: Dict[str, int] = {
    "run_start":   100,
    "run_end":     200,
    "trial_start":  30,
    "cue_grasp":    11,
    "cue_touch":    12,
    "go_grasp":     21,
    "go_touch":     22,
    "iti_start":    40,
}

# Priorité de tri — même onset → visuel < parport < son < marqueur
_ACTION_PRIORITY: Dict[str, int] = {
    "visual_fixation":    0,
    "visual_instruction": 0,
    "parport_event":      1,
    "sound_cue":          2,
    "sound_go":           2,
    "marker":             3,
}

# FIXED: seuil minimal entre deux triggers port parallèle (secondes)
MIN_TRIGGER_SPACING_S: float = 0.005


# ═══════════════════════════════════════════════════════════════════════

class MotorPlanning(BaseTask):
    """
    Sequential multi-run motor planning task with PTB audio scheduling.

    Parameters
    ----------
    run_sequence : list[str] | None
        Séquence d'effecteurs. Défaut : ['hand','tool'] × 4.
    sound_preload_s : float
        Fenêtre de pré-chargement PTB avant onset (défaut 0.100 s).
    audio_hw_delay_s : float
        Latence DAC → haut-parleur non gérée par PTB (défaut 0.0 s).
        Calibrer à l'oscilloscope. Le trigger est décalé de cette valeur.
    random_seed : int | None
        Graine PRNG. None = dérivée de l'horloge.
        Loguée dans chaque fichier de données.
    """

    def __init__(
        self,
        win: visual.Window,
        nom: str,
        session: str = "01",
        # ── Séquence de runs ──
        run_sequence: Optional[List[str]] = None,
        # ── Nombre d'essais ──
        n_trials_per_condition: int = 20,
        max_consecutive: int = 3,
        # ── Timing (secondes) ──
        preview_duration: float = 2.0,
        plan_duration: float = 5.5,
        plan_jitter: float = 0.5,
        execute_duration: float = 2.0,
        iti_duration: float = 8.0,
        initial_baseline: float = 10.0,
        final_baseline: float = 10.0,
        # ── Audio scheduling ──
        sound_preload_s: float = 0.100,
        audio_hw_delay_s: float = 0.0,
        # ── Sons ──
        sound_dir: str = "sounds",
        grasp_sound_file: str = "grasp.wav",
        touch_sound_file: str = "touch.wav",
        go_beep_file: str = "beep.wav",
        go_beep_freq: float = 1000.0,
        go_beep_volume: float = 0.7,
        cue_fallback_duration: float = 0.5,
        go_fallback_duration: float = 0.5,
        # ── Reproductibilité ──
        random_seed: Optional[int] = None,
        # ── Misc ──
        enregistrer: bool = True,
        eyetracker_actif: bool = False,
        parport_actif: bool = True,
        event_codes: Optional[Dict[str, int]] = None,
        # FIXED: plus de **kwargs — une typo lèvera TypeError
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

        # ── FIXED: seed PRNG avant toute randomisation ──
        if random_seed is None:
            random_seed = int(datetime.now().timestamp() * 1000) % (2**32)
        self.random_seed: int = random_seed
        random.seed(self.random_seed)
        self.logger.ok(f"Random seed : {self.random_seed}")

        # ── Séquence de runs ──
        if run_sequence is None:
            run_sequence = ["hand", "tool"] * 4
        self.run_sequence: List[str] = [e.lower() for e in run_sequence]
        self.n_runs: int = len(self.run_sequence)
        if self.n_runs == 0:
            raise ValueError("run_sequence ne peut pas être vide.")
        for eff in self.run_sequence:
            if eff not in EFFECTORS:
                raise ValueError(
                    f"Effecteur inconnu '{eff}'. Valides : {EFFECTORS}"
                )

        # ── Design ──
        self.n_trials_per_condition: int = n_trials_per_condition
        if self.n_trials_per_condition < 1:
            raise ValueError("n_trials_per_condition doit être ≥ 1")
        self.n_trials_total: int = len(CONDITIONS) * n_trials_per_condition
        self.max_consecutive: int = max_consecutive
        if self.max_consecutive < 1:
            raise ValueError("max_consecutive doit être ≥ 1")

        # ── Timing ──
        self.preview_duration: float = max(0.5, preview_duration)
        self.plan_duration: float    = plan_duration
        self.plan_jitter: float      = plan_jitter
        self.execute_duration: float = execute_duration
        self.iti_duration: float     = iti_duration
        self.initial_baseline: float = initial_baseline
        self.final_baseline: float   = final_baseline

        # ── Audio scheduling ──
        self.sound_preload_s: float  = sound_preload_s
        self.audio_hw_delay_s: float = audio_hw_delay_s

        # ── Sons ──
        self.sound_dir: str               = sound_dir
        self.grasp_sound_file: str        = grasp_sound_file
        self.touch_sound_file: str        = touch_sound_file
        self.go_beep_file: str            = go_beep_file
        self.go_beep_freq: float          = go_beep_freq
        self.go_beep_volume: float        = go_beep_volume
        self.cue_fallback_duration: float = cue_fallback_duration
        self.go_fallback_duration: float  = go_fallback_duration

        # ── FIXED: validation codes trigger uniques ──
        self.event_codes: Dict[str, int] = (
            event_codes if event_codes else dict(DEFAULT_EVENT_CODES)
        )
        self._validate_trigger_codes()

        # ── État runtime ──
        self.global_records: List[Dict[str, Any]] = []
        self.current_run_records: List[Dict[str, Any]] = []
        self.timeline: List[Dict[str, Any]] = []
        self.current_run: int = 0
        self.current_effector: str = ""

        # ── PsychoPy Sound objects ──
        self._sounds: Dict[str, Sound] = {}

        # ── PTB availability ──
        self._ptb_available: bool = _HAS_PTB

        # ── Init chain ──
        self._measure_frame_rate()
        self._setup_visual_stimuli()
        self._setup_sounds()
        self._check_sound_durations()

        # ── Log PTB status ──
        if self._ptb_available:
            self.logger.ok(
                f"psychtoolbox DISPONIBLE — scheduling activé "
                f"(preload={self.sound_preload_s * 1000:.0f} ms, "
                f"hw_delay={self.audio_hw_delay_s * 1000:.1f} ms)"
            )
        else:
            self.logger.warn(
                "psychtoolbox NON DISPONIBLE — "
                "timing audio dégradé (play immédiat). "
                "Installez : pip install psychtoolbox"
            )

        # ── Durée estimée ──
        trial_dur = (
            self.preview_duration
            + self.plan_duration
            + self.execute_duration
            + self.iti_duration
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
            f"~{session_dur / 60:.0f} min total | "
            f"seed={self.random_seed}"
        )

    # ═══════════════════════════════════════════════════════════════════
    #  INIT HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _validate_trigger_codes(self) -> None:
        """FIXED: vérifie unicité des codes trigger — échoue fort."""
        values = [v for v in self.event_codes.values() if v != 0]
        if len(values) != len(set(values)):
            dupes = [v for v in set(values) if values.count(v) > 1]
            raise ValueError(
                f"Codes trigger dupliqués détectés : {dupes}. "
                f"La disambiguation EMG requiert des codes uniques."
            )
        self.logger.ok(
            f"Codes trigger validés : {len(self.event_codes)} entrées uniques"
        )

    def _measure_frame_rate(self) -> None:
        measured = self.win.getActualFrameRate(
            nIdentical=10, nMaxFrames=100, threshold=1,
        )
        self.frame_rate: float = measured if measured else 60.0
        self.frame_duration_s: float = 1.0 / self.frame_rate
        self.logger.log(
            f"Frame rate : {self.frame_rate:.1f} Hz → "
            f"{self.frame_duration_s * 1000:.2f} ms/frame"
        )

    # ═══════════════════════════════════════════════════════════════════
    #  VISUAL STIMULI
    # ═══════════════════════════════════════════════════════════════════

    def _setup_visual_stimuli(self) -> None:
        self.cue_stim = visual.TextStim(
            self.win,
            text="",
            height=0.06,
            color="white",
            pos=(0.0, 0.0),
            wrapWidth=1.5,
            font="Arial",
            bold=False,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  SOUND SETUP
    # ═══════════════════════════════════════════════════════════════════

    def _setup_sounds(self) -> None:
        """Pré-charge tous les stimuli audio avec fallback gracieux."""
        backend = "PTB" if self._ptb_available else "sounddevice"
        self.logger.log(f"Audio backend : {backend} | OS={sys.platform}")

        base = Path(self.root_dir) / self.sound_dir

        def _safe_load(
            value: Any,
            name: str,
            fallback_freq: float = 400,
            fallback_dur: float = 0.5,
        ) -> Optional[Sound]:
            try:
                snd = Sound(value=value, name=name)
                self.logger.ok(f"Chargé : {name} ({value})")
                return snd
            except Exception as e:
                self.logger.warn(f"Échec chargement {name} ({value}) : {e}")
                try:
                    snd = Sound(
                        value=fallback_freq,
                        secs=fallback_dur,
                        volume=self.go_beep_volume,
                        name=f"{name}_fallback",
                    )
                    self.logger.warn(
                        f"Fallback ton {fallback_freq} Hz pour {name}"
                    )
                    return snd
                except Exception as e2:
                    self.logger.err(
                        f"Fallback AUSSI échoué pour {name} : {e2}"
                    )
                    return None

        # ── Cue Grasp ──
        grasp_path = base / self.grasp_sound_file
        self._sounds["grasp"] = _safe_load(
            str(grasp_path) if grasp_path.exists() else 400,
            "grasp_cue",
            400,
            self.cue_fallback_duration,
        )

        # ── Cue Touch ──
        touch_path = base / self.touch_sound_file
        self._sounds["touch"] = _safe_load(
            str(touch_path) if touch_path.exists() else 600,
            "touch_cue",
            600,
            self.cue_fallback_duration,
        )

        # ── Go beep ──
        go_path = base / self.go_beep_file
        if go_path.exists():
            self._sounds["go"] = _safe_load(str(go_path), "go_beep_wav")
        else:
            self.logger.warn("beep.wav introuvable → fallback tone")
            self._sounds["go"] = _safe_load(
                self.go_beep_freq,
                "go_beep_fallback",
                self.go_beep_freq,
                self.go_fallback_duration,
            )

        # ── Vérifier les sons critiques ──
        missing = [k for k, v in self._sounds.items() if v is None]
        if missing:
            raise RuntimeError(
                f"Sons critiques non chargés : {missing}. "
                f"Vérifiez votre configuration audio."
            )

        # ── Warmup driver audio ──
        try:
            warmup = Sound(value=100, secs=0.01, volume=0.0, name="warmup")
            warmup.play()
            core.wait(0.05)
            warmup.stop()
            self.logger.ok("Audio driver amorcé.")
        except Exception as e:
            self.logger.warn(f"Audio warmup échoué (non bloquant) : {e}")

    def _check_sound_durations(self) -> None:
        """FIXED: vérifie que les sons ne risquent pas de se chevaucher."""
        for name, snd in self._sounds.items():
            try:
                dur = snd.getDuration()
                if dur is not None:
                    self.logger.log(f"Son '{name}' : {dur:.3f} s")
                    if dur > 2.0:
                        self.logger.warn(
                            f"Son '{name}' dure {dur:.2f} s — "
                            f"risque de chevauchement avec l'événement suivant"
                        )
                else:
                    self.logger.log(f"Son '{name}' : durée inconnue")
            except Exception:
                self.logger.log(f"Son '{name}' : durée indisponible")

    # ═══════════════════════════════════════════════════════════════════
    #  PSEUDO-RANDOMISATION
    # ═══════════════════════════════════════════════════════════════════

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
        max_attempts: int = 10_000,
    ) -> List[str]:
        """
        Séquence pseudo-aléatoire : jamais plus de max_c consécutifs.

        Phase 1 : shuffle-and-check (rapide).
        Phase 2 : construction incrémentale (garanti).
        """
        pool = [it for it in items for _ in range(reps)]
        n = len(pool)

        # Phase 1
        for _ in range(max_attempts):
            random.shuffle(pool)
            if MotorPlanning._check_max_consecutive(pool, max_c):
                return list(pool)

        # Phase 2 — fallback déterministe
        seq: List[str] = []
        remaining = {it: reps for it in items}
        for _ in range(n):
            available = [
                it
                for it in items
                if remaining[it] > 0
                and not (
                    len(seq) >= max_c
                    and all(s == it for s in seq[-max_c:])
                )
            ]
            if not available:
                available = [it for it in items if remaining[it] > 0]
            chosen = random.choice(available)
            seq.append(chosen)
            remaining[chosen] -= 1
        return seq

    def _build_trial_order(self) -> List[str]:
        """Construit l'ordre pseudo-aléatoire pour le run courant."""
        order = self._pseudo_random_max_consecutive(
            CONDITIONS,
            self.n_trials_per_condition,
            max_c=self.max_consecutive,
        )
        # FIXED: raise au lieu d'assert — pas désactivable par -O
        if not self._check_max_consecutive(order, self.max_consecutive):
            raise ValueError(
                f"Contrainte max {self.max_consecutive} consécutifs violée"
            )
        if len(order) != self.n_trials_total:
            raise ValueError(
                f"Nombre d'essais incorrect : "
                f"{len(order)} ≠ {self.n_trials_total}"
            )
        return order

    # ═══════════════════════════════════════════════════════════════════
    #  TIMELINE CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════

    def _add_event(
        self, onset_s: float, action: str, **kwargs: Any
    ) -> None:
        event: Dict[str, Any] = {
            "onset_s": round(onset_s, 6),
            "action": action,
            "_priority": _ACTION_PRIORITY.get(action, 9),
        }
        event.update(kwargs)
        self.timeline.append(event)

    def _build_trial_events(
        self, t: float, trial_idx: int, condition: str
    ) -> float:
        """
        Ajoute tous les événements pour UN essai commençant à t.
        Returns : onset du prochain essai.

        FIXED: architecture trigger simplifiée — 4 triggers port max/essai :
          trial_start (30) → cue (11/12) → go (21/22) → iti (40)
          Espacement garanti > 2 s par la structure temporelle.
          Pas de triggers redondants (preview = trial_start, execute = go).
        """
        trial_start_t = t
        jitter = random.uniform(-self.plan_jitter, self.plan_jitter)
        plan_dur = max(2.0, self.plan_duration + jitter)

        # ── TRIAL START + PREVIEW (fusionnés — un seul trigger) ──
        self._add_event(
            t,
            "marker",
            label="trial_start",
            trial_index=trial_idx,
            condition=condition,
            effector=self.current_effector,
            plan_duration_s=round(plan_dur, 4),
            jitter_s=round(jitter, 4),
        )
        if self.parport_actif:
            self._add_event(
                t,
                "parport_event",
                label="trial_start_trigger",
                trial_index=trial_idx,
                condition=condition,
                pin_code=self.event_codes["trial_start"],
            )
        self._add_event(
            t + 0.001,
            "visual_fixation",
            label="preview_start",
            trial_index=trial_idx,
            condition=condition,
        )

        # ── CUE AUDIO (trigger envoyé par _execute_sound_event) ──
        t_cue = trial_start_t + self.preview_duration
        self._add_event(
            t_cue,
            "sound_cue",
            label=f"cue_{condition}",
            trial_index=trial_idx,
            condition=condition,
            sound_id=condition,
            pin_code=self.event_codes[f"cue_{condition}"],
        )

        # ── GO BEEP (trigger envoyé par _execute_sound_event) ──
        t_go = t_cue + plan_dur
        self._add_event(
            t_go,
            "sound_go",
            label="go_beep",
            trial_index=trial_idx,
            condition=condition,
            sound_id="go",
            pin_code=self.event_codes[f"go_{condition}"],
        )
        # Marqueur CSV-only — pas de trigger port (redondant avec go)
        self._add_event(
            t_go + 0.001,
            "marker",
            label="execute_start",
            trial_index=trial_idx,
            condition=condition,
        )

        # ── ITI ──
        t_iti = t_go + self.execute_duration
        self._add_event(
            t_iti,
            "marker",
            label="iti_start",
            trial_index=trial_idx,
            condition=condition,
        )
        if self.parport_actif:
            self._add_event(
                t_iti,
                "parport_event",
                label="iti_trigger",
                trial_index=trial_idx,
                pin_code=self.event_codes["iti_start"],
            )
        self._add_event(
            t_iti + 0.002,
            "visual_fixation",
            label="iti_fixation",
            trial_index=trial_idx,
        )

        # ── TRIAL END (marqueur CSV-only) ──
        t_end = t_iti + self.iti_duration
        self._add_event(
            t_end,
            "marker",
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
            t,
            "marker",
            label="run_start",
            effector=self.current_effector,
            run_number=self.current_run,
            n_trials=len(trial_order),
            trial_order=str(trial_order),
            random_seed=self.random_seed,
        )
        if self.parport_actif:
            self._add_event(
                t + 0.001,
                "parport_event",
                label="run_start_trigger",
                pin_code=self.event_codes["run_start"],
            )

        # ── Baseline initiale ──
        self._add_event(t + 0.002, "visual_fixation", label="baseline_start")
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
                t + 0.001,
                "parport_event",
                label="run_end_trigger",
                pin_code=self.event_codes["run_end"],
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

    # ═══════════════════════════════════════════════════════════════════
    #  TIMELINE VALIDATION
    # ═══════════════════════════════════════════════════════════════════

    def _validate_timeline(self) -> None:
        """FIXED: validation étendue — collisions, espacement, monotonie."""
        if not self.timeline:
            return

        visual_onsets: set[float] = set()
        sound_onsets: set[float] = set()
        parport_times: List[float] = []

        for evt in self.timeline:
            if evt["action"].startswith("visual_"):
                visual_onsets.add(evt["onset_s"])
            elif evt["action"].startswith("sound_"):
                sound_onsets.add(evt["onset_s"])
            elif evt["action"] == "parport_event":
                parport_times.append(evt["onset_s"])

        # 1. Collision flip / son
        collisions = visual_onsets & sound_onsets
        if collisions:
            self.logger.err(
                f"COLLISION flip/son : {len(collisions)} détectée(s) !"
            )
        else:
            self.logger.ok("Timeline : aucune collision flip/son.")

        # 2. Espacement sons vs preload
        sound_times = sorted(sound_onsets)
        for i in range(1, len(sound_times)):
            gap = sound_times[i] - sound_times[i - 1]
            if gap < self.sound_preload_s + 0.010:
                self.logger.warn(
                    f"Sons trop rapprochés : {gap:.3f} s "
                    f"(preload={self.sound_preload_s:.3f} s) "
                    f"à t={sound_times[i]:.3f} s"
                )

        # 3. Espacement triggers (port + sons qui envoient un trigger)
        all_trigger_times = sorted(parport_times + sound_times)
        for i in range(1, len(all_trigger_times)):
            gap = all_trigger_times[i] - all_trigger_times[i - 1]
            if gap < MIN_TRIGGER_SPACING_S:
                self.logger.warn(
                    f"Triggers trop rapprochés : {gap * 1000:.1f} ms "
                    f"< {MIN_TRIGGER_SPACING_S * 1000:.0f} ms minimum "
                    f"à t={all_trigger_times[i]:.3f} s"
                )

        # 4. Monotonie
        onsets = [e["onset_s"] for e in self.timeline]
        if any(onsets[i] > onsets[i + 1] for i in range(len(onsets) - 1)):
            self.logger.err("Timeline NON monotone après tri !")

        # 5. Pas d'onset négatif
        if any(t < 0 for t in onsets):
            self.logger.err("Onset négatif détecté dans la timeline !")

    def _save_planned_timeline(self) -> None:
        """Sauvegarde la timeline planifiée sur disque (CSV)."""
        if not self.enregistrer or not self.timeline:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=all_keys, extrasaction="ignore"
                )
                writer.writeheader()
                for evt in self.timeline:
                    writer.writerow(
                        {k: v for k, v in evt.items() if k != "_priority"}
                    )
            self.logger.ok(f"Planned timeline → {path}")
        except Exception as e:
            self.logger.err(f"Échec sauvegarde planned : {e}")

    # ═══════════════════════════════════════════════════════════════════
    #  TIMELINE EXECUTION
    # ═══════════════════════════════════════════════════════════════════

    def _wait_until(
        self, target_s: float, high_precision: bool = False
    ) -> None:
        """
        Attend que task_clock atteigne target_s.
        high_precision : sleep puis busy-wait les 2 dernières ms.
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

    def _execute_sound_event(
        self, event: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Exécute un événement sonore avec pre-scheduling PTB.

        FIXED: retourne les temps dans les DEUX horloges pour un
        calcul d'erreur correct. Le trigger est envoyé au moment
        (target_ptb + audio_hw_delay_s) pour compenser la latence DAC.

        Returns
        -------
        dict avec :
            actual_task_s : onset dans le référentiel task_clock
            actual_ptb_s  : onset dans le référentiel PTB (NaN si pas PTB)
        """
        target_onset = event["onset_s"]
        sound_id = (
            "go"
            if event["action"] == "sound_go"
            else event.get("sound_id", "grasp")
        )
        snd = self._sounds.get(sound_id)
        pin_code = event.get("pin_code", 0)

        result: Dict[str, float] = {
            "actual_task_s": 0.0,
            "actual_ptb_s": float("nan"),
        }

        # Son manquant → trigger seul au bon moment
        if snd is None:
            self.logger.warn(f"Son introuvable : {sound_id}")
            self._wait_until(target_onset, high_precision=True)
            if self.parport_actif and pin_code:
                self.ParPort.send_trigger(pin_code)
            result["actual_task_s"] = self.task_clock.getTime()
            return result

        # ═══════════════════════════════════════════
        # PTB PATH
        # ═══════════════════════════════════════════
        if self._ptb_available:

            # Attente jusqu'à la fenêtre de preload
            preload_t = max(0.0, target_onset - self.sound_preload_s)
            self._wait_until(preload_t, high_precision=False)

            # Snapshot des deux horloges
            now_task = self.task_clock.getTime()
            now_ptb = ptb.GetSecs()
            delta = target_onset - now_task

            # Trop tard pour scheduler → fallback immédiat
            if delta < 0.010:
                self.logger.warn(
                    f"LATE scheduling ({delta * 1000:.1f} ms) "
                    f"pour {event.get('label', '?')}"
                )
                self._wait_until(target_onset, high_precision=True)
                if self.parport_actif and pin_code:
                    self.ParPort.send_trigger(pin_code)
                snd.stop()
                snd.play()
                result["actual_task_s"] = self.task_clock.getTime()
                return result

            # Temps cible PTB
            target_ptb = now_ptb + delta

            # Temps trigger = onset audio + compensation DAC
            trigger_ptb = target_ptb + self.audio_hw_delay_s

            # Pre-schedule audio
            snd.stop()
            snd.play(when=target_ptb)

            # Busy-wait sur horloge PTB
            while ptb.GetSecs() < trigger_ptb:
                pass

            # FIXED: capturer le temps PTB exact du trigger
            actual_ptb = ptb.GetSecs()
            if self.parport_actif and pin_code:
                self.ParPort.send_trigger(pin_code)

            # FIXED: reconvertir dans le référentiel task_clock
            result["actual_task_s"] = now_task + (actual_ptb - now_ptb)
            result["actual_ptb_s"] = actual_ptb

        # ═══════════════════════════════════════════
        # FALLBACK (pas de PTB)
        # ═══════════════════════════════════════════
        else:
            self._wait_until(target_onset, high_precision=True)
            if self.parport_actif and pin_code:
                self.ParPort.send_trigger(pin_code)
            snd.stop()
            snd.play()
            result["actual_task_s"] = self.task_clock.getTime()

        return result

    def _dispatch_event(self, event: Dict[str, Any]) -> float:
        """Exécute un événement non-sonore. Retourne task_clock time."""
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

        # marker — aucune action hardware
        return self.task_clock.getTime()

    # ── Record ───────────────────────────────────────────────────────

    def _build_execution_record(
        self,
        event: Dict[str, Any],
        actual_task_s: float,
        actual_ptb_s: float = float("nan"),
    ) -> Dict[str, Any]:
        """
        FIXED: scheduling_error_ms corrigé de audio_hw_delay_s
        pour les événements sonores → mesure le jitter pur.
        """
        error_ms = (actual_task_s - event["onset_s"]) * 1000.0
        if event["action"].startswith("sound_"):
            error_ms -= self.audio_hw_delay_s * 1000.0

        return {
            "participant": self.nom,
            "session": self.session,
            "random_seed": self.random_seed,
            "effector": self.current_effector,
            "run_number": self.current_run,
            "event_index": event.get("event_index", ""),
            "action": event["action"],
            "label": event.get("label", ""),
            "onset_planned_s": event["onset_s"],
            "onset_actual_s": round(actual_task_s, 6),
            "onset_ptb_s": (
                round(actual_ptb_s, 6)
                if not math.isnan(actual_ptb_s)
                else ""
            ),
            "scheduling_error_ms": round(error_ms, 3),
            "audio_hw_delay_ms": (
                round(self.audio_hw_delay_s * 1000, 3)
                if event["action"].startswith("sound_")
                else ""
            ),
            "trial_index": event.get("trial_index", ""),
            "condition": event.get("condition", ""),
            "sound_id": event.get("sound_id", ""),
            "pin_code": event.get("pin_code", ""),
            "plan_duration_s": event.get("plan_duration_s", ""),
            "jitter_s": event.get("jitter_s", ""),
            "trial_duration_s": event.get("trial_duration_s", ""),
        }

    # ── Boucle d'exécution principale ────────────────────────────────

    def _execute_timeline(self) -> None:
        """
        Parcourt la timeline pré-calculée.

        FIXED:
          - Quit check avant CHAQUE événement (y compris sons)
          - GC réactivé pendant les ITI et baselines (fenêtres sûres)
          - Retour PTB → task_clock cohérent
        """
        n_events = len(self.timeline)
        self.logger.log(
            f"Exécution run {self.current_run} "
            f"({self.current_effector}) : {n_events} événements …"
        )

        # Labels pendant lesquels on peut relâcher le GC
        _GC_SAFE_LABELS = frozenset(
            {"baseline_start", "final_baseline_start", "iti_start"}
        )

        gc.disable()

        try:
            for i, event in enumerate(self.timeline):
                is_sound = event["action"].startswith("sound_")

                # ── FIXED: quit check systématique ──
                self.should_quit()

                # ── FIXED: GC pendant les fenêtres sûres ──
                if event.get("label") in _GC_SAFE_LABELS:
                    gc.enable()
                    gc.collect()
                    gc.disable()

                # ── Dispatch ──
                if is_sound:
                    timing = self._execute_sound_event(event)
                    actual_t = timing["actual_task_s"]
                    actual_ptb = timing.get("actual_ptb_s", float("nan"))
                else:
                    self._wait_until(event["onset_s"])
                    actual_t = self._dispatch_event(event)
                    actual_ptb = float("nan")

                # ── Enregistrement ──
                record = self._build_execution_record(
                    event, actual_t, actual_ptb
                )
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
                    cond = event.get("condition", "?")
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
            r
            for r in self.current_run_records
            if r["action"].startswith("sound_")
            and isinstance(r["scheduling_error_ms"], (int, float))
        ]
        if not sound_records:
            return

        errors = [abs(r["scheduling_error_ms"]) for r in sound_records]
        errors_sorted = sorted(errors)
        n = len(errors_sorted)

        mean_err = sum(errors) / n
        max_err = max(errors)
        p50 = errors_sorted[n // 2]
        p95 = errors_sorted[int(n * 0.95)] if n >= 20 else max_err
        p99 = errors_sorted[int(n * 0.99)] if n >= 100 else max_err
        n_over_1 = sum(1 for e in errors if e > 1.0)
        n_over_2 = sum(1 for e in errors if e > 2.0)

        self.logger.log(
            f"Run {self.current_run} timing sons : "
            f"{n} events | mean={mean_err:.3f} ms | "
            f"p50={p50:.3f} ms | p95={p95:.3f} ms | "
            f"p99={p99:.3f} ms | max={max_err:.3f} ms | "
            f">1 ms:{n_over_1} | >2 ms:{n_over_2}"
        )

    # ═══════════════════════════════════════════════════════════════════
    #  SAUVEGARDE
    # ═══════════════════════════════════════════════════════════════════

    def _save_run_data(self) -> Optional[str]:
        if not self.enregistrer or not self.current_run_records:
            return None
        return self.save_data(
            data_list=self.current_run_records,
            filename_suffix=(
                f"_{self.current_effector}_run{self.current_run:02d}"
            ),
        )

    # ═══════════════════════════════════════════════════════════════════
    #  UI — INSTRUCTIONS & PAUSES
    # ═══════════════════════════════════════════════════════════════════

    def _show_session_instructions(self) -> None:
        trial_dur = (
            self.preview_duration
            + self.plan_duration
            + self.execute_duration
            + self.iti_duration
        )
        run_dur_min = (
            self.initial_baseline
            + self.n_trials_total * trial_dur
            + self.final_baseline
        ) / 60.0
        total_min = run_dur_min * self.n_runs
        ptb_status = "OUI (sub-ms)" if self._ptb_available else "NON (dégradé)"

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
        eff_label = (
            "la MAIN" if self.current_effector == "hand" else "l'OUTIL"
        )
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
        """Reset clock → t = 0 pour le run."""
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
        next_eff = self.run_sequence[self.current_run]  # 0-indexed lookup
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

    # ═══════════════════════════════════════════════════════════════════
    #  MAIN ENTRY — SEQUENTIAL MULTI-RUN
    # ═══════════════════════════════════════════════════════════════════

    def run(self) -> None:
        """Exécute tous les runs en séquence."""
        finished = False

        try:
            self._show_session_instructions()

            for run_idx, effector in enumerate(self.run_sequence):
                self.current_run = run_idx + 1
                self.current_effector = effector
                self.current_run_records = []

                # ── Build timeline ──
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