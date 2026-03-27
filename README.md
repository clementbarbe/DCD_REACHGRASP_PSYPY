# Motor Planning Task — Grasp / Touch × Hand / Tool

Tâche de planification motrice avec co-enregistrement EMG, inspirée de
Gallivan et al. (2013, *eLife*).
Conçue pour l'étude des représentations motrices anticipatoires — CENIR,
Institut du Cerveau (ICM).

## Principe

1. Une croix de fixation apparaît (**Preview**, 2 s)
2. Une instruction audio indique l'action à planifier :
   **« Grasp »** (saisir l'objet) ou **« Touch »** (toucher l'objet)
3. Le participant prépare mentalement le mouvement (**Plan**, 5.5 s ± jitter)
4. Un **bip sonore** (1000 Hz) signale l'exécution (**Go**)
5. Le participant exécute le mouvement puis revient en position de départ
6. Intervalle inter-essai (**ITI**, 8 s)

L'effecteur (main nue ou outil inversé) est constant au sein d'un run
et alterne entre les runs.

## Design

| Paramètre | Valeur par défaut |
|-----------|-------------------|
| Runs | 8 (alternance hand / tool) |
| Essais / run | 40 (20 grasp + 20 touch) |
| Max consécutifs identiques | 3 |
| Durée estimée / run | ~12 min |
| Durée totale session | ~100 min (pauses incluses) |

## Structure d'un essai
Preview (2 s)  →  Cue audio  →  Plan (5.5 ± 0.5 s)  →  Go beep  →  Execute (2 s)  →  ITI (8 s)
│                │                                      │
fixation     "Grasp"/"Touch"                        trigger EMG
trigger EMG                            + onset son

## Timing audio — Pre-scheduling PTB

Pour garantir une synchronisation sub-milliseconde entre le trigger EMG
et l'onset audio :

1. **100 ms avant** l'onset cible : `snd.play(when=T_ptb)`
   → PsychToolbox pré-remplit le buffer DAC
2. **À l'onset exact** : busy-wait → trigger port parallèle
   → le son et le trigger arrivent simultanément (< 1 ms)

**Prérequis** : `pip install psychtoolbox`
(sans PTB, le fallback `sounddevice` introduit ~5-20 ms de jitter)

**Validation** : mesurer le délai trigger→onset audio sur oscilloscope
avant toute collecte de données.

## Pseudo-randomisation

L'ordre des essais est pseudo-aléatoire avec la contrainte
**jamais plus de 3 essais consécutifs de la même condition**.

Algorithme en deux phases :
1. Shuffle-and-check (rapide, stochastique)
2. Construction incrémentale (fallback déterministe, garanti)

## Paramètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `run_sequence` | `[hand,tool] × 4` | Séquence d'effecteurs par run |
| `n_trials_per_condition` | `20` | Essais par condition par run |
| `max_consecutive` | `3` | Répétitions max d'affilée |
| `preview_duration` | `2.0 s` | Durée fixation avant instruction |
| `plan_duration` | `5.5 s` | Durée de planification (centre du jitter) |
| `plan_jitter` | `± 0.5 s` | Jitter uniforme sur la phase Plan |
| `execute_duration` | `2.0 s` | Fenêtre d'exécution |
| `iti_duration` | `8.0 s` | Intervalle inter-essai |
| `initial_baseline` | `10.0 s` | Baseline de début de run |
| `final_baseline` | `10.0 s` | Baseline de fin de run |
| `go_beep_freq` | `1000 Hz` | Fréquence du bip Go |
| `sound_preload_s` | `0.100 s` | Marge de pré-scheduling PTB |

## Prérequis

- Python 3.10+
- PsychoPy 2025.1.1
- psychtoolbox (`pip install psychtoolbox`)
- numpy

Fichiers audio attendus dans `sounds/` :
- `grasp.wav` — instruction « Grasp »
- `touch.wav` — instruction « Touch »

Si absents, des tons purs de remplacement sont générés automatiquement
(400 Hz pour grasp, 600 Hz pour touch).

## Déroulement d'une session
Instructions session → ESPACE
│
├─ Run 1 (hand) → ESPACE → exécution → sauvegarde
│   └─ Pause inter-run → ESPACE
├─ Run 2 (tool) → ESPACE → exécution → sauvegarde
│   └─ Pause inter-run → ESPACE
├─ ...
└─ Run 8 (tool) → ESPACE → exécution → sauvegarde
│
Écran de fin → sauvegarde combinée

## Données

Les résultats sont sauvegardés dans `data/motor_planning/` :

| Fichier | Contenu |
|---------|---------|
| `*_hand_run01_*_planned.csv` | Timeline pré-calculée (avant exécution) |
| `*_hand_run01_*_incremental.csv` | Backup événement par événement |
| `*_hand_run01_*.csv` | Données finales du run |
| `*_all_runs_*.csv` | Données combinées (tous les runs) |

## Codes triggers EMG

| Code | Événement |
|------|-----------|
| 100 | Début de run |
| 200 | Fin de run |
| 1 | Début d'essai |
| 2 | Début preview |
| 10 | Cue « Grasp » |
| 20 | Cue « Touch » |
| 30 | Go beep (grasp) |
| 40 | Go beep (touch) |
| 50 | Début exécution |
| 60 | Début ITI |

## Référence

Gallivan, J. P., McLean, D. A., Valyear, K. F., & Culham, J. C. (2013).
Decoding the neural mechanisms of human tool use.
*eLife*, 2, e00425. https://doi.org/10.7554/eLife.00425

## Auteur

Clément BARBE — CENIR, Institut du Cerveau (ICM), Paris