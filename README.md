# Stimulation Électrique — Somatotopie Digitale & Tâche de Prédiction

## Vue d'ensemble

Protocole de stimulation électrique digitale en IRMf pour l'étude de la somatotopie corticale et des mécanismes de prédiction sensorielle.

## Protocole

### Phase 1 — Finger Mapping

Cartographie somatotopique par block design ON/OFF.

| Paramètre              | Valeur par défaut |
|-------------------------|-------------------|
| Blocs ON                | 20                |
| Blocs OFF               | 20                |
| Durée bloc ON           | 10 s              |
| Durée bloc OFF          | 10 s ± 5 s       |
| Doigts stimulés         | D1, D2, D3, D4   |
| Stims par doigt par bloc| 5                 |
| Total stims par bloc ON | 20                |
| ISI (inter-stim)        | 500 ms            |
| Séquence                | Pseudo-aléatoire  |
| Durée estimée           | ~6 min 40 s       |

**Contraintes de séquence :**
- Pas deux stimulations consécutives sur le même doigt
- Chaque doigt stimulé exactement 5 fois par bloc ON

---

### Phase 2 — Tâche de Prédiction

4 conditions expérimentales, chacune avec la même structure de blocs ON/OFF que le mapping.

| Condition | Code | Doigts | Séquence      | D4              |
|-----------|------|--------|---------------|-----------------|
| FP        | 50   | 4      | Prédictible   | Stimulé         |
| TP        | 51   | 3      | Prédictible   | Omission        |
| FR        | 52   | 4      | Aléatoire     | Stimulé         |
| TR        | 53   | 3      | Aléatoire     | Omission        |

**Séquence prédictible** : D1 → D2 → D3 → D4 (cyclique)

**Omission** : Le slot temporel de D4 est préservé (même ISI), aucune stimulation n'est envoyée. Cela permet de mesurer la réponse cérébrale à l'absence d'un stimulus attendu.

**Pause entre conditions** : 3 min (configurable)

**Durée estimée** : ~35 min (4 conditions × ~7 min + 3 pauses × 3 min)

