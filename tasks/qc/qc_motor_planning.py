# qc_connectelec.py
"""
QC Dashboard — Stimulation Électrique (Mapping & Prediction)
=============================================================
Compatible avec les deux formats de données :
    - ANCIEN : time_s, target_time_s, stim_index (toutes les lignes = stim)
    - NOUVEAU (timeline) : onset_actual_s, onset_planned_s, action, label, ...
      (les lignes incluent markers/visuels → filtrage sur action)

Panneaux :
    1. Précision temporelle (scheduling_error_ms)
    2. ISI intra-bloc vs cible 500 ms
    3. Distribution des doigts par condition
    4. Vérification des omissions par condition
    5. Durées des blocs ON
    6. Scheduling error au fil du temps (dérive)
"""

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

FINGER_ORDER = ["D1", "D2", "D3", "D4"]
CONDITION_COLORS = {
    "FP": "#2ca02c",
    "TP": "#d62728",
    "FR": "#1f77b4",
    "TR": "#ff7f0e",
    "mapping": "#9467bd",
}

# Colonnes canoniques utilisées en interne par le QC
_CANONICAL = {
    "time_s":               "time_s",
    "target_time_s":        "target_time_s",
    "stim_index":           "stim_index",
    "scheduling_error_ms":  "scheduling_error_ms",
    "condition":            "condition",
    "block_index":          "block_index",
    "finger":               "finger",
    "is_omission":          "is_omission",
}

# Mapping nouveau format → canonique
_NEW_TO_CANONICAL = {
    "onset_actual_s":       "time_s",
    "onset_planned_s":      "target_time_s",
    "stim_index_in_block":  "stim_index",
}


# ═════════════════════════════════════════════════════════════════════════════
# FORMAT DETECTION & NORMALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def _detect_format(columns: set) -> str:
    """
    Détecte si le CSV est au format ancien ou nouveau (timeline).

    Returns:
        'new' si colonnes timeline détectées
        'old' si colonnes legacy détectées
        Raises ValueError sinon
    """
    new_markers = {"onset_actual_s", "onset_planned_s", "action"}
    old_markers = {"time_s", "target_time_s", "stim_index"}

    if new_markers.issubset(columns):
        return "new"
    elif old_markers.issubset(columns):
        return "old"
    else:
        missing_new = new_markers - columns
        missing_old = old_markers - columns
        raise ValueError(
            f"Format non reconnu.\n"
            f"  Colonnes présentes : {sorted(columns)}\n"
            f"  Manquant (nouveau) : {missing_new}\n"
            f"  Manquant (ancien)  : {missing_old}"
        )


def _normalize_to_canonical(df: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """
    Renomme les colonnes vers le format canonique interne.
    Pour le nouveau format, filtre uniquement les événements de stimulation.
    """
    if fmt == "new":
        # ── Filtrer uniquement les stim events ──
        if "action" in df.columns:
            df = df[
                df["action"].isin(["stim_deliver", "stim_omit"])
            ].copy()

        # ── Renommer ──
        rename_map = {}
        for new_col, canon_col in _NEW_TO_CANONICAL.items():
            if new_col in df.columns and canon_col not in df.columns:
                rename_map[new_col] = canon_col

        if rename_map:
            df = df.rename(columns=rename_map)

    # ── Vérification des colonnes canoniques requises ──
    required_canonical = {
        "condition", "block_index", "finger",
        "is_omission", "time_s", "target_time_s", "scheduling_error_ms",
    }
    # stim_index peut être absent dans certains cas anciens
    present = set(df.columns)
    missing = required_canonical - present
    if missing:
        raise ValueError(f"Colonnes canoniques manquantes après normalisation : {missing}")

    # ── Assurer que stim_index existe ──
    if "stim_index" not in df.columns:
        # Reconstruire à partir de l'ordre dans chaque bloc
        df = df.sort_values(["block_index", "time_s"]).reset_index(drop=True)
        df["stim_index"] = df.groupby("block_index").cumcount()

    # ── Conversion types ──
    for col in ["time_s", "target_time_s", "scheduling_error_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["block_index"] = pd.to_numeric(df["block_index"], errors="coerce")
    df["stim_index"] = pd.to_numeric(df["stim_index"], errors="coerce")

    # is_omission peut être True/False string ou bool
    if df["is_omission"].dtype == object:
        df["is_omission"] = df["is_omission"].map(
            {"True": True, "False": False, "1": True, "0": False,
             "true": True, "false": False, True: True, False: False}
        ).fillna(False)
    df["is_omission"] = df["is_omission"].astype(bool)

    return df.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# LOAD
# ═════════════════════════════════════════════════════════════════════════════

def _load_and_validate(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier non trouvé : {csv_path}")

    df_raw = pd.read_csv(csv_path)
    columns = set(df_raw.columns)

    fmt = _detect_format(columns)

    # Garder les métadonnées du run pour l'affichage
    run_type = "unknown"
    run_number = 0
    if "run_type" in df_raw.columns:
        run_type = df_raw["run_type"].iloc[0]
    if "run_number" in df_raw.columns:
        run_number = df_raw["run_number"].iloc[0]

    df = _normalize_to_canonical(df_raw, fmt)

    # Réinjecter les métadonnées si perdues au filtrage
    if "run_type" not in df.columns:
        df["run_type"] = run_type
    if "run_number" not in df.columns:
        df["run_number"] = run_number

    return df, fmt


# ═════════════════════════════════════════════════════════════════════════════
# COMPUTED COLUMNS — ISI STRICTEMENT INTRA-BLOC
# ═════════════════════════════════════════════════════════════════════════════

def _add_computed_columns(df: pd.DataFrame):
    df = df.sort_values(["block_index", "stim_index"]).reset_index(drop=True)

    # ── ISI intra-bloc uniquement ──
    df["isi_ms"] = np.nan
    for b_idx, block in df.groupby("block_index"):
        block = block.sort_values("stim_index")
        if len(block) < 2:
            continue
        times = block["time_s"].values
        isis = np.diff(times) * 1000.0
        df.loc[block.index[1:], "isi_ms"] = isis

    # ── Durée bloc ON ──
    block_stats = []
    for b_idx, block in df.groupby("block_index"):
        if block.empty:
            continue
        block_stats.append({
            "block_index": b_idx,
            "condition": block["condition"].iloc[0],
            "n_stims": len(block),
            "t_first": block["time_s"].min(),
            "t_last": block["time_s"].max(),
            "duration_stim_span_s": block["time_s"].max() - block["time_s"].min(),
        })

    df_blocks = pd.DataFrame(block_stats)

    df = df.merge(
        df_blocks[["block_index", "duration_stim_span_s"]],
        on="block_index",
        how="left",
    )

    return df, df_blocks


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION RULES
# ═════════════════════════════════════════════════════════════════════════════

def _run_checks(df: pd.DataFrame, df_blocks: pd.DataFrame) -> list:
    msgs = []

    # ── 1. 20 stims par bloc ──
    counts = df.groupby("block_index").size()
    all_20 = (counts == 20).all()
    msgs.append(
        f"{'PASS' if all_20 else 'FAIL'} | "
        f"20 stim/bloc : {counts.value_counts().to_dict()}"
    )

    # ── 2. 5 stim/doigt/bloc ──
    finger_counts = (
        df.groupby(["block_index", "finger"]).size().unstack(fill_value=0)
    )
    all_5 = (finger_counts == 5).all().all()
    msgs.append(
        f"{'PASS' if all_5 else 'FAIL'} | "
        f"5 stim/doigt/bloc"
    )

    # ── 3. FP/TP séquence prédictible ──
    for cond in ["FP", "TP"]:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        expected = (FINGER_ORDER * 5)[:20]
        ok_blocks = 0
        total_blocks = 0
        for b_idx in sub["block_index"].unique():
            seq = (
                sub[sub["block_index"] == b_idx]
                .sort_values("stim_index")["finger"]
                .tolist()
            )
            total_blocks += 1
            if seq == expected:
                ok_blocks += 1
        msgs.append(
            f"{'PASS' if ok_blocks == total_blocks else 'FAIL'} | "
            f"{cond} séquence prédictible : {ok_blocks}/{total_blocks} blocs"
        )

    # ── 4. FR/TR/mapping no-repeat ──
    for cond in ["FR", "TR", "mapping"]:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        violations = 0
        total_blocks = 0
        for b_idx in sub["block_index"].unique():
            seq = (
                sub[sub["block_index"] == b_idx]
                .sort_values("stim_index")["finger"]
                .tolist()
            )
            total_blocks += 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i - 1]:
                    violations += 1
        msgs.append(
            f"{'PASS' if violations == 0 else 'FAIL'} | "
            f"{cond} no-repeat : {violations} violations / {total_blocks} blocs"
        )

    # ── 5. Omissions ──
    for cond in ["FP", "FR", "mapping"]:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        n_omit = int(sub["is_omission"].sum())
        msgs.append(
            f"{'PASS' if n_omit == 0 else 'FAIL'} | "
            f"{cond} omissions = {n_omit} (attendu 0)"
        )

    for cond in ["TP", "TR"]:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        omit_per_block = sub.groupby("block_index")["is_omission"].sum()
        all_5_omit = (omit_per_block == 5).all()
        omit_fingers = sub[sub["is_omission"] == True]["finger"].unique()
        only_d4 = set(omit_fingers) == {"D4"}
        msgs.append(
            f"{'PASS' if all_5_omit and only_d4 else 'FAIL'} | "
            f"{cond} omissions : {omit_per_block.unique()} par bloc, "
            f"doigts omis = {list(omit_fingers)}"
        )

    # ── 6. Timing ──
    err = df["scheduling_error_ms"].dropna()
    if not err.empty:
        max_err = err.abs().max()
        pct_over_2ms = (err.abs() > 2.0).mean() * 100
        msgs.append(
            f"{'PASS' if max_err < 5.0 else 'WARN'} | "
            f"Timing : mean={err.mean():.3f}ms, max={max_err:.3f}ms, "
            f">2ms={pct_over_2ms:.1f}%"
        )

    # ── 7. ISI intra-bloc ──
    isi = df["isi_ms"].dropna()
    if not isi.empty:
        isi_ok = isi.between(490, 510).mean() * 100
        msgs.append(
            f"{'PASS' if isi_ok > 95 else 'WARN'} | "
            f"ISI intra-bloc : μ={isi.mean():.1f}ms, σ={isi.std():.1f}ms, "
            f"{isi_ok:.0f}% dans [490-510ms]"
        )

    # ── 8. Condition balance (prediction only) ──
    pred = df[df["condition"].isin(["FP", "TP", "FR", "TR"])]
    if not pred.empty:
        cond_counts = pred.groupby("condition")["block_index"].nunique()
        balanced = (cond_counts == 5).all() if len(cond_counts) == 4 else False
        msgs.append(
            f"{'PASS' if balanced else 'FAIL'} | "
            f"Condition balance : {cond_counts.to_dict()}"
        )

    # ── 9. Durée blocs ──
    spans = df_blocks["duration_stim_span_s"]
    if not spans.empty:
        span_ok = spans.between(9.3, 9.7).all()
        msgs.append(
            f"{'PASS' if span_ok else 'WARN'} | "
            f"Durée blocs (span stim) : μ={spans.mean():.3f}s, "
            f"range=[{spans.min():.3f}, {spans.max():.3f}]"
        )

    return msgs


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

def _plot_dashboard(df, df_blocks, csv_path, checks, qc_dir):
    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"QC Stimulation Électrique — {os.path.basename(csv_path)}",
        fontsize=15, fontweight="bold",
    )

    conds_present = [
        c for c in ["FP", "TP", "FR", "TR", "mapping"]
        if c in df["condition"].unique()
    ]

    # ════════════════════════════════════════════════════════════════
    # 1. SCHEDULING ERROR
    # ════════════════════════════════════════════════════════════════
    ax = axes[0, 0]
    err = df["scheduling_error_ms"].dropna()
    if not err.empty:
        sns.histplot(err, kde=True, ax=ax, color="steelblue", bins=40)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.axvline(err.mean(), color="red", ls="-", lw=1,
                   label=f"μ = {err.mean():.3f} ms")
        ax.axvspan(-2, 2, alpha=0.1, color="green", label="±2 ms")
        ax.set_title(
            f"1. Précision Temporelle\n"
            f"μ={err.mean():.3f} ms | σ={err.std():.3f} ms | "
            f"max |err|={err.abs().max():.3f} ms"
        )
    else:
        ax.text(0.5, 0.5, "Pas de données", transform=ax.transAxes, ha="center")
        ax.set_title("1. Précision Temporelle")
    ax.set_xlabel("Scheduling Error (ms)")
    ax.legend(fontsize=8)

    # ════════════════════════════════════════════════════════════════
    # 2. ISI INTRA-BLOC
    # ════════════════════════════════════════════════════════════════
    ax = axes[0, 1]
    isi = df["isi_ms"].dropna()
    if not isi.empty:
        isi_clean = isi[isi.between(400, 600)]
        if not isi_clean.empty:
            sns.histplot(isi_clean, kde=True, ax=ax, color="purple", bins=50)
            ax.axvline(500, color="black", ls="--", lw=1.5, label="Cible 500 ms")
            ax.axvline(isi_clean.mean(), color="red", ls="-", lw=1,
                       label=f"μ = {isi_clean.mean():.2f} ms")
            ax.axvspan(498, 502, alpha=0.15, color="green", label="±2 ms")
            pct_in = len(isi_clean) / len(isi) * 100
            ax.set_title(
                f"2. ISI Intra-Bloc\n"
                f"μ={isi_clean.mean():.2f} ms | σ={isi_clean.std():.2f} ms | "
                f"{pct_in:.0f}% dans [400-600]"
            )
        else:
            ax.text(0.5, 0.5, "ISI hors range",
                    transform=ax.transAxes, ha="center")
            ax.set_title("2. ISI Intra-Bloc")
        ax.set_xlabel("Inter-Stimulus Interval (ms)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Pas de données ISI",
                transform=ax.transAxes, ha="center")
        ax.set_title("2. ISI Intra-Bloc")

    # ════════════════════════════════════════════════════════════════
    # 3. FINGER DISTRIBUTION BY CONDITION
    # ════════════════════════════════════════════════════════════════
    ax = axes[0, 2]
    finger_cond = (
        df.groupby(["condition", "finger"])
        .size()
        .reset_index(name="count")
    )
    finger_cond["finger"] = pd.Categorical(
        finger_cond["finger"], categories=FINGER_ORDER, ordered=True
    )
    # Palette restreinte aux conditions présentes
    palette = {c: CONDITION_COLORS[c] for c in conds_present
               if c in CONDITION_COLORS}
    sns.barplot(
        data=finger_cond, x="finger", y="count",
        hue="condition", palette=palette, ax=ax,
    )
    ax.set_title("3. Distribution Doigts × Condition")
    ax.set_ylabel("Nombre total de stims")
    ax.legend(fontsize=7, title="Condition")

    # ════════════════════════════════════════════════════════════════
    # 4. OMISSIONS BY CONDITION
    # ════════════════════════════════════════════════════════════════
    ax = axes[1, 0]
    omit_data = (
        df[df["is_omission"] == True]
        .groupby(["condition", "finger"])
        .size()
        .reset_index(name="n_omissions")
    )
    if not omit_data.empty:
        omit_data["finger"] = pd.Categorical(
            omit_data["finger"], categories=FINGER_ORDER, ordered=True
        )
        palette_omit = {c: CONDITION_COLORS[c] for c in omit_data["condition"].unique()
                        if c in CONDITION_COLORS}
        sns.barplot(
            data=omit_data, x="finger", y="n_omissions",
            hue="condition", palette=palette_omit, ax=ax,
        )
        ax.set_title("4. Omissions par Condition × Doigt")
        ax.set_ylabel("Nombre d'omissions")
    else:
        ax.text(0.5, 0.5, "Aucune omission\n(normal si mapping/FP/FR)",
                transform=ax.transAxes, ha="center", fontsize=12)
        ax.set_title("4. Omissions par Condition × Doigt")

    # ════════════════════════════════════════════════════════════════
    # 5. BLOCK DURATIONS (stim span)
    # ════════════════════════════════════════════════════════════════
    ax = axes[1, 1]
    if not df_blocks.empty:
        colors_list = [
            CONDITION_COLORS.get(c, "gray") for c in df_blocks["condition"]
        ]
        ax.bar(
            df_blocks["block_index"],
            df_blocks["duration_stim_span_s"],
            color=colors_list, edgecolor="black", linewidth=0.5,
        )
        ax.axhline(9.5, color="red", ls="--", lw=1.5, label="Cible 9.5 s")
        ax.set_ylim(
            max(0, df_blocks["duration_stim_span_s"].min() - 0.5),
            df_blocks["duration_stim_span_s"].max() + 0.5,
        )
        ax.set_title(
            f"5. Durée Blocs ON (1ère→dernière stim)\n"
            f"μ={df_blocks['duration_stim_span_s'].mean():.3f} s | "
            f"cible=9.500 s"
        )
        ax.set_xlabel("Bloc #")
        ax.set_ylabel("Durée (s)")
        patches = [
            mpatches.Patch(color=CONDITION_COLORS[c], label=c)
            for c in conds_present
        ]
        ax.legend(handles=patches, fontsize=7)
    else:
        ax.text(0.5, 0.5, "Pas de données blocs",
                transform=ax.transAxes, ha="center")
        ax.set_title("5. Durée Blocs ON")

    # ════════════════════════════════════════════════════════════════
    # 6. SCHEDULING ERROR OVER TIME (dérive)
    # ════════════════════════════════════════════════════════════════
    ax = axes[1, 2]
    for cond in conds_present:
        sub = df[df["condition"] == cond].sort_values("time_s")
        color = CONDITION_COLORS.get(cond, "gray")
        ax.scatter(
            sub["time_s"], sub["scheduling_error_ms"],
            c=color, s=8, alpha=0.6, label=cond,
        )

    ax.axhline(0, color="black", ls="--", lw=1)
    ax.axhline(2, color="red", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(-2, color="red", ls=":", lw=0.8, alpha=0.5)

    # Rolling mean pour visualiser la dérive
    df_sorted = df.sort_values("time_s")
    err_sorted = df_sorted["scheduling_error_ms"].dropna()
    if len(err_sorted) > 20:
        rolling = err_sorted.rolling(20, center=True).mean()
        ax.plot(
            df_sorted.loc[rolling.index, "time_s"],
            rolling,
            color="black", lw=1.5, alpha=0.7, label="Rolling mean (20)",
        )

    ax.set_title("6. Dérive Temporelle\nScheduling error au fil du run")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Scheduling Error (ms)")
    ax.legend(fontsize=7, markerscale=2)

    # ════════════════════════════════════════════════════════════════
    # CHECK BOX
    # ════════════════════════════════════════════════════════════════
    check_text = "\n".join(checks)
    fig.text(
        0.5, 0.01, check_text,
        fontsize=7, fontfamily="monospace", ha="center", va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    png_name = os.path.basename(csv_path).replace(".csv", "_QC.png")
    save_path = os.path.join(qc_dir, png_name)
    plt.savefig(save_path, dpi=120)
    plt.close()

    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY
# ═════════════════════════════════════════════════════════════════════════════

def qc_connectelec(csv_path: str):
    print(f"{'═' * 60}")
    print(f"  QC Stimulation Électrique")
    print(f"  {os.path.basename(csv_path)}")
    print(f"{'═' * 60}")

    df, fmt = _load_and_validate(csv_path)
    print(f"  ✓ {len(df)} stimulations chargées (format: {fmt})")

    run_type = df["run_type"].iloc[0] if "run_type" in df.columns else "unknown"
    run_num = df["run_number"].iloc[0] if "run_number" in df.columns else 0
    print(f"  ✓ Type: {run_type} | Run: {run_num}")

    df, df_blocks = _add_computed_columns(df)

    checks = _run_checks(df, df_blocks)
    for msg in checks:
        status = msg.split("|")[0].strip()
        icon = "✓" if status == "PASS" else ("⚠" if status == "WARN" else "✗")
        print(f"  {icon} {msg}")

    csv_dir = os.path.dirname(csv_path) or "."
    qc_dir = os.path.join(csv_dir, "qc")
    os.makedirs(qc_dir, exist_ok=True)

    save_path = _plot_dashboard(df, df_blocks, csv_path, checks, qc_dir)
    print(f"\n  ✓ Dashboard sauvegardé : {save_path}")

    # ── Summary CSV ──
    isi = df["isi_ms"].dropna()
    err = df["scheduling_error_ms"].dropna()
    summary = {
        "csv_file":               os.path.basename(csv_path),
        "data_format":            fmt,
        "run_type":               run_type,
        "run_number":             run_num,
        "n_blocks":               df["block_index"].nunique(),
        "n_stims_total":          len(df),
        "n_delivered":            int((~df["is_omission"]).sum()),
        "n_omissions":            int(df["is_omission"].sum()),
        "sched_error_mean_ms":    round(err.mean(), 4) if not err.empty else None,
        "sched_error_std_ms":     round(err.std(), 4) if not err.empty else None,
        "sched_error_max_abs_ms": round(err.abs().max(), 4) if not err.empty else None,
        "isi_intrablock_mean_ms": round(isi.mean(), 2) if not isi.empty else None,
        "isi_intrablock_std_ms":  round(isi.std(), 2) if not isi.empty else None,
        "block_span_mean_s":      round(
            df_blocks["duration_stim_span_s"].mean(), 4
        ) if not df_blocks.empty else None,
        "pct_timing_over_1ms":    round(
            (err.abs() > 1).mean() * 100, 1
        ) if not err.empty else None,
        "pct_timing_over_2ms":    round(
            (err.abs() > 2).mean() * 100, 1
        ) if not err.empty else None,
    }
    summary_path = os.path.join(
        qc_dir, os.path.basename(csv_path).replace(".csv", "_summary.csv")
    )
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"  ✓ Summary CSV : {summary_path}")

    # ── Verdict ──
    n_fail = sum(1 for m in checks if m.startswith("FAIL"))
    n_warn = sum(1 for m in checks if m.startswith("WARN"))
    if n_fail == 0 and n_warn == 0:
        print(f"\n  ✅ VERDICT : ALL PASS")
    elif n_fail == 0:
        print(f"\n  ⚠️  VERDICT : {n_warn} WARNING(s)")
    else:
        print(f"\n  ❌ VERDICT : {n_fail} FAIL(s), {n_warn} WARNING(s)")

    print(f"{'═' * 60}\n")

