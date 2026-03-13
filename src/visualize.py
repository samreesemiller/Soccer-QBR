# ── 1. Feature importance ──────────────────────────────────────────────────
def plot_feature_importance(model, feature_cols: list, save_path: str = None):
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values()
 
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [ACCENT if v == importances.max() else "#4fc3f7" for v in importances]
    bars = ax.barh(importances.index, importances.values, color=colors, height=0.6)
 
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score")
    ax.axvline(importances.mean(), color="#ff7043", linestyle="--",
               linewidth=1, label=f"Mean ({importances.mean():.3f})")
    ax.legend(fontsize=9)
 
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="#f0f0f0")
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ── 2. Success rate by Area (pitch zone) ──────────────────────────────────
def plot_success_by_zone(combinations: pd.DataFrame, save_path: str = None):
    df = _decode(combinations.copy())
    zone_order = ["Defense", "Mid", "Attack"]
 
    avg = (df.groupby("Area")["predicted_success_prob"]
             .mean()
             .reindex(zone_order))
 
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = [ACCENT if v == avg.max() else "#4fc3f7" for v in avg]
    bars = ax.bar(avg.index, avg.values, color=bar_colors, width=0.5, zorder=3)
    ax.set_ylim(0, 1)
    ax.set_title("Avg. Predicted Pass Success by Pitch Zone", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Predicted Success Probability")
    ax.grid(axis="y", zorder=0, alpha=0.4)
 
    for bar, val in zip(bars, avg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                f"{val:.1%}", ha="center", fontsize=11, fontweight="bold")
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ── 3. Heatmap — Distance × Direction ────────────────────────────────────
def plot_heatmap_distance_direction(combinations: pd.DataFrame, save_path: str = None):
    df = _decode(combinations.copy())
    pivot = df.pivot_table(
        values="predicted_success_prob",
        index="Direction",
        columns="Distance",
        aggfunc="mean"
    )
    direction_order = ["Forward", "Neutral", "Back"]
    pivot = pivot.reindex([d for d in direction_order if d in pivot.index])
 
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        pivot, ax=ax, cmap=HEAT_PALETTE, annot=True, fmt=".0%",
        linewidths=0.5, linecolor="#333",
        cbar_kws={"label": "Success Prob."},
        vmin=0, vmax=1
    )
    ax.set_title("Pass Success Probability — Distance × Direction", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Direction")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ── 4. Defenders vs success ───────────────────────────────────────────────
def plot_defenders_impact(combinations: pd.DataFrame, save_path: str = None):
    avg = (combinations.groupby("Min. Def.")["predicted_success_prob"]
                       .mean())
 
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(avg.index, avg.values, color=ACCENT, marker="o",
            linewidth=2.5, markersize=8, zorder=3)
    ax.fill_between(avg.index, avg.values, alpha=0.15, color=ACCENT)
    ax.set_title("Effect of Nearby Defenders on Pass Success", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Min. Defenders Nearby")
    ax.set_ylabel("Avg. Predicted Success Prob.")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.4)
 
    for x, y in zip(avg.index, avg.values):
        ax.text(x, y + 0.02, f"{y:.1%}", ha="center", fontsize=10, fontweight="bold")
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ── 5. Pitch zone map ─────────────────────────────────────────────────────
def plot_pitch_zone_map(combinations: pd.DataFrame, save_path: str = None):
    df = _decode(combinations.copy())
    zone_prob = df.groupby("Area")["predicted_success_prob"].mean().to_dict()
    zones = {"Defense": (0, 0, 40, 80), "Mid": (40, 0, 40, 80), "Attack": (80, 0, 40, 80)}
 
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(PITCH_GREEN)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Pass Completion Probability by Pitch Zone", fontsize=13,
                 fontweight="bold", pad=14)
 
    # Pitch outline & centre line
    for rect_args in [
        dict(xy=(0, 0), width=120, height=80, linewidth=2, edgecolor=PITCH_LINE, facecolor="none"),
    ]:
        ax.add_patch(mpatches.Rectangle(**rect_args))
    ax.axvline(60, color=PITCH_LINE, linewidth=1.5, linestyle="--", alpha=0.6)
    ax.axvline(40, color=PITCH_LINE, linewidth=1.5, linestyle="--", alpha=0.6)
    ax.axvline(80, color=PITCH_LINE, linewidth=1.5, linestyle="--", alpha=0.6)
 
    cmap = plt.get_cmap("RdYlGn")
    for zone, (x, y, w, h) in zones.items():
        prob = zone_prob.get(zone, 0.5)
        color = cmap(prob)
        rect = mpatches.Rectangle((x, y), w, h, linewidth=0,
                                   facecolor=color, alpha=0.65)
        ax.add_patch(rect)
        ax.text(x + w / 2, h / 2 + 4, zone,
                ha="center", va="center", fontsize=13,
                fontweight="bold", color="white",
                bbox=dict(facecolor="black", alpha=0.4, boxstyle="round,pad=0.3"))
        ax.text(x + w / 2, h / 2 - 8, f"{prob:.1%}",
                ha="center", va="center", fontsize=16,
                fontweight="bold", color="white")
 
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Success Probability", color="#f0f0f0")
    cbar.ax.yaxis.set_tick_params(color="#f0f0f0")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#f0f0f0")
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ── 6. All plots at once ──────────────────────────────────────────────────
def plot_all(model, feature_cols: list, combinations: pd.DataFrame, save_dir: str = None):
    """Run all visualizations in one call."""
    def path(name):
        return f"{save_dir}/{name}.png" if save_dir else None
 
    plot_feature_importance(model, feature_cols,          save_path=path("feature_importance"))
    plot_success_by_zone(combinations,                    save_path=path("success_by_zone"))
    plot_heatmap_distance_direction(combinations,         save_path=path("heatmap_distance_direction"))
    plot_defenders_impact(combinations,                   save_path=path("defenders_impact"))
    plot_pitch_zone_map(combinations,                     save_path=path("pitch_zone_map"))