import numpy as np
import os
import sys
import json
import glob
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CONFIG = {
    "n_clusters":         2,     # K for KMeans. 2 lets inliers/outliers separate naturally.
                                 # If you have very few matches (<4), falls back to k=1.
    "min_cluster_size":   2,     # Winning cluster must have at least this many members.
    "max_ratio_spread":   0.15,  # After voting, inlier ratios must be within ±15% of consensus.
                                 # Raise if your floor plans mix very different scales.
}
# ---------------------------------------------------------------------------


def compute_ratios(matches: list[dict]) -> np.ndarray:
    """
    For each matched (line, text) pair compute:
        ratio = px_length / text_value   (pixels per real-world unit)

    Returns a 1-D float array, one ratio per match.
    Pairs where text_value == 0 are excluded (returns mask too).
    """
    ratios = []
    valid  = []
    for m in matches:
        val = m.get("text_value") or 0
        if val <= 0:
            valid.append(False)
            ratios.append(np.nan)
        else:
            valid.append(True)
            ratios.append(m["px_length"] / val)
    return np.array(ratios), np.array(valid, dtype=bool)


def vote_scale(matches: list[dict]) -> dict:
    """
    Run K-Means on the px/unit ratios to find a consensus scale.

    Returns a result dict:
    {
        "scale_px_per_unit":  float  — consensus pixels-per-unit ratio,
        "inliers":            list[dict] — matches that agree with consensus,
        "outliers":           list[dict] — matches rejected as noise,
        "all_ratios":         list[float],
        "cluster_centers":    list[float],
        "winning_cluster":    int,
    }
    """
    ratios, valid_mask = compute_ratios(matches)

    valid_matches = [m for m, v in zip(matches, valid_mask) if v]
    valid_ratios  = ratios[valid_mask]

    if len(valid_ratios) == 0:
        raise ValueError("No valid (px_length / text_value) pairs found. "
                         "Check that scale_matches.json has numeric text_value entries.")

    # ── K-Means ──────────────────────────────────────────────────────────────
    k = min(CONFIG["n_clusters"], len(valid_ratios))   # can't have more clusters than points
    X = valid_ratios.reshape(-1, 1)

    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_.flatten()

    # ── Pick the largest cluster ──────────────────────────────────────────────
    cluster_sizes  = np.bincount(labels)
    winning_label  = int(np.argmax(cluster_sizes))
    winning_center = float(centers[winning_label])

    # ── Spread filter: refine within the winning cluster ──────────────────────
    # Discard any point more than max_ratio_spread away from the winning center
    spread = CONFIG["max_ratio_spread"]
    inlier_mask = (
        (labels == winning_label) &
        (np.abs(valid_ratios - winning_center) / winning_center <= spread)
    )

    inlier_ratios = valid_ratios[inlier_mask]

    if len(inlier_ratios) < CONFIG["min_cluster_size"]:
        # Fallback: use all points in winning cluster without spread filter
        inlier_mask   = labels == winning_label
        inlier_ratios = valid_ratios[inlier_mask]

    consensus_scale = float(np.mean(inlier_ratios))

    inliers  = [m for m, keep in zip(valid_matches, inlier_mask) if keep]
    outliers = [m for m, keep in zip(valid_matches, inlier_mask) if not keep]

    return {
        "scale_px_per_unit": round(consensus_scale, 6),
        "inliers":           inliers,
        "outliers":          outliers,
        "all_ratios":        valid_ratios.tolist(),
        "cluster_centers":   centers.tolist(),
        "winning_cluster":   winning_label,
    }


def apply_scale(matches: list[dict], scale_px_per_unit: float) -> list[dict]:
    """
    For every match (including outliers if you pass them), compute
        real_dimension = px_length / scale_px_per_unit
    Returns a new list of dicts with 'real_dimension' added.
    """
    results = []
    for m in matches:
        real = m["px_length"] / scale_px_per_unit
        results.append({**m, "real_dimension": round(real, 3)})
    return results


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_vote(result: dict, save_path: str = None):
    """
    Strip chart of px/unit ratios — inliers (green) vs outliers (red),
    plus cluster centers (dashed) and consensus (solid blue).
    """
    ratios   = np.array(result["all_ratios"])
    centers  = result["cluster_centers"]
    scale    = result["scale_px_per_unit"]
    n_in     = len(result["inliers"])
    n_out    = len(result["outliers"])

    # Label each ratio as inlier or outlier
    inlier_ratios  = [m["px_length"] / m["text_value"] for m in result["inliers"]]
    outlier_ratios = [m["px_length"] / m["text_value"] for m in result["outliers"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: strip chart ─────────────────────────────────────────────────────
    ax1.scatter(inlier_ratios,  [1]*len(inlier_ratios),  color="limegreen",
                s=80, zorder=3, label=f"Inliers ({n_in})")
    ax1.scatter(outlier_ratios, [1]*len(outlier_ratios), color="red",
                s=80, zorder=3, marker="x", label=f"Outliers ({n_out})")
    for c in centers:
        ax1.axvline(c, color="orange", linestyle="--", linewidth=1.2,
                    label=f"Cluster center {c:.4f}")
    ax1.axvline(scale, color="dodgerblue", linewidth=2,
                label=f"Consensus scale {scale:.4f} px/unit")
    ax1.set_xlabel("px / real-world unit")
    ax1.set_yticks([])
    ax1.set_title("Scale Vote — Ratio Distribution")
    ax1.legend(fontsize=8)

    # ── Right: histogram ──────────────────────────────────────────────────────
    ax2.hist(ratios, bins=max(5, len(ratios)//2), color="steelblue",
             edgecolor="white", alpha=0.8)
    ax2.axvline(scale, color="dodgerblue", linewidth=2,
                label=f"Consensus {scale:.4f}")
    ax2.set_xlabel("px / real-world unit")
    ax2.set_ylabel("count")
    ax2.set_title("Ratio Histogram")
    ax2.legend(fontsize=8)

    plt.suptitle(f"K-Means Scale Voting  |  k={len(centers)}  |  "
                 f"consensus = {scale:.4f} px/unit", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot -> {save_path}")
    else:
        plt.show()
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
    MATCHES_JSON = os.path.join(PROJECT_ROOT, "data/crops_number/output/scale_matches.json")
    OUT_DIR      = os.path.join(PROJECT_ROOT, "data/crops_number/output")

    if not os.path.exists(MATCHES_JSON):
        print(f"scale_matches.json not found at {MATCHES_JSON}")
        print("Run scale_match.py first.")
        sys.exit(1)

    with open(MATCHES_JSON) as f:
        matches = json.load(f)

    print(f"Loaded {len(matches)} matched (line, text) pairs from scale_matches.json\n")

    # ── Vote ──────────────────────────────────────────────────────────────────
    result = vote_scale(matches)
    scale  = result["scale_px_per_unit"]

    print(f"{'='*55}")
    print(f"  K-Means Scale Voting  (k={CONFIG['n_clusters']})")
    print(f"{'='*55}")
    print(f"  Cluster centers (px/unit): "
          f"{[round(c,4) for c in result['cluster_centers']]}")
    print(f"  Winning cluster:           #{result['winning_cluster']}")
    print(f"  Inliers / total:           "
          f"{len(result['inliers'])} / {len(matches)}")
    print(f"\n  ► Consensus scale:  {scale:.6f} px/unit")
    print(f"  ► 1 unit = {1/scale:.3f} px  |  1 px = {scale:.6f} units\n")

    # ── Apply scale to inliers ────────────────────────────────────────────────
    final = apply_scale(result["inliers"], scale)

    print(f"  {'#':<4} {'px_len':>8} {'text_val':>10} {'ratio':>9} {'real_dim':>10}  raw_text")
    print(f"  {'-'*60}")
    for i, m in enumerate(final, 1):
        ratio = m["px_length"] / m["text_value"]
        print(f"  {i:<4} {m['px_length']:>8.1f} {m['text_value']:>10} "
              f"{ratio:>9.4f} {m['real_dimension']:>10.2f}  \"{m['text_raw']}\"")

    if result["outliers"]:
        print(f"\n  Rejected outliers:")
        for m in result["outliers"]:
            ratio = m["px_length"] / m["text_value"]
            print(f"    px={m['px_length']:.1f}  val={m['text_value']}  "
                  f"ratio={ratio:.4f}  text=\"{m['text_raw']}\"")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "scale_px_per_unit": scale,
        "unit_per_px":       round(1.0 / scale, 6),
        "n_inliers":         len(result["inliers"]),
        "n_outliers":        len(result["outliers"]),
        "final_dimensions":  final,
    }
    out_json = os.path.join(OUT_DIR, "scale_result.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    visualize_vote(result, save_path=os.path.join(OUT_DIR, "scale_vote.png"))
