#!/usr/bin/env python3
"""Plot the trajectory CPU/JIT/GPU validation + scaling figures."""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(__file__)


def main():
    val = np.load(os.path.join(HERE, "trajectory_cpu_vs_gpu_results.npz"))
    scaling = np.load(os.path.join(HERE, "trajectory_batch_scaling.npz"))

    fig = plt.figure(figsize=(18, 13))
    gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.30)

    # ── Row 0: endpoint scatter (4 reference trajectories) ──────────
    z0 = val["ref_z0"]; v0 = val["ref_v0"]
    zf_jit = val["ref_zf_jit"]; vf_jit = val["ref_vf_jit"]
    zf_gpu = val["ref_zf_gpu"]; vf_gpu = val["ref_vf_gpu"]
    zf_py = val["ref_zf_py"]; vf_py = val["ref_vf_py"]
    labels = [f"(z₀={z*1e3:.1f}mm, v₀={v:+.1f})" for z, v in zip(z0, v0)]

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(zf_py * 1e3,  vf_py,  s=140, marker="o", label="Python",
               facecolors="none", edgecolors="b", linewidths=2)
    ax.scatter(zf_jit * 1e3, vf_jit, s=80,  marker="s", label="JIT",
               facecolors="none", edgecolors="g", linewidths=2)
    ax.scatter(zf_gpu * 1e3, vf_gpu, s=30,  marker="x", label="CUDA",
               c="r", linewidths=2)
    for lbl, z, v in zip(labels, zf_jit, vf_jit):
        ax.annotate(lbl, (z * 1e3, v), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("final z (mm)")
    ax.set_ylabel("final v (m/s)")
    ax.set_title("Endpoints of 4 reference trajectories\n(circles overlap = perfect agreement)",
                 fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Row 0: per-trajectory residual ──────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    rms_z = val["ref_traj_rms_z"]; rms_v = val["ref_traj_rms_v"]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, rms_z + 1e-30, width, label="z RMS (m)", color="C0")
    ax.bar(x + width/2, rms_v + 1e-30, width, label="v RMS (m/s)", color="C1")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([l.split(" (")[0] for l in
                                          ["trapped", "damping", "mixed", "escaping"]],
                                         rotation=15, ha="right")
    ax.set_ylabel("JIT vs CUDA full-trajectory RMS")
    ax.set_title("Per-trajectory RMS residual: CPU JIT vs CUDA",
                 fontsize=11, fontweight="bold")
    ax.axhline(1e-6, color="gray", ls="--", lw=0.7,
               label="1e-6 m / m/s tolerance")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # ── Row 1: batch 100-particle scatter ────────────────────────────
    zf_jit_b = val["batch_zf_jit"]
    vf_jit_b = val["batch_vf_jit"]
    zf_gpu_b = val["batch_zf_gpu"]
    vf_gpu_b = val["batch_vf_gpu"]
    nact = val["batch_n_actual"]
    trapped = nact == int(val["t_max"] / val["dt"])

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(zf_jit_b[trapped] * 1e3, vf_jit_b[trapped], s=40,
               marker="o", facecolors="none", edgecolors="g",
               label="JIT (trapped)")
    ax.scatter(zf_jit_b[~trapped] * 1e3, vf_jit_b[~trapped], s=60,
               marker="x", c="orange", label="JIT (escaped)")
    ax.scatter(zf_gpu_b * 1e3, vf_gpu_b, s=10, marker=".", c="r",
               label="CUDA (all)")
    ax.set_xlabel("final z (mm)")
    ax.set_ylabel("final v (m/s)")
    ax.set_title(f"100-particle ensemble endpoints "
                 f"(t_max={val['t_max']:.2f} s)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    diff_z = zf_jit_b - zf_gpu_b
    diff_v = vf_jit_b - vf_gpu_b
    ax.hist(np.log10(np.abs(diff_z) + 1e-30), bins=40, alpha=0.6,
            color="C0", label="Δz")
    ax.hist(np.log10(np.abs(diff_v) + 1e-30), bins=40, alpha=0.6,
            color="C1", label="Δv")
    ax.set_xlabel("log₁₀ |JIT − CUDA|")
    ax.set_ylabel("count")
    ax.set_title("Batch endpoint residual distribution",
                 fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Row 2: scaling sweep ─────────────────────────────────────────
    N = scaling["N"]
    t_jit = scaling["t_jit"]
    t_gpu = scaling["t_gpu"]
    speedup = scaling["speedup"]

    ax = fig.add_subplot(gs[2, 0])
    ax.loglog(N, t_jit, "g-o", lw=1.7, label="CPU JIT (sequential)")
    ax.loglog(N, t_gpu, "r-s", lw=1.7, label="CUDA (parallel)")
    ax.set_xlabel("N particles")
    ax.set_ylabel("wall time (s)")
    ax.set_title("Batch trajectory wall-time scaling "
                 f"(30 000 Euler steps each)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    # Annotate cross-over
    crossover_idx = np.argmax(speedup >= 1.0)
    if speedup[crossover_idx] >= 1.0:
        ax.axvline(N[crossover_idx], color="gray", ls="--", lw=0.6)
        ax.text(N[crossover_idx], min(t_gpu) * 0.7,
                f"GPU breaks even ≈ N = {N[crossover_idx]}",
                fontsize=8, ha="center")

    ax = fig.add_subplot(gs[2, 1])
    ax.semilogx(N, speedup, "k-o", lw=2)
    ax.axhline(1.0, color="gray", ls="--", lw=0.7)
    ax.set_xlabel("N particles")
    ax.set_ylabel("CUDA speed-up vs sequential CPU JIT")
    ax.set_title("Speed-up vs ensemble size",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    for n, s in zip(N, speedup):
        ax.annotate(f"{s:.1f}×", (n, s), fontsize=8,
                    xytext=(0, 6), textcoords="offset points",
                    ha="center")

    fig.suptitle("SrOH DC-MOT 1D trajectory:  Python ↔ Numba JIT ↔ CUDA",
                 fontsize=15, fontweight="bold", y=0.995)

    out = os.path.join(HERE, "trajectory_cpu_vs_gpu.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out}")


if __name__ == "__main__":
    main()
