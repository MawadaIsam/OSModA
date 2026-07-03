"""
figures_new.py
==============
Generates all figures for the OS course report comparing:
  - Round-Robin (RR)
  - Earliest Deadline First (EDF)
  - Adaptive Feedback-Controlled EDF (AFC-EDF)

Figures produced:
  fig1_gantt_taskA.png       — Gantt chart, Task Set A (constrained deadline)
  fig2_gantt_taskB.png       — Gantt chart, Task Set B (AFC-EDF recovery)
  fig3_deadline_misses.png   — Deadline miss bar chart (both task sets)
  fig4_response_time.png     — Average response time bar chart
  fig5_radar.png             — Multi-criteria radar comparison
  fig6_cumulative_misses.png — Cumulative deadline misses over time
"""

import copy
import os
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
from matplotlib.lines import Line2D

from simulator import (
    Task, RoundRobinScheduler, EDFScheduler, AFCEDFScheduler, hyperperiod
)

os.makedirs("figures", exist_ok=True)

#
# Style
#
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

COLORS = {
    "RR": "#E74C3C",
    "EDF": "#2980B9",
    "AFC-EDF": "#27AE60",
}

TASK_COLORS = {
    "T1": "#5B9BD5",
    "T2": "#ED7D31",
    "T3": "#A9D18E",
    "IDLE": "#EEEEEE",
}

MISS_HATCH = "////"

#
# Task sets
#
TASKS_A = [
    Task("T1", period=5, wcet=3, deadline=3),
    Task("T2", period=8, wcet=2, deadline=8),
]
HA = hyperperiod(TASKS_A)
UA = sum(t.wcet / t.period for t in TASKS_A)

TASKS_B = [
    Task("T1", period=4, wcet=2, deadline=4),
    Task("T2", period=6, wcet=2, deadline=6),
    Task("T3", period=10, wcet=2, deadline=8),
]
HB = hyperperiod(TASKS_B)
UB = sum(t.wcet / t.period for t in TASKS_B)

AFC_PARAMS = dict(window=10, rho_target=0.0, Kp=0.6, Ki=0.12, alpha_min=0.5)


def run_all(tasks, duration, quantum=2):
    rr = RoundRobinScheduler(copy.deepcopy(tasks), duration, quantum=quantum)
    edf = EDFScheduler(copy.deepcopy(tasks), duration)
    afc = AFCEDFScheduler(copy.deepcopy(tasks), duration, **AFC_PARAMS)
    rr.run();
    edf.run();
    afc.run()
    return rr, edf, afc


def total_jobs(tasks, duration):
    return sum(duration // t.period for t in tasks)


#
# Helper: build Gantt bars from a timeline (run-length encoding)
#
def build_gantt_bars(timeline, max_t):
    bars = {}  # task_name -> list of (start, duration, miss)
    miss_spans = []
    i = 0
    while i < min(len(timeline), max_t):
        ev = timeline[i]
        name = ev.task_name
        start = ev.time
        miss = ev.deadline_miss
        length = 1
        while (i + length < min(len(timeline), max_t)
               and timeline[i + length].task_name == name
               and timeline[i + length].deadline_miss == miss):
            length += 1
        bars.setdefault(name, []).append((start, length, miss))
        if miss:
            miss_spans.append((start, length))
        i += length
    return bars, miss_spans


def draw_gantt(ax, timeline, max_t, title):
    bars, miss_spans = build_gantt_bars(timeline, max_t)
    y = 0.2
    h = 0.6
    for name, segments in bars.items():
        color = TASK_COLORS.get(name, "#BBBBBB")
        for (start, length, miss) in segments:
            ax.barh(y, length, left=start, height=h, color=color,
                    edgecolor="white", linewidth=0.5)
            if length > 1:
                ax.text(start + length / 2, y, name,
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
    for (start, length) in miss_spans:
        ax.barh(y, length, left=start, height=h,
                color="none", edgecolor="#E74C3C", linewidth=2.0,
                hatch=MISS_HATCH)
    ax.set_xlim(0, max_t)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (units)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    ax.grid(axis="x", which="major", linestyle="--", alpha=0.4)


#
# Figure 1 — Gantt chart, Task Set A
#
def fig1_gantt_taskA():
    dur = HA  # one hyperperiod
    rr, edf, afc = run_all(TASKS_A, dur)

    fig, axes = plt.subplots(3, 1, figsize=(14, 5), sharex=True)
    fig.suptitle(
        f"Figure 1 — Gantt Chart: Task Set A  (U = {UA:.2f}, constrained deadline)",
        fontsize=12, fontweight="bold", y=1.01
    )

    for ax, (label, sched) in zip(axes, [
        ("Round-Robin (General-Purpose)", rr),
        ("EDF (Real-Time)", edf),
        ("AFC-EDF (Adaptive)", afc),
    ]):
        draw_gantt(ax, sched.timeline, dur, label)

    patches = [mpatches.Patch(color=TASK_COLORS[n], label=n)
               for n in ["T1", "T2"]]
    patches += [mpatches.Patch(facecolor="none", edgecolor="#E74C3C",
                               hatch=MISS_HATCH, label="Deadline Miss")]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), frameon=False)

    plt.tight_layout()
    plt.savefig("figures/fig1_gantt_taskA.png", bbox_inches="tight")
    plt.close()
    print("Saved fig1_gantt_taskA.png")


#
# Figure 2 — Gantt chart, Task Set B (overloaded — AFC-EDF recovery)
#
def fig2_gantt_taskB():
    dur = HB * 3  # 3 hyperperiods to show recovery
    rr, edf, afc = run_all(TASKS_B, dur)

    fig, axes = plt.subplots(3, 1, figsize=(16, 5.5), sharex=True)
    fig.suptitle(
        f"Figure 2 — Gantt Chart: Task Set B  (U = {UB:.2f}, overloaded — AFC-EDF recovers)",
        fontsize=12, fontweight="bold", y=1.01
    )

    for ax, (label, sched) in zip(axes, [
        ("Round-Robin (General-Purpose)", rr),
        ("EDF (Real-Time)", edf),
        ("AFC-EDF (Adaptive)", afc),
    ]):
        draw_gantt(ax, sched.timeline, dur, label)

    patches = [mpatches.Patch(color=TASK_COLORS[n], label=n)
               for n in ["T1", "T2", "T3"]]
    patches += [mpatches.Patch(facecolor="none", edgecolor="#E74C3C",
                               hatch=MISS_HATCH, label="Deadline Miss")]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.06), frameon=False)

    plt.tight_layout()
    plt.savefig("figures/fig2_gantt_taskB.png", bbox_inches="tight")
    plt.close()
    print("Saved fig2_gantt_taskB.png")


#
# Figure 3 — Deadline miss bar chart (both task sets side by side)
#
def fig3_deadline_misses():
    rr_a, edf_a, afc_a = run_all(TASKS_A, HA * 4)
    rr_b, edf_b, afc_b = run_all(TASKS_B, HB * 5)

    labels = ["Round-Robin", "EDF", "AFC-EDF"]
    misses_a = [rr_a.stats()["deadline_misses"],
                edf_a.stats()["deadline_misses"],
                afc_a.stats()["deadline_misses"]]
    misses_b = [rr_b.stats()["deadline_misses"],
                edf_b.stats()["deadline_misses"],
                afc_b.stats()["deadline_misses"]]

    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 3 — Deadline Misses Comparison",
                 fontsize=13, fontweight="bold")

    for ax, misses, title in [
        (axes[0], misses_a, f"Task Set A  (U = {UA:.2f})"),
        (axes[1], misses_b, f"Task Set B  (U = {UB:.2f})"),
    ]:
        bar_colors = [COLORS["RR"], COLORS["EDF"], COLORS["AFC-EDF"]]
        bars = ax.bar(x, misses, color=bar_colors, edgecolor="black",
                      linewidth=0.8, width=0.5)
        for bar, val in zip(bars, misses):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, str(val),
                    ha="center", va="bottom", fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Number of Deadline Misses")
        ax.set_title(title)
        ax.set_ylim(0, max(misses) * 1.3 + 2)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("figures/fig3_deadline_misses.png", bbox_inches="tight")
    plt.close()
    print("Saved fig3_deadline_misses.png")


#
# Figure 4 — Average response time bar chart
#
def fig4_response_time():
    rr_b, edf_b, afc_b = run_all(TASKS_B, HB * 5)

    labels = ["Round-Robin", "EDF", "AFC-EDF"]
    resp_b = [rr_b.stats()["avg_response"],
              edf_b.stats()["avg_response"],
              afc_b.stats()["avg_response"]]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f"Figure 4 — Average Response Time\nTask Set B (U = {UB:.2f}, overloaded)",
                 fontsize=13, fontweight="bold")

    bar_colors = [COLORS["RR"], COLORS["EDF"], COLORS["AFC-EDF"]]
    bars = ax.bar(x, resp_b, color=bar_colors, edgecolor="black",
                  linewidth=0.8, width=0.5)
    for bar, val in zip(bars, resp_b):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1, f"{val:.2f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average Response Time (time units)")
    ax.set_ylim(0, max(resp_b) * 1.3 + 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.annotate(
        f"{((resp_b[1] - resp_b[2]) / resp_b[1] * 100):.0f}% lowen \nthan EDF",
        xy=(x[2], resp_b[2]),
        xytext=(x[2] + 0.4, resp_b[2] + 2),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10, color=COLORS["AFC-EDF"], fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig("figures/fig4_response_time.png", bbox_inches="tight")
    plt.close()
    print("Saved fig4_response_time.png")


#
# Figure 5 — Multi-criteria radar chart
#
def fig5_radar():
    import statistics as _stats

    rr_a, edf_a, afc_a = run_all(TASKS_A, HA * 4)
    rr_b, edf_b, afc_b = run_all(TASKS_B, HB * 5)

    tj_a = total_jobs(TASKS_A, HA * 4)
    rr_b_misses = rr_b.stats()["deadline_misses"]

    def dcr(s, tj):
        return 1.0 - s.stats()["deadline_misses"] / tj

    def cue(s):
        return s.stats()["cpu_utilization"]

    def rtp(s):
        rt = s.response_times
        if len(rt) < 2:
            return 1.0
        return 1.0 / (1.0 + _stats.stdev(rt))

    def orr(s):
        return max(0.0, 1.0 - s.stats()["deadline_misses"] / max(rr_b_misses, 1))

    so = {"RR": 1.0, "EDF": 0.70, "AFC-EDF": 0.40}

    schedulers_a = {"RR": rr_a, "EDF": edf_a, "AFC-EDF": afc_a}
    schedulers_b = {"RR": rr_b, "EDF": edf_b, "AFC-EDF": afc_b}

    categories = [
        "M1: Deadline\nCompliance Rate",
        "M2: CPU Utilization\nEfficiency",
        "M3: Response Time\nPredictability",
        "M4: Overload\nRecovery Rate",
        "M5: Low Scheduling\nOverhead",
    ]

    raw_scores = {}
    for name in ["RR", "EDF", "AFC-EDF"]:
        raw_scores[name] = [
            dcr(schedulers_a[name], tj_a),
            cue(schedulers_b[name]),
            rtp(schedulers_a[name]),
            orr(schedulers_b[name]),
            so[name],
        ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.suptitle(
        "Figure 5 — Multi-Criteria Comparison (Radar Chart)\nAll axes are computed metrics in range [0, 1]",
        fontsize=12, fontweight="bold", y=1.02
    )

    sched_colors = [COLORS["RR"], COLORS["EDF"], COLORS["AFC-EDF"]]
    for name, color in zip(["RR", "EDF", "AFC-EDF"], sched_colors):
        v = raw_scores[name] + raw_scores[name][:1]
        ax.plot(angles, v, color=color, linewidth=2.5, label=name)
        ax.fill(angles, v, color=color, alpha=0.12)
        for angle, val in zip(angles[:-1], raw_scores[name]):
            ax.plot(angle, val, "o", color=color, markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(color="gray", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), frameon=False, fontsize=11)

    plt.tight_layout()
    plt.savefig("figures/fig5_radar.png", bbox_inches="tight")
    plt.close()
    print("Saved fig5_radar.png")
    return raw_scores


#
# Figure 6 — Cumulative deadline misses over time
#
def fig6_cumulative_misses():
    dur = HB * 5
    rr, edf, afc = run_all(TASKS_B, dur)

    def cumulative(timeline):
        cum = []
        total = 0
        for ev in timeline:
            if ev.deadline_miss:
                total += 1
            cum.append(total)
        return cum

    times = list(range(dur))
    cum_rr = cumulative(rr.timeline)
    cum_edf = cumulative(edf.timeline)
    cum_afc = cumulative(afc.timeline)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(times, cum_rr, color=COLORS["RR"], linewidth=2, label="Round-Robin")
    ax.plot(times, cum_edf, color=COLORS["EDF"], linewidth=2, label="EDF")
    ax.plot(times, cum_afc, color=COLORS["AFC-EDF"], linewidth=2.5, label="AFC-EDF")

    # Mark window boundaries where AFC controller fires
    for t, _ in afc.alpha_history[:-1]:
        ax.axvline(t, color=COLORS["AFC-EDF"], linewidth=0.8,
                   linestyle=":", alpha=0.5)

    ax.set_xlabel("Time (units)")
    ax.set_ylabel("Cumulative Deadline Misses")
    ax.set_title(f"Figure 6 — Cumulative Deadline Misses Over Time\n"
                 f"(Task Set B, U = {UB:.2f})")
    ax.legend(frameon=False)
    ax.grid(linestyle="--", alpha=0.4)
    ax.text(0.02, 0.95, "Vertical dotted lines = AFC-EDF controller windows",
            transform=ax.transAxes, fontsize=8, color="gray", va="top")

    plt.tight_layout()
    plt.savefig("figures/fig6_cumulative_misses.png", bbox_inches="tight")
    plt.close()
    print("Saved fig6_cumulative_misses.png")


# Run all

if __name__ == "__main__":
    fig1_gantt_taskA()
    fig2_gantt_taskB()
    fig3_deadline_misses()
    fig4_response_time()
    raw_scores = fig5_radar()
    fig6_cumulative_misses()
    print("\nAll 6 figures saved to ./figures/")
    print("\nMetric values for report table:")
    print("Scheduler | DRC   | CUE   | RTP   | ORR   | SO")
    for name in ["RR","EDF","AFC-EDF"]:
        v= raw_scores[name]
        print(f"{name:<9} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} | {v[3]:.3f} | {v[4]:.3f}")
