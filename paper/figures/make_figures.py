"""Figures for the paper, regenerated from the raw result files.

Fig. 1  sigma_distance.pdf : same-token-pair sigma vs positional distance, post-RoPE and
        RoPE-removed (key j re-rotated onto position i), per model; break-even band 0.25-0.32.
        Right panel: distribution of per-(layer, head) median oracle sigma.
        Inputs: runs/local_repro/perhead_derot_{llama,qwen3}.json and, when present,
        runs/modal/review/perhead_derot_{mistral,qwen25}.json, runs/local_repro/sigma_nonrope_gpt2medium.json.
Fig. 2  frontier.pdf : LongBench macro mean (dev split) vs NIAH accuracy at 5 % and 2 % budgets,
        for the selectors and merging operators that have BOTH measurements (the v8 channel
        ablations have no needle run and are therefore absent); each merge is joined by an arrow
        to its own eviction base, and the one operator that moves its base to the right is a star.
        Inputs: runs/modal/{dev8k,dev8k_v8,dev8k_bias,dev8k_nobias,dev8k_c02}/persample_*.json,
        runs/modal/niah8k*/*.csv.

    python paper/figures/make_figures.py
"""
import csv
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
BE = (0.25, 0.32)

# Figures are drawn at the ICLR text width (5.5 in) and included at \linewidth, so the point
# sizes below are the sizes on the page: nothing smaller than 6.5 pt.
TEXT_W = 5.5
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42})


def load_json(p):
    p = os.path.join(ROOT, p)
    return json.load(open(p)) if os.path.exists(p) else None


# ----------------------------------------------------------------------------- Fig. 1
def fig_sigma_distance():
    models = [
        ("Llama-3.1-8B", "runs/local_repro/perhead_derot_llama.json"),
        ("Qwen3-1.7B", "runs/local_repro/perhead_derot_qwen3.json"),
        ("Mistral-7B", "runs/modal/review/perhead_derot_mistral.json"),
        ("Qwen2.5-7B", "runs/modal/review/perhead_derot_qwen25.json"),
    ]
    gpt2 = load_json("runs/local_repro/sigma_nonrope_gpt2medium.json")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXT_W, 2.6), gridspec_kw={"width_ratios": [1.45, 1]})
    xs_lab = ["1", "2", "3-4", "5-8", "9-64", "65-512", "513+"]
    keys = ["1-1", "2-2", "3-4", "5-8", "9-64", "65-512", "513-1000000"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    any_model = False
    for ci, (name, path) in enumerate(models):
        d = load_json(path)
        if not d:
            continue
        any_model = True
        s = d["same_token"]
        post = [s[k]["sigma_post_median"] for k in keys if k in s]
        der = [s[k]["sigma_derot_median"] for k in keys if k in s]
        x = list(range(len(post)))
        ax.plot(x, post, "-o", color=colors[ci], ms=3)
        ax.plot(x, der, "--s", color=colors[ci], ms=3, alpha=0.8)
        meds = [h["median"] for h in d["heads"]]
        ax2.hist(meds, bins=28, range=(0, 3.5), histtype="step", color=colors[ci], label=name, density=True)
    if gpt2:
        s = gpt2["same_token_summary"]
        g = [s[k]["sigma_median"] for k in keys if k in s]
        ax.plot(list(range(len(g))), g, "-^", color="k", ms=3)
        ax2.hist([h for h in gpt2.get("head_medians", [])] or [], bins=28, range=(0, 3.5), histtype="step",
                 color="k", label="GPT-2 medium", density=True)
    ax.axhspan(BE[0], BE[1], color="red", alpha=0.15, lw=0)
    ax.text(0.05, BE[1] + 0.04, "value-only break-even", color="red", fontsize=7)
    ax.set_xticks(range(len(xs_lab)))
    ax.set_xticklabels(xs_lab)
    ax.set_xlabel("distance between two occurrences of a token")
    ax.set_ylabel("median $\\sigma$ (key gap)")
    ax.set_ylim(0, 2.5)
    # line style carries the RoPE condition; colour carries the model (legend on the right panel)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="0.3", ls="-", marker="o", ms=3, label="post-RoPE"),
                       Line2D([], [], color="0.3", ls="--", marker="s", ms=3, label="RoPE removed")],
              loc="upper left", frameon=False)
    ax2.axvspan(BE[0], BE[1], color="red", alpha=0.15, lw=0)
    ax2.set_xlim(0, 3.5)          # six GPT-2 head rows at 5.1-5.4 lie off the axis (caption says so)
    ax2.set_xlabel("per-head median oracle $\\sigma$")
    ax2.set_ylabel("density over (layer, head)")
    # the model legend sits above both panels, clear of the histogram bars
    h, l = ax2.get_legend_handles_labels()
    h = [Line2D([], [], color=x.get_edgecolor(), lw=1.4) for x in h]
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "sigma_distance.pdf"))
    print("wrote sigma_distance.pdf", "(models found)" if any_model else "(NO MODEL DATA)")


# ----------------------------------------------------------------------------- Fig. 2
NAMES = {
    "snapkv;observation_window=32;per_head=false;pool_kernel=7": ("SnapKV (pooled)", "s"),
    "snapkv;observation_window=32;per_head=false": ("SnapKV (unpooled)", "s"),
    "pyramidkv;observation_window=32;per_head=false": ("PyramidKV", "D"),
    "otkv7;auto_regime=false": ("VWS", "o"),
    "otkv7": ("VWS + regime detector", "o"),
    "h2o;per_head=false": ("H2O", "^"),
    "h2o": ("H2O (per-head)", "^"),
    "streamingllm": ("StreamingLLM", "v"),
    "cam": ("CaM (merge)", "x"),
    "otkv7;consolidate=true": ("our consolidation (keys averaged)", "x"),
    "otkv7;consolidate=true;consolidate_gate=cos": ("compensated (cos gate)", "x"),
    "kvmerger;per_head=false": ("KVMerger on H2O", "x"),
    "kvmerger;observation_window=32;per_head=false;pool_kernel=7": ("KVMerger on SnapKV", "x"),
    "kvmerger;observation_window=32;per_head=false;pool_kernel=7;merge_threshold=0.5": ("KVMerger on SnapKV ($\\delta$=0.5)", "x"),
    "otkv8;observation_window=32;per_head=false;pool_kernel=7": ("keys fixed, mass+mixture", "*"),
    "otkv8;observation_window=32;per_head=false;pool_kernel=7;force_theta=0": ("\\quad mass only", "x"),
    "otkv8;observation_window=32;per_head=false;pool_kernel=7;force_c=1": ("\\quad mixture only", "x"),
}
BASE_OF = {  # merge operator -> its eviction base
    "cam": "h2o;per_head=false",
    "otkv7;consolidate=true": "otkv7;auto_regime=false",
    "otkv7;consolidate=true;consolidate_gate=cos": "otkv7;auto_regime=false",
    "kvmerger;per_head=false": "h2o;per_head=false",
    "kvmerger;observation_window=32;per_head=false;pool_kernel=7": "snapkv;observation_window=32;per_head=false;pool_kernel=7",
    "kvmerger;observation_window=32;per_head=false;pool_kernel=7;merge_threshold=0.5": "snapkv;observation_window=32;per_head=false;pool_kernel=7",
    "otkv8;observation_window=32;per_head=false;pool_kernel=7": "snapkv;observation_window=32;per_head=false;pool_kernel=7",
    "otkv8;observation_window=32;per_head=false;pool_kernel=7;force_theta=0": "snapkv;observation_window=32;per_head=false;pool_kernel=7",
    "otkv8;observation_window=32;per_head=false;pool_kernel=7;force_c=1": "snapkv;observation_window=32;per_head=false;pool_kernel=7",
}


def longbench_means(d):
    out = {}
    for p in glob.glob(os.path.join(ROOT, d, "persample_*.json")):
        j = json.load(open(p))
        if j.get("task") == "niah":
            continue
        out.setdefault(j["method"], {})[j["task"]] = j["scores"]
    means = {}
    for m, tasks in out.items():
        if len(tasks) < 8:
            continue
        vals = [x for t in sorted(tasks) for x in tasks[t]]
        means[m] = 100 * st.mean(vals)
    return means


def niah_acc(patterns):
    """The NIAH export is a two-line header ('Task,niah' / 'Model,overall_accuracy')
    followed by 'method,accuracy' rows."""
    acc = {}
    for pattern in patterns:
        for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
            with open(p) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 2 or parts[0] in ("Task", "Model"):
                        continue
                    # two-column NIAH export, or a combined export whose last
                    # column is overall_accuracy
                    try:
                        acc.setdefault(parts[0], float(parts[-1]))
                    except ValueError:
                        pass
    return acc


# marker, colour, filled: selectors blue, merging operators warm colours, the one merge that
# beats its base green; each method keeps one style in both panels
STYLE = {
    "SnapKV (pooled)": ("s", "tab:blue", True),
    "SnapKV (unpooled)": ("s", "tab:blue", False),
    "PyramidKV": ("D", "tab:cyan", True),
    "VWS": ("o", "navy", True),
    "VWS + regime detector": ("o", "navy", False),
    "H2O": ("^", "slategray", True),
    "H2O (per-head)": ("^", "slategray", False),
    "StreamingLLM": ("v", "slategray", True),
    "CaM (merge)": ("x", "tab:brown", True),
    "our consolidation (keys averaged)": ("X", "tab:red", True),
    "compensated (cos gate)": ("P", "tab:red", True),
    "KVMerger on H2O": ("x", "tab:purple", True),
    "KVMerger on SnapKV": ("x", "tab:orange", True),
    "KVMerger on SnapKV ($\\delta$=0.5)": ("+", "tab:orange", True),
    "keys fixed, mass+mixture": ("*", "tab:green", True),
}


def _frontier_points(dset, npat):
    lb = {}
    for one in dset:
        lb.update(longbench_means(one))
    ni = niah_acc(npat)
    return {m: (lb[m], ni[m]) for m in NAMES if m in lb and m in ni}


def _draw(ax, pts, handles):
    for m, (x, y) in pts.items():
        label = NAMES[m][0]
        mk, col, filled = STYLE[label]
        size = 110 if mk == "*" else 36
        kw = dict(color=col) if filled or mk in "x+" else dict(facecolors="none", edgecolors=col)
        handles[label] = ax.scatter(x, y, marker=mk, s=size, zorder=3, linewidths=1.1, clip_on=False, **kw)
    for merge, base in BASE_OF.items():
        if merge in pts and base in pts:
            col = STYLE[NAMES[merge][0]][1]
            ax.annotate("", xy=pts[merge], xytext=pts[base],
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.9, alpha=0.85,
                                        shrinkA=0, shrinkB=0, mutation_scale=8))
    ax.grid(alpha=0.3)


def fig_frontier():
    p5 = _frontier_points(
        ["runs/modal/dev8k", "runs/modal/dev8k_v8", "runs/modal/dev8k_bias", "runs/modal/dev8k_nobias"],
        ["runs/modal/niah8k/niah8k_c05*.csv", "runs/modal/niah8k/niah8k_abl*.csv",
         "runs/modal/niah8k/niah8k_cons*.csv", "runs/modal/niah8k_kvm/*.csv", "runs/modal/niah8k_v8/*.csv"])
    p2 = _frontier_points(["runs/modal/dev8k_c02"], ["runs/modal/niah8k/niah8k_c02*.csv"])
    # the 5 % panel is split: H2O and KVMerger-on-H2O sit ~3 points left of the selector cluster
    far = {m: v for m, v in p5.items() if v[0] < 23.0}
    near = {m: v for m, v in p5.items() if v[0] >= 23.0}
    fig, (axL, axR, ax2) = plt.subplots(1, 3, figsize=(TEXT_W, 3.4), sharey=True,
                                        gridspec_kw={"width_ratios": [0.55, 1.45, 1.1], "wspace": 0.08})
    handles = {}
    _draw(axL, p5, handles)
    _draw(axR, p5, handles)
    _draw(ax2, p2, handles)
    xs = [v[0] for v in far.values()]
    axL.set_xlim(min(xs) - 0.35, max(xs) + 0.35)
    xs = [v[0] for v in near.values()]
    axR.set_xlim(min(xs) - 0.2, max(xs) + 0.2)
    xs = [v[0] for v in p2.values()]
    ax2.set_xlim(min(xs) - 0.12, max(xs) + 0.12)
    for a in (axL, axR, ax2):
        a.set_ylim(0.34, 1.05)
    # break marks between the two halves of the 5 % panel
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axR.tick_params(axis="y", left=False)
    d = 0.015
    for a, xs_ in ((axL, (1 - d, 1 + d)), (axR, (-d * 1.6, d * 1.6))):
        kw = dict(transform=a.transAxes, color="k", clip_on=False, lw=0.8)
        a.plot(xs_, (-d * 2, d * 2), **kw)
        a.plot(xs_, (1 - d * 2, 1 + d * 2), **kw)
    axL.set_ylabel("NIAH accuracy")
    fig.text(0.40, 0.955, "budget 5 %", ha="center", fontsize=8.5)
    ax2.set_title("budget 2 %")
    fig.supxlabel("LongBench macro mean (dev split)", y=0.255, fontsize=8)
    order = [k for k in STYLE if k in handles]
    fig.legend([handles[k] for k in order], order, loc="lower center", ncol=3,
               frameon=False, handletextpad=0.3, columnspacing=1.2, fontsize=6.8,
               bbox_to_anchor=(0.53, 0.0))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.38)
    fig.savefig(os.path.join(OUT, "frontier.pdf"))
    print("wrote frontier.pdf")


if __name__ == "__main__":
    fig_sigma_distance()
    fig_frontier()
