"""Per-task rows of the deployed-merge appendix table (tab:deploy) from the
raw per-sample files. Usage: python paper/figures/make_deploy_table.py"""
import glob, json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAP = "_snapkv_observation_window_32_per_head_false_pool_kernel_7.json"
COLS = [("runs/modal/dev8k_v8", "_otkv8_observation_window_32_per_head_false_pool_kernel_7.json"),
        ("runs/modal/dev8k_bias", "_force_theta_0.json"),
        ("runs/modal/dev8k_nobias", "_force_c_1.json"),
        ("runs/modal/dev8k", "_kvmerger_observation_window_32_per_head_false_pool_kernel_7.json"),
        ("runs/modal/ports_a", "_keepkv.json"),
        ("runs/modal/ports_c", "_selkv_selector_snapkv.json")]
def load(d, s):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, d, "persample_*" + s))):
        J = json.load(open(f)); out[J["task"]] = np.array(J["scores"])
    return out
base = load("runs/modal/dev8k", SNAP)
cols = [load(d, s) for d, s in COLS]
tasks = sorted(base)
SHORT = {"multifieldqa_en": "mfqa", "gov_report": "gov", "multi_news": "mnews",
         "2wikimqa": "2wiki", "hotpotqa": "hotpot", "triviaqa": "trivia"}
for t in tasks:
    cells = [f"{100*(c[t][:50] - base[t][:50]).mean():+.2f}" if c else "---" for c in cols]
    nch = int((cols[0][t][:50] != base[t][:50]).sum()) if cols[0] else 0
    print(f"{SHORT.get(t, t)} & {100*base[t][:50].mean():.2f} & " + " & ".join(cells) + f" & {nch}\\\\")
print("\\midrule")
cells = [f"$\\mathbf{{{np.mean([100*(c[t][:50]-base[t][:50]).mean() for t in tasks]):+.2f}}}$" if i == 0
         else f"{np.mean([100*(c[t][:50]-base[t][:50]).mean() for t in tasks]):+.2f}"
         for i, c in enumerate(cols) if c]
nch = sum(int((cols[0][t][:50] != base[t][:50]).sum()) for t in tasks)
print(f"mean & {np.mean([100*base[t][:50].mean() for t in tasks]):.2f} & " + " & ".join(cells) + f" & {nch}\\\\")
# the caption quotes how many samples the two ported operators change
for (d, suf), name in zip(COLS[4:], ("KeepKV", "SelKV")):
    c = load(d, suf)
    if c:
        print(f"% {name} changes {sum(int((c[t][:50] != base[t][:50]).sum()) for t in tasks)} of 400 samples")
