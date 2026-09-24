"""Recompute the paper's load-bearing numbers from the raw per-sample files.

Covers: the two KVMerger rows of the matched-protocol table; the deployed merge
and both of its channel ablations (mean, both bootstrap intervals as quoted,
permutation and sign tests, per-task sign count, drop-one-task sensitivity); the
two direct contrasts and the interaction; the baseline-identity check; the v8
mechanism table; the consolidation provenance identity; and the single-number
claims. It does NOT check the remaining rows of tab:matched or tab:main, the
theory sections, or the appendix protocol text.

Recomputes, from the raw per-sample files and run JSONs, the numbers the paper
quotes, and reports any mismatch. Run before every submission:

    python experiments/verify_paper_numbers.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = open(os.path.join(ROOT, "paper/main.tex")).read()
FAILS, CHECKS = [], 0


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def tex_has(pattern):
    """Is this literal string (after whitespace folding) in the paper?"""
    flat = re.sub(r"\s+", " ", TEX)
    return re.sub(r"\s+", " ", pattern) in flat


def paired(dirname, method_suffix, ref_suffix, tasks=None, ref_dirname=None):
    """Paired per-sample difference of two methods (the reference may live in
    another run directory, e.g. the baseline sweep the method was added to)."""
    d = os.path.join(ROOT, "runs/modal", dirname)
    rd = os.path.join(ROOT, "runs/modal", ref_dirname or dirname)
    diffs, n_tasks = [], 0
    for f in sorted(glob.glob(os.path.join(d, "persample_*" + method_suffix))):
        J = json.load(open(f))
        task = J["task"]
        if tasks and task not in tasks:
            continue
        rf = os.path.join(rd, f"persample_{task}{ref_suffix}")
        if not os.path.exists(rf):
            continue
        R = json.load(open(rf))
        x, r = np.array(J["scores"]), np.array(R["scores"])
        n = min(len(x), len(r))
        diffs.append(100 * (x[:n] - r[:n]))
        n_tasks += 1
    if not diffs:
        return None
    dd = np.concatenate(diffs)
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(dd, len(dd)).mean() for _ in range(20000)])
    return dd.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(dd), n_tasks


SNAP = "_snapkv_observation_window_32_per_head_false_pool_kernel_7.json"


def _load_runs(dirname, suffix):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "runs/modal", dirname, "persample_*" + suffix))):
        J = json.load(open(f)); out[J["task"]] = np.array(J["scores"])
    return out


def stats_of(dirname, suffix, ref_dirname="dev8k"):
    """Every inference the paper quotes for one operator, from the raw files."""
    from scipy import stats as _s
    d = os.path.join(ROOT, "runs/modal", dirname)
    rd = os.path.join(ROOT, "runs/modal", ref_dirname)
    per, names = [], []
    for f in sorted(glob.glob(os.path.join(d, "persample_*" + suffix))):
        J = json.load(open(f)); t = J["task"]
        R = json.load(open(os.path.join(rd, f"persample_{t}{SNAP}")))
        x, r = np.array(J["scores"]), np.array(R["scores"]); n = min(len(x), len(r))
        per.append(100 * (x[:n] - r[:n])); names.append(t)
    dd = np.concatenate(per); means = np.array([x.mean() for x in per])
    rng = np.random.default_rng(0)
    cb = np.array([means[rng.integers(0, len(means), len(means))].mean() for _ in range(200000)])
    perm = np.array([(dd * rng.choice([-1, 1], len(dd))).mean() for _ in range(200000)])
    # exact sign-flip randomisation over the eight task means (all 2^8 assignments)
    import itertools
    flips = np.array([np.array(f) for f in itertools.product([-1, 1], repeat=len(means))])
    task_p = float((np.abs(flips @ means) / len(means) >= abs(means.mean()) - 1e-12).mean())
    nz = dd[dd != 0]
    keep = [i for i, t in enumerate(names) if t != "triviaqa"]
    return {"mean": dd.mean(), "task_lo": np.percentile(cb, 2.5), "task_hi": np.percentile(cb, 97.5),
            "perm_p": (np.abs(perm) >= abs(dd.mean())).mean(), "task_p": task_p,
            "wins": int((nz > 0).sum()), "losses": int((nz < 0).sum()),
            "sign_p": _s.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue,
            "n_pos": int((means > 0).sum()), "drop_trivia": means[keep].mean()}

print("== matched-protocol table (tab:matched), dev split ==")
for label, suffix, ref, want in (
    ("KVMerger on pooled SnapKV, published delta=0.75",
     "_kvmerger_observation_window_32_per_head_false_pool_kernel_7.json", SNAP, -0.58),
    ("KVMerger on pooled SnapKV, delta=0.5",
     "_kvmerger_observation_window_32_per_head_false_pool_kernel_7_merge_threshold_0.5.json", SNAP, -0.06),
):
    r = paired("dev8k", suffix, ref)
    if r is None:
        check(label, False, "source files missing")
        continue
    m, lo, hi, n, _ = r
    check(f"{label}: {m:+.2f} [{lo:+.2f},{hi:+.2f}] (n={n})",
          abs(m - want) < 0.02, f"paper says {want:+.2f}")

print("\n== deployed v8 merge on LongBench dev (sec:classb) ==")
r = paired("dev8k_v8", "_otkv8_observation_window_32_per_head_false_pool_kernel_7.json", SNAP, ref_dirname="dev8k")
if r is None:
    check("v8 vs pooled SnapKV", False, "source files missing")
else:
    m, lo, hi, n, nt = r
    check(f"v8 vs pooled SnapKV: {m:+.2f} [{lo:+.2f},{hi:+.2f}] over {nt} tasks, n={n}",
          tex_has(f"$+{m:.2f}$ LongBench") and n == 400, "paper quotes +0.70 LongBench over 400 paired samples")
    # the inference the abstract and section 6 rest on, recomputed here
    st_ = stats_of("dev8k_v8", "_otkv8_observation_window_32_per_head_false_pool_kernel_7.json")
    check(f"task-clustered CI [{st_['task_lo']:+.2f},{st_['task_hi']:+.2f}] is the one quoted",
          tex_has(f"$[{st_['task_lo']:+.2f},{st_['task_hi']:+.2f}]$"))
    check(f"permutation p over samples {st_['perm_p']:.3f} and over task means {st_['task_p']:.3f}",
          tex_has(f"$p={st_['perm_p']:.3f}$") and tex_has(f"${st_['task_p']:.3f}$"))
    check(f"the unresolved sign test is disclosed ({st_['wins']}/{st_['losses']}, p={st_['sign_p']:.2f})",
          tex_has(f"${st_['wins']}/{st_['losses']}$") and tex_has(f"$p={st_['sign_p']:.2f}$"))
    check(f"{st_['n_pos']} of 8 tasks positive, and dropping triviaqa leaves {st_['drop_trivia']:+.2f}",
          tex_has("seven of eight tasks positive") and tex_has(f"$+{st_['drop_trivia']:.2f}$"))
    check(f"paper states the task count ({nt} of eight)",
          tex_has("two of eight") if nt == 2 else tex_has("400 paired"),
          f"{nt} tasks completed")

print("\n== KeepKV and SelKV ports (tab:matched) ==")
# Each operator is paired against ITS OWN base: the two operator rows against the
# pooled-SnapKV run every other row uses, the own-selector row against the same
# selector with merging switched off (which is not SnapKV).
for label, d, suffix, rd, rsuf in (
    ("KeepKV operator on pooled SnapKV", "ports_a", "_keepkv.json", "dev8k", SNAP),
    ("SelKV operator on pooled SnapKV", "ports_c", "_selkv_selector_snapkv.json", "dev8k", SNAP),
    ("SelKV, own selector", "ports_b", "_selkv_selector_selkv.json",
     "ports_b", "_selkv_selector_selkv_merge_false.json"),
    ("SelKV, own selector @25%", "ports_d25", "_selkv_selector_selkv.json",
     "ports_d25", "_selkv_selector_selkv_merge_false.json"),
    ("KeepKV @25% vs its own base", "ports_g25", "_keepkv.json",
     "ports_g25", "_keepkv_merge_false.json"),
    ("KeepKV, vote bias off", "ports_f", "_keepkv_votes_bias_false.json", "dev8k", SNAP),
    ("KeepKV, middle targets only", "ports_e", "_keepkv_targets_middle.json", "dev8k", SNAP),
    ("KeepKV, keys held fixed", "ports_h", "_keepkv_keys_fixed.json", "dev8k", SNAP),
):
    r = paired(d, suffix, rsuf, ref_dirname=rd)
    if r is None:
        check(label, False, "source files missing"); continue
    m, lo, hi, n, nt = r
    quoted = any(tex_has(f"$[{lo+a:+.2f},{hi+b:+.2f}]$")
                 for a in (-0.02, -0.01, 0.0, 0.01, 0.02) for b in (-0.02, -0.01, 0.0, 0.01, 0.02))
    check(f"{label}: {m:+.2f} [{lo:+.2f},{hi:+.2f}] over {nt} tasks, n={n}",
          tex_has(f"${m:+.2f}$") and quoted and nt == 8 and n == 400,
          "paper must quote this mean and interval")

# the contrast that attributes KeepKV's loss to the key rewrite: same merges, key write removed
A = _load_runs("ports_h", "_keepkv_keys_fixed.json")
B = _load_runs("ports_a", "_keepkv.json")
ts = sorted(set(A) & set(B))
if len(ts) == 8:
    per = [100 * (A[t][:50] - B[t][:50]) for t in ts]
    dd = np.concatenate(per); means = np.array([p.mean() for p in per])
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(dd, len(dd)).mean() for _ in range(20000)])
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    check(f"keys held fixed - keys moved: {dd.mean():+.2f} [{lo:+.2f},{hi:+.2f}], {int((means>0).sum())}/8 tasks",
          tex_has(f"${dd.mean():+.2f}$") and tex_has(f"$[{lo:+.2f},{hi:+.2f}]$"),
          "paper must quote this contrast")

# The operators are largely inert at this budget; the paper says so with these rates.
for label, d, method, num, den in (
    ("KeepKV cosine candidates", "ports_a", "keepkv", "candidates", "evicted"),
    ("SelKV routed (own selector)", "ports_b", "selkv;selector=selkv", "routed", "evicted"),
    ("SelKV routed (pooled SnapKV)", "ports_c", "selkv;selector=snapkv", "routed", "evicted"),
):
    path = os.path.join(ROOT, "runs/modal", d, "merge_stats.jsonl")
    if not os.path.exists(path):
        check(label, False, "merge_stats.jsonl missing"); continue
    tot = {}
    for line in open(path):
        rec = json.loads(line)
        if rec["method"] != method:
            continue
        for k, v in rec["stats"].items():
            tot[k] = tot.get(k, 0.0) + v
    frac = 100 * tot[num] / tot[den]
    check(f"{label}: {frac:.1f}% of evicted tokens ({tot[num]:.3g} of {tot[den]:.3g})",
          tex_has(f"${frac:.1f}\\%$"),        # one decimal: "$5\\%$" is the protocol's budget
          "paper must quote this rate")

print("\n== the deployed merge and its two channel ablations (tab:matched, tab:deploy) ==")
for label, d, suffix, want in (
    ("keys fixed, mass and mixture", "dev8k_v8", "_otkv8_observation_window_32_per_head_false_pool_kernel_7.json", +0.70),
    ("  mass channel only (theta=0)", "dev8k_bias", "_force_theta_0.json", +0.08),
    ("  value mixture only (c=1)", "dev8k_nobias", "_force_c_1.json", +0.20),
    ("keys averaged (KVMerger)", "dev8k", "_kvmerger_observation_window_32_per_head_false_pool_kernel_7.json", -0.58),
):
    r = paired(d, suffix, SNAP, ref_dirname="dev8k")
    if r is None:
        check(label, False, "source files missing"); continue
    m, lo, hi, n, nt = r
    # bootstrap endpoints move by ~0.01 between draws, so accept any interval
    # within that of the recomputed one rather than an exact string
    quoted = any(tex_has(f"$[{lo+a:+.2f},{hi+b:+.2f}]$")
                 for a in (-0.02, -0.01, 0.0, 0.01, 0.02) for b in (-0.02, -0.01, 0.0, 0.01, 0.02))
    check(f"{label}: {m:+.2f} [{lo:+.2f},{hi:+.2f}] over {nt} tasks, n={n}",
          abs(m - want) < 0.02 and nt == 8 and quoted,
          f"paper says {want:+.2f} over 8 tasks and must quote this interval")

print("\n== direct contrasts between the construction and its ablations ==")
for label, a_dir, a_suf, b_dir, b_suf in (
    ("joint - mass only", "dev8k_v8", "_otkv8_observation_window_32_per_head_false_pool_kernel_7.json",
     "dev8k_bias", "_force_theta_0.json"),
    ("joint - mixture only", "dev8k_v8", "_otkv8_observation_window_32_per_head_false_pool_kernel_7.json",
     "dev8k_nobias", "_force_c_1.json"),
):
    A = _load_runs(a_dir, a_suf); B = _load_runs(b_dir, b_suf)
    ts = sorted(set(A) & set(B))
    per = [100 * (A[t][:50] - B[t][:50]) for t in ts]
    means = np.array([x.mean() for x in per])
    rng = np.random.default_rng(0)
    cb = np.array([means[rng.integers(0, len(means), len(means))].mean() for _ in range(200000)])
    lo, hi = np.percentile(cb, 2.5), np.percentile(cb, 97.5)
    ok = any(tex_has(f"$+{means.mean():.2f}$ $[{lo+a:+.2f},{hi+b:+.2f}]$")
             for a in (-0.02, -0.01, 0.0, 0.01, 0.02) for b in (-0.02, -0.01, 0.0, 0.01, 0.02))
    check(f"{label}: {means.mean():+.2f} [{lo:+.2f},{hi:+.2f}] clustered by task", ok,
          "paper must quote this contrast and interval")
# the interaction the paper calls unresolved
J = _load_runs("dev8k_v8", "_otkv8_observation_window_32_per_head_false_pool_kernel_7.json")
M = _load_runs("dev8k_bias", "_force_theta_0.json")
X = _load_runs("dev8k_nobias", "_force_c_1.json")
S0 = _load_runs("dev8k", SNAP)
ts = sorted(set(J) & set(M) & set(X) & set(S0))
inter = np.array([(100 * (J[t][:50] - M[t][:50] - X[t][:50] + S0[t][:50])).mean() for t in ts])
rng = np.random.default_rng(0)
cb = np.array([inter[rng.integers(0, len(inter), len(inter))].mean() for _ in range(200000)])
lo, hi = np.percentile(cb, 2.5), np.percentile(cb, 97.5)
ok = any(tex_has(f"$[{lo+a:+.2f},{hi+b:+.2f}]$")
         for a in (-0.02, -0.01, 0.0, 0.01, 0.02) for b in (-0.02, -0.01, 0.0, 0.01, 0.02))
check(f"interaction {inter.mean():+.2f} [{lo:+.2f},{hi:+.2f}] is quoted as unresolved",
      tex_has(f"$+{inter.mean():.2f}$") and ok
      and (tex_has("not resolved") or tex_has("unresolved")))

print("\n== the baseline every paired comparison rests on ==")
import numpy as _np
same = 0
for f in sorted(glob.glob(os.path.join(ROOT, "runs/modal/dev8k_fresh", "persample_*" + SNAP))):
    J = json.load(open(f))
    O = json.load(open(os.path.join(ROOT, "runs/modal/dev8k", os.path.basename(f))))
    same += int(_np.allclose(J["scores"], O["scores"]))
check(f"pooled SnapKV re-run from scratch reproduces the original on {same} of 8 tasks", same == 8)

print("\n== v8 mechanism table (tab:v8) vs run JSONs ==")
NAMES = {"qwen3-1.7b": "Qwen3-1.7B", "llama-3.1-8b-instruct": "Llama-3.1-8B"}
for path in sorted(p for p in glob.glob(os.path.join(ROOT, "runs/local_repro/v8_offline_*_b*.json"))
                   if "prefix_v0" not in p):
    S = json.load(open(path))["summary"]
    tag = os.path.basename(path).split("v8_offline_")[1].rsplit("_b", 1)[0]
    b = json.load(open(path))["meta"]["budget"]
    g = S["ot_gain_decode"]["cluster_bootstrap"]
    row = f"{100*(1-S['conditions']['ot']['rel_decode_mean']):+.1f} / {100*(1-S['conditions']['ot']['rel_decode_late_mean']):+.1f}"
    check(f"{NAMES.get(tag, tag)} {100*b:.0f}%: table row {row} present, gain {100*g['mean']:.1f}% "
          f"[{100*g['lo']:.1f},{100*g['hi']:.1f}]", tex_has(row), f"looked for '{row}' in tab:v8")

print("\n== provenance: a merge must be paired against ITS OWN base ==")
# `otkv7` (bare) in runs/modal/dev8k predates commit 4735f18, which flipped the
# v7 default to auto_regime=false, so it carries the regime detector and is NOT
# the base of the later consolidation run; `otkv7;auto_regime=false` is. The
# identity below is the check: consolidation - v7(no detector) must equal
# (consolidation - v7(detector)) + (v7(detector) - v7(no detector)).
c_arf = paired("dev8k", "_otkv7_consolidate_true.json", "_otkv7_auto_regime_false.json")
c_v7 = paired("dev8k", "_otkv7_consolidate_true.json", "_otkv7.json")
v7_arf = paired("dev8k", "_otkv7.json", "_otkv7_auto_regime_false.json")
if None not in (c_arf, c_v7, v7_arf):
    check(f"consolidation @5% against its own base (no detector): {c_arf[0]:+.2f} "
          f"[{c_arf[1]:+.2f},{c_arf[2]:+.2f})", abs(c_arf[0] + 0.72) < 0.02, "paper says -0.72")
    check(f"the detector, not merging, accounts for {v7_arf[0]:+.2f} of it "
          f"(identity {c_v7[0]:+.2f} {v7_arf[0]:+.2f} = {c_v7[0]+v7_arf[0]:+.2f})",
          abs((c_v7[0] + v7_arf[0]) - c_arf[0]) < 0.01)

print("\n== single-number claims ==")
niah = glob.glob(os.path.join(ROOT, "runs/modal/niah8k_v8/*.csv"))
if niah:
    last = open(niah[0]).read().strip().splitlines()[-1]
    acc = float(last.split(",")[-1])
    check(f"deployed v8 NIAH-25 accuracy {acc:.2f}", tex_has(f"(${acc:.2f}$)") or tex_has(f"${acc:.2f}$"),
          "paper: needle retrieval unchanged")
ppl = glob.glob(os.path.join(ROOT, "runs/modal/ppl8k_v8/*.csv"))
if ppl:
    last = open(ppl[0]).read().strip().splitlines()[-1]
    val = float(last.split(",")[4])
    base = 4.49
    check(f"deployed v8 wikitext ppl {val:.3f} vs base {base} = {100*(base-val)/base:.1f}% better",
          tex_has(f"${val:.2f}$") and tex_has("7.9"), f"paper quotes 4.13 / 7.9%")

print(f"\n{CHECKS} checks, {len(FAILS)} failed" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
