"""LaTeX rows of the v8 intervention table (Appendix, tab:v8) from runs/local_repro/v8_offline_*.json.
Rows = constructions, columns = model x budget; each cell = decode-row gain over all 64 steps / over the
last 16 steps (late rows, held out from the online re-calibration too), in % of the eviction error.
Usage: python paper/figures/make_v8_table.py"""
import glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs/local_repro")
NAMES = {"qwen3-1.7b": "Qwen3-1.7B", "llama-3.1-8b-instruct": "Llama-3.1-8B"}
COND = [("ot", "prefill-window fit, matching"), ("greedy", "\\quad greedy instead of matching"),
        ("nobias", "\\quad no bias channel ($c=1$)"), ("biasonly", "\\quad bias only ($\\theta=0$)"), ("random", "\\quad random pairing, same proposed count"),
        ("massc", "\\quad $c$ from the mass fit only (no joint search)"), ("online", "online: fit on decode steps 1--32, $\\gamma$ on 33--48"),
        ("oracle", "decode-fitted reference (in sample, $\\gamma$ on the same rows)")]
runs = {}
for path in sorted(p for p in glob.glob(os.path.join(RUN_DIR, "v8_offline_*_b*.json")) if "prefix_v0" not in p):
    J = json.load(open(path)); S = J["summary"]; b = J["meta"]["budget"]
    tag = os.path.basename(path).split("v8_offline_")[1].rsplit("_b", 1)[0]
    runs[(NAMES.get(tag, tag), b)] = S
cols = sorted(runs.keys(), key=lambda x: (x[0] != "Qwen3-1.7B", x[1]))
for key in cols:
    S = runs[key]; g = S["ot_gain_decode"]["cluster_bootstrap"]; oc = S["oracle_gain_decode"]["cluster_bootstrap"]
    gg = S["ot_vs_greedy_decode"]["cluster_bootstrap_greedy_minus_ot"]; on = S["online_gain_decode_late"]
    ol = S["ot_gain_decode_late"]; orl = S["oracle_gain_decode_late"]; j = S["massc_gain_decode"]
    c = S["conditions"]
    print(f"% {key[0]} {key[1]:g}: heads {S['n_heads']} invalid {S['n_invalid']}; ot {100*g['mean']:.1f} [{100*g['lo']:.1f},{100*g['hi']:.1f}] "
          f"median {100*S['ot_gain_decode']['median']:.1f} trimmed {100*S['ot_gain_decode']['mean_without_top10pct_heads']:.1f} "
          f"worst {S['ot_gain_decode']['worst_head_rel']:.2f} g>0 {c['ot']['frac_gamma_positive']:.2f} mass {100*c['ot']['merged_mass_mean']:.1f}%; "
          f"oracle {100*oc['mean']:.1f} [{100*oc['lo']:.1f},{100*oc['hi']:.1f}]; greedy-ot {100*gg['mean']:.2f} [{100*gg['lo']:.2f},{100*gg['hi']:.2f}]; "
          f"massc {100*j['mean']:.1f} [{100*j['lo']:.1f},{100*j['hi']:.1f}]; LATE ot {100*ol['mean']:.1f} [{100*ol['lo']:.1f},{100*ol['hi']:.1f}] "
          f"online {100*on['mean']:.1f} [{100*on['lo']:.1f},{100*on['hi']:.1f}] (g>0 {c['online']['frac_gamma_positive']:.2f}) "
          f"oracle {100*orl['mean']:.1f} [{100*orl['lo']:.1f},{100*orl['hi']:.1f}]; nobias {100*(1-c['nobias']['rel_decode_mean']):.1f} random {100*(1-c['random']['rel_decode_mean']):.1f}; "
          f"biasonly {100*S.get('biasonly_gain_decode', {'mean': float('nan')})['mean']:.1f}; H1 {S['gate_H1_deployable_merge']} H3 {S['gate_H3_ot_beats_greedy']}")
head = "construction & " + " & ".join(f"{m} {100*b:.0f}\\%" for m, b in cols) + "\\\\"
print(head)
print("\\midrule")
for cnd, label in COND:
    cells = []
    for key in cols:
        r = runs[key]["conditions"][cnd]
        cells.append(f"{100*(1-r['rel_decode_mean']):+.1f} / {100*(1-r['rel_decode_late_mean']):+.1f}")
    print(f"{label} & " + " & ".join(cells) + "\\\\")
print("\\midrule")
for lab, fn in (("applied merges per head (prefill fit)", lambda S: f"{S['conditions']['ot']['n_merged_mean']:.0f}"),
                ("applied evicted mass (prefill fit)", lambda S: f"{100*S['conditions']['ot']['merged_mass_mean']:.1f}\\%"),
                ("heads ${>}5\\%$ worse (prefill fit)", lambda S: f"{100*S['conditions']['ot']['frac_heads_worse_than_baseline_5pct']:.0f}\\%"),
                ("worst head, error ratio (prefill fit)", lambda S: f"{S['ot_gain_decode']['worst_head_rel']:.2f}")):
    print(f"{lab} & " + " & ".join(fn(runs[k]) for k in cols) + "\\\\")
