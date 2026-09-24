"""Spawn an evaluation on the *deployed* ot-kv-framework-runner app.

`modal run` ties the remote call to the local client: a laptop DNS blip sends a
cancellation and kills a half-finished sweep (it did, on 2026-09-01). Deploying
once and spawning detaches the call completely -- the function runs to
completion server-side and results land on the volume regardless of what
happens locally.

    modal deploy misc/modal/evaluate.py       # once per code change
    python misc/modal/spawn_eval.py --methods "otkv7" --tasks longbench ...

Prints the FunctionCall id; progress is visible via `modal app logs` or by
listing the save_dir on the volume.
"""
import argparse

import modal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=None, help="comma-separated method strings (required unless --script)")
    ap.add_argument("--tasks", default="longbench")
    ap.add_argument("--longbench-tasks", dest="longbench_tasks",
                    default="qasper,multifieldqa_en,hotpotqa,2wikimqa,gov_report,multi_news,triviaqa,samsum")
    ap.add_argument("--longbench-offset", dest="longbench_offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-length", dest="max_length", type=int, default=8000)
    ap.add_argument("--compression-size", dest="compression_size", type=float, default=0.05)
    ap.add_argument("--recent-size", dest="recent_size", type=float, default=0.1)
    ap.add_argument("--sink-size", dest="sink_size", type=int, default=4)
    ap.add_argument("--mode", default="prefill")
    ap.add_argument("--per-head", dest="per_head", default="true")
    ap.add_argument("--prefill-fraction", dest="prefill_fraction", type=float, default=0.5)
    ap.add_argument("--wiki-docs", dest="wiki_docs", default="")
    ap.add_argument("--save-dir", dest="save_dir", default="/ot_kv_data/runs")
    ap.add_argument("--filename", default="")
    ap.add_argument("--model-id", dest="model_id", default="/ot_kv_data/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra evaluator kwargs as key=value (e.g. context_intervals=10 depth_intervals=10)")
    ap.add_argument("--script", nargs=argparse.REMAINDER, default=None,
                    help="instead of an evaluation, spawn run_script with this argv "
                         "(e.g. --script experiments/centroid_gap.py --model /ot_kv_data/models/X --device cuda ...)")
    args = ap.parse_args()

    if args.script:
        fn = modal.Function.from_name("ot-kv-framework-runner", "run_script")
        call = fn.spawn(args.script)
        print(f"spawned run_script: {call.object_id}")
        return

    if not args.methods:
        ap.error("--methods is required unless --script is given")
    fn = modal.Function.from_name("ot-kv-framework-runner", "run_framework_on_modal")

    kwargs = dict(
        compression_size=args.compression_size,
        recent_size=args.recent_size,
        sink_size=args.sink_size,
        mode=args.mode,
        per_head=str(args.per_head).lower() == "true",
        longbench_truncate=True,
        haystack_dir="/ot_kv_data/datasets/niah/PaulGrahamEssays",
        longbench_dir="/ot_kv_data/datasets/LongBench_Dataset",
        longbench_tasks=args.longbench_tasks,
        longbench_offset=args.longbench_offset,
        save_dir=args.save_dir,
        sinkhorn_iters=50,
    )
    if args.wiki_docs:
        kwargs["wiki_docs"] = args.wiki_docs
    if args.filename:
        kwargs["filename"] = args.filename
    for kv in args.extra:
        key, _, val = kv.partition("=")
        try:
            kwargs[key] = int(val)
        except ValueError:
            try:
                kwargs[key] = float(val)
            except ValueError:
                kwargs[key] = val

    call = fn.spawn(
        model_id=args.model_id,
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        tasks=[t.strip() for t in args.tasks.split(",") if t.strip()],
        prefill_fraction=args.prefill_fraction,
        max_length=args.max_length,
        limit=args.limit if args.limit > 0 else None,
        **kwargs,
    )
    print(f"spawned: {call.object_id}")


if __name__ == "__main__":
    main()
