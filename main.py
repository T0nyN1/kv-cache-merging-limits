import argparse
import os
import time

from evaluation.models.wrapper import EvaluatorHFLM
from evaluation.tasks.registry import get_evaluator
from utils import set_device, export_results


def _int_or_float(value):
    """Parse a size argument as int when integral, else float.

    `compression_size`, `recent_size` and `sink_size` are overloaded: an int
    means an absolute token count, a float means a ratio. argparse hands string
    values through untyped, and `_update_budget` only recognises int/float, so a
    raw string would silently collapse the budget to 0. This keeps the semantics.
    """
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip()
    try:
        return int(s)
    except ValueError:
        return float(s)


def _coerce(text: str):
    low = text.strip().lower()
    if low in ("", "none", "null"):
        return None            # `key=` is the natural way to write "unset"
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _split_method(method: str):
    """Split "name;key=value;key=value" into (name, overrides).

    Lets one evaluation run compare several configurations of the same cache
    without reloading the model: `otkv3;select_mode=kmeans:50;select_decode=true`.
    Results are keyed by the full string, so variants stay distinguishable.
    """
    parts = method.split(";")
    overrides = {}
    for part in parts[1:]:
        key, sep, value = part.partition("=")
        if sep:
            overrides[key.strip()] = _coerce(value)
    return parts[0].strip(), overrides


def get_cache_config(method: str, kwargs: dict):
    method, overrides = _split_method(method)
    cache_class, cache_kwargs = _get_cache_config(method, kwargs)
    cache_kwargs.update(overrides)
    return cache_class, cache_kwargs


def _get_cache_config(method: str, kwargs: dict):
    match method.lower():
        case "baseline":
            from baselines.baseline import BaselineCache
            return BaselineCache, {}

        case "h2o":
            from baselines.h2o import H2OCache
            return H2OCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": kwargs.get('sink_size', 4),
                "per_head": kwargs.get('per_head', False),
                "mode": kwargs.get('mode', 'prefill'),
            }

        case "otkv":
            from core.ot_kv import OTKVCache
            return OTKVCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": kwargs.get('sink_size', 4),
                "gamma": kwargs.get('otkv_gamma', 1.0),
                "epsilon": kwargs.get('otkv_epsilon', 0.01),
                "transport_mode": kwargs.get('otkv_transport_mode', "soft"),
                "compress_interval": kwargs.get('otkv_compress_interval', 32),
                "target_beta": kwargs.get("target_beta", 0.0),
                "sinkhorn_iters": kwargs.get("sinkhorn_iters", 50),
                "per_head": kwargs.get('per_head', False),
                "mode": kwargs.get('mode', 'prefill'),
            }
        case "otkv3":
            from core.ot_kv_v3 import OTKVv3Cache
            return OTKVv3Cache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": kwargs.get('sink_size', 4),
                "merge": kwargs.get('v3_merge', True),
                "merge_strength": kwargs.get('v3_merge_strength', 0.5),
                "select_mode": kwargs.get('v3_select_mode', 'topk'),
                "gate_sigma": kwargs.get('v3_gate_sigma', 0.0),
                "top_r": kwargs.get('v3_top_r', 16),
                "epsilon": kwargs.get('v3_epsilon', 0.05),
                "sinkhorn_iters": kwargs.get('v3_sinkhorn_iters', 5),
                "capacity_beta": kwargs.get('v3_capacity_beta', 1.0),
                "select_decode": kwargs.get('v3_select_decode', False),
                "cost_alpha": kwargs.get('v3_cost_alpha', 0.8),
                "observation_window": kwargs.get('observation_window', None),
                "compress_interval": kwargs.get('otkv_compress_interval', 32),
                "per_head": kwargs.get('per_head', False),
                "mode": kwargs.get('mode', 'prefill'),
            }

        case "otkv4":
            from core.ot_kv_v4 import OTKVv4Cache
            cfg = {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": kwargs.get('sink_size', 4),
                "preset": kwargs.get('v4_preset', 'qa'),
                "per_head": kwargs.get('per_head', True),
                "mode": kwargs.get('mode', 'prefill'),
            }
            for key in ("layer_budget", "select_mode", "select_scope", "merge", "merge_strength",
                        "gate_sigma", "cost_alpha", "epsilon", "top_r", "compress_interval",
                        "observation_window", "split_scores", "select_decode", "capacity_beta"):
                if f"v4_{key}" in kwargs:
                    cfg[key] = kwargs[f"v4_{key}"]
            return OTKVv4Cache, cfg

        case "otkv7":
            from core.ot_kv_v7 import OTKVv7Cache
            cfg = {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": kwargs.get('sink_size', 4),
                "per_head": kwargs.get('per_head', True),
                "mode": kwargs.get('mode', 'prefill'),
            }
            return OTKVv7Cache, cfg

        case "otkv6":
            from core.ot_kv_v6 import OTKVv6Cache
            cfg = {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": kwargs.get('sink_size', 4),
                "per_head": kwargs.get('per_head', True),
                "mode": kwargs.get('mode', 'prefill'),
            }
            return OTKVv6Cache, cfg

        case "otkv5":
            from core.ot_kv_v5 import OTKVv5Cache
            cfg = {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": kwargs.get('sink_size', 4),
                "per_head": kwargs.get('per_head', True),
                "mode": kwargs.get('mode', 'prefill'),
            }
            return OTKVv5Cache, cfg

        case "otkv_joint":
            from core.ot_kv_joint import OTKVCache
            return OTKVCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": kwargs.get('sink_size', 4),
                "gamma": kwargs.get('otkv_gamma', 1.0),
                "epsilon": kwargs.get('otkv_epsilon', 0.01),
                "transport_mode": kwargs.get('otkv_transport_mode', "soft"),
                "compress_interval": kwargs.get('otkv_compress_interval', 32),
                "target_beta": kwargs.get("target_beta", 0.0),
                "sinkhorn_iters": kwargs.get("sinkhorn_iters", 50),
                "per_head": kwargs.get('per_head', False),
                "mode": kwargs.get('mode', 'prefill'),
            }

        case "cam":
            from baselines.cam import CaMCache
            return CaMCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": kwargs.get('sink_size', 4),
                "per_head": False,
                "mode": kwargs.get('mode', 'prefill'),
            }

        case "kvmerger":
            from baselines.kvmerger import KVMergerCache
            return KVMergerCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": kwargs.get('sink_size', 4),
                "observation_window": kwargs.get('observation_window', None),
                "per_head": False,
                "mode": kwargs.get('mode', 'prefill'),
            }

        case "otkv8":
            from core.otkv_v8_cache import OTKVv8Cache
            cfg = {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": kwargs.get('sink_size', 4),
                "observation_window": kwargs.get('observation_window', 32),
                "pool_kernel": kwargs.get('pool_kernel', 7),
                "per_head": False,
                "mode": kwargs.get('mode', 'prefill'),
            }
            for key in ("row_window", "fit_rows", "sel_rows", "max_dist", "max_cand", "n_blocks",
                        "c_max", "joint_c", "min_gain", "gammas", "solver", "merge", "force_theta", "force_c"):
                if key in kwargs:
                    cfg[key] = kwargs[key]
            return OTKVv8Cache, cfg

        case "streamingllm":
            from baselines.streamingllm import StreamingLLMCache
            return StreamingLLMCache, {
                "compression_size": kwargs.get('compression_size', 1.0),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": 4 if kwargs.get('sink_size') is None else kwargs.get('sink_size'),
                "mode": kwargs.get('mode', 'prefill'),
            }

        case "snapkv":
            from baselines.snapkv import SnapKVCache
            return SnapKVCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": 0 if kwargs.get('sink_size') is None else kwargs.get('sink_size'),
                "observation_window": kwargs.get('observation_window', None),
                "per_head": kwargs.get('per_head', False),
                "mode": kwargs.get('mode', 'prefill'),
            }

        case "keepkv":
            from baselines.keepkv import KeepKVCache
            return KeepKVCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": 0 if kwargs.get('sink_size') is None else kwargs.get('sink_size'),
                "per_head": False,
                "mode": kwargs.get('mode', 'prefill'),
            }

        case "selkv":
            from baselines.selkv import SelKVCache
            return SelKVCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": 0 if kwargs.get('sink_size') is None else kwargs.get('sink_size'),
                "per_head": False,
                "mode": kwargs.get('mode', 'prefill'),
            }

        case "pyramidkv":
            from baselines.pyramidkv import PyramidKVCache
            return PyramidKVCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": 0 if kwargs.get('sink_size') is None else kwargs.get('sink_size'),
                "pyramid_low_scale": kwargs.get('pyramid_low_scale', 1.5),
                "pyramid_high_scale": kwargs.get('pyramid_high_scale', 0.5),
                "per_head": kwargs.get('per_head', False),
                "mode": kwargs.get('mode', 'prefill'),
            }

        case "echokv":
            from baselines.echokv import EchoKVCache
            return EchoKVCache, {
                "compression_size": kwargs.get('compression_size', 0.5),
                "recent_size": kwargs.get('recent_size', 0.1),
                "sink_size": 0 if kwargs.get('sink_size') is None else kwargs.get('sink_size'),
                "max_representative_scan": kwargs.get('max_representative_scan', None),
                "per_head": kwargs.get('per_head', False),
                "mode": kwargs.get('mode', 'prefill'),
            }

        case _:
            raise ValueError(f"Unknown method: {method}")


def main(model_id, methods, tasks, **kwargs):
    device = set_device()
    print(f"\n{'=' * 60}")
    print(f"🚀 Starting Multi-Evaluation Pipeline")
    print(f"Model  : {model_id}")
    print(f"Tasks  : {', '.join(tasks)}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Using device: {device}")
    print(f"{'=' * 60}\n")

    print(f">>> [Init] Loading Large Language Model ONCE into VRAM...")
    model_wrapper = EvaluatorHFLM(
        pretrained=model_id,
        cache_class=None,
        cache_kwargs={},
        prefill_fraction=kwargs.get("prefill_fraction", 0.2),
        max_length=kwargs.get("max_length", 4096),
        device=device,
    )
    print(f">>> [Init] Model loaded successfully!\n")

    summary_results = {task: {} for task in tasks}

    for task in tasks:
        print(f"\n{'=' * 60}")
        print(f"📌 Task: {task.upper()}")
        print(f"{'=' * 60}")

        try:
            evaluator_class = get_evaluator(task)
        except Exception as e:
            print(f"[Error] Failed to load evaluator for task '{task}': {e}")
            continue

        for method in methods:
            print(f"\n---> Evaluating Method: [{method.upper()}] on [{task}]")

            # Labels the per-sample merge-statistics lines the compensated ports append to
            # MERGE_STATS_PATH (baselines/compensated_base.py), so several methods in one
            # sweep stay separable in the same file.
            os.environ["OTKV_METHOD"] = method
            save_dir = kwargs.get("save_dir")
            if save_dir:
                os.environ["MERGE_STATS_PATH"] = os.path.join(save_dir, "merge_stats.jsonl")

            try:
                cache_class, cache_kwargs = get_cache_config(method, kwargs)
                model_wrapper.cache_class = cache_class
                model_wrapper.cache_kwargs = cache_kwargs
            except Exception as e:
                print(f"[Error] Failed to configure method '{method}': {e}")
                summary_results[task][method] = "Config Error"
                continue

            try:
                start_time = time.time()
                evaluator_instance = evaluator_class(model_wrapper=model_wrapper,
                                                     method_name=method, **kwargs)
                result = evaluator_instance.evaluate()
                elapsed = time.time() - start_time

                summary_results[task][method] = result
                print(f"     ✅ Done in {elapsed:.2f}s | Result: {result}")
            except Exception as e:
                import traceback
                print(f"     ❌ [Evaluation Failed] {str(e)}")
                traceback.print_exc()
                summary_results[task][method] = f"Error: {str(e)}"

    print("\n\n" + "=" * 60)
    print("🏆 FINAL EVALUATION SUMMARY")
    print("=" * 60)
    for task, method_res in summary_results.items():
        print(f"\n🔹 TASK: {task}")
        print(f"{'Method':<15} | {'Result':<20}")
        print("-" * 40)
        for method, res in method_res.items():
            res_str = f"{res:.4f}" if isinstance(res, float) else str(res)
            print(f"{method:<15} | {res_str:<20}")
    print("=" * 60 + "\n")

    export_results(summary_results, kwargs.get("save_dir", None), kwargs.get("filename", None))


def run():
    parser = argparse.ArgumentParser(description="OT-KV & KV Compression Evaluation Framework (v2: Multi-Run)")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model repository ID or local path")
    parser.add_argument("--tasks", type=str, nargs='+', default=["wikitext"],
                        choices=["wikitext", "niah", "longbench", "profile_niah", "kv_recovery", "retrieval"],
                        help="Evaluation task names (space separated, e.g., wikitext niah)")
    parser.add_argument("--methods", type=str, nargs='+', default=["baseline"],
                        help="KV Cache compression methods (space separated, e.g., baseline h2o snapkv). "
                             "Append ';key=value' pairs to override that method's kwargs for one run, "
                             "e.g. 'otkv3;select_mode=kmeans:50;select_decode=true'.")
    parser.add_argument("--compression_size", type=_int_or_float, default=0.5,
                        help="Target KV Cache budget: float ratio (0.5 = keep 50%) or int token count")
    parser.add_argument("--recent_size", type=_int_or_float, default=0.1,
                        help="Local/recent window: float ratio of budget or int token count")
    parser.add_argument("--sink_size", type=_int_or_float, default=4,
                        help="Initial/sink tokens to retain: int token count or float ratio of budget")
    parser.add_argument("--mode", type=str, default="prefill", choices=["prefill", "entire"],
                        help="Budget policy: 'prefill' fixes budget at prefill end; 'entire' recomputes it each decode step")
    parser.add_argument("--per_head", type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="Use per-head cache pruning instead of global")
    parser.add_argument("--otkv_compress_interval", type=int, default=32,
                        help="Run OTKV decode-time OT compression every N decode steps")
    parser.add_argument("--observation_window", type=int, default=None,
                        help="Score tokens with the last W queries only (SnapKV-style). Essential for "
                             "prompt-ending-in-a-question tasks (LongBench, NIAH); leave unset for "
                             "open-ended perplexity, where all-query accumulation is better.")
    parser.add_argument("--v3_select_mode", type=str, default="topk",
                        help="otkv3 anchor selection: 'topk' (heavy hitters), 'adaptive:B' (recommended, "
                             "B~4: per-head coverage bonus scaled by how diffuse the head is), or "
                             "'kmeans:P'. adaptive:4 cuts KL to dense by ~20%% at a small cost in "
                             "needle-retrieval accuracy; see docs/ot_kv_v3.md")
    parser.add_argument("--v3_merge", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="otkv3: transport evicted value mass onto the anchors (False = pure eviction)")
    parser.add_argument("--v3_merge_strength", type=float, default=0.5,
                        help="otkv3: shrinkage on the transported mass; 0.5 matches the measured noise "
                             "in the historical-attention-ratio coefficient")
    parser.add_argument("--v3_select_decode", type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="otkv3: re-run coverage selection at decode-time compressions. Recommended "
                             "with --v3_select_mode kmeans:P, whose coverage anchors are otherwise "
                             "eroded by plain top-k maintenance")
    parser.add_argument("--v3_gate_sigma", type=float, default=0.0,
                        help="otkv3: soft reliability gate exp(-cost/sigma) on the transport plan")
    parser.add_argument("--prefill_fraction", type=float, default=0.1,
                        help="Fraction of document used for the initial prefill stage in PPL testing")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length. For LongBench, inputs longer than this will be middle-truncated. Set to -1 to disable truncation.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of samples for evaluation (for quick debugging)")
    parser.add_argument("--wiki_docs", type=str, default=None,
                        help="Wikitext document numbers to evaluate, e.g. 1-10 or 1,2,3,4")
    parser.add_argument("--haystack_dir", type=str, default="datasets/niah/PaulGrahamEssays", )
    parser.add_argument("--longbench_dir", type=str, default="datasets/LongBench_dataset", )
    parser.add_argument("--longbench_tasks", type=str, default="all")
    parser.add_argument("--longbench_offset", type=int, default=0,
                        help="Skip the first N samples of every LongBench task, so a dev split "
                             "(offset 0) and a test split (offset 50) never share samples")
    parser.add_argument("--context_intervals", type=int, default=5,
                        help="NIAH: number of context lengths between 1000 and max_length (5 -> 25 positions with "
                             "depth_intervals=5; 10 x 10 is the paper's 100-position run)")
    parser.add_argument("--depth_intervals", type=int, default=5,
                        help="NIAH: number of needle depths in [0, 1]")
    parser.add_argument("--save_dir", type=str, default="runs")
    parser.add_argument("--filename", type=str, default=None,
                        help="Stem of the summary CSV written to save_dir (default: eval_results); "
                             "the Modal launcher passed this through kwargs, the CLI needs a flag")

    args = parser.parse_args()
    main_kwargs = vars(args)
    main(**main_kwargs)


if __name__ == "__main__":
    run()
