import os
import sys

import modal

app = modal.App("ot-kv-framework-runner")
data_volume = modal.Volume.from_name("ot_kv_data")

eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.11.0",
        "transformers==5.7.0",
        "accelerate",
        "lm-eval",
        "wonderwords",
        "nltk",
        "datasets",
        "tiktoken",
        "hf_transfer",
        "rouge",  # LongBench summarization (ROUGE-L)
        "jieba",  # LongBench Chinese tasks
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(
        local_path=".",
        remote_path="/app",
        ignore=[".git", "__pycache__", ".idea", ".vscode", "venv", "env",
                # local artefacts and large local data never needed server-side
                "paper", "runs", "models", "datasets"]
    )
)


@app.function(
    image=eval_image,
    gpu="H200",
    volumes={"/ot_kv_data": data_volume},
    timeout=21600,   # a full multi-method LongBench sweep runs past the 2 h default
)
def run_framework_on_modal(
        model_id: str,
        methods: list[str],
        tasks: list[str],
        prefill_fraction: float,
        max_length: int,
        limit: int = None,
        **kwargs
):
    sys.path.append("/app")
    os.chdir("/app")

    os.environ["HF_DATASETS_CACHE"] = "/ot_kv_data/datasets"
    os.environ["HF_HOME"] = "/ot_kv_data/models"

    from main import main as custom_main

    print("=" * 60)
    print("Launching modal evaluation pipeline...")
    print(f"Model:   {model_id}")
    print(f"Methods:   {', '.join(methods)}")
    print(f"Tasks:   {', '.join(tasks)}")
    print(f"Hyperparameters: {kwargs}")
    print("=" * 60)

    save_dir = kwargs.get("save_dir", None)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        os.environ["OTKV8_STATS_PATH"] = os.path.join(save_dir, "otkv8_stats.txt")   # per-sample merge timings
    custom_main(
        model_id=model_id,
        methods=methods,
        tasks=tasks,
        prefill_fraction=prefill_fraction,
        max_length=max_length,
        limit=limit,
        **kwargs
    )

    data_volume.commit()


@app.function(
    image=eval_image,
    gpu="A100-80GB",
    volumes={"/ot_kv_data": data_volume},
    timeout=7200,
)
def run_sigma_families(models: list[str], out_dir: str = "/ot_kv_data/runs/sigma",
                       docs: int = 3, prefill: int = 2560):
    import subprocess
    sys.path.append("/app")
    os.chdir("/app")
    os.environ["HF_DATASETS_CACHE"] = "/ot_kv_data/datasets"
    os.environ["HF_HOME"] = "/ot_kv_data/models"
    os.makedirs(out_dir, exist_ok=True)
    for m in models:
        safe = m.replace("/", "_")
        try:
            subprocess.run(
                [sys.executable, "experiments/sigma_families.py", "--model", m,
                 "--device", "cuda", "--dtype", "bfloat16",
                 "--docs", str(docs), "--prefill", str(prefill),
                 "--out", os.path.join(out_dir, f"sigma_{safe}.json")],
                check=True)
        except subprocess.CalledProcessError as e:
            print(f"[sigma] {m} FAILED: {e}")
        data_volume.commit()


@app.function(
    image=eval_image,
    gpu="A100-80GB",
    volumes={"/ot_kv_data": data_volume},
    timeout=7200,
)
def run_script(argv: list[str], out_dir: str = "/ot_kv_data/runs/review"):
    """Run any experiments/ script server-side (e.g. centroid_gap.py or
    sigma_perhead_derot.py on a model that does not fit locally). argv is the
    full argument list after `python`; paths under /ot_kv_data resolve on the
    volume. The volume is committed afterwards."""
    import subprocess
    sys.path.append("/app")
    os.chdir("/app")
    os.environ["HF_DATASETS_CACHE"] = "/ot_kv_data/datasets"
    os.environ["HF_HOME"] = "/ot_kv_data/models"
    os.makedirs(out_dir, exist_ok=True)
    print("[run_script]", " ".join(argv))
    rc = subprocess.run([sys.executable] + argv).returncode
    data_volume.commit()
    print(f"[run_script] exit {rc}")
    return rc


@app.local_entrypoint()
def run(
        model_id: str = "/ot_kv_data/models/Llama-3.1-8B-Instruct",
        methods: str = "otkv3,h2o",
        tasks: str = "longbench",
        prefill_fraction: float = 0.5,
        max_length: int = 8000,
        limit: int = 0,
        wiki_docs: str = "",
        compression_size: float = 0.5,
        recent_size: float = 0.1,
        sink_size: int = 4,
        mode: str = "prefill",
        per_head: bool = True,
        longbench_truncate: bool = True,
        longbench_tasks: str = "qasper",
        longbench_offset: int = 0,
        observation_window: int = 0,
        save_dir: str = "/ot_kv_data/runs",
        filename: str = "",
):
    # methods are comma separated; a method may carry ';key=value' overrides, so
    # only the commas split entries (see main._split_method)
    method_list = [m.strip() for m in methods.split(",") if m.strip()]
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]

    kwargs = {
        "compression_size": compression_size,
        "recent_size": recent_size,
        "sink_size": sink_size,
        "mode": mode,
        "per_head": per_head,
        "longbench_truncate": longbench_truncate,
        "haystack_dir": "/ot_kv_data/datasets/niah/PaulGrahamEssays",
        "longbench_dir": "/ot_kv_data/datasets/LongBench_Dataset",
        "longbench_tasks": longbench_tasks,
        "longbench_offset": longbench_offset,
        "save_dir": save_dir,
        "sinkhorn_iters": 50,
    }
    if observation_window > 0:
        kwargs["observation_window"] = observation_window
    if wiki_docs:
        kwargs["wiki_docs"] = wiki_docs
    if filename:
        kwargs["filename"] = filename

    eval_limit = limit if limit > 0 else None

    run_framework_on_modal.remote(
        model_id=model_id,
        methods=method_list,
        tasks=task_list,
        prefill_fraction=prefill_fraction,
        max_length=max_length,
        limit=eval_limit,
        **kwargs
    )
