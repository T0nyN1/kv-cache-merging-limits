from typing import Dict, Any

from .base_evaluator import BaseEvaluator
from .longbench_metrics import dataset2metric, score_prediction
from .registry import register_task


@register_task("longbench")
class LongBenchEvaluator(BaseEvaluator):

    def _dump_per_sample(self, task, per_sample):
        import json
        import os
        import re

        save_dir = self.args.get("save_dir")
        if not save_dir or not per_sample:
            return
        method = self.args.get("method_name", "unknown")
        # Truncating a long config string collides: two variants differing only
        # in a trailing kwarg map to the same file and silently overwrite each
        # other. Keep a readable prefix and disambiguate with a hash of the full
        # name.
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", method)
        if len(safe) > 100:
            import hashlib
            safe = safe[:100] + "-" + hashlib.sha1(method.encode()).hexdigest()[:8]
        os.makedirs(save_dir, exist_ok=True)
        offset = int(self.args.get("longbench_offset", 0) or 0)
        suffix = f"@{offset}" if offset else ""
        path = os.path.join(save_dir, f"persample_{task}_{safe}{suffix}.json")
        with open(path, "w") as f:
            json.dump({"task": task, "method": method, "offset": offset,
                       "scores": per_sample}, f)
        print(f"    [per-sample scores -> {path}]")

    def evaluate(self) -> Dict[str, Any]:
        import json
        import os
        import urllib.request
        import zipfile
        import torch
        from tqdm import tqdm

        max_length = self.args.get('max_length', 7500)
        tokenizer = self.model_wrapper.tokenizer
        model = self.model_wrapper._model

        longbench_dir = self.args.get("longbench_dir", "./datasets/LongBench_dataset")
        data_folder = os.path.join(longbench_dir, "data")

        if not os.path.exists(data_folder):
            print(f"\n[*] Downloading LongBench directly to {longbench_dir}...")
            os.makedirs(longbench_dir, exist_ok=True)
            zip_url = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
            zip_path = os.path.join(longbench_dir, "data.zip")
            urllib.request.urlretrieve(zip_url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(longbench_dir)

        tasks_arg = self.args.get('longbench_tasks', "all")

        if tasks_arg.strip().lower() == "all" or not tasks_arg:
            tasks = [f.replace(".jsonl", "") for f in os.listdir(data_folder) if f.endswith(".jsonl")]
            print(f"\n[*] No specific tasks provided. Found {len(tasks)} tasks in dataset directory.")
        else:
            tasks = tasks_arg.split(",")

        all_results = {}
        for task in tasks:
            task = task.strip()
            metric_name = getattr(dataset2metric.get(task, None), "__name__", "qa_f1_score (fallback)")
            print(f"\n[*] Running LongBench task: {task}  (metric: {metric_name})")
            file_path = os.path.join(data_folder, f"{task}.jsonl")

            if not os.path.exists(file_path):
                print(f"[!] Warning: Task file {file_path} not found. Skipping...")
                continue

            dataset = [json.loads(line) for line in open(file_path, 'r', encoding='utf-8')]
            offset = int(self.args.get('longbench_offset', 0) or 0)
            limit = self.args.get('limit', None)
            if limit:
                dataset = dataset[offset:offset + limit]
            elif offset:
                dataset = dataset[offset:]

            task_score = 0.0
            truncated_count = 0
            per_sample = []
            for item in tqdm(dataset, desc=f"Evaluating {task}"):
                context = item['context']
                query = item['input']
                answers = item["answers"]
                all_classes = item.get("all_classes", None)

                if query.strip():
                    prompt = f"Please read the following context and answer the question.\n\nContext:\n{context}\n\nQuestion:\n{query}"
                else:
                    prompt = context
                messages = [{"role": "user", "content": prompt}]
                input_tensor = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )["input_ids"].to(model.device)

                if max_length > 0 and input_tensor.shape[1] > max_length:
                    half = max_length // 2
                    input_tensor = torch.cat([input_tensor[:, :half], input_tensor[:, -half:]], dim=1)
                    truncated_count += 1

                custom_cache = self.model_wrapper._setup_cache_and_hooks()

                with torch.no_grad():
                    output_ids = model.generate(
                        input_tensor,
                        attention_mask=torch.ones_like(input_tensor),
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        past_key_values=custom_cache,
                        use_cache=True
                    )
                self._cleanup_cache_and_hooks(custom_cache)

                generated_tokens = output_ids[0][input_tensor.shape[1]:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                sample_score = score_prediction(task, response, answers, all_classes=all_classes)
                task_score += sample_score
                per_sample.append(sample_score)

            score = task_score / len(dataset) if len(dataset) > 0 else 0.0
            all_results[task] = {
                "score": score,
                "metric": metric_name,
                "tested_samples": len(dataset),
                "truncated_samples": truncated_count,
            }
            # Per-sample scores let the caller pair methods on identical inputs.
            # LongBench F1 varies enough across documents that unpaired means over
            # 50 samples cannot separate methods a few points apart.
            self._dump_per_sample(task, per_sample)
            if max_length > 0 and truncated_count > 0:
                print(f"    [!] {truncated_count}/{len(dataset)} samples middle-truncated to "
                      f"max_length={max_length}; pass --max_length -1 to evaluate full context.")
            print(f"-> {task} SCORE ({metric_name}): {score * 100:.2f}%")

        return {"longbench": all_results}
