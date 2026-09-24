import glob
import os
from typing import Dict, Any

import torch

from .base_evaluator import BaseEvaluator
from .registry import register_task


@register_task("niah")
class NIAHEvaluator(BaseEvaluator):

    def evaluate(self) -> Dict[str, Any]:
        print("\n[*] Running Needle In A Haystack Evaluation...")
        tokenizer = self.model_wrapper.tokenizer
        model = self.model_wrapper._model
        device = getattr(self.model_wrapper, 'device', model.device)

        haystack_dir = self.args.get('haystack_dir', "datasets/niah/PaulGrahamEssays")
        max_length = self.model_wrapper.max_length
        context_intervals = self.args.get('context_intervals', 5)
        depth_intervals = self.args.get('depth_intervals', 5)
        limit = self.args.get('limit', None)
        limit = None if limit is None or int(limit) <= 0 else int(limit)

        needle = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
        question = "What is the best thing to do in San Francisco?"

        if not os.path.exists(haystack_dir):
            return {"needlehaystack": {"error": f"Haystack dir not found: {haystack_dir}"}}

        text_files = glob.glob(os.path.join(haystack_dir, "*.txt"))
        full_text = "".join(open(f, 'r', encoding='utf-8').read() + "\n\n" for f in text_files)
        full_text_tokens = tokenizer.encode(full_text, add_special_tokens=False)

        context_lengths = [int(x) for x in torch.linspace(1000, max_length, context_intervals)]
        depths = [float(x) for x in torch.linspace(0, 1, depth_intervals)]
        results_list = []

        for length in context_lengths:
            for depth in depths:
                if limit is not None and len(results_list) >= limit:
                    accuracy = sum(r['score'] for r in results_list) / len(results_list) if results_list else 0
                    return {"needlehaystack": {"overall_accuracy": accuracy}}


                if device == "cuda":
                    torch.cuda.synchronize()
                    mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)

                    print(f"\n" + "=" * 50)
                    print(f"[New Task] Length: {length:<4} | Depth: {depth:.2f}")
                    print(f"[VRAM Base] Allocated: {mem_allocated:.2f} GB | Reserved: {mem_reserved:.2f} GB")

                    if mem_allocated > 18.0:
                        print(f"[WARNING] Suspected memory leak detected! Baseline occupancy is abnormally high!")
                    print("=" * 50)

                context_tokens = full_text_tokens[:length]
                insert_idx = int(depth * len(context_tokens))

                context_part1 = tokenizer.decode(context_tokens[:insert_idx])
                context_part2 = tokenizer.decode(context_tokens[insert_idx:])
                context_with_needle = f"{context_part1}\n{needle}\n{context_part2}"

                prompt = f"Context:\n{context_with_needle}\n\nQuestion: {question}\nAnswer:"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                custom_cache = self.model_wrapper._setup_cache_and_hooks()

                with torch.no_grad():
                    output_ids = model.generate(
                        inputs.input_ids,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        past_key_values=custom_cache,
                        use_cache=True
                    )
                self._cleanup_cache_and_hooks(custom_cache)

                generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).lower()

                score = 1 if "dolores park" in response or "sandwich" in response else 0
                results_list.append({"length": length, "depth": depth, "score": score})
                print(f"\n[Length: {length:<4} | Depth: {depth:.2f}] Score: {score}")

        accuracy = sum(r['score'] for r in results_list) / len(results_list) if results_list else 0
        # per-position dump so methods can be compared with an exact paired test
        # (the summary CSV only carries the mean)
        save_dir = self.args.get("save_dir")
        if save_dir and results_list:
            import json
            import re
            method = self.args.get("method_name", "unknown")
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", method)[:100]
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"persample_niah_{safe}.json")
            with open(path, "w") as f:
                json.dump({"task": "niah", "method": method, "results": results_list}, f)
            print(f"    [per-sample NIAH results -> {path}]")
        return {"needlehaystack": {"overall_accuracy": accuracy}}
