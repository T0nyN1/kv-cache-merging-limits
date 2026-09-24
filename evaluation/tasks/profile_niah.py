from typing import Dict, Any

from .base_evaluator import BaseEvaluator
from .registry import register_task


@register_task("profile_niah")
class ProfileNIAHEvaluator(BaseEvaluator):

    def evaluate(self) -> Dict[str, Any]:
        import torch
        import time
        import os
        import glob
        from transformers import LogitsProcessorList, LogitsProcessor

        print("\n[*] Running System Profiler with NIAH Corpus...")

        tokenizer = self.model_wrapper.tokenizer
        model = self.model_wrapper._model

        device = getattr(self.model_wrapper, 'device', model.device)

        def synchronize_device():
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()

        prompt_length = self.args.get('max_length', 4000)
        generate_length = self.args.get('profiler_gen_length', 128)
        haystack_dir = self.args.get('haystack_dir', "./datasets/PaulGrahamEssays")

        text_files = glob.glob(os.path.join(haystack_dir, "*.txt"))
        if not text_files:
            print(f"[!] Error: No .txt files found in {haystack_dir}.")
            return {"profile_niah": {}}

        full_text = ""
        for f in text_files:
            with open(f, 'r', encoding='utf-8') as file:
                full_text += file.read() + "\n\n"

        full_text_tokens = tokenizer.encode(full_text, add_special_tokens=False)

        while len(full_text_tokens) < prompt_length:
            full_text_tokens += full_text_tokens

        context_tokens = full_text_tokens[:prompt_length]
        real_prompt = tokenizer.decode(context_tokens)
        inputs = tokenizer(real_prompt, return_tensors="pt").to(device)

        print(f"-> Target Prefill Length: {inputs.input_ids.shape[1]} tokens")
        print(f"-> Target Generate Length: {generate_length} tokens")

        print(f"-> Performing Warm-up on {device.type.upper()}...")
        with torch.no_grad():
            _ = model.generate(
                inputs.input_ids[:, :128],
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        synchronize_device()

        # Device-agnostic memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif device.type == 'mps':
            torch.mps.empty_cache()

        custom_cache = self.model_wrapper._setup_cache_and_hooks()

        def get_current_kv_bytes(cache_obj):
            current_bytes = 0
            if cache_obj is not None:
                if hasattr(cache_obj, "layers"):
                    for layer in cache_obj.layers:
                        if hasattr(layer, "keys") and layer.keys is not None:
                            current_bytes += layer.keys.numel() * layer.keys.element_size()
                        if hasattr(layer, "values") and layer.values is not None:
                            current_bytes += layer.values.numel() * layer.values.element_size()
                elif hasattr(cache_obj, "key_cache") and hasattr(cache_obj, "value_cache"):
                    for k in cache_obj.key_cache:
                        if k is not None:
                            current_bytes += k.numel() * k.element_size()
                    for v in cache_obj.value_cache:
                        if v is not None:
                            current_bytes += v.numel() * v.element_size()
            return current_bytes

        class ProfilerTracker(LogitsProcessor):
            def __init__(self, cache_obj):
                self.start_time = None
                self.ttft = None
                self.cache_obj = cache_obj
                self.max_kv_bytes = 0

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                if self.ttft is None and self.start_time is not None:
                    synchronize_device()
                    self.ttft = time.time() - self.start_time

                is_compressed_state = getattr(self.cache_obj, "_prefill_finalized", True)

                if is_compressed_state:
                    current_bytes = get_current_kv_bytes(self.cache_obj)
                    if current_bytes > self.max_kv_bytes:
                        self.max_kv_bytes = current_bytes

                return scores

        tracker = ProfilerTracker(custom_cache)
        logits_processor = LogitsProcessorList([tracker])

        print("-> Running Benchmark...")

        synchronize_device()
        start_time = time.time()
        tracker.start_time = start_time

        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=generate_length,
                min_new_tokens=generate_length,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                past_key_values=custom_cache,
                use_cache=True,
                logits_processor=logits_processor
            )

        final_bytes = get_current_kv_bytes(custom_cache)
        max_bytes = max(tracker.max_kv_bytes, final_bytes)
        kv_cache_mb = max_bytes / (1024 ** 2)

        self._cleanup_cache_and_hooks(custom_cache)

        synchronize_device()
        end_time = time.time()

        total_time = end_time - start_time
        ttft = tracker.ttft if tracker.ttft else total_time
        decode_time = total_time - ttft
        decode_tokens = max(generate_length - 1, 1)

        decode_tps = decode_tokens / decode_time if decode_time > 0 else 0
        overall_tps = generate_length / total_time

        print(f"\n--- Profiling Results ---")
        print(f"-> Context Length: {inputs.input_ids.shape[1]} tokens")
        print(f"-> Total Time: {total_time:.4f} s")
        print(f"-> Time To First Token (TTFT): {ttft:.4f} s")
        print(f"-> Pure Decode Time ({decode_tokens} tokens): {decode_time:.4f} s")
        print(f"-> Pure Decode Throughput: {decode_tps:.2f} tokens/s")
        print(f"-> Overall Throughput: {overall_tps:.2f} tokens/s")
        print(f"-> Peak Compressed KV Cache Memory: {kv_cache_mb:.2f} MB")
        print("---------------------------------------")

        return {
            "profile_niah": {
                "context_length_tokens": inputs.input_ids.shape[1],
                "total_time_s": total_time,
                "ttft_s": ttft,
                "pure_decode_time_s": decode_time,
                "pure_decode_throughput_tps": decode_tps,
                "overall_throughput_tps": overall_tps,
                "kv_cache_mb": kv_cache_mb,
            }
        }