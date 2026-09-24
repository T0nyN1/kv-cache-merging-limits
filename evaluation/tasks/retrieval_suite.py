"""RULER-style synthetic retrieval suite.

The public llamastack/ruler mirror carries 1-12 examples per task at 8k —
too few to measure anything — so this evaluator regenerates the four core
RULER recipes (Hsieh et al., 2024) deterministically from seeds, using the
same haystack corpus as the NIAH task:

  niah_single     one magic number, one question (RULER niah_single_2)
  niah_multikey   four keyed numbers, ask for one — three in-context distractors
  niah_multiquery four keyed numbers, ask for all four
  vt              variable-tracking: one 4-hop assignment chain plus two
                  distractor chains; name every variable holding the value

Scoring is RULER's: fraction of expected answer strings present in the
generated text. Per-method per-sample dumps mirror the LongBench evaluator so
comparisons stay paired.
"""

import glob
import json
import os
import random
import re
import string
from typing import Dict, Any, List, Tuple

from .base_evaluator import BaseEvaluator
from .registry import register_task

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
         "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
         "xray yankee zulu amber basil cedar dahlia elm fern garnet hazel iris jade "
         "kelp lotus maple nectar opal pearl quartz rowan sage thistle umber violet "
         "willow yarrow zephyr").split()

TASKS = ("niah_single", "niah_multikey", "niah_multiquery", "vt")


def _needle(key: str, num: str) -> str:
    return f"One of the special magic numbers for {key} is: {num}."


def _build_niah(rng: random.Random, n_needles: int, n_queries: int
                ) -> Tuple[List[str], str, List[str]]:
    keys = rng.sample(WORDS, n_needles)
    nums = [str(rng.randint(1000000, 9999999)) for _ in keys]
    needles = [_needle(k, n) for k, n in zip(keys, nums)]
    q_idx = rng.sample(range(n_needles), n_queries)
    if n_queries == 1:
        i = q_idx[0]
        question = (f"What is the special magic number for {keys[i]} mentioned "
                    f"in the provided text?")
        answers = [nums[i]]
    else:
        named = ", ".join(keys[i] for i in q_idx)
        question = (f"What are the special magic numbers for {named} mentioned "
                    f"in the provided text?")
        answers = [nums[i] for i in q_idx]
    return needles, question, answers


def _build_vt(rng: random.Random) -> Tuple[List[str], str, List[str]]:
    def chain(n_hops):
        names = ["".join(rng.choices(string.ascii_uppercase, k=3)) + str(rng.randint(10, 99))
                 for _ in range(n_hops)]
        value = str(rng.randint(10000, 99999))
        stmts = [f"VAR {names[0]} = {value}"]
        stmts += [f"VAR {names[i]} = VAR {names[i - 1]}" for i in range(1, n_hops)]
        return names, value, stmts
    tgt_names, tgt_value, tgt_stmts = chain(4)
    stmts = list(tgt_stmts)
    for _ in range(2):
        _, _, s = chain(4)
        stmts += s
    rng.shuffle(stmts)
    question = (f"Find all variables that are assigned the value {tgt_value} in "
                f"the text above, either directly or through another variable. "
                f"Answer with the variable names.")
    return stmts, question, tgt_names


@register_task("retrieval")
class RetrievalSuiteEvaluator(BaseEvaluator):

    def _dump_per_sample(self, task, per_sample):
        save_dir = self.args.get("save_dir")
        if not save_dir or not per_sample:
            return
        method = self.args.get("method_name", "unknown")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", method)[:100]
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"persample_ret_{task}_{safe}.json")
        with open(path, "w") as f:
            json.dump({"task": task, "method": method, "scores": per_sample}, f)
        print(f"    [per-sample scores -> {path}]")

    def evaluate(self) -> Dict[str, Any]:
        import torch
        from tqdm import tqdm

        tokenizer = self.model_wrapper.tokenizer
        model = self.model_wrapper._model
        max_length = int(self.args.get("max_length", 8000))
        n_samples = int(self.args.get("limit") or 25)
        tasks_arg = str(self.args.get("retrieval_tasks", "all"))
        tasks = TASKS if tasks_arg in ("all", "") else tuple(t.strip() for t in tasks_arg.split(","))

        files = sorted(glob.glob(os.path.join(
            self.args.get("haystack_dir", "datasets/niah/PaulGrahamEssays"), "*.txt")))
        corpus = "".join(open(f, encoding="utf-8").read() + "\n\n" for f in files)
        corpus_ids = tokenizer.encode(corpus, add_special_tokens=False)

        results = {}
        for task in tasks:
            per_sample = []
            for si in tqdm(range(n_samples), desc=f"retrieval/{task}"):
                rng = random.Random(10_000 * hash(task) % 999983 + si)
                if task == "niah_single":
                    inserts, question, answers = _build_niah(rng, 1, 1)
                elif task == "niah_multikey":
                    inserts, question, answers = _build_niah(rng, 4, 1)
                elif task == "niah_multiquery":
                    inserts, question, answers = _build_niah(rng, 4, 4)
                elif task == "vt":
                    inserts, question, answers = _build_vt(rng)
                else:
                    raise ValueError(f"unknown retrieval task {task}")

                ctx_budget = max_length - 420
                start = rng.randint(0, max(0, len(corpus_ids) - ctx_budget - 1))
                hay = tokenizer.decode(corpus_ids[start:start + ctx_budget])
                # insert statements at random depths, deepest first so earlier
                # offsets stay valid
                text = hay
                for stmt in sorted(inserts, key=lambda _: rng.random()):
                    at = rng.randint(0, len(text))
                    text = text[:at] + "\n" + stmt + "\n" + text[at:]

                prompt = (f"Read the following text carefully.\n\n{text}\n\n"
                          f"Question: {question}\nAnswer:")
                messages = [{"role": "user", "content": prompt}]
                input_tensor = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt",
                )["input_ids"].to(model.device)
                if input_tensor.shape[1] > max_length:
                    half = max_length // 2
                    input_tensor = torch.cat(
                        [input_tensor[:, :half], input_tensor[:, -half:]], dim=1)

                cache = self.model_wrapper._setup_cache_and_hooks()
                with torch.no_grad():
                    out = model.generate(
                        input_tensor,
                        attention_mask=torch.ones_like(input_tensor),
                        max_new_tokens=96, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        past_key_values=cache, use_cache=True)
                self._cleanup_cache_and_hooks(cache)
                resp = tokenizer.decode(out[0][input_tensor.shape[1]:],
                                        skip_special_tokens=True)
                hit = sum(1 for a in answers if a in resp) / len(answers)
                per_sample.append(hit)

            score = sum(per_sample) / len(per_sample)
            results[task] = {"score": score, "samples": n_samples}
            self._dump_per_sample(task, per_sample)
            print(f"-> retrieval/{task}: {score * 100:.1f}%")

        return {"retrieval": results}
