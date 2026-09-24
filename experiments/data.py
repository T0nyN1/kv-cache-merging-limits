"""Local dataset helpers for the OT-KV benchmark.

Everything here is designed to work offline from the HF cache so the bench can
run on a laptop without network access.
"""
import os
import random
import re


def load_wikitext_docs(min_tokens=2000, max_docs=8, tokenizer=None, split="test"):
    """Return long wikitext-2 documents (document-level split), longest first."""
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split=split)
    docs = []
    for row in ds:
        text = row["page"]
        if len(text) < min_tokens * 3:
            continue
        docs.append(text)
    docs.sort(key=len, reverse=True)

    out = []
    for text in docs:
        if tokenizer is not None:
            n = len(tokenizer.encode(text, add_special_tokens=False))
            if n < min_tokens:
                continue
        out.append(text)
        if len(out) >= max_docs:
            break
    return out


NEEDLE_TEMPLATE = "The secret access code for the {key} vault is {value}."
NEEDLE_QUESTION = "\n\nQuestion: What is the secret access code for the {key} vault?\nAnswer: The secret access code for the {key} vault is"

_KEYS = ["amber", "cobalt", "granite", "juniper", "meridian", "obsidian", "quartz", "verdant"]


def build_niah_samples(haystack_docs, tokenizer, context_tokens=3000, depths=(0.15, 0.35, 0.55, 0.75, 0.92), seed=0):
    """Needle-in-a-haystack built from local wikitext text (no download needed).

    A single sentence carrying a random 6-digit code is inserted at a fractional
    depth of an otherwise irrelevant context; the model must read it back.
    """
    rng = random.Random(seed)
    haystack = "\n\n".join(haystack_docs)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", haystack) if len(s.strip()) > 40]

    samples = []
    for depth in depths:
        key = rng.choice(_KEYS)
        value = f"{rng.randint(100000, 999999)}"
        needle = NEEDLE_TEMPLATE.format(key=key, value=value)

        # grow a context of ~context_tokens sentences
        chunk, n_tok = [], 0
        i = rng.randrange(0, max(1, len(sentences) - 400))
        while n_tok < context_tokens and i < len(sentences):
            chunk.append(sentences[i])
            n_tok += len(tokenizer.encode(sentences[i], add_special_tokens=False)) + 1
            i += 1

        insert_at = int(len(chunk) * depth)
        chunk.insert(insert_at, needle)
        context = " ".join(chunk)
        prompt = context + NEEDLE_QUESTION.format(key=key)
        samples.append({"prompt": prompt, "answer": value, "depth": depth, "key": key})
    return samples
