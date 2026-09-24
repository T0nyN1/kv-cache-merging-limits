from typing import Dict, Any

from .base_evaluator import BaseEvaluator
from .registry import register_task


def _parse_wiki_docs(wiki_docs: str) -> list[int] | None:
    if wiki_docs is None:
        return None

    doc_spec = str(wiki_docs).strip()
    if not doc_spec:
        return None

    sample_ids = []
    seen = set()

    for chunk in doc_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        if "-" in chunk:
            bounds = [part.strip() for part in chunk.split("-", 1)]
            if len(bounds) != 2 or not bounds[0].isdigit() or not bounds[1].isdigit():
                raise ValueError(f"Invalid --wiki-docs range: {chunk}")

            start_doc = int(bounds[0])
            end_doc = int(bounds[1])
            if start_doc <= 0 or end_doc <= 0:
                raise ValueError("--wiki-docs uses 1-based document numbers; values must be >= 1")
            if end_doc < start_doc:
                raise ValueError(f"Invalid --wiki-docs range: {chunk}")

            doc_numbers = range(start_doc, end_doc + 1)
        else:
            if not chunk.isdigit():
                raise ValueError(f"Invalid --wiki-docs document id: {chunk}")

            doc_number = int(chunk)
            if doc_number <= 0:
                raise ValueError("--wiki-docs uses 1-based document numbers; values must be >= 1")

            doc_numbers = [doc_number]

        for doc_number in doc_numbers:
            sample_id = doc_number - 1
            if sample_id not in seen:
                seen.add(sample_id)
                sample_ids.append(sample_id)

    if not sample_ids:
        raise ValueError("--wiki-docs did not contain any valid document ids")

    return sample_ids


@register_task("wikitext")
class WikitextEvaluator(BaseEvaluator):

    def evaluate(self) -> Dict[str, Any]:
        from lm_eval import simple_evaluate
        print("\n[*] Running lm-eval task: ['wikitext']")

        limit = self.args.get('limit', None)
        wiki_docs = self.args.get('wiki_docs', None)
        samples = None

        sample_ids = _parse_wiki_docs(wiki_docs)
        if sample_ids is not None:
            samples = {"wikitext": sample_ids}
            limit = None
            print(f"[*] Wikitext docs: {wiki_docs} -> sample ids {sample_ids}")

        results = simple_evaluate(
            model=self.model_wrapper,
            tasks=["wikitext"],
            limit=limit,
            samples=samples,
        )
        return results['results']
