"""Citation-grounded generation with a small local instruction-tuned model.

The prompt forces answers to cite excerpt numbers and defines the refusal
string ``INSUFFICIENT_CONTEXT``. Generation metrics here are explicitly
PROXIES (citation existence, lexical groundedness, refusal behaviour) — they
are not a substitute for human evaluation, and hallucination risk remains.

Citation metrics measure EXISTENCE and RANGE only: "citation_existence_rate"
checks that a non-refusal answer cites at least one excerpt that was actually
provided to the model, and "fabricated_citation_rate" counts answers that emit
a bracketed id outside the provided range. NEITHER verifies that the cited
passage supports the claim — no entailment/NLI model is run.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from fin_rag_research_assistant import config

REFUSAL_STRING = "INSUFFICIENT_CONTEXT"
EXCERPT_PATTERN = re.compile(r"\[(\d+)\]")  # citations like [1], [2]


def is_refusal(answer: str) -> bool:
    """True when the model reply is (or contains) the exact refusal string."""
    return REFUSAL_STRING in answer.strip()


def build_prompt(question: str, excerpts: list[str]) -> list[dict, ...]:
    """Chat-template messages: numbered excerpts + question + fixed instruction.

    The instruction string is included verbatim (unit-tested).
    """
    numbered = "\n\n".join(
        f"[{i + 1}] {text}" for i, text in enumerate(excerpts)
    )
    system = (
        "You are a research assistant answering questions about Federal "
        "Reserve Beige Book economic reports. "
        + config.GEN_INSTRUCTION
        + "."
    )
    user = (
        f"Excerpts:\n\n{numbered}\n\nQuestion: {question}\n\n"
        f"{config.GEN_CITATION_REMINDER}\n\nAnswer:"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def find_citations(answer: str, n_excerpts: int) -> tuple[list[int], list[int]]:
    """Scan bracketed citation ids BEFORE any filtering.

    Returns ``(valid_ids, fabricated_ids)`` where

    - ``valid_ids`` is the deduplicated, order-of-appearance list of ids that
      point at excerpts that actually exist (1..n_excerpts), and
    - ``fabricated_ids`` is the deduplicated, order-of-appearance list of ids
      outside that range (the model cited an excerpt it was never given).

    Only ``valid_ids`` may be used as evidence downstream; ``fabricated_ids``
    are recorded so the fabricated-citation rate can be reported.
    """
    valid: list[int] = []
    fabricated: list[int] = []
    for match in EXCERPT_PATTERN.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= n_excerpts:
            if number not in valid:
                valid.append(number)
        elif number not in fabricated:
            fabricated.append(number)
    return valid, fabricated


def parse_citations(answer: str, n_excerpts: int) -> list[int]:
    """Extract cited excerpt numbers; invalid ones are dropped.

    Backward-compatible thin wrapper over :func:`find_citations`: returns only
    the deduplicated list of citations that point at excerpt ids that actually
    exist (1..n_excerpts), in order of first appearance.
    """
    return find_citations(answer, n_excerpts)[0]


def answer_precision(answer: str, excerpts: list[str], cited: list[int]) -> float:
    """Fraction of the answer's non-stopword tokens that appear in the cited
    excerpts (precision side of the groundedness F1).

    PROXY metric: the denominator is ONLY the answer's unique non-stopword
    tokens, so — unlike the F1 below — it does not shrink merely because the
    cited 800-token excerpts contain many other words. It is reported as a
    separate, more denominator-robust companion to the F1.
    """
    if not cited:
        return 0.0
    answer_tokens = _answer_tokens(answer)
    evidence_tokens = _evidence_tokens(excerpts, cited)
    if not answer_tokens or not evidence_tokens:
        return 0.0
    overlap = len(answer_tokens & evidence_tokens)
    return overlap / len(answer_tokens)


def lexical_groundedness(answer: str, excerpts: list[str], cited: list[int]) -> float:
    """Token-overlap F1 between the answer and the cited excerpts (heuristic).

    PROXY metric: a high value means the answer shares vocabulary with the
    cited evidence; it does NOT certify factual correctness.

    Documented denominator limitation: the recall denominator is the set of ALL
    unique non-stopword tokens of the cited excerpts, which here are full
    800-token chunks — so recall is structurally tiny and the F1 should always
    be read alongside :func:`answer_precision` (same numerator, answer-only
    denominator).
    """
    if not cited:
        return 0.0
    answer_tokens = _answer_tokens(answer)
    evidence_tokens = _evidence_tokens(excerpts, cited)
    if not answer_tokens or not evidence_tokens:
        return 0.0
    overlap = len(answer_tokens & evidence_tokens)
    precision = overlap / len(answer_tokens)
    recall = overlap / len(evidence_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _answer_tokens(answer: str) -> set[str]:
    stop = _STOPWORDS
    return {tok.lower() for tok in re.findall(r"[A-Za-z0-9]+", answer)} - stop


def _evidence_tokens(excerpts: list[str], cited: list[int]) -> set[str]:
    stop = _STOPWORDS
    evidence_tokens: set[str] = set()
    for idx in cited:
        evidence_tokens |= {
            tok.lower() for tok in re.findall(r"[A-Za-z0-9]+", excerpts[idx - 1])
        }
    return evidence_tokens - stop


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "are",
    "was", "were", "on", "as", "with", "that", "this", "it", "by", "at",
    "from", "be", "its", "our", "their", "we", "you", "not", "but", "also",
}


@dataclass
class GenerationResult:
    qid: str
    question: str
    prompt_excerpts: list[str]
    retrieved_chunk_ids: list[str]  # provenance: excerpt i -> retrieved chunk id
    raw_answer: str
    citations: list[int]  # in-range ids (the only ones used as evidence)
    fabricated_citations: list[int]  # emitted ids outside the provided range
    citation_existence: bool  # non-refusal answer cites >= 1 provided excerpt
    citation_valid: bool  # existence AND no fabricated ids (refusals are valid)
    used_refusal: bool
    groundedness_f1: float
    answer_precision: float
    latency_s: float
    n_prompt_chars: int


def generate_answer(
    question: str,
    excerpts: list[str],
    model,
    tokenizer,
    max_new_tokens: int = config.GEN_MAX_NEW_TOKENS,
) -> tuple[str, float]:
    """Run one grounded generation; returns (raw_text, latency_seconds).

    Caller supplies a loaded transformers model + tokenizer (heavy import
    stays with the caller so this module imports without torch).
    """
    import torch  # guarded: only called when the rag stack is installed

    messages = build_prompt(question, excerpts)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic greedy decoding for reproducibility
        )
    latency = time.perf_counter() - start
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, latency


def evaluate_generation_result(
    qid: str,
    question: str,
    excerpts: list[str],
    retrieved_ids: list[str],
    raw_answer: str,
    latency_s: float,
) -> GenerationResult:
    """Score one raw answer with the proxy metrics.

    Citation metrics are EXISTENCE/RANGE checks, NOT entailment: no model
    verifies that the cited passage supports the claim.

    - ``citation_existence``: a non-refusal answer must cite at least one
      excerpt that was actually provided (the excerpts passed to the prompt ARE
      the retrieved top-k, so "in range" is the operational check).
    - ``fabricated_citations``: bracketed ids emitted by the model that point
      OUTSIDE the provided range (counted before filtering; they are never used
      as evidence). Any fabricated id makes ``citation_valid`` False.
    - A refusal makes no factual claim, so it needs no citation.
    """
    citations, fabricated = find_citations(raw_answer, len(excerpts))
    used_refusal = is_refusal(raw_answer)
    existence = used_refusal or bool(citations)
    valid = existence and not fabricated
    grounded = lexical_groundedness(raw_answer, excerpts, citations)
    precision = answer_precision(raw_answer, excerpts, citations)
    return GenerationResult(
        qid=qid,
        question=question,
        prompt_excerpts=excerpts,
        retrieved_chunk_ids=list(retrieved_ids),
        raw_answer=raw_answer,
        citations=citations,
        fabricated_citations=fabricated,
        citation_existence=existence,
        citation_valid=valid,
        used_refusal=used_refusal,
        groundedness_f1=grounded,
        answer_precision=precision,
        latency_s=latency_s,
        n_prompt_chars=sum(len(e) for e in excerpts),
    )


def _row_field(row: GenerationResult | dict, field: str):
    """Read a field from either a GenerationResult or its serialized dict."""
    if isinstance(row, dict):
        return row[field]
    return getattr(row, field)


def refusal_correctness(results: list[GenerationResult | dict], expected: dict[str, bool]) -> dict:
    """Refusal behaviour summary: did the model refuse when it should?"""
    rows = []
    for res in results:
        if _row_field(res, "qid") not in expected:
            continue
        used = bool(_row_field(res, "used_refusal"))
        qid = _row_field(res, "qid")
        rows.append(
            {
                "qid": qid,
                "expected_refusal": expected[qid],
                "used_refusal": used,
                "correct": used == expected[qid],
            }
        )
    correct = sum(1 for r in rows if r["correct"])
    return {
        "rows": rows,
        "n": len(rows),
        "n_correct": correct,
        "accuracy": correct / len(rows) if rows else float("nan"),
    }


def load_generation_model(model_id: str = config.GEN_MODEL_ID):
    """Load Qwen2.5-0.5B-Instruct on CPU (heavy imports live here).

    transformers >= 5 renamed ``torch_dtype`` to ``dtype``; we try the plain
    call first (default CPU dtype) and fall back to explicit kwargs for 4.x.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    except TypeError:  # pragma: no cover - transformers 4.x path
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="float32")
    model.eval()
    return model, tokenizer
