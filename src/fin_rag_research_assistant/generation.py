"""Citation-grounded generation with a small local instruction-tuned model.

The prompt forces answers to cite excerpt numbers and defines the refusal
string ``INSUFFICIENT_CONTEXT``. Generation metrics here are explicitly
PROXIES (citation validity, lexical groundedness, refusal behaviour) — they are
not a substitute for human evaluation, and hallucination risk remains.
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


def parse_citations(answer: str, n_excerpts: int) -> list[int]:
    """Extract cited excerpt numbers; invalid ones are dropped.

    Returns the deduplicated list of citations that point at excerpt ids that
    actually exist (1..n_excerpts), in order of first appearance.
    """
    cited: list[int] = []
    for match in EXCERPT_PATTERN.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= n_excerpts and number not in cited:
            cited.append(number)
    return cited


def lexical_groundedness(answer: str, excerpts: list[str], cited: list[int]) -> float:
    """Token-overlap F1 between the answer and the cited excerpts (heuristic).

    PROXY metric: a high value means the answer shares vocabulary with the
    cited evidence; it does NOT certify factual correctness.
    """
    if not cited:
        return 0.0
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "are",
        "was", "were", "on", "as", "with", "that", "this", "it", "by", "at",
        "from", "be", "its", "our", "their", "we", "you", "not", "but", "also",
    }
    answer_tokens = {
        tok.lower() for tok in re.findall(r"[A-Za-z0-9]+", answer)
    } - stop
    evidence_tokens: set[str] = set()
    for idx in cited:
        evidence_tokens |= {
            tok.lower() for tok in re.findall(r"[A-Za-z0-9]+", excerpts[idx - 1])
        }
    evidence_tokens -= stop
    if not answer_tokens or not evidence_tokens:
        return 0.0
    overlap = len(answer_tokens & evidence_tokens)
    precision = overlap / len(answer_tokens)
    recall = overlap / len(evidence_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class GenerationResult:
    qid: str
    question: str
    prompt_excerpts: list[str]
    retrieved_chunk_ids: list[str]  # provenance: excerpt i -> retrieved chunk id
    raw_answer: str
    citations: list[int]
    citation_valid: bool  # citations exist (point at retrieved excerpts, in range)
    used_refusal: bool
    groundedness_f1: float
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

    Citation validity: a non-refusal answer must cite at least one excerpt and
    every cited id must point at an excerpt that exists AND was actually
    retrieved (the excerpts passed to the prompt ARE the retrieved top-k, so
    "in range" is the operational check; out-of-range ids are fabricated
    citations).
    """
    citations = parse_citations(raw_answer, len(excerpts))
    used_refusal = is_refusal(raw_answer)
    valid = True
    if not used_refusal and not citations:
        valid = False  # asserted a fact with zero citations
    if any(cid < 1 or cid > len(excerpts) for cid in citations):
        valid = False  # fabricated citation id (outside the retrieved range)
    grounded = lexical_groundedness(raw_answer, excerpts, citations)
    return GenerationResult(
        qid=qid,
        question=question,
        prompt_excerpts=excerpts,
        retrieved_chunk_ids=list(retrieved_ids),
        raw_answer=raw_answer,
        citations=citations,
        citation_valid=valid,
        used_refusal=used_refusal,
        groundedness_f1=grounded,
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
