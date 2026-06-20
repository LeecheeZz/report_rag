from __future__ import annotations

import re
from typing import Sequence

from .text_utils import normalize_text

UNCERTAIN_ANSWER = "我不确定"
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def build_generation_prompt(
    query: str,
    results: Sequence[dict],
    context_chunks: int,
) -> str:
    contexts = []
    for index, result in enumerate(results[:context_chunks], 1):
        text = normalize_text(result["text"])
        contexts.append(
            f"[{index}] 来源: {result['source']} 页码: {result['pages']}\n{text}"
        )
    context = "\n\n".join(contexts)
    return (
        "你是一个严谨的行业研报问答助手。请只根据给定资料回答问题，"
        "不要编造资料中没有的信息。\n"
        "每个事实性结论后必须标注证据编号，例如（证据[1]）。"
        "只能引用资料中存在的编号，不能引用不存在的编号。"
        f"如果资料不足以回答问题，只回答“{UNCERTAIN_ANSWER}”。\n\n"
        f"资料:\n{context}\n\n"
        f"问题: {query}\n\n"
        "回答:"
    )


def extract_citation_numbers(answer: str) -> set[int]:
    return {int(match) for match in CITATION_PATTERN.findall(answer)}


def is_uncertain_answer(answer: str) -> bool:
    return answer.strip().startswith(UNCERTAIN_ANSWER)


def validate_citations(
    answer: str,
    results: Sequence[dict],
    context_chunks: int,
) -> bool:
    if is_uncertain_answer(answer):
        return True

    citations = extract_citation_numbers(answer)
    max_reference = min(context_chunks, len(results))
    if not citations or max_reference <= 0:
        return False
    return all(1 <= citation <= max_reference for citation in citations)


def format_citation_sources(
    answer: str,
    results: Sequence[dict],
    context_chunks: int,
) -> str:
    citations = sorted(extract_citation_numbers(answer))
    max_reference = min(context_chunks, len(results))
    lines = []
    for citation in citations:
        if 1 <= citation <= max_reference:
            result = results[citation - 1]
            lines.append(
                f"[{citation}] {result['source']}  pages:{result['pages']}  "
                f"chunk_id:{result['chunk_id']}"
            )
    if not lines:
        return answer
    return f"{answer}\n\n引用来源:\n" + "\n".join(lines)


class QwenGenerator:
    def __init__(self, model_path: str, use_fp16: bool = True) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing transformers/torch. Install dependencies before generation."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_cuda = torch.cuda.is_available()
        load_kwargs = {"trust_remote_code": True}
        if use_cuda and use_fp16:
            load_kwargs["torch_dtype"] = torch.float16
        if use_cuda:
            load_kwargs["device_map"] = "auto"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except ImportError:
            load_kwargs.pop("device_map", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        self.model.eval()
        if not hasattr(self.model, "hf_device_map"):
            device = "cuda" if use_cuda else "cpu"
            self.model.to(device)

    def generate_answer(
        self,
        query: str,
        results: Sequence[dict],
        context_chunks: int,
        max_input_tokens: int,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        prompt = build_generation_prompt(query, results, context_chunks)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        device = next(self.model.parameters()).device
        inputs = {name: value.to(device) for name, value in inputs.items()}
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False

        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

