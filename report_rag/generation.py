from __future__ import annotations

from typing import Sequence

from .text_utils import normalize_text


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
        "不要编造资料中没有的信息；如果资料不足，请明确说明。\n\n"
        f"资料:\n{context}\n\n"
        f"问题: {query}\n\n"
        "回答:"
    )


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

