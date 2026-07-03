from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SourceChunk:
    id: str
    text: str
    metadata: Dict[str, Any]


class Generator:
    """Generative QA pipeline using an instruction-tuned transformer model."""

    PROMPT_TEMPLATE = (
        "SYSTEM:\nYou are a research assistant. You are provided with context chunks from a PDF. "
        "Synthesize an answer ONLY from the provided text. If the answer is not in the context, state 'Information not found in document.' "
        "Append the source metadata to each sentence.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n\n"
        "INSTRUCTIONS:\n- Answer only from the provided context.\n- If context is insufficient, say 'Information not found in document.'\n"
        "- Cite each sentence with its source metadata.\n\n"
    )

    def __init__(self, model_name: str = "meta-llama/Llama-3-8b-instruct", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        )

    def synthesize(self, question: str, chunks: List[SourceChunk], max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        context_text = self._build_context(chunks)
        prompt = self.PROMPT_TEMPLATE.format(context=context_text, question=question)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _build_context(self, chunks: List[SourceChunk]) -> str:
        formatted: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            source = self._format_metadata(chunk.metadata)
            formatted.append(f"[{idx}] {chunk.text}\nSOURCE: {source}")
        return "\n\n".join(formatted)

    @staticmethod
    def _format_metadata(metadata: Dict[str, Any]) -> str:
        if isinstance(metadata, dict):
            source = metadata.get("source") or metadata.get("source_id") or "unknown"
            page = metadata.get("page")
            if page is not None:
                return f"{source}:page={page}"
            return str(source)
        return str(metadata)


__all__ = ["Generator", "SourceChunk"]
