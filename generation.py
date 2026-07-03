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
    """Generative QA engine using an instruction-tuned causal LLM.

    - Loads a model (Llama/Mistral) via `transformers`.
    - Uses a strict prompt template that forces the model to answer using ONLY the provided context.
    - Appends a SOURCES metadata section to the final output.
    """

    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if "cuda" in self.device else torch.float32, device_map="auto")

    PROMPT_TEMPLATE = (
        "You are a precise research assistant. Use ONLY the provided CONTEXT to answer the QUESTION. "
        "Do NOT invent facts, speculate, or use knowledge beyond the CONTEXT. If the CONTEXT is insufficient, reply 'Insufficient information in the provided context.'\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n\n"
        "INSTRUCTIONS:\n- Synthesize a concise, accurate answer using ONLY the context.\n- Inline-cite relevant chunks by number (e.g., [1], [2]) where appropriate.\n- After the answer, list a SOURCES section mapping each used citation number to its metadata.\n\n"
    )

    def synthesize(self, question: str, chunks: List[SourceChunk], max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        # Build numbered context
        numbered: List[str] = []
        for i, c in enumerate(chunks, start=1):
            meta = c.metadata or {}
            meta_str = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else ""
            numbered.append(f"[{i}] {c.text}\n--METADATA: {meta_str}")

        context_text = "\n\n".join(numbered)
        prompt = self.PROMPT_TEMPLATE.format(context=context_text, question=question)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False)
        answer = self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

        # Append SOURCES listing for all provided chunks (model is instructed to cite by number)
        sources_lines = [f"[{i}] {c.metadata}" for i, c in enumerate(chunks, start=1)]
        return f"{answer}\n\nSOURCES:\n" + "\n".join(sources_lines)


__all__ = ["Generator", "SourceChunk"]
