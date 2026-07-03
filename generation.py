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
        "SYSTEM:\nYou are a research assistant. You are provided with context chunks from a PDF. "
        "Synthesize an answer ONLY using the provided text. If the answer is not in the context, state 'Information not found in document.' "
        "Append the source metadata to each sentence.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n\n"
        "INSTRUCTIONS:\n- Answer only from the provided context.\n- If context is insufficient, say 'Information not found in document.'\n"
        "- Cite each sentence with the corresponding source metadata.\n\n"
    )

    def synthesize(self, question: str, chunks: List[SourceChunk], max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        # Build numbered context
        numbered: List[str] = []
        for i, c in enumerate(chunks, start=1):
            meta = c.metadata or {}
            if isinstance(meta, dict):
                source = str(meta.get("source", "")).strip()
                page = meta.get("page")
                if source:
                    meta_str = f"{source}"
                    if page is not None:
                        meta_str += f":page={page}"
                else:
                    meta_str = str(meta)
            else:
                meta_str = str(meta)
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
