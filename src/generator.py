from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class T5Generator:
    def __init__(self, model_name: str = "google/flan-t5-base", device: str | None = None, max_new_tokens: int = 128, temperature: float = 0.0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @staticmethod
    def build_prompt(question: str, context_chunks: List[str]) -> str:
        context = "\n\n".join([f"[Doc {i+1}] {c}" for i, c in enumerate(context_chunks)])
        return f"Answer the question using the context. If unknown, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
