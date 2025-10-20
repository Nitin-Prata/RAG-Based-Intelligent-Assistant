import os
import json
from typing import List, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _concat_fields(row: Dict, text_fields: List[str]) -> str:
    parts = []
    for field in text_fields:
        value = row.get(field)
        if value is None:
            continue
        parts.append(str(value))
    return "\n\n".join(parts).strip()


def _chunk_by_token_length(text: str, tokenizer, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)
        if end == len(tokens):
            break
        start = end - overlap if overlap > 0 else end
        start = max(0, start)
    return chunks


def load_and_prepare_corpus(dataset_name: str,
                             dataset_split: str,
                             text_fields: List[str],
                             title_field: str,
                             chunk_size_tokens: int,
                             chunk_overlap_tokens: int,
                             embedding_model_name: str,
                             output_path: str,
                             seed: int = 42) -> pd.DataFrame:
    """Loads dataset, chunks into ~token-sized segments, and saves JSONL with metadata.

    Returns a DataFrame with columns: [id, title, paragraph_id, text].
    """
    _ensure_dir(output_path)

    dset = load_dataset(dataset_name, split=dataset_split)
    dset = dset.shuffle(seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, use_fast=True)

    records: List[Dict] = []
    paragraph_id = 0

    for ex in tqdm(dset, desc="Chunking corpus"):
        title = str(ex.get(title_field, "")) if title_field in ex else ""
        full_text = _concat_fields(ex, text_fields)
        chunks = _chunk_by_token_length(full_text, tokenizer, chunk_size_tokens, chunk_overlap_tokens)
        for idx, chunk in enumerate(chunks):
            rec = {
                "id": f"para{paragraph_id}_chunk{idx}",
                "title": title,
                "paragraph_id": paragraph_id,
                "text": chunk
            }
            records.append(rec)
        paragraph_id += 1

    # Save to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return pd.DataFrame.from_records(records)


def read_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_corpus_dataframe(path: str) -> pd.DataFrame:
    rows = read_jsonl(path)
    return pd.DataFrame.from_records(rows) if rows else pd.DataFrame(columns=["id", "title", "paragraph_id", "text"])
