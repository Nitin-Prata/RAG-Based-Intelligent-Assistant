# Retrieval-Augmented Generation (RAG) for Domain-Specific Q&A

A research-quality RAG system that answers user questions by combining sparse (BM25) and dense (FAISS + sentence-transformers) retrieval with a small free LLM (Flan-T5). Built for reproducibility and hackathon-ready deployment.

## Highlights

- Hybrid retrieval: **BM25 + dense** with weighted fusion
- Small, free models: **multi-qa-MiniLM-L6** for embeddings, **Flan-T5** for generation
- Clean pipeline modules under `src/`
- Research notebook with metrics, plots, and qualitative analyses

## Quick Start

1. Create and activate a Python 3.10+ virtual environment (already present as `venv/`).
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook notebooks/RAG_Pipeline.ipynb
   ```

## Dataset

Uses a small open dataset by default (`ag_news`) via Hugging Face Datasets. You can switch to another dataset (e.g., `wiki_snippets`) by editing `config.json`.

## Configuration

Adjust hyperparameters and paths in `config.json`:

- `bm25_weight`, `dense_weight`, `top_k`, `chunk_size_tokens`, `generation_model_name`, etc.

## Project Structure

```
RAG-System/
  ├─ data/                # corpus, embeddings, FAISS index, eval set
  ├─ notebooks/           # research notebook
  ├─ src/                 # pipeline modules
  │  ├─ data_loader.py
  │  ├─ retriever.py
  │  ├─ generator.py
  │  └─ evaluate.py
  ├─ outputs/             # metrics, plots, examples
  ├─ config.json
  ├─ requirements.txt
  └─ README.md
```

## Results

The notebook produces:

- Exact Match (EM) and F1 metrics
- Retrieval score distribution plots
- Comparison of baseline vs. RAG
- Example questions with retrieved contexts and answers

## Reproducibility

- Deterministic seeds where possible
- Single-command setup and notebook to reproduce experiments

## License

MIT
