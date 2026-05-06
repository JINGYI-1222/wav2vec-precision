# wav2vec Precision Analysis

This project investigates how reducing numerical precision affects the geometry of wav2vec2 speech representations and whether it alters conclusions about inter- and intra-speaker variability.

## Dataset

Russian–French interference corpus (19 speakers, 16 target words, ~6 repetitions each).
Download from: https://www.ortolang.fr/market/corpora/ru-fr_interference/v1

Place the data at: `data/ru-fr_interference/`

## Pipeline

This project uses DVC to manage the pipeline. To reproduce:

```bash
pip install -r requirements.txt
dvc repro
```

## Stages

1. `extract_features` – Extract wav2vec2 embeddings (mean pooling over word segments)
2. `convert_precision` – Convert to float64, float32, float16, and 8-bit quantisation
3. `compute_distances` – Compute intra/inter-speaker cosine distances
4. `visualize` – Generate figures

## Results

| Precision | Storage | Intra mean | Inter mean | Ratio |
|-----------|---------|------------|------------|-------|
| float64 | 13.3 MB | 0.06740844 | 0.12577885 | 1.8659 |
| float32 | 6.7 MB | 0.06740844 | 0.12577885 | 1.8659 |
| float16 | 3.3 MB | 0.06740831 | 0.12577860 | 1.8659 |
| int8 | 1.7 MB | 0.06742129 | 0.12579292 | 1.8658 |