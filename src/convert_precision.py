import pickle
import numpy as np
import os

INPUT = 'outputs/embeddings_float32.pkl'
OUTPUT = 'outputs/embeddings_all_precisions.pkl'

def quantize_to_8bit(embeddings_f32):
    """Linear quantization from float32 to uint8 (per-dimension)."""
    min_vals = embeddings_f32.min(axis=0)
    max_vals = embeddings_f32.max(axis=0)
    scale = (max_vals - min_vals) / 255.0
    scale = np.where(scale == 0, 1.0, scale)
    quantized = np.round((embeddings_f32 - min_vals) / scale).astype(np.uint8)
    return quantized, min_vals, scale

def dequantize(quantized, min_vals, scale):
    """Reconstruct float32 from uint8."""
    return quantized.astype(np.float32) * scale + min_vals

def main():
    with open(INPUT, 'rb') as f:
        records = pickle.load(f)

    emb_f32 = np.array([r['embedding'] for r in records])

    emb_f64 = emb_f32.astype(np.float64)
    emb_f16 = emb_f32.astype(np.float16)
    emb_int8, min_vals, scale = quantize_to_8bit(emb_f32)
    emb_int8_reconstructed = dequantize(emb_int8, min_vals, scale)

    result = {
        'records': records,
        'float64': emb_f64,
        'float32': emb_f32,
        'float16': emb_f16,
        'int8': emb_int8_reconstructed,
        'int8_raw': emb_int8,
        'int8_min': min_vals,
        'int8_scale': scale
    }

    os.makedirs('outputs', exist_ok=True)
    with open(OUTPUT, 'wb') as f:
        pickle.dump(result, f)

    print(f"Storage comparison:")
    print(f"  float64: {emb_f64.nbytes/1e6:.2f} MB")
    print(f"  float32: {emb_f32.nbytes/1e6:.2f} MB")
    print(f"  float16: {emb_f16.nbytes/1e6:.2f} MB")
    print(f"  int8:    {emb_int8.nbytes/1e6:.2f} MB")
    print(f"Saved to {OUTPUT}")

if __name__ == '__main__':
    main()