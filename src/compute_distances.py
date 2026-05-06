import pickle
import numpy as np
import pandas as pd
import time
import os
from sklearn.metrics.pairwise import cosine_distances

INPUT = 'outputs/embeddings_all_precisions.pkl'
OUTPUT = 'outputs/distances.pkl'

TARGET_WORDS = ['rue', 'roue', 'cache', 'sous', 'sur', 'roule',
                'tsarine', 'hier', 'tulle', 'reste', 'gabriel',
                'juxtaposer', 'pas', "j'en", 'chie', 'divan']

def compute_intra_inter_distances(embeddings, records):
    df = pd.DataFrame({
        'speaker': [r['speaker'] for r in records],
        'word': [r['word'] for r in records],
        'idx': range(len(records))
    })

    intra_distances = []
    inter_distances = []

    for word in TARGET_WORDS:
        word_df = df[df['word'] == word]
        word_embs = embeddings[word_df['idx'].values]
        speakers = word_df['speaker'].values
        dist_matrix = cosine_distances(word_embs)

        for i in range(len(word_df)):
            for j in range(i+1, len(word_df)):
                d = dist_matrix[i, j]
                if speakers[i] == speakers[j]:
                    intra_distances.append(d)
                else:
                    inter_distances.append(d)

    return np.array(intra_distances), np.array(inter_distances)

def main():
    with open(INPUT, 'rb') as f:
        data = pickle.load(f)

    records = data['records']
    results = {}

    for name in ['float64', 'float32', 'float16', 'int8']:
        embs = data[name]
        start = time.time()
        intra, inter = compute_intra_inter_distances(embs, records)
        elapsed = time.time() - start
        results[name] = {
            'intra': intra,
            'inter': inter,
            'time': elapsed
        }
        print(f"{name}: intra={intra.mean():.8f}, inter={inter.mean():.8f}, "
              f"ratio={inter.mean()/intra.mean():.6f}, time={elapsed:.3f}s")

    os.makedirs('outputs', exist_ok=True)
    with open(OUTPUT, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved to {OUTPUT}")

if __name__ == '__main__':
    main()