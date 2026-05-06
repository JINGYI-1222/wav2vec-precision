import os
import pickle
import pandas as pd
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Model, AutoProcessor
from tqdm import tqdm

# Paths
BASE = 'data/ru-fr_interference/2/'
WAV_DIR = BASE + 'wav_et_textgrids/FRcorp_textgrids_only/'
OUTPUT = 'outputs/embeddings_float32.pkl'

TARGET_WORDS = ['rue', 'roue', 'cache', 'sous', 'sur', 'roule',
                'tsarine', 'hier', 'tulle', 'reste', 'gabriel',
                'juxtaposer', 'pas', "j'en", 'chie', 'divan']

def get_word_embedding(audio, sr, word_start, word_end, processor, model, device):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state[0]
    total_frames = hidden_states.shape[0]
    total_duration = len(audio) / sr
    fps = total_frames / total_duration
    start_frame = int(word_start * fps)
    end_frame = min(int(word_end * fps), total_frames)
    word_frames = hidden_states[start_frame:end_frame]
    embedding = word_frames.mean(dim=0)
    return embedding.cpu().numpy()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    model.eval()

    metadata = pd.read_csv(BASE + 'metadata_RUFR.csv', sep=';')
    spk_to_L1 = dict(zip(metadata['spk'], metadata['L1']))

    records = []
    for spk in tqdm(os.listdir(WAV_DIR)):
        spk_dir = os.path.join(WAV_DIR, spk)
        if not os.path.isdir(spk_dir):
            continue
        for f in os.listdir(spk_dir):
            if not f.endswith('_words.csv'):
                continue
            csv_path = os.path.join(spk_dir, f)
            wav_path = csv_path.replace('_words.csv', '.wav')
            if not os.path.exists(wav_path):
                continue
            df = pd.read_csv(csv_path, sep=';', header=None, names=['word', 'start', 'end'])
            df['word'] = df['word'].str.strip()
            target_rows = df[df['word'].isin(TARGET_WORDS)]
            if target_rows.empty:
                continue
            audio, sr = librosa.load(wav_path, sr=16000)
            for _, row in target_rows.iterrows():
                emb = get_word_embedding(audio, sr, row['start'], row['end'], processor, model, device)
                records.append({
                    'speaker': spk,
                    'word': row['word'],
                    'L1': spk_to_L1.get(spk, 'unknown'),
                    'embedding': emb
                })

    os.makedirs('outputs', exist_ok=True)
    with open(OUTPUT, 'wb') as f:
        pickle.dump(records, f)
    print(f"Saved {len(records)} records to {OUTPUT}")

if __name__ == '__main__':
    main()