"""
    data/
      ├── audio/  <classA>/file.wav  <classB>/file.wav
      ├── docs/   <classX>/file.txt  <classY>/file.txt
      ├── images/ <classC>/file.png …
      └── video/  <classZ>/file.mp4 …
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from SRM import SpikeSequence, SpikeResponseModel, WeightMatrix  

import audio_encoding  
import doc_spike_encoding  
import image_phase_encoding 
import video_spike_encoding  

from RecurrentNetwork import RecurrentNetwork  

def to_spike_sequence(spikes: List[float] | np.ndarray) -> SpikeSequence:
    return SpikeSequence(np.asarray(spikes, dtype=float))

def _label_from_path(file_path: Path, modality_root: Path) -> str:
    
    try:
        return file_path.relative_to(modality_root).parts[0]
    except ValueError:  
        return "unknown"

def encode_audio(audio_root: Path, rate: int = 100) -> List[Tuple[str, SpikeSequence, str]]:
    enc = audio_encoding.PoissonEncoder(rate=rate)
    triples: List[Tuple[str, SpikeSequence, str]] = []
    for wav in audio_root.glob("**/*.*"):
        if wav.is_file() and wav.suffix.lower() in {".wav", ".mp3", ".flac"}:
            seq_raw = audio_encoding.process_single_file(str(wav), enc)
            triples.append((str(wav), to_spike_sequence(seq_raw), _label_from_path(wav, audio_root)))
    return triples

def encode_docs(doc_root: Path) -> List[Tuple[str, SpikeSequence, str]]:
    files = [p for p in doc_root.glob("**/*.txt")]
    seqs = doc_spike_encoding.process_multiple_files([str(p) for p in files])
    return [
        (str(fp), to_spike_sequence(seq), _label_from_path(fp, doc_root))
        for fp, seq in zip(files, seqs)
    ]

def encode_images(img_root: Path) -> List[Tuple[str, SpikeSequence, str]]:
    triples = []
    for img in img_root.glob("**/*.*"):
        if img.is_file() and img.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            spike_map = image_phase_encoding.image_to_delayed_phase_spike_encoding(str(img))
            flat = (
                np.concatenate([np.flatnonzero(v) for v in spike_map.values()])
                if spike_map else np.array([])
            )
            triples.append((str(img), to_spike_sequence(flat), _label_from_path(img, img_root)))
    return triples


def encode_videos(vid_root: Path) -> List[Tuple[str, SpikeSequence, str]]:
    triples = []
    for vid in vid_root.glob("**/*.*"):
        if vid.is_file() and vid.suffix.lower() in {".mp4", ".avi", ".mkv"}:
            frames = video_spike_encoding.process_single_video_and_generate_spikes(str(vid))
            flat = np.fromiter((t for frame in frames for t in frame), float)
            triples.append((str(vid), to_spike_sequence(flat), _label_from_path(vid, vid_root)))
    return triples

def build_dataset(base: Path) -> List[Dict]:
    dataset: List[Dict] = []

    dataset += [
        {"path": p, "spikes": s, "label": l}
        for p, s, l in encode_audio(base / "audio")
    ]
    dataset += [
        {"path": p, "spikes": s, "label": l}
        for p, s, l in encode_docs(base / "docs")
    ]
    dataset += [
        {"path": p, "spikes": s, "label": l}
        for p, s, l in encode_images(base / "images")
    ]
    dataset += [
        {"path": p, "spikes": s, "label": l}
        for p, s, l in encode_videos(base / "video")
    ]

    return dataset

def gaussian_kernel_similarity(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    return float(np.sum(np.exp(-((x - y) ** 2) / (2 * sigma ** 2))))

def nearest_neighbor_accuracy(dataset: List[Dict]) -> float:
    n = len(dataset)
    if n < 2:
        return 0.0
    correct = 0
    for i, sample in enumerate(dataset):
        best_sim = -1.0
        best_label = None
        x_i = sample["spikes"].spikes
        for j, other in enumerate(dataset):
            if i == j:
                continue
            sim = gaussian_kernel_similarity(x_i, other["spikes"].spikes)
            if sim > best_sim:
                best_sim = sim
                best_label = other["label"]
        if best_label == sample["label"]:
            correct += 1
    return correct / n

def train_network(dataset: List[Dict], hidden: int = 16, synapses: int = 4, epochs: int = 5):
    print(f"Building network: inputs=1, hidden={hidden}, synapses={synapses}")
    weights = WeightMatrix(1, hidden, synapses)
    neurons = [SpikeResponseModel(decayT=5, decayR=20, thresh=1.0) for _ in range(hidden)]
    net = RecurrentNetwork(neurons, weights) 

    for ep in range(1, epochs + 1):
        for sample in dataset:
            net.forward(sample["spikes"])  
            net.learning()                  
        print(f"Epoch {ep}/{epochs} finished.")
    return net

def main():
    base = Path("data")
    if not base.exists():
        raise FileNotFoundError("Expected ./data folder with audio/docs/images/video sub‑dirs.")

    dataset = build_dataset(base)
    if not dataset:
        raise RuntimeError("No data found – please populate the data/ folder.")

    labels = {d["label"] for d in dataset}
    print(f"Loaded {len(dataset)} samples across {len(labels)} categories: {sorted(labels)}")

    train_network(dataset)

    acc = nearest_neighbor_accuracy(dataset)
    print(f"\nNearest‑neighbor leave‑one‑out accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
