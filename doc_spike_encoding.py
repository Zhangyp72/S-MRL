import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from SpikingNeuronModel import SRM

class PoissonEncoding:
    HUGE_VAL = float(1000)

    def __init__(self, rate=100):
        self.rate = rate
        self.tn = 0
        self.T = 1000 / rate

    def set_rate(self, rate):
        if rate <= 0:
            raise ValueError("Rate must be positive.")
        self.rate = rate
        self.T = 1000 / rate

    def spike(self):
        self.tn += self._update()
        return self.tn

    def _update(self):
        if self.T <= 0:
            return self.HUGE_VAL
        return -self.T * np.log(1.0 - np.random.rand())

    def spike_encoding(self, duration):
        if duration <= 0:
            raise ValueError("Duration must be positive.")
        spike_sequence = []
        t = self.spike()
        while t <= duration:
            spike_sequence.append(t)
            t = self.spike()
        return spike_sequence


def read_text_file(file_path):
    
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentences.append(line.strip().split())
    return sentences


def encode_word2vec(sentences, vector_size=100, window=5, min_count=1, workers=4):
    
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    word_vectors = []
    for sentence in sentences:
        for word in sentence:
            if word in model.wv:
                word_vectors.append(model.wv[word])
    return np.array(word_vectors)


def compute_statistics(word_vectors):
 
    means = np.mean(word_vectors, axis=1)
    stds = np.std(word_vectors, axis=1)
    variances = np.var(word_vectors, axis=1)
    return means, stds, variances


def normalize_variances(variances, min_rate=1):
  
    variances[variances == 0] = 1e-6
    min_val = np.min(variances)
    max_val = np.max(variances)
    if max_val - min_val <= 1e-6:
        return np.ones_like(variances) * min_rate
    normalized = (variances - min_val) / (max_val - min_val) * 50
    normalized += 1e-6
    normalized[normalized < min_rate] = min_rate
    return normalized


def generate_spike_sequences_from_matrix(matrix, duration=100, min_rate=1):
    
    spike_sequences = []
    for row in matrix:
        row_sequences = []
        for value in row:
            if value < min_rate:
                value = min_rate
            poisson_encoder = PoissonEncoding(rate=value)
            spike_sequence = poisson_encoder.spike_encoding(duration)
            row_sequences.append(spike_sequence)
        spike_sequences.append(row_sequences)
    return spike_sequences


def flatten_spike_sequences(spike_sequences):
    
    flattened = []
    for row_sequences in spike_sequences:
        for sequence in row_sequences:
            flattened.extend(sequence)
    return flattened


def process_multiple_files(file_paths, duration=100, vector_size=100, min_rate=1):
   
    all_spike_sequences = []

    for file_path in file_paths:
        #print(f"Processing file: {file_path}")
        sentences = read_text_file(file_path)
        if not sentences:
            print(f"File {file_path} is empty or invalid.")
            all_spike_sequences.append([])
            continue
        word_vectors = encode_word2vec(sentences, vector_size=vector_size)
        if word_vectors.size == 0:
            #print(f"No valid word vectors for file {file_path}.")
            all_spike_sequences.append([])
            continue
        means, stds, variances = compute_statistics(word_vectors)
        normalized_variances = normalize_variances(variances, min_rate=min_rate)
        normalized_variance_matrix = normalized_variances.reshape(-1, 1)
        spike_sequences = generate_spike_sequences_from_matrix(
            normalized_variance_matrix, duration=duration, min_rate=min_rate
        )
        flattened_spike_sequences = flatten_spike_sequences(spike_sequences)
        all_spike_sequences.append(flattened_spike_sequences)

    return all_spike_sequences


if __name__ == "__main__":
    
    directory_path = ".\data\\"  
    file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".txt")]

   
    duration = 100  
    vector_size = 300  
    min_rate = 1  

    
    all_spike_sequences = process_multiple_files(file_paths, duration=duration, vector_size=vector_size, min_rate=min_rate)

    print("all spikes sequences is: ")
    print(all_spike_sequences[0])
    print("all spikes sequences is: ")
    print(all_spike_sequences[1])
    print("the length of all spikes sequences is: ")
    print(len(all_spike_sequences))
   