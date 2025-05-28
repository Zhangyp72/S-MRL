import os
import numpy as np
from scipy.io import wavfile
import librosa
import random


class PoissonEncoder:

    def __init__(self, rate=100):
       
        self.rate = rate  

    def encode(self, audio_signal, sampling_rate):
       
        duration = len(audio_signal) / sampling_rate  
        num_samples = len(audio_signal)  

        normalized_signal = (audio_signal - np.min(audio_signal)) / (np.max(audio_signal) - np.min(audio_signal))

        spike_times = []
        for i in range(num_samples):
            prob = normalized_signal[i] * self.rate / sampling_rate
            if random.random() < prob:  
                spike_time = i / sampling_rate
                spike_times.append(spike_time)

        return np.array(spike_times)


def load_audio_file(file_path):
   
    try:
      
        print(f"Attempting to load {file_path} with librosa...")
        audio_signal, sampling_rate = librosa.load(file_path, sr=None)
        print(f"Loaded {file_path} with librosa successfully!")
    except Exception as librosa_error:
        try:
           
            print(f"Librosa failed to load {file_path}. Trying scipy...")
            sampling_rate, audio_signal = wavfile.read(file_path)
            if audio_signal.ndim > 1:
                audio_signal = audio_signal[:, 0]
            print(f"Loaded {file_path} with scipy successfully!")
        except Exception as scipy_error:
            print(f"Failed to load audio file: {file_path}")
            print(f"Librosa Error: {librosa_error}")
            print(f"Scipy Error: {scipy_error}")
            return None, None
    return sampling_rate, audio_signal


def process_audio_directory(directory, encoder, valid_extensions=(".wav", ".mp3", ".flac")):
  
    results = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                sampling_rate, audio_signal = load_audio_file(file_path)
                if audio_signal is not None:
                    try:
                        spike_times = encoder.encode(audio_signal, sampling_rate)
                        results[file_path] = spike_times
                        print(f"Successfully encoded {file_path}")
                    except Exception as encoding_error:
                        print(f"Failed to encode {file_path}: {encoding_error}")
                else:
                    print(f"Skipping {file_path} due to loading failure.")
    return results


if __name__ == "__main__":
   
    poisson_encoder = PoissonEncoder(rate=100)

    audio_directory = ".\\Audio"  

    encoding_results = process_audio_directory(audio_directory, poisson_encoder)
    print(encoding_results)
