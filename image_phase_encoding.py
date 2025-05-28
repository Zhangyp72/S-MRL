import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def read_image_to_grayscale(image_path):
    
    try:
       
        image = Image.open(image_path).convert('L')
        return np.array(image)
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def delayed_phase_spike_encoding(pixel_value, freq, time_step, delay_time):
    
    amplitude = pixel_value / 255.0

    duration = 1.0  
    time = np.arange(0, duration, time_step)

    signal = amplitude * np.cos(2 * np.pi * freq * time)
    zero_crossings = (np.roll(signal, -1) > 0) & (signal <= 0)
    delay_samples = int(delay_time / time_step)
    spike = np.zeros_like(zero_crossings, dtype=int)
    for i in range(len(zero_crossings)):
        if zero_crossings[i] and i + delay_samples < len(zero_crossings):
            spike[i + delay_samples] = 1

    return spike

def image_to_delayed_phase_spike_encoding(image_path, freq=10, time_step=0.001, delay_time=0.02):

    gray_image = read_image_to_grayscale(image_path)
    if gray_image is None:
        return None

    spike_map = {}

    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            pixel_value = gray_image[i, j]
            spike = delayed_phase_spike_encoding(pixel_value, freq, time_step, delay_time)
            spike_map[(i, j)] = spike

    return spike_map

image_path = ".\\images\\"  
spike_map = image_to_delayed_phase_spike_encoding(image_path, freq=10, time_step=0.001, delay_time=0.02)

if spike_map:
 
    print("Pulse Encoding for Selected Pixels:")
    for key in list(spike_map.keys())[:50]: 
        print(f"Pixel {key}: {spike_map[key][:100]}") 
