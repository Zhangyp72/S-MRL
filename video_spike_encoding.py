import cv2
import os
import numpy as np


def read_video_and_calculate_variances(video_path):
   
    if not os.path.isfile(video_path):
        print(f"file {video_path} does not exist！")
        return []
 
    cap = cv2.VideoCapture(video_path)
  
    if not cap.isOpened():
        print(f"unable to open video file：{video_path}")
        print("Possible causes: Lack of codec support, corrupted video files, or OpenCV environment issues.")
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    print(f"\nHandling video files: {video_path}, total number of frames: {frame_count}")

    variances = []  
    frame_index = 0  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Waring: unable to read frame {frame_index}，possibly end of video or frame corruption.")
            break

        if frame is None:
            print(f"Waring： {frame_index}th frame is empty.")
            variances.append(None)  
        else:
           
            variance = np.var(frame)
            variances.append(variance)
            print(f"The pixel variance of the {frame_index}th frame: {variance:.2f}")

        frame_index += 1
    cap.release()
    return variances


def normalize_variances(variances):

    valid_variances = [v for v in variances if v is not None]
    if not valid_variances:
        print("There are no valid frames of variance data to normalize.")
        return []

    min_variance = min(valid_variances)
    max_variance = max(valid_variances)
    print(f"Minimum: {min_variance:.2f}, Maximum variance: {max_variance:.2f}")

    normalized_variances = [
        (v - min_variance) / (max_variance - min_variance) if v is not None else None
        for v in variances
    ]

    return normalized_variances


def generate_poisson_spikes(normalized_variances, time_steps=100, max_rate=20):
  
    spike_sequences = [] 

    for idx, value in enumerate(normalized_variances):
        if value is None:
            spike_sequences.append([]) 
            continue

        rate = value * max_rate

        spikes = []
        for t in range(time_steps):
            if np.random.rand() < rate / time_steps:  
                spikes.append(t)

        spike_sequences.append(spikes)
        print(f"The {idx}th frame Poisson spike sequence: {spikes}")

    return spike_sequences


def process_video_directory_and_generate_spikes(video_dir):
   
    if not os.path.isdir(video_dir):
        print(f"Folder {video_dir} does not exist，please check the path！")
        return {}
  
    supported_formats = ['.mp4', '.avi', '.mkv', '.mov', '.flv']

    video_files = [f for f in os.listdir(video_dir) if os.path.splitext(f)[1].lower() in supported_formats]

    if not video_files:
        return {}

    video_spike_sequences = {}  

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        
        variances = read_video_and_calculate_variances(video_path)

        normalized_variances = normalize_variances(variances)

        spike_sequences = generate_poisson_spikes(normalized_variances)
        video_spike_sequences[video_file] = spike_sequences
    return video_spike_sequences

if __name__ == "__main__":
 
    video_directory = ".\\video"
    video_spike_sequences_data = process_video_directory_and_generate_spikes(video_directory)

    for video_name, spike_sequences in video_spike_sequences_data.items():
        print(video_name)
        print(spike_sequences)
