import numpy as np
import torch
from tqdm import tqdm
from timeit import default_timer as timer

from rtmdet import RTMDetPipeline

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = RTMDetPipeline(device=device)

    num_frames = 1000
    frame_height = 480
    frame_width = 640
    channels = 3

    average_time = 0.0

    for _ in tqdm(range(num_frames)):
        frame = np.random.randint(0, 256, (frame_height, frame_width, channels), dtype=np.uint8)

        start_time = timer()
        results = pipeline.predict([frame], retina=True)
        end_time = timer()

        inference_time = end_time - start_time
        average_time += inference_time

    average_time /= num_frames
    print(f"Average inference time: {average_time:.4f} seconds")
