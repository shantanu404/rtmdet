import cv2
import torch
import numpy as np
from timeit import default_timer as timer

from rtmdet.model import RTMDetPipeline

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = RTMDetPipeline()
    pipeline.to(device)
    pipeline.eval()

    video_path = "examples/demo.mp4"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pre_processing_start = timer()
        img_batch, _, _ = pipeline._prepare_images([frame])
        img_batch = np.stack(img_batch)
        img_batch = torch.from_numpy(img_batch).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            img_batch = pipeline.preprocess(img_batch)
        pre_processing_end = timer()

        print(f"Pre-processing time: {pre_processing_end - pre_processing_start:.4f} seconds")

        inference_start = timer()
        with torch.no_grad():
            features = pipeline.model(img_batch)
        inference_end = timer()

        print(f"Inference time: {inference_end - inference_start:.4f} seconds")

        post_processing_start = timer()
        with torch.no_grad():
            results = pipeline.postprocess(features)
        post_processing_end = timer()

        print(f"Post-processing time: {post_processing_end - post_processing_start:.4f} seconds")

    cap.release()