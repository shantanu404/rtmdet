import cv2
import numpy as np
import torch
from rtmdet import RTMDetPipeline
from tqdm import tqdm

def annotate_frame(img, out):
    all_masks = np.zeros_like(img)
    for mask, bbox, score, label in zip(out.masks, out.bboxes, out.scores, out.labels):
        x1, y1, x2, y2 = map(int, bbox)
        random_color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), random_color, 2)
        cv2.putText(
            img,
            f"{label}:{score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            random_color,
            2,
        )
        mask = mask.cpu().numpy()
        all_masks[mask] = random_color
    concatenated_image = np.concatenate((img, all_masks), axis=1)
    return concatenated_image

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = RTMDetPipeline(
        device=device,
        weight_path='weights/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth'
    )
    cap = cv2.VideoCapture("examples/demo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter("examples/output.avi", fourcc, fps, (width*2, height))

    batch_size = 64
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    pbar = tqdm(total=total_frames, desc="Processing frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) == batch_size:
            results = pipeline.predict(frames, retina=True)
            for img, out in zip(frames, results):
                annotated = annotate_frame(img, out)
                out_video.write(annotated)
                processed_frames += 1
                pbar.update(1)
            frames = []
    if frames:
        results = pipeline.predict(frames, retina=True)
        for img, out in zip(frames, results):
            annotated = annotate_frame(img, out)
            out_video.write(annotated)
            processed_frames += 1
            pbar.update(1)
    pbar.close()
    cap.release()
    out_video.release()
