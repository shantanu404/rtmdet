import cv2
import numpy as np
import torch

from rtmdet import RTMDetPipeline

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = RTMDetPipeline(
        device=device,
        weight_path='weights/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth'
    )

    img = cv2.imread("examples/demo.jpg")
    out = pipeline.predict([img], retina=True)[0]

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
    cv2.imwrite("examples/output.jpg", concatenated_image)
