import cv2
import numpy as np
import torch

from rtmdet.model import RTMDetPipeline

if __name__ == "__main__":
    state = torch.load("weights/rtmdet_ins_x_8xb16_300e_coco_10162025.pth")
    pipeline = RTMDetPipeline()
    pipeline.load_state_dict(state, strict=True)
    pipeline = pipeline.to("cuda")

    img = cv2.imread("examples/demo.jpg")
    og_img = img.copy()

    out = pipeline.predict([img])[0]

    all_masks = np.zeros_like(og_img)

    for mask, bbox, score, label in zip(out.masks, out.bboxes, out.scores, out.labels):
        x1, y1, x2, y2 = map(int, bbox)
        random_color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]

        cv2.rectangle(og_img, (x1, y1), (x2, y2), random_color, 2)
        cv2.putText(
            og_img,
            f"{label}:{score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            random_color,
            2,
        )

        mask = mask.cpu().numpy().astype(np.uint8) * 255
        all_masks[mask == 255] = random_color

    concatenated_image = np.concatenate((og_img, all_masks), axis=1)
    cv2.imwrite("examples/output.jpg", concatenated_image)

    # torch.jit.trace(pipeline, torch.randn(1, 3, 640, 640)).save("weights/rtmdet_ins.pt")