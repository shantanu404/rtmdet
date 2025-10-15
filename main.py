import cv2
import numpy as np
import torch

from rtmdet.ckpt_loader import check_params_update, load_mmdet_checkpoint
from rtmdet.model import RTMDet

if __name__ == "__main__":
    model = RTMDet(num_classes=80)
    state = load_mmdet_checkpoint(
        "weights/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth"
    )

    check_params_update(model, state)
    model.load_state_dict(state, strict=True)

    means = [103.530, 116.280, 123.675]
    stds = [57.375, 57.120, 58.395]

    img = cv2.imread("examples/demo.jpg")
    h, w = img.shape[:2]

    # Resize keeping aspect ratio
    scale = min(640 / h, 640 / w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh))

    # Pad to 640x640
    top = 0
    bottom = 640 - nh
    left = 0
    right = 640 - nw
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114]
    )
    img = cv2.resize(img, (640, 640))
    img_meta = {"img_shape": (640, 640, 3)}
    og_img = img.copy()

    img = torch.from_numpy(img).unsqueeze(0).float()
    img = img.permute(0, 3, 1, 2)  # NHWC to NCHW
    img = (img - torch.tensor(means).view(1, 3, 1, 1)) / torch.tensor(stds).view(
        1, 3, 1, 1
    )

    results = model.predict(img, [img_meta])
    out = results[0]

    all_masks = np.zeros((640, 640, 3), dtype=np.uint8)

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

    # Get rid of padding
    all_masks = all_masks[0:nh, 0:nw]
    og_img = og_img[0:nh, 0:nw]

    concatenated_image = np.concatenate((og_img, all_masks), axis=1)
    cv2.imwrite("examples/output.jpg", concatenated_image)
