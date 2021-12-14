from Detection import RetinaDetection
import cv2
from conf import cfg
import numpy as np


image_path = "./sample.jpg"
img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
img = np.float32(img_raw)
detector = RetinaDetection(cfg)
dets = detector.forward(img)

for b in dets:
    if b[4] < cfg["vis_thres"]:
        continue
    text = "{:.4f}".format(b[4])
    b = list(map(int, b))
    cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    cx = b[0]
    cy = b[1] + 12
    cv2.putText(img_raw, text, (cx, cy),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    # landms
    cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
    cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
    cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
    cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
    cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
# save image

name = "test_success.jpg"
cv2.imwrite(name, img_raw)
