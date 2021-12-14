import cv2
import numpy as np
from utils import PriorBox, py_cpu_nms, decode, decode_landm

class RetinaDetection:
    def __init__(self, cfg, model_path="./checkpoints/faceDetector_RFB.onnx"):
        self.cfg = cfg
        self.model_path = model_path
        self.load(self.model_path)

    def load(self, model_path):
        self.model = cv2.dnn.readNetFromONNX(model_path)

    def forward(self, image):
        """image: should be BGR image"""
        img, scale, resize, im_height, im_width = self.preprocess(image)
        self.model.setInput(img)
        loc, conf, landms = self.model.forward(["output0", "532","531"])
        #post_process
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        prior_data = priorbox.forward()
        boxes = decode(np.squeeze(loc,axis=0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        scores = np.squeeze(conf, axis=0)[:, 1]
        landms = decode_landm(np.squeeze(landms, axis=0), prior_data, self.cfg['variance'])
        scale1 = np.array([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
        landms = landms * scale1 / resize
        # ignore low scores
        inds = np.where(scores > self.cfg["confidence_threshold"])[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.cfg["top_k"]]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.cfg["nms_threshold"])
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.cfg["keep_top_k"], :]
        landms = landms[:self.cfg["keep_top_k"], :]

        dets = np.concatenate((dets, landms), axis=1)
        return dets

    def preprocess(self,img):
        target_size = self.cfg["long_side"]
        max_size = self.cfg["long_side"]
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if self.cfg["origin_size"]:
            resize = 1
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img, scale, resize, im_height, im_width

