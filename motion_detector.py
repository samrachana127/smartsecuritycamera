import cv2
import numpy as np


class MotionDetector:
    def __init__(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=40,
            detectShadows=True
        )
        self.heatmap_acc = None

    def detect(self, frame):
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        fg_mask = self.bg_sub.apply(blurred)

        _, fg_mask = cv2.threshold(
            fg_mask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(
            fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(
            fg_mask, kernel, iterations=2)

        if self.heatmap_acc is None:
            self.heatmap_acc = np.zeros(
                (frame.shape[0], frame.shape[1]),
                dtype=np.float32)

        self.heatmap_acc += (fg_mask / 255.0)
        self.heatmap_acc *= 0.95

        heatmap_norm = cv2.normalize(
            self.heatmap_acc, None, 0, 255,
            cv2.NORM_MINMAX)
        heatmap_uint8 = heatmap_norm.astype(np.uint8)
        heatmap_colour = cv2.applyColorMap(
            heatmap_uint8, cv2.COLORMAP_JET)

        cntrs, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        motion_boxes = []
        for c in cntrs:
            if cv2.contourArea(c) < 400:
                continue
            x, y, w, h = cv2.boundingRect(c)
            motion_boxes.append((x, y, x + w, y + h))

        return motion_boxes, fg_mask, heatmap_colour