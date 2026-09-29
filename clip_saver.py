import cv2
import os
from datetime import datetime


class ClipSaver:
    def __init__(self, output_folder="saved_clips"):
        self.output_folder = output_folder
        self.writer = None
        self.is_recording = False
        self.frames_after = 0
        self.frames_after_limit = 60

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created folder: {output_folder}")

    def start_recording(self, frame):
        if self.is_recording:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_folder}/incident_{timestamp}.avi"
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(
            filename, fourcc, 20.0, (w, h))
        self.is_recording = True
        self.frames_after = 0
        print(f"Recording started: {filename}")

    def save_frame(self, frame):
        if self.is_recording and self.writer:
            self.writer.write(frame)

    def stop_recording(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.is_recording = False
        self.frames_after = 0
        print("Recording saved!")

    def update(self, frame, human_found):
        if human_found:
            if not self.is_recording:
                self.start_recording(frame)
            self.frames_after = 0
            self.save_frame(frame)
        else:
            if self.is_recording:
                self.save_frame(frame)
                self.frames_after += 1
                if self.frames_after >= self.frames_after_limit:
                    self.stop_recording()