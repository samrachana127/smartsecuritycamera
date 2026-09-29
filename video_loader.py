import cv2

class VideoLoader:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)

    def is_open(self):
        return self.cap.isOpened()

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def get_info(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Resolution: {width}x{height} | FPS: {fps}")

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()