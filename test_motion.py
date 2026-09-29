import cv2
from video_loader import VideoLoader
from motion_detector import MotionDetector
from clip_saver import ClipSaver

loader = VideoLoader(source="warehouse.mp4")
detector = MotionDetector()
saver = ClipSaver(output_folder="saved_clips")

while loader.is_open():
    ret, frame = loader.read_frame()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    boxes, mask = detector.detect(frame)

    human_found = False

    for (x1, y1, x2, y2) in boxes:
        w = x2 - x1
        h = y2 - y1
        area = w * h
        ratio = h / max(w, 1)

        bottom_half = y2 > 270
        tall_enough = h > 40
        upright = ratio > 0.8
        not_too_big = area < 80000

        if bottom_half and tall_enough and upright and not_too_big:
            human_found = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.rectangle(frame, (x1, y1 - 28), (x2, y1), (0, 0, 255), -1)
            cv2.putText(frame, "HUMAN", (x1 + 4, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 22), (x2, y1), (255, 255, 0), -1)
            cv2.putText(frame, "MOTION", (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 2)

    # Update clip saver every frame
    saver.update(frame, human_found)

    # Show recording indicator
    if saver.is_recording:
        cv2.circle(frame, (930, 20), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (900, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)

    # Top bar
    cv2.rectangle(frame, (0, 0), (960, 38), (20, 20, 20), -1)
    cv2.putText(frame, "SecureVision | Northampton Warehouse",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (200, 200, 200), 2)

    # Alert banner
    if human_found:
        cv2.rectangle(frame, (0, 38), (960, 72), (0, 0, 180), -1)
        cv2.putText(frame, "  !! HUMAN DETECTED !!",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, 38), (960, 72), (20, 60, 20), -1)
        cv2.putText(frame, "  Monitoring...",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (150, 255, 150), 2)

    cv2.imshow("SecureVision", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == 27:
        print("Quitting...")
        break

saver.stop_recording()
loader.release()
print("Done!")