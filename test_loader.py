from video_loader import VideoLoader

loader = VideoLoader(source=0)

if not loader.is_open():
    print("ERROR: Camera not found!")
else:
    print("Camera opened successfully!")
    loader.get_info()
    ret, frame = loader.read_frame()
    print("Frame shape:", frame.shape)

loader.release()