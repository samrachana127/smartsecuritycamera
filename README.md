# Smart Security Camera

A Python and OpenCV based security camera dashboard that detects motion, 
tracks activity, and saves video clips of incidents automatically.

## What it does

- Loads video from a camera feed or video file
- Detects motion in real time
- Saves short video clips whenever motion is detected
- Displays everything through a simple desktop dashboard (built with Tkinter)

## Files

| File | What it does |
|---|---|
| `dashboard.py` | Main app — the dashboard UI you run |
| `video_loader.py` | Handles reading video from a camera or file |
| `motion_detector.py` | Detects motion between frames |
| `clip_saver.py` | Saves a video clip when motion is detected |
| `test_loader.py` | Tests for the video loader |
| `test_motion.py` | Tests for motion detection |

## How to run it

1. Clone this repo
2. Install the required packages:
pip install -r requirements.txt

3. Run the dashboard:
python dashboard.py

## Tech used

- Python
- OpenCV (video processing and motion detection)
- Tkinter (dashboard UI)
- NumPy
- Pillow (image handling)

## Project background

Built as a university project (CSY3058, Media Technology) — a warehouse 
security scenario where the camera monitors an area, flags motion, and 
keeps a record of incidents as saved clips.
