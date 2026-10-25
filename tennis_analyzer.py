import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

print("Starting TennisAI...")

device = 0 if torch.cuda.is_available() else "cpu"

if device == 0:
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

video_path = input(
    "Enter video filename (or 0 for webcam): "
).strip()

source = int(video_path) if video_path == "0" else video_path

model = YOLO("yolo11n-pose.pt")

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("ERROR: Cannot open video or camera.")
    print("Check the video filename and location.")
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
if not 0 < fps <= 120:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = Path("runs/tennis_analysis.mp4")
output_path.parent.mkdir(parents=True, exist_ok=True)

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    print("ERROR: Cannot create output video.")
    raise SystemExit(1)

frames = 0

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model.predict(
            frame,
            device=device,
            imgsz=640,
            conf=0.35,
            verbose=False
        )

        annotated = results[0].plot()

        cv2.putText(
            annotated,
            "TennisAI Pose Analysis",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        writer.write(annotated)
        cv2.imshow("TennisAI", annotated)

        frames += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

print("Frames processed:", frames)
print("Saved video:", output_path.resolve())

