import cv2
import torch
import torch.nn as nn

from pathlib import Path
from torchvision import models, transforms
from ultralytics import YOLO


# =====================================
# Configuration
# =====================================

MODEL_PATH = Path("models/tennis_actions_resnet18.pth")
POSE_MODEL_PATH = "yolo11n-pose.pt"

OUTPUT_DIR = Path("runs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "tennis_action_analysis.mp4"

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================
# Load action classification model
# =====================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Action model not found: {MODEL_PATH.resolve()}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()


# =====================================
# Image preprocessing
# =====================================

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def classify_action(frame):
    """
    Classify the dominant visual content of a frame.
    """

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_tensor = preprocess(rgb_frame).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = probabilities.max(dim=1)

    action = class_names[prediction.item()]
    confidence = confidence.item()

    return action, confidence


# =====================================
# Select video or webcam
# =====================================

print("TennisAI - Action Video Analysis")
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

source_text = input(
    "Enter video filename/full path, or 0 for webcam: "
).strip().strip('"')

source = 0 if source_text == "0" else source_text

if isinstance(source, str) and not Path(source).is_file():
    raise FileNotFoundError(
        f"Video file not found: {Path(source).resolve()}"
    )


# =====================================
# Initialize video and pose model
# =====================================

pose_model = YOLO(POSE_MODEL_PATH)

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    raise RuntimeError("Could not open the video or webcam.")

fps = cap.get(cv2.CAP_PROP_FPS)

if not 0 < fps <= 120:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")


# =====================================
# Process frames
# =====================================

frame_count = 0

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        # Pose estimation
        pose_results = pose_model.predict(
            frame,
            device=DEVICE,
            imgsz=640,
            conf=0.35,
            verbose=False
        )

        annotated = pose_results[0].plot()

        # Frame-level action classification
        action, confidence = classify_action(frame)

        label = f"{action}: {confidence * 100:.1f}%"

        cv2.putText(
            annotated,
            label,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        writer.write(annotated)

        cv2.imshow("TennisAI - Action Analysis", annotated)

        frame_count += 1

        if frame_count % 100 == 0:
            print("Processed frames:", frame_count)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

print("\nAnalysis completed.")
print("Frames processed:", frame_count)
print("Output video:", OUTPUT_PATH.resolve())