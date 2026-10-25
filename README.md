# TennisAI - Biomechanical Motion Analysis

AI-powered tennis action classification and biomechanical motion analysis using deep learning and computer vision.

## Features
- Tennis action classification: backhand, forehand, ready position, and serve.
- Human pose estimation using YOLO Pose.
- 2D joint-angle analysis.
- CSV export and biomechanical visualizations.
- Experimental heuristic motion-risk flags.

## Models
- ResNet18: tennis action classification.
- YOLO11n-Pose: human pose estimation.

## Evaluation
- Test accuracy: 77%%.
- Macro F1-score: 0.76.
- Four tennis action classes.

## Important Limitation
The motion-risk indicators are experimental heuristics, not clinically validated injury predictions. ACL injury prediction has not been validated using labeled injury data.

## Installation
Create a Python virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Run
```bash
python tennis_biomechanics.py
python tennis_action_video.py
python -m streamlit run app.py
```

## Hardware
Tested with NVIDIA Quadro T2000 GPU and CUDA.

## License
No license has been assigned yet.
