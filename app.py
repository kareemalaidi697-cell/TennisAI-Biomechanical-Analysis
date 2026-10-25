from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PLOTS = RUNS / "biomechanics_plots"
RISK = RUNS / "risk_analysis"
VIDEO_CANDIDATES = [
    RUNS / "tennis_biomechanics.mp4",
    RUNS / "tennis_action_analysis.mp4",
    ROOT / "tennis_video.mp4",
]
ANGLE_CSV = RUNS / "tennis_joint_angles.csv"
RISK_CSV = RISK / "tennis_motion_features.csv"

st.set_page_config(page_title="TennisAI | Performance Analytics", page_icon="🎾", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #081827 0%, #102d3a 55%, #123e43 100%); color: #f4f7fb;}
[data-testid="stSidebar"] {background: #07131f;}
h1,h2,h3 {color:#f6fbff;}
div[data-testid="stMetric"] {background:rgba(255,255,255,.07); padding:16px; border-radius:12px; border:1px solid rgba(255,255,255,.12);}
.small-note {font-size: .9rem; color:#cbd5e1;}
</style>
""", unsafe_allow_html=True)

st.title("🎾 TennisAI")
st.subheader("AI-Powered Athletic Performance Analytics")
st.caption("Real-Time Biomechanical Motion Analysis • Tennis Action Recognition • Exploratory Movement Indicators")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose a section", ["Overview", "Biomechanical Analysis", "Movement Indicators", "Model Performance", "Files & Limitations"])
    st.divider()
    st.caption("Graduation project prototype")
    st.caption("Research/demo use — not a medical diagnostic system.")

def load_csv(path):
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return None

angles = load_csv(ANGLE_CSV)
risk = load_csv(RISK_CSV)
video = next((p for p in VIDEO_CANDIDATES if p.exists()), None)

if page == "Overview":
    st.markdown("### Project overview")
    st.write("This prototype combines computer-vision-based tennis action classification with 2D pose estimation and joint-angle tracking from video.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Action classes", "4")
    c2.metric("Classifier test accuracy", "77%")
    c3.metric("Frames analyzed", f"{len(angles):,}" if angles is not None else "Not found")
    c4.metric("Video result", "Available" if video else "Not found")
    st.info("The action classifier recognizes backhand, forehand, ready position, and serve. The biomechanics component estimates 2D joint angles from pose keypoints.")
    if video:
        st.markdown("### Processed video")
        st.video(str(video))
    else:
        st.warning("No processed video found in the expected locations. Place the generated MP4 inside the runs folder.")
    st.markdown("### Pipeline")
    st.code("Input video → YOLO pose estimation → 2D joint-angle features\nInput frame → ResNet18 transfer-learning classifier → action label\nJoint-angle CSV → exploratory movement indicators and visualizations", language="text")

elif page == "Biomechanical Analysis":
    st.markdown("### 2D biomechanical motion analysis")
    if angles is not None:
        st.success(f"Loaded {len(angles):,} frame records from {ANGLE_CSV.relative_to(ROOT)}.")
        st.dataframe(angles.head(20), use_container_width=True)
        st.download_button("Download joint-angle CSV", angles.to_csv(index=False).encode("utf-8"), "tennis_joint_angles.csv", "text/csv")
    else:
        st.warning("Joint-angle CSV not found. Expected: runs/tennis_joint_angles.csv")
    found = False
    for p in [PLOTS / "knees_angles.png", PLOTS / "hips_angles.png", PLOTS / "elbows_angles.png", PLOTS / "shoulders_angles.png"]:
        if p.exists():
            found = True
            st.markdown(f"#### {p.stem.replace('_',' ').title()}")
            st.image(str(p), use_container_width=True)
    if not found:
        st.warning("No biomechanics plots found in runs/biomechanics_plots.")
    st.caption("Angles are estimated in 2D image coordinates. They are sensitive to camera view, occlusion, keypoint confidence, and frame-to-frame detection jitter.")

elif page == "Movement Indicators":
    st.markdown("### Exploratory movement indicators")
    if risk is not None:
        st.success(f"Loaded {len(risk):,} frame records.")
        st.dataframe(risk.head(20), use_container_width=True)
        st.download_button("Download movement features CSV", risk.to_csv(index=False).encode("utf-8"), "tennis_motion_features.csv", "text/csv")
        flag_cols = [c for c in risk.columns if "review" in c.lower() or "risk" in c.lower() or "flag" in c.lower()]
        if flag_cols:
            col = flag_cols[0]
            vals = risk[col].astype(str).str.lower().isin(["true","1","yes"])
            a,b = st.columns(2)
            a.metric("Frames flagged for review", f"{int(vals.sum()):,}")
            b.metric("Share flagged", f"{vals.mean()*100:.1f}%")
        st.warning("These flags are heuristic prompts for reviewing motion, not probabilities of ACL injury and not a clinical diagnosis.")
        for p in [RISK / "knee_angles_review.png", RISK / "knee_angular_speed.png"]:
            if p.exists():
                st.image(str(p), caption=p.stem.replace("_"," ").title(), use_container_width=True)
    else:
        st.warning("Movement feature CSV not found. Expected: runs/risk_analysis/tennis_motion_features.csv")

elif page == "Model Performance":
    st.markdown("### Action recognition model")
    a,b,c = st.columns(3)
    a.metric("Test accuracy", "77%")
    b.metric("Macro F1-score", "0.76")
    c.metric("Validation accuracy", "81%")
    st.write("Model: ResNet18 with transfer learning. Classes: backhand, forehand, ready position, serve.")
    report = pd.DataFrame({
        "Class": ["Backhand", "Forehand", "Ready position", "Serve"],
        "Precision": [0.80,0.69,0.78,0.81],
        "Recall": [0.53,0.79,0.83,0.92],
        "F1-score": [0.64,0.73,0.81,0.86],
        "Test samples": [75,75,75,75],
    })
    st.dataframe(report, use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(8,3.5))
    ax.bar(report["Class"], report["F1-score"])
    ax.set_ylim(0,1)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1-score")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Metrics are from the held-out action-classification test split; they do not measure injury prediction performance.")
    model_path = ROOT / "models" / "tennis_actions_resnet18.pth"
    st.write("Checkpoint:", "Found" if model_path.exists() else "Not found")

else:
    st.markdown("### Files and scientific limitations")
    expected = [
        ROOT / "models" / "tennis_actions_resnet18.pth",
        ANGLE_CSV,
        RISK_CSV,
        PLOTS / "knees_angles.png",
        PLOTS / "hips_angles.png",
        PLOTS / "elbows_angles.png",
        PLOTS / "shoulders_angles.png",
    ]
    for p in expected:
        st.write(("✅" if p.exists() else "⚠️"), str(p.relative_to(ROOT)), "—", "available" if p.exists() else "missing")
    st.markdown("#### Important limitations")
    st.markdown("""
- The classifier recognizes four tennis action classes; it is not an injury classifier.
- Pose-derived angles are 2D estimates and are not equivalent to laboratory-grade 3D biomechanics.
- Movement flags are heuristic review indicators only. They are not validated ACL injury probabilities.
- Reliable injury prediction requires appropriately labeled injury/outcome data, athlete-level separation between train/test sets, and independent validation.
- This prototype is intended for research demonstration and must not be used for medical decisions.
""")
    st.markdown("#### Reproducibility")
    st.write("Keep the model checkpoint and the generated CSV/PNG/MP4 outputs in their existing relative folders. The dashboard reads those files; it does not silently rerun long video inference jobs.")
