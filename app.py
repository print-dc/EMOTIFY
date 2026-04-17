from __future__ import annotations

import os
import threading
from collections import Counter, deque
from pathlib import Path
from typing import Optional

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODEL_PATH = SRC_DIR / "model.h5"
FACE_CASCADE_PATH = SRC_DIR / "haarcascade_frontalface_default.xml"
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
POSE_LANDMARK = mp_pose.PoseLandmark


def build_emotion_model() -> Sequential:
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation="relu"),
        Dropout(0.5),
        Dense(7, activation="softmax"),
    ])
    model.load_weights(str(MODEL_PATH))
    return model


@st.cache_resource
def load_emotion_model() -> Sequential:
    return build_emotion_model()


@st.cache_resource
def load_face_cascade() -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError(f"Failed to load face cascade from {FACE_CASCADE_PATH}")
    return cascade


def create_pose_estimator() -> mp_pose.Pose:
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _landmark_xy(landmarks, landmark_id: POSE_LANDMARK):
    landmark = landmarks[landmark_id.value]
    return landmark.x, landmark.y, landmark.visibility


def _distance(point_a, point_b) -> float:
    return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))


def infer_action(pose_landmarks) -> tuple[str, float]:
    if pose_landmarks is None:
        return "pose not detected", 0.0

    landmarks = pose_landmarks.landmark

    try:
        nose_x, nose_y, nose_v = _landmark_xy(landmarks, POSE_LANDMARK.NOSE)
        ls_x, ls_y, ls_v = _landmark_xy(landmarks, POSE_LANDMARK.LEFT_SHOULDER)
        rs_x, rs_y, rs_v = _landmark_xy(landmarks, POSE_LANDMARK.RIGHT_SHOULDER)
        lw_x, lw_y, lw_v = _landmark_xy(landmarks, POSE_LANDMARK.LEFT_WRIST)
        rw_x, rw_y, rw_v = _landmark_xy(landmarks, POSE_LANDMARK.RIGHT_WRIST)
        lh_x, lh_y, lh_v = _landmark_xy(landmarks, POSE_LANDMARK.LEFT_HIP)
        rh_x, rh_y, rh_v = _landmark_xy(landmarks, POSE_LANDMARK.RIGHT_HIP)
    except IndexError:
        return "pose not detected", 0.0

    wrist_visible = lw_v > 0.55 and rw_v > 0.55
    shoulder_visible = ls_v > 0.55 and rs_v > 0.55
    hip_visible = lh_v > 0.5 and rh_v > 0.5

    if wrist_visible and shoulder_visible:
        wrists_above_shoulders = lw_y < ls_y - 0.04 and rw_y < rs_y - 0.04
        wrists_far_apart = abs(lw_x - rw_x) > abs(ls_x - rs_x) * 1.25
        if wrists_above_shoulders and wrists_far_apart:
            return "celebrating", 0.93
        if wrists_above_shoulders:
            return "hands raised", 0.85
        if lw_y < ls_y - 0.04:
            return "left hand raised", 0.78
        if rw_y < rs_y - 0.04:
            return "right hand raised", 0.78

    if nose_v > 0.55:
        left_face_distance = _distance((lw_x, lw_y), (nose_x, nose_y)) if lw_v > 0.55 else 9.0
        right_face_distance = _distance((rw_x, rw_y), (nose_x, nose_y)) if rw_v > 0.55 else 9.0
        if min(left_face_distance, right_face_distance) < 0.12:
            return "thinking / face touch", 0.8

    if hip_visible and wrist_visible:
        left_hip_distance = _distance((lw_x, lw_y), (lh_x, lh_y))
        right_hip_distance = _distance((rw_x, rw_y), (rh_x, rh_y))
        if left_hip_distance < 0.13 and right_hip_distance < 0.13:
            return "hands on hips", 0.74

    if shoulder_visible and nose_v > 0.55:
        shoulder_mid_x = (ls_x + rs_x) / 2
        offset = nose_x - shoulder_mid_x
        if offset > 0.09:
            return "leaning right", 0.68
        if offset < -0.09:
            return "leaning left", 0.68

    if shoulder_visible:
        return "neutral stance", 0.58

    return "pose not detected", 0.0


def analyze_frame(
    frame_bgr: np.ndarray,
    model: Sequential,
    face_cascade: cv2.CascadeClassifier,
    pose_estimator: mp_pose.Pose,
    model_lock: threading.Lock,
    action_history: Optional[deque[str]] = None,
    emotion_history: Optional[deque[str]] = None,
) -> tuple[np.ndarray, dict[str, object]]:
    frame = frame_bgr.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose_estimator.process(rgb)
    action_label, action_confidence = infer_action(pose_results.pose_landmarks)

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if action_history is not None and action_label != "pose not detected":
        action_history.append(action_label)
        action_label = Counter(action_history).most_common(1)[0][0]

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

    dominant_emotion = "no face detected"
    dominant_confidence = 0.0

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        resized = cv2.resize(roi_gray, (48, 48)).astype("float32") / 255.0
        cropped_img = np.expand_dims(np.expand_dims(resized, axis=-1), axis=0)

        with model_lock:
            prediction = model.predict(cropped_img, verbose=0)[0]

        maxindex = int(np.argmax(prediction))
        emotion = EMOTIONS[maxindex]
        confidence = float(prediction[maxindex])

        if confidence >= dominant_confidence:
            dominant_emotion = emotion
            dominant_confidence = confidence

        cv2.rectangle(frame, (x, y - 10), (x + w, y + h + 10), (37, 99, 235), 2)
        cv2.putText(
            frame,
            f"{emotion} {confidence:.0%}",
            (x, max(30, y - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if emotion_history is not None and dominant_emotion != "no face detected":
        emotion_history.append(dominant_emotion)
        dominant_emotion = Counter(emotion_history).most_common(1)[0][0]

    overlay_lines = [
        f"Action: {action_label} ({action_confidence:.0%})" if action_confidence else "Action: waiting for pose",
        f"Emotion: {dominant_emotion} ({dominant_confidence:.0%})" if dominant_confidence else "Emotion: waiting for face",
    ]

    for idx, line in enumerate(overlay_lines):
        cv2.putText(
            frame,
            line,
            (20, 35 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 180),
            2,
            cv2.LINE_AA,
        )

    summary = {
        "action": action_label,
        "action_confidence": round(action_confidence, 3),
        "emotion": dominant_emotion,
        "emotion_confidence": round(dominant_confidence, 3),
        "faces_detected": int(len(faces)),
    }
    return frame, summary


class EmotionActionProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.model = load_emotion_model()
        self.face_cascade = load_face_cascade()
        self.pose_estimator = create_pose_estimator()
        self.model_lock = threading.Lock()
        self.action_history: deque[str] = deque(maxlen=10)
        self.emotion_history: deque[str] = deque(maxlen=10)
        self.latest_result: dict[str, object] = {
            "action": "waiting for pose",
            "action_confidence": 0.0,
            "emotion": "waiting for face",
            "emotion_confidence": 0.0,
            "faces_detected": 0,
        }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        annotated_frame, summary = analyze_frame(
            image,
            self.model,
            self.face_cascade,
            self.pose_estimator,
            self.model_lock,
            self.action_history,
            self.emotion_history,
        )
        self.latest_result = summary
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


def main() -> None:
    st.set_page_config(page_title="EMOTIFY Live", page_icon=":camera:", layout="wide")

    if not MODEL_PATH.exists():
        st.error(f"Missing model weights at {MODEL_PATH}")
        st.stop()

    if not FACE_CASCADE_PATH.exists():
        st.error(f"Missing Haar cascade at {FACE_CASCADE_PATH}")
        st.stop()

    st.title("EMOTIFY Live")
    st.caption("Browser webcam emotion detection with pose-based action recognition.")

    with st.sidebar:
        st.subheader("How It Works")
        st.write("1. Open the app over HTTPS.")
        st.write("2. Click Start and allow camera access.")
        st.write("3. The browser streams frames for live emotion and action analysis.")
        st.info("This app uses a trained facial emotion model plus pose heuristics for action recognition.")

    col_live, col_upload = st.columns([1.4, 1])

    with col_live:
        st.subheader("Live Webcam")
        webrtc_streamer(
            key="emotify-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=EmotionActionProcessor,
            async_processing=True,
        )

    with col_upload:
        st.subheader("Quick Test With an Image")
        uploaded_file = st.file_uploader("Upload a frame", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            pose_estimator = create_pose_estimator()
            preview, summary = analyze_frame(
                input_image,
                load_emotion_model(),
                load_face_cascade(),
                pose_estimator,
                threading.Lock(),
            )
            metric_cols = st.columns(3)
            metric_cols[0].metric("Emotion", str(summary["emotion"]))
            metric_cols[1].metric("Action", str(summary["action"]))
            metric_cols[2].metric("Faces", str(summary["faces_detected"]))
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Processed image", use_container_width=True)
        else:
            st.write("Upload a photo if you want to test the model without opening the webcam.")


if __name__ == "__main__":
    main()
