import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

try:
    import mediapipe as mp
except ImportError:
    mp = None

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize MediaPipe Holistic for full-body tracking when available
if mp is not None and hasattr(mp, "solutions"):
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
else:
    mp_holistic = None
    mp_drawing = None

# Command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
mode = ap.parse_args().mode

# Define possible emotions from FER-2013 dataset
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Action mapping based on emotions
ACTION_TO_EMOTION = {
    "smiling": "happy",
    "crying": "sad",
    "jumping with joy": "happy",
    "fighting": "angry",
    "laughing": "happy"
}

def suggest_action(emotion):
    """Suggests an action based on detected emotion."""
    actions = [action for action, mapped_emotion in ACTION_TO_EMOTION.items() if mapped_emotion == emotion]
    return actions[0] if actions else "No action suggestion available"

# Function to plot accuracy and loss curves
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(model_history.history['accuracy'])
    axs[0].plot(model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')
    
    axs[1].plot(model_history.history['loss'])
    axs[1].plot(model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

# Define data generators
train_dir = os.path.join(BASE_DIR, 'data', 'train')
val_dir = os.path.join(BASE_DIR, 'data', 'test')
num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50
model_path = os.path.join(BASE_DIR, 'model.h5')
face_cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

# Create CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

if mode == "train":
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            "Training data not found. Place the FER-2013 image folders in "
            f"'{train_dir}' and '{val_dir}' before running train mode."
        )

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(48, 48), batch_size=batch_size, color_mode="grayscale", class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(48, 48), batch_size=batch_size, color_mode="grayscale", class_mode='categorical')

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001, decay=1e-6),
        metrics=['accuracy']
    )
    model_info = model.fit(
        train_generator, steps_per_epoch=num_train // batch_size,
        epochs=num_epoch, validation_data=validation_generator,
        validation_steps=num_val // batch_size)
    plot_model_history(model_info)
    model.save_weights(model_path)

elif mode == "display":
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pretrained weights were not found at '{model_path}'. "
            "Train the model first or add the downloaded model.h5 file to the src folder."
        )

    if not os.path.exists(face_cascade_path):
        raise FileNotFoundError(f"Face cascade file missing at '{face_cascade_path}'.")

    model.load_weights(model_path)
    cv2.ocl.setUseOpenCL(False)
    emotion_dict = {i: EMOTIONS[i] for i in range(len(EMOTIONS))}
    cap = cv2.VideoCapture(0)
    facecasc = cv2.CascadeClassifier(face_cascade_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Check camera permissions and try again.")
    
    holistic_context = (
        mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        if mp_holistic is not None else None
    )

    if holistic_context is None:
        print("MediaPipe holistic landmarks are unavailable. Running emotion detection without pose tracking.")

    if holistic_context is not None:
        holistic_manager = holistic_context
    else:
        class _NullContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        holistic_manager = _NullContext()

    with holistic_manager as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if holistic is not None:
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)

                # Convert back to BGR for OpenCV
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                # Draw pose and hand landmarks when MediaPipe supports them
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Face detection for emotion analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img, verbose=0)
                maxindex = int(np.argmax(prediction))
                detected_emotion = emotion_dict[maxindex]
                action_suggestion = suggest_action(detected_emotion)

                cv2.putText(frame, f"{detected_emotion} ({action_suggestion})", (x+20, y-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Emotion & Action Tracking', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
else:
    raise ValueError("Invalid mode. Use '--mode train' or '--mode display'.")
