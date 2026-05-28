import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import tempfile
from dotenv import load_dotenv

load_dotenv()

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

import google.protobuf.message_factory

if not hasattr(google.protobuf.message_factory, "GetMessageClass"):
    try:
        google.protobuf.message_factory.GetMessageClass = (
            google.protobuf.message_factory.MessageFactory().GetMessageClass
        )
        print("Applied protobuf compatibility patch")
    except Exception as exc:
        print(f"Could not apply protobuf compatibility patch: {exc}")


# ============================================================================
# Notebook-aligned GCN helpers
# ============================================================================

def relative_normalize(lh, rh, pose):
    if np.any(rh):
        origin = rh.reshape(21, 3)[0]
    elif np.any(lh):
        origin = lh.reshape(21, 3)[0]
    else:
        origin = np.zeros(3)

    lh = lh.reshape(21, 3) - origin
    rh = rh.reshape(21, 3) - origin

    pose = pose.reshape(33, 4)
    pose[:, :3] -= origin

    l_shoulder = pose[11][:3]
    r_shoulder = pose[12][:3]
    scale = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6

    lh /= scale
    rh /= scale
    pose[:, :3] /= scale

    return np.concatenate([lh.flatten(), rh.flatten(), pose.flatten()])


def get_adjacency_matrix():
    num_joints = 75
    A = np.zeros((num_joints, num_joints), dtype=np.float32)

    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    for i, j in hand_connections:
        A[i, j] = 1
        A[j, i] = 1

    for i, j in hand_connections:
        A[i + 21, j + 21] = 1
        A[j + 21, i + 21] = 1

    pose_connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
        (15, 17), (15, 19), (15, 21),
        (16, 18), (16, 20), (16, 22),
        (27, 31), (28, 32),
    ]

    for i, j in pose_connections:
        A[i + 42, j + 42] = 1
        A[j + 42, i + 42] = 1

    A[0, 57] = 1
    A[57, 0] = 1
    A[21, 58] = 1
    A[58, 21] = 1

    np.fill_diagonal(A, 1)
    D = np.diag(np.sum(A, axis=1) ** -0.5)
    A_norm = D @ A @ D

    return A_norm.astype(np.float32)


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, out_features, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.activation_name = activation
        self.activation_fn = tf.keras.activations.get(activation)

    def build(self, input_shape):
        in_features = input_shape[-1]
        self.W = self.add_weight(
            shape=(in_features, self.out_features),
            initializer="glorot_uniform",
            trainable=True,
            name="gcn_weight",
        )
        self.b = self.add_weight(
            shape=(self.out_features,),
            initializer="zeros",
            trainable=True,
            name="gcn_bias",
        )

    def call(self, x, A):
        support = tf.matmul(x, self.W) + self.b
        output = tf.matmul(A, support)
        return self.activation_fn(output)

    def get_config(self):
        config = super().get_config()
        config.update({"out_features": self.out_features, "activation": self.activation_name})
        return config


class GCNStep(tf.keras.layers.Layer):
    def __init__(self, out_features, A, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.activation_name = activation
        self.A_init = A
        self.gcn = GCNLayer(out_features, activation=activation)

    def build(self, input_shape):
        self.A = self.add_weight(
            shape=self.A_init.shape,
            initializer=tf.keras.initializers.Constant(self.A_init),
            trainable=False,
            name="adjacency",
        )
        super().build(input_shape)

    def call(self, x):
        A_batch = tf.tile(tf.expand_dims(self.A, 0), [tf.shape(x)[0], 1, 1])
        return self.gcn(x, A_batch)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.out_features)

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_features": self.out_features,
            "activation": self.activation_name,
            "A": self.A_init.tolist(),
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["A"] = np.array(config["A"], dtype=np.float32)
        return cls(**config)


def build_gcn_lstm_model(num_joints, joint_features, seq_len, num_classes, A):
    inputs = tf.keras.Input(shape=(seq_len, num_joints * joint_features))

    x = tf.keras.layers.Reshape((seq_len, num_joints, joint_features))(inputs)

    x = tf.keras.layers.TimeDistributed(
        GCNStep(64, A, activation="relu", name="gcn_step1"), name="td_gcn1"
    )(x)

    x = tf.keras.layers.TimeDistributed(
        GCNStep(128, A, activation="relu", name="gcn_step2"), name="td_gcn2"
    )(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(), name="td_flatten")(x)

    x = tf.keras.layers.Masking(mask_value=0.0)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2), name="bilstm1"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=False, dropout=0.2), name="bilstm2"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def reshape_landmarks_for_gcn(X):
    samples, seq_len, _ = X.shape
    num_joints = 75

    out = np.zeros((samples, seq_len, num_joints, 4), dtype=np.float32)

    lh = X[:, :, 0:63].reshape(samples, seq_len, 21, 3)
    out[:, :, 0:21, :3] = lh

    rh = X[:, :, 63:126].reshape(samples, seq_len, 21, 3)
    out[:, :, 21:42, :3] = rh

    pose = X[:, :, 126:258].reshape(samples, seq_len, 33, 4)
    out[:, :, 42:75, :] = pose

    return out


def extract_landmarks_dummy():
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    pose = np.zeros(33 * 4)
    return np.concatenate([lh, rh, pose])


# ============================================================================
# Configuration
# ============================================================================

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "outputs", "best_isl_gcn_model.keras"))
DATA_PATH = os.getenv("DATA_PATH", os.path.join(BASE_DIR, "outputs", "processed_landmarks.pkl"))
DATASET_PATH = os.getenv(
    "DATASET_PATH",
    os.path.join(BASE_DIR, "Indian Sign Language Greetings Dataset - Sub Variant of INCLUDE"),
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_FRAMES = 60
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))


def ok(data=None, status=200):
    payload = {"ok": True}
    if data:
        payload.update(data)
    return jsonify(payload), status


def fail(message, status=400, **extra):
    payload = {"ok": False, "error": message}
    payload.update(extra)
    return jsonify(payload), status


@app.errorhandler(Exception)
def handle_exception(error):
    if isinstance(error, HTTPException):
        return fail(error.description, error.code)
    import traceback
    traceback.print_exc()
    return fail(str(error), 500)


print("Loading class names...")
try:
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    class_names = data["class_names"]
    print(f"Classes loaded: {class_names}")
except Exception as exc:
    print(f"Error loading class names: {exc}")
    class_names = []


print("Building and loading model...")
try:
    A = get_adjacency_matrix()
    model = build_gcn_lstm_model(
        num_joints=75,
        joint_features=4,
        seq_len=MAX_FRAMES,
        num_classes=len(class_names),
        A=A,
    )
    model(np.zeros((1, MAX_FRAMES, 75 * 4), dtype=np.float32))
    model.load_weights(MODEL_PATH)
    print("Model loaded successfully")
except Exception as exc:
    print(f"Error loading model: {exc}")
    model = None


mp_holistic = mp.solutions.holistic


# ============================================================================
# Notebook-identical inference preprocessing
# ============================================================================

def extract_landmarks(results):
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    if results.pose_landmarks:
        pose = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ]).flatten()
    else:
        pose = np.zeros(33 * 4)

    return relative_normalize(lh, rh, pose)


def process_video(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        dummy_landmarks = extract_landmarks_dummy()
        return np.array([dummy_landmarks] * max_frames, dtype=np.float32)

    landmark_sequence = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for i, frame in enumerate(frames):
            if i >= max_frames:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            landmark_sequence.append(extract_landmarks(results))

    if len(landmark_sequence) > 0:
        while len(landmark_sequence) < max_frames:
            landmark_sequence.append(np.zeros_like(landmark_sequence[0]))
    else:
        dummy_landmarks = extract_landmarks_dummy()
        landmark_sequence = [dummy_landmarks] * max_frames

    return np.array(landmark_sequence[:max_frames], dtype=np.float32)


def predict_landmarks(landmark_seq):
    landmark_seq = landmark_seq[np.newaxis]
    landmark_gcn = reshape_landmarks_for_gcn(landmark_seq)
    landmark_input = landmark_gcn.reshape(1, MAX_FRAMES, 75 * 4)

    probs = model.predict(landmark_input, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    prediction = class_names[pred_idx] if confidence >= CONFIDENCE_THRESHOLD else "Uncertain"

    return {
        "prediction": prediction,
        "confidence": confidence,
        "all_predictions": [
            {"sign": class_names[i], "score": float(probs[i])}
            for i in np.argsort(probs)[-3:][::-1]
        ],
    }


# ============================================================================
# Routes
# ============================================================================

@app.route("/health", methods=["GET"])
def health():
    return ok({
        "status": "ok",
        "model_loaded": model is not None,
        "classes": class_names,
        "sentence_generation": bool(GEMINI_API_KEY),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return fail("Model not loaded", 500)

    if "video" not in request.files:
        return fail("No video file provided", 400)

    video_file = request.files["video"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        video_path = tmp.name
        video_file.save(video_path)

    try:
        landmark_seq = process_video(video_path, max_frames=MAX_FRAMES)

        if np.all(landmark_seq == 0):
            return ok({
                "prediction": "No Motion",
                "confidence": 0.0,
                "message": "No landmarks detected in video",
            })

        return ok(predict_landmarks(landmark_seq))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return fail(str(exc), 500)
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


@app.route("/api/generate-sentence", methods=["POST"])
def generate_sentence():
    payload = request.get_json(silent=True) or {}
    words = payload.get("words", [])

    if not isinstance(words, list):
        return fail("words must be a list", 400)

    cleaned_words = [
        str(word).strip()
        for word in words
        if str(word).strip() and str(word).strip().lower() not in {"no motion", "uncertain"}
    ]

    if not cleaned_words:
        return fail("No predicted words were provided", 400)

    prompt = f"""You are an intelligent Indian Sign Language sentence reconstruction assistant.

The input is a sequence of predicted sign-language words from a real-time gesture recognition system.

The sequence may:
* contain repeated words
* miss grammar words
* miss articles
* contain imperfect ordering

Your task:
* reconstruct the most natural and meaningful English sentence possible
* preserve intended meaning
* avoid hallucinating unrelated content
* keep sentence concise and human-like

Input words:
{cleaned_words}

Return ONLY the corrected sentence."""

    try:
        from langchain_core.messages import HumanMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.2,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return ok({"sentence": response.content.strip(), "words": cleaned_words})
    except ImportError:
        fallback = " ".join(cleaned_words).strip()
        fallback = fallback[:1].upper() + fallback[1:]
        if not fallback.endswith((".", "!", "?")):
            fallback += "."
        return ok({
            "sentence": fallback,
            "words": cleaned_words,
            "warning": "LangChain Gemini packages are not installed; returned a local fallback sentence.",
        })
    except Exception as exc:
        return fail(f"Sentence generation failed: {exc}", 502)


@app.route("/api/signs", methods=["GET"])
def get_signs():
    signs = []
    if os.path.exists(DATASET_PATH):
        for class_name in sorted(os.listdir(DATASET_PATH)):
            class_dir = os.path.join(DATASET_PATH, class_name)
            if not os.path.isdir(class_dir):
                continue

            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith((".mp4", ".webm", ".mov")):
                    signs.append({
                        "name": class_name,
                        "category": class_name[:1].upper(),
                        "video_url": f"http://localhost:5000/videos/{class_name}/{filename}",
                    })
                    break

    return ok({"signs": signs})


@app.route("/videos/<path:filename>")
def serve_video(filename):
    from flask import send_from_directory
    return send_from_directory(DATASET_PATH, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)