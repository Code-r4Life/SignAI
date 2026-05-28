import cv2
import os
import pickle
import numpy as np
import mediapipe as mp
import tensorflow as tf

def relative_normalize(lh, rh, pose):
      # Choose wrist as origin
      if np.any(rh):
          origin = rh.reshape(21,3)[0]
      elif np.any(lh):
          origin = lh.reshape(21,3)[0]
      else:
          origin = np.zeros(3)

      lh = lh.reshape(21,3) - origin
      rh = rh.reshape(21,3) - origin

      pose = pose.reshape(33,4)
      pose[:,:3] -= origin

      # Scale by shoulder width
      l_shoulder = pose[11][:3]
      r_shoulder = pose[12][:3]
      scale = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6

      lh /= scale
      rh /= scale
      pose[:,:3] /= scale

      return np.concatenate([lh.flatten(), rh.flatten(), pose.flatten()])

def get_adjacency_matrix():
    """
    Define adjacency for: left_hand(21) + right_hand(21) + pose(33) = 75 joints
    Indices: 0-20 = left hand, 21-41 = right hand, 42-74 = pose
    """
    num_joints = 75
    A = np.zeros((num_joints, num_joints), dtype=np.float32)

    # MediaPipe hand connections (same for both hands)
    hand_connections = [
        (0,1),(1,2),(2,3),(3,4),          # thumb
        (0,5),(5,6),(6,7),(7,8),           # index
        (0,9),(9,10),(10,11),(11,12),      # middle
        (0,13),(13,14),(14,15),(15,16),    # ring
        (0,17),(17,18),(18,19),(19,20),    # pinky
        (5,9),(9,13),(13,17)               # palm
    ]

    # Left hand (offset 0)
    for i, j in hand_connections:
        A[i, j] = 1; A[j, i] = 1

    # Right hand (offset 21)
    for i, j in hand_connections:
        A[i+21, j+21] = 1; A[j+21, i+21] = 1

    # MediaPipe pose connections (offset 42)
    pose_connections = [
        (0,1),(1,2),(2,3),(3,7),           # face left
        (0,4),(4,5),(5,6),(6,8),           # face right
        (9,10),                            # mouth
        (11,12),(11,13),(13,15),           # left arm
        (12,14),(14,16),                   # right arm
        (11,23),(12,24),(23,24),           # torso
        (23,25),(25,27),(27,29),(29,31),   # left leg
        (24,26),(26,28),(28,30),(30,32),   # right leg
        (15,17),(15,19),(15,21),           # left hand tips
        (16,18),(16,20),(16,22),           # right hand tips
        (27,31),(28,32)
    ]

    for i, j in pose_connections:
        A[i+42, j+42] = 1; A[j+42, i+42] = 1

    # Cross-body connections: wrists to pose wrists
    # Left hand wrist (0) → Pose left wrist (15+42=57)
    A[0, 57] = 1; A[57, 0] = 1
    # Right hand wrist (21) → Pose right wrist (16+42=58)
    A[21, 58] = 1; A[58, 21] = 1

    # Add self-loops
    np.fill_diagonal(A, 1)

    # Normalize: D^(-1/2) * A * D^(-1/2)
    D = np.diag(np.sum(A, axis=1) ** -0.5)
    A_norm = D @ A @ D

    return A_norm.astype(np.float32)

# ============================================================================
# GCN LAYER & MODEL
# ============================================================================

class GCNLayer(tf.keras.layers.Layer):

    def __init__(self, out_features, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.activation_name = activation
        self.activation_fn = tf.keras.activations.get(activation)

    def build(self, input_shape):
        in_features = input_shape[-1]
        self.W = self.add_weight(shape=(in_features, self.out_features),
                                 initializer='glorot_uniform', trainable=True, name='gcn_weight')
        self.b = self.add_weight(shape=(self.out_features,),
                                 initializer='zeros', trainable=True, name='gcn_bias')

    def call(self, x, A):
        support = tf.matmul(x, self.W) + self.b
        output = tf.matmul(A, support)
        return self.activation_fn(output)

    def get_config(self):
        config = super().get_config()
        config.update({'out_features': self.out_features, 'activation': self.activation_name})
        return config


class GCNStep(tf.keras.layers.Layer):
    """Per-frame GCN with baked-in adjacency — fully serializable, no Lambda"""

    def __init__(self, out_features, A, activation='relu', **kwargs):
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
            name='adjacency'
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
            'out_features': self.out_features,
            'activation': self.activation_name,
            'A': self.A_init.tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['A'] = np.array(config['A'], dtype=np.float32)
        return cls(**config)


def build_gcn_lstm_model(num_joints, joint_features, seq_len, num_classes, A):

    inputs = tf.keras.Input(shape=(seq_len, num_joints * joint_features))

    x = tf.keras.layers.Reshape((seq_len, num_joints, joint_features))(inputs)

    x = tf.keras.layers.TimeDistributed(
        GCNStep(64, A, activation='relu', name='gcn_step1'), name='td_gcn1'
    )(x)

    x = tf.keras.layers.TimeDistributed(
        GCNStep(128, A, activation='relu', name='gcn_step2'), name='td_gcn2'
    )(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten(), name='td_flatten'
    )(x)

    x = tf.keras.layers.Masking(mask_value=0.0)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2), name='bilstm1'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=False, dropout=0.2), name='bilstm2'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def reshape_landmarks_for_gcn(X):
    """
    Convert flat landmark array → per-joint feature array.

    Input X shape:  (samples, seq_len, 258)
      258 = left_hand(63) + right_hand(63) + pose(132)
           = 21*3  +  21*3  +  33*4

    Output shape: (samples, seq_len, 75, 4)
      75 joints, each with up to 4 features (x, y, z, [visibility or 0])
    """
    samples, seq_len, _ = X.shape
    num_joints = 75  # 21 + 21 + 33

    out = np.zeros((samples, seq_len, num_joints, 4), dtype=np.float32)

    # Left hand: joints 0-20, features x,y,z (no visibility → pad 0)
    lh = X[:, :, 0:63].reshape(samples, seq_len, 21, 3)
    out[:, :, 0:21, :3] = lh

    # Right hand: joints 21-41
    rh = X[:, :, 63:126].reshape(samples, seq_len, 21, 3)
    out[:, :, 21:42, :3] = rh

    # Pose: joints 42-74, features x,y,z,visibility
    pose = X[:, :, 126:258].reshape(samples, seq_len, 33, 4)
    out[:, :, 42:75, :] = pose

    return out  # (samples, seq_len, 75, 4)


# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = 'outputs/best_isl_gcn_model.keras'
PKL_PATH   = 'outputs/processed_landmarks.pkl'
MAX_FRAMES = 60
CAPTURE_DURATION = 4  # seconds

# ── Load class names ─────────────────────────────────────────────────────────
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)
class_names = data['class_names']
print(f"Classes: {class_names}")

# ── Build model + load weights ───────────────────────────────────────────────
A = get_adjacency_matrix()
model = build_gcn_lstm_model(
    num_joints=75, joint_features=4,
    seq_len=MAX_FRAMES, num_classes=len(class_names), A=A
)
model(np.zeros((1, MAX_FRAMES, 75 * 4), dtype=np.float32))
model.load_weights(MODEL_PATH)
print("✅ Model loaded")

# ── MediaPipe ────────────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic

# ── Webcam ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

print(f"🎥 Starting live detection. Press 'q' to quit.")

last_prediction = "Waiting..."
last_confidence = 0.0
frame_buffer    = []
is_capturing    = True
capture_start   = cv2.getTickCount() / cv2.getTickFrequency()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display      = frame.copy()
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        # ── Collect frames ───────────────────────────────────────────────────
        if is_capturing:
            frame_buffer.append(frame.copy())
            elapsed   = current_time - capture_start
            remaining = CAPTURE_DURATION - elapsed

            # ── Process when clip is full ────────────────────────────────────
            if elapsed >= CAPTURE_DURATION:
                print("🔄 Processing...")
                is_capturing = False

                landmark_sequence = []
                for i, f in enumerate(frame_buffer):
                    if i >= MAX_FRAMES:
                        break
                    rgb     = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb)

                    lh   = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
                           if results.left_hand_landmarks else np.zeros(21 * 3)
                    rh   = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
                           if results.right_hand_landmarks else np.zeros(21 * 3)
                    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten() \
                           if results.pose_landmarks else np.zeros(33 * 4)

                    landmark_sequence.append(relative_normalize(lh, rh, pose))

                # Pad
                while len(landmark_sequence) < MAX_FRAMES:
                    landmark_sequence.append(np.zeros_like(landmark_sequence[0]))
                landmark_sequence = np.array(landmark_sequence[:MAX_FRAMES])

                # Predict
                if not np.all(landmark_sequence == 0):
                    seq      = landmark_sequence[np.newaxis]
                    gcn_in   = reshape_landmarks_for_gcn(seq)
                    gcn_flat = gcn_in.reshape(1, MAX_FRAMES, 75 * 4)
                    probs    = model.predict(gcn_flat, verbose=0)[0]
                    pred_idx        = np.argmax(probs)
                    last_prediction = class_names[pred_idx]
                    last_confidence = float(probs[pred_idx])
                    print(f"✅ {last_prediction} ({last_confidence:.2f})")
                else:
                    last_prediction = "No hands detected"
                    last_confidence = 0.0

                # Reset for next clip
                frame_buffer  = []
                is_capturing  = True
                capture_start = cv2.getTickCount() / cv2.getTickFrequency()

        # ── UI overlay ───────────────────────────────────────────────────────
        h, w = display.shape[:2]

        if is_capturing:
            elapsed   = current_time - capture_start
            remaining = CAPTURE_DURATION - elapsed
            cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display, f"REC  {remaining:.1f}s",
                        (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, "Processing...",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.rectangle(display, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.putText(display, f"Prediction : {last_prediction}",
                    (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Confidence : {last_confidence:.2f}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow('ISL Live Detection', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("👋 Stopped.")