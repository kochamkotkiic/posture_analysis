import sys
import os
import time
import cv2
import joblib
import numpy as np
import mediapipe as mp
import asyncio
import websockets
import json
import base64
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.core.normalizer import normalize_features

MODEL_PATH = os.path.join(BASE_DIR, "models", "posture_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
PROFILES_DIR = os.path.join(BASE_DIR, "data", "profiles")

LANDMARKS_TO_USE = {
    "nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12
}


def extract_features(landmarks):
    features = []
    for name, idx in LANDMARKS_TO_USE.items():
        lm = landmarks[idx]
        features.extend([lm.x, lm.y, lm.z])
    return features


# Ładujemy model z Jupytera na samym początku
print("\n[AI] Wczytuję model sieci...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None


async def posture_server(websocket):
    print("\n[SERWER] Interfejs połączony! Przesyłam listę profili...")

    # 1. Zbieramy listę plików .json z profilami
    profiles_list = []
    if os.path.exists(PROFILES_DIR):
        for f in glob.glob(os.path.join(PROFILES_DIR, "*.json")):
            profiles_list.append(os.path.basename(f).replace(".json", ""))

    # 2. Wysyłamy profile do Electrona
    await websocket.send(json.dumps({"type": "profiles_list", "data": profiles_list}))

    # 3. Czekamy, aż użytkownik kliknie profil w Electronie
    msg = await websocket.recv()
    client_data = json.loads(msg)

    if client_data.get("type") == "select_profile":
        profile_name = client_data["name"]
        profile_path = os.path.join(PROFILES_DIR, f"{profile_name}.json")

        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        calibration = profile_data.get("calibration")
        if not calibration:
            print(f"[BŁĄD] Profil {profile_name} nie ma kalibracji!")
            return

        print(f"\n[SERWER] Załadowano profil '{profile_name}'. Odpalam kamerę!")

        # 4. START KAMERY
        cap = cv2.VideoCapture(0)
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        predictions = []
        SMOOTH_N = 10
        bad_since = None
        bad_seconds = 0.0

        with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                label = 0
                confidence = 1.0

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                    )
                    raw_features = extract_features(results.pose_landmarks.landmark)
                    norm_features = normalize_features(raw_features, calibration)

                    X = np.array(norm_features).reshape(1, -1)
                    if scaler: X = scaler.transform(X)

                    pred = model.predict(X)[0]
                    proba = model.predict_proba(X)[0]
                    confidence = float(proba[pred])

                    predictions.append(pred)
                    if len(predictions) > SMOOTH_N: predictions.pop(0)
                    label = 1 if predictions.count(1) > SMOOTH_N // 2 else 0

                now = time.time()
                if label == 1:
                    if bad_since is None: bad_since = now
                    bad_seconds = now - bad_since
                else:
                    bad_since = None
                    bad_seconds = 0.0

                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                send_data = {
                    "type": "video_frame",
                    "label": int(label),
                    "confidence": confidence,
                    "bad_seconds": float(bad_seconds),
                    "image": frame_base64
                }

                try:
                    await websocket.send(json.dumps(send_data))
                    await asyncio.sleep(0.03)
                except websockets.exceptions.ConnectionClosed:
                    print("\n[SERWER] Zamknięto okno aplikacji. Python gotowy na kolejne połączenie.")
                    break

        cap.release()


async def main():
    print("\n🚀 Otwieram bramy WebSockets (Czekam na Electrona)...")
    async with websockets.serve(posture_server, "localhost", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())