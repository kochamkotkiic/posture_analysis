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

    # Funkcja pomocnicza do pobierania listy profili
    def get_profiles_list():
        profiles = []
        if os.path.exists(PROFILES_DIR):
            for f in glob.glob(os.path.join(PROFILES_DIR, "*.json")):
                profiles.append(os.path.basename(f).replace(".json", ""))
        return profiles

    # 1. Wysyłamy początkową listę profili
    await websocket.send(json.dumps({"type": "profiles_list", "data": get_profiles_list()}))

    # 2. Czekamy na akcję użytkownika (wybór profilu lub stworzenie nowego)
    while True:
        msg = await websocket.recv()
        client_data = json.loads(msg)

        # Obsługa tworzenia nowego profilu
        if client_data.get("type") == "create_profile":
            profile_name = client_data.get("name", "").strip()
            if profile_name:
                print(f"\n[SERWER] Rozpoczęcie kalibracji dla profilu: {profile_name}")

                # Rozpocznij kalibrację w Electronie
                cap = cv2.VideoCapture(0)
                mp_pose = mp.solutions.pose
                mp_drawing = mp.solutions.drawing_utils

                samples = []
                collecting = False
                start_time = None
                calibration_data = None

                with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1) as pose:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            await asyncio.sleep(0.01)
                            continue

                        frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = pose.process(rgb)

                        visibility = 0.0
                        progress = 0.0

                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                            )

                            lm = results.pose_landmarks.landmark
                            visibility = (lm[11].visibility + lm[12].visibility) / 2.0

                            # Zbieranie próbek
                            if collecting and visibility > 0.6:
                                sample = {}
                                for name, idx in LANDMARKS_TO_USE.items():
                                    p = lm[idx]
                                    sample[f"{name}_x"] = p.x
                                    sample[f"{name}_y"] = p.y
                                    sample[f"{name}_z"] = p.z
                                samples.append(sample)

                        # Sprawdź czas kalibracji
                        if collecting and start_time:
                            elapsed = time.time() - start_time
                            progress = min(elapsed / 5.0, 1.0)

                            if elapsed >= 5.0:
                                if len(samples) >= 10:
                                    # Uśrednij próbki
                                    keys = samples[0].keys()
                                    calibration_data = {}
                                    for key in keys:
                                        values = [s[key] for s in samples]
                                        calibration_data[key] = float(np.mean(values))
                                    print(f"✅ Zebrano {len(samples)} próbek kalibracyjnych!")
                                else:
                                    print("❌ Za mało próbek!")
                                    samples = []
                                    collecting = False
                                    start_time = None

                                if calibration_data:
                                    break

                        # Wyślij klatkę do Electrona
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')

                        send_data = {
                            "type": "calibration_frame",
                            "image": frame_base64,
                            "visibility": float(visibility),
                            "collecting": collecting,
                            "progress": float(progress)
                        }

                        try:
                            await websocket.send(json.dumps(send_data))
                            await asyncio.sleep(0.03)

                            # Sprawdź czy nadeszła komenda
                            try:
                                msg = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                                cmd = json.loads(msg)
                                if cmd.get("type") == "start_calibration" and not collecting:
                                    collecting = True
                                    start_time = time.time()
                                    samples = []
                                    print("▶ Rozpoczęto zbieranie próbek...")
                                elif cmd.get("type") == "cancel_calibration":
                                    print("❌ Anulowano kalibrację")
                                    cap.release()
                                    await websocket.send(json.dumps({"type": "profiles_list", "data": get_profiles_list()}))
                                    break
                            except asyncio.TimeoutError:
                                pass

                        except websockets.exceptions.ConnectionClosed:
                            print("\n[SERWER] Połączenie zamknięte podczas kalibracji")
                            cap.release()
                            return

                cap.release()

                # Zapisz profil jeśli kalibracja się powiodła
                if calibration_data:
                    if not os.path.exists(PROFILES_DIR):
                        os.makedirs(PROFILES_DIR)

                    profile_path = os.path.join(PROFILES_DIR, f"{profile_name}.json")
                    profile_content = {
                        "name": profile_name,
                        "calibration": calibration_data,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }

                    with open(profile_path, "w", encoding="utf-8") as f:
                        json.dump(profile_content, f, indent=2, ensure_ascii=False)

                    print(f"✅ Profil '{profile_name}' zapisany w {profile_path}")
                    await websocket.send(json.dumps({"type": "profile_created", "name": profile_name}))

            continue

        # Obsługa odświeżania listy profili
        if client_data.get("type") == "get_profiles":
            await websocket.send(json.dumps({"type": "profiles_list", "data": get_profiles_list()}))
            continue

        # Obsługa wyboru profilu
        if client_data.get("type") == "select_profile":
            break

    # 3. Kontynuuj z wybranym profilem
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