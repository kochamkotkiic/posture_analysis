import sys
import os
import cv2
import joblib
import os
import pandas as pd
import numpy as np
import mediapipe as mp
import json

PROFILE_NAME = "laptop_wyzej"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
FRAMES_DIR = os.path.join(BASE_DIR, "data", "raw_frames", "test_naturalny_1")
OUTPUT_CSV = "wyniki_z_klatek.csv"


# Standardowa konfiguracja ścieżek projektu
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


def analyze_recorded_frames(frames_dir):
    print(f"--- START ANALIZY KLATEK ---")

    # 1. Wczytanie modelu ML
    print("Wczytywanie modelu ML...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    # 2. Wczytanie profilu kalibracyjnego
    profile_path = os.path.join(PROFILES_DIR, f"{PROFILE_NAME}.json")
    if not os.path.exists(profile_path):
        print(f"❌ BŁĄD: Nie znaleziono profilu {profile_path}")
        return

    with open(profile_path, "r", encoding="utf-8") as f:
        profile_data = json.load(f)
    calibration = profile_data["calibration"]
    print(f"Wczytano bazę kalibracyjną dla profilu: {PROFILE_NAME}")

    # 3. Przygotowanie listy klatek
    if not os.path.exists(frames_dir):
        print(f"❌ BŁĄD: Nie znaleziono folderu z klatkami: {frames_dir}")
        return

    all_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    total_frames = len(all_frames)
    print(f"Znaleziono {total_frames} klatek do analizy. Rozpoczynam...")

    results = []
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1) as pose:
        for i, filename in enumerate(all_frames):

            # Wypisywanie postępu co 500 klatek, żebyś wiedziała, że program działa
            if i % 500 == 0:
                print(f"Przetworzono {i} / {total_frames} klatek...")

            filepath = os.path.join(frames_dir, filename)
            frame = cv2.imread(filepath)

            if frame is None:
                continue

            # ==========================================
            # ⚠️ UWAGA: EFEKT LUSTRA
            # Jeśli w pliku server.py (w Live aplikacji) masz frame = cv2.flip(frame, 1)
            # to tutaj też musi to zostać odznaczone!
            frame = cv2.flip(frame, 1)
            # ==========================================

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            label = -1  # Domyślnie -1 (brak osoby)

            if res.pose_landmarks:
                # Wyciągnięcie i normalizacja cech
                raw_features = extract_features(res.pose_landmarks.landmark)
                norm_features = normalize_features(raw_features, calibration)

                # Formatowanie pod model sklearn
                X = np.array(norm_features).reshape(1, -1)
                if scaler:
                    X = scaler.transform(X)

                # Predykcja
                label = int(model.predict(X)[0])

            results.append({"frame": filename, "prediction": label})

    # 4. Zapis do CSV
    df = pd.DataFrame(results)

    # Zapiszmy to w folderze Twojego projektu, tak żeby Jupyter łatwo to znalazł
    output_path = os.path.join(BASE_DIR, "data", OUTPUT_CSV)
    # Zabezpieczenie na wypadek uruchomienia w złym miejscu:
    # Jeśli BASE_DIR zawiedzie, zapisze w bieżącym folderze
    try:
        df.to_csv(output_path, index=False)
        print(f"\n✅ ZAKOŃCZONO! Zapisano wyniki pomyślnie w pliku: {output_path}")
    except Exception as e:
        local_path = OUTPUT_CSV
        df.to_csv(local_path, index=False)
        print(f"\n✅ ZAKOŃCZONO! Zapisano awaryjnie w bieżącym folderze jako: {local_path}")


if __name__ == "__main__":
    analyze_recorded_frames(FRAMES_DIR)