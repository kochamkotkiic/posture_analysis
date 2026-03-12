import cv2
import mediapipe as mp
import pandas as pd
import os
import time

# ── konfiguracja ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "posture_data.csv")
COLLECT_INTERVAL = 0.1   # co ile sekund zapisujemy próbkę (10 Hz)

# Punkty których używamy (górna połowa ciała)
LANDMARKS_TO_USE = {
    "nose":           0,
    "left_eye":       2,
    "right_eye":      5,
    "left_ear":       7,
    "right_ear":      8,
    "left_shoulder":  11,
    "right_shoulder": 12,
}

# ── inicjalizacja MediaPipe ───────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_features(landmarks) -> dict:
    """Wyciąga współrzędne X,Y,Z wybranych punktów jako płaski słownik."""
    features = {}
    for name, idx in LANDMARKS_TO_USE.items():
        lm = landmarks[idx]
        features[f"{name}_x"] = round(lm.x, 6)
        features[f"{name}_y"] = round(lm.y, 6)
        features[f"{name}_z"] = round(lm.z, 6)
    return features


def draw_status(frame, label: str, count: int, mode: str, visibility: float = 0.0):
    color = (0, 200, 0) if mode == "good" else (0, 0, 220) if mode == "bad" else (180, 180, 180)
    cv2.rectangle(frame, (0, 0), (640, 60), (30, 30, 30), -1)
    cv2.putText(frame, f"Tryb: {label}  |  Probki: {count}",
                (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # ← NOWE: pasek widoczności barków
    vis_color = (0, 220, 0) if visibility > 0.7 else (0, 165, 255) if visibility > 0.4 else (0, 0, 220)
    cv2.putText(frame, f"Widocznosc barkow: {visibility:.2f}",
                (15, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, vis_color, 2)

    cv2.rectangle(frame, (0, frame.shape[0] - 50), (640, frame.shape[0]), (30, 30, 30), -1)
    cv2.putText(frame, "G = dobra postawa  |  B = zla postawa  |  Q = koniec",
                (10, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

def main():
    os.makedirs("../../data", exist_ok=True)

    # Wczytaj istniejące dane jeśli plik już istnieje
    if os.path.exists(OUTPUT_PATH):
        existing_df = pd.read_csv(OUTPUT_PATH)
        records = existing_df.to_dict("records")
        print(f"Wczytano {len(records)} istniejących próbek.")
    else:
        records = []

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mode = "idle"   # "idle" | "good" | "bad"
    last_saved = 0.0

    with mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1
    ) as pose:

        print("\n🟢 Kamera uruchomiona!")
        print("Naciśnij G aby nagrywać DOBRĄ postawę")
        print("Naciśnij B aby nagrywać ZŁĄ postawę")
        print("Naciśnij Q aby zapisać i zakończyć\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Błąd kamery!")
                break

            frame = cv2.flip(frame, 1)  # lustrzane odbicie
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # ── rysowanie szkieletu ───────────────────────────
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )

            # ── zbieranie danych ──────────────────────────────
            now = time.time()
            if mode != "idle" and results.pose_landmarks:
                if now - last_saved >= COLLECT_INTERVAL:
                    features = extract_features(results.pose_landmarks.landmark)
                    features["label"] = 0 if mode == "good" else 1
                    records.append(features)
                    last_saved = now

            # ── HUD ───────────────────────────────────────────
            # pobierz widoczność barków
            visibility = 0.0
            if results.pose_landmarks:
                left_vis = results.pose_landmarks.landmark[11].visibility
                right_vis = results.pose_landmarks.landmark[12].visibility
                visibility = (left_vis + right_vis) / 2.0

            label_text = "DOBRA (0)" if mode == "good" else "ZLA (1)" if mode == "bad" else "PAUZA"
            draw_status(frame, label_text, len(records), mode, visibility)

            cv2.imshow("PostureGuard - Zbieranie danych", frame)

            # ── klawisze ─────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("g") or key == ord("G"):
                mode = "good"
                print(f"▶ Nagrywam DOBRĄ postawę... (próbki: {len(records)})")
            elif key == ord("b") or key == ord("B"):
                mode = "bad"
                print(f"▶ Nagrywam ZŁĄĄ postawę... (próbki: {len(records)})")
            elif key == ord(" "):
                mode = "idle"
                print("⏸ Pauza")
            elif key == ord("q") or key == ord("Q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    # ── zapis do CSV ──────────────────────────────────────────
    if records:
        df = pd.DataFrame(records)
        df.to_csv(OUTPUT_PATH, index=False)
        good = (df["label"] == 0).sum()
        bad  = (df["label"] == 1).sum()
        print(f"\n✅ Zapisano {len(df)} próbek → {OUTPUT_PATH}")
        print(f"   Dobra postawa (0): {good} próbek")
        print(f"   Zła postawa  (1): {bad} próbek")
    else:
        print("Brak danych do zapisania.")


if __name__ == "__main__":
    main()