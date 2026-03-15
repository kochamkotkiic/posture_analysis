import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

OUTPUT_PATH      = os.path.join(BASE_DIR, "data", "posture_data.csv")
COLLECT_INTERVAL = 0.1

LANDMARKS_TO_USE = {
    "nose":           0,
    "left_eye":       2,
    "right_eye":      5,
    "left_ear":       7,
    "right_ear":      8,
    "left_shoulder":  11,
    "right_shoulder": 12,
}

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_raw_features(landmarks) -> dict:
    features = {}
    for name, idx in LANDMARKS_TO_USE.items():
        lm = landmarks[idx]
        features[f"{name}_x"] = round(lm.x, 6)
        features[f"{name}_y"] = round(lm.y, 6)
        features[f"{name}_z"] = round(lm.z, 6)
    return features


def normalize_features(raw: dict, baseline: dict) -> dict:
    """Odejmuje wartości bazowe (kalibrację) od surowych współrzędnych."""
    normalized = {}
    for key in raw:
        normalized[key] = round(raw[key] - baseline.get(key, 0.0), 6)
    return normalized


def run_calibration_phase(cap, pose) -> dict | None:
    """
    Faza kalibracji przed nagrywaniem.
    Zbiera próbki przez 5 sekund i zwraca uśrednione wartości bazowe.
    """
    print("\n" + "="*50)
    print("  FAZA KALIBRACJI")
    print("="*50)
    print("Usiądź PROSTO i naciśnij SPACJĘ aby rozpocząć...")

    CALIBRATION_SECONDS = 5
    samples    = []
    collecting = False
    start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        visibility = 0.0
        if results.pose_landmarks:
            lm        = results.pose_landmarks.landmark
            visibility = (lm[11].visibility + lm[12].visibility) / 2.0

            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
            )

            if collecting and visibility > 0.6:
                samples.append(extract_raw_features(lm))

        # sprawdź czy czas minął
        if collecting and start_time:
            elapsed  = time.time() - start_time
            progress = min(elapsed / CALIBRATION_SECONDS, 1.0)
            if elapsed >= CALIBRATION_SECONDS:
                if len(samples) >= 10:
                    print(f"✅ Kalibracja zakończona! Zebrano {len(samples)} próbek.")
                    cv2.destroyWindow("Kalibracja")
                    return _average_samples(samples)
                else:
                    print("❌ Za mało próbek — spróbuj ponownie!")
                    samples, collecting, start_time = [], False, None
        else:
            progress = 0.0

        _draw_calibration_ui(frame, collecting, progress, visibility, CALIBRATION_SECONDS)
        cv2.imshow("Kalibracja", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and not collecting and visibility > 0.6:
            collecting = True
            start_time = time.time()
            samples    = []
            print("▶ Zbieranie próbek kalibracyjnych...")
        elif key == ord("q"):
            return None

    return None


def _average_samples(samples: list[dict]) -> dict:
    keys   = samples[0].keys()
    result = {}
    for key in keys:
        result[key] = float(np.mean([s[key] for s in samples]))
    return result


def _draw_calibration_ui(frame, collecting, progress, visibility, total_secs):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.putText(frame, "KALIBRACJA — usiądź prosto!",
                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    vis_color = (0, 220, 0) if visibility > 0.7 else (0, 165, 255) if visibility > 0.4 else (0, 0, 220)
    cv2.putText(frame, f"Widoczność barków: {visibility:.2f}",
                (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vis_color, 2)

    if not collecting:
        cv2.putText(frame, "Naciśnij SPACJĘ gdy widoczność jest zielona",
                    (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
    else:
        bar_x, bar_y, bar_w, bar_h = 15, 120, w - 30, 25
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 200, 0), -1)
        secs_left = max(0, total_secs - int(progress * total_secs))
        cv2.putText(frame, f"Nie ruszaj się... {secs_left}s",
                    (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

    cv2.rectangle(frame, (0, h - 35), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "SPACJA = start kalibracji  |  Q = anuluj",
                (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


def draw_status(frame, label: str, count: int, mode: str, visibility: float = 0.0):
    color = (0, 200, 0) if mode == "good" else (0, 0, 220) if mode == "bad" else (180, 180, 180)
    cv2.rectangle(frame, (0, 0), (640, 60), (30, 30, 30), -1)
    cv2.putText(frame, f"Tryb: {label}  |  Próbki: {count}",
                (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    vis_color = (0, 220, 0) if visibility > 0.7 else (0, 165, 255) if visibility > 0.4 else (0, 0, 220)
    cv2.putText(frame, f"Widoczność barków: {visibility:.2f}",
                (15, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vis_color, 2)

    cv2.rectangle(frame, (0, frame.shape[0] - 45), (640, frame.shape[0]), (30, 30, 30), -1)
    cv2.putText(frame, "G = dobra postawa  |  B = zla postawa  |  SPACJA = pauza  |  Q = koniec",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1
    ) as pose:

        # ── FAZA 1: kalibracja ────────────────────────────────
        baseline = run_calibration_phase(cap, pose)
        if baseline is None:
            print("Anulowano — brak kalibracji.")
            cap.release()
            cv2.destroyAllWindows()
            return

        # ── FAZA 2: nagrywanie ────────────────────────────────
        # wczytaj istniejące dane
        if os.path.exists(OUTPUT_PATH):
            existing_df = pd.read_csv(OUTPUT_PATH)
            records     = existing_df.to_dict("records")
            print(f"\nWczytano {len(records)} istniejących próbek.")
        else:
            records = []

        mode       = "idle"
        last_saved = 0.0

        print("\n🟢 Nagrywanie!")
        print("G = dobra postawa | B = zła postawa | SPACJA = pauza | Q = koniec\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            visibility = 0.0

            if results.pose_landmarks:
                lm         = results.pose_landmarks.landmark
                visibility = (lm[11].visibility + lm[12].visibility) / 2.0

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )

                now = time.time()
                if mode != "idle" and visibility > 0.6:
                    if now - last_saved >= COLLECT_INTERVAL:
                        raw        = extract_raw_features(lm)
                        normalized = normalize_features(raw, baseline)
                        normalized["label"] = 0 if mode == "good" else 1
                        records.append(normalized)
                        last_saved = now

            label_text = "DOBRA (0)" if mode == "good" else "ZLA (1)" if mode == "bad" else "PAUZA"
            draw_status(frame, label_text, len(records), mode, visibility)
            cv2.imshow("PostureGuard - Zbieranie danych", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("g"), ord("G")):
                mode = "good"
                print(f"▶ Nagrywam DOBRĄ postawę... (próbki: {len(records)})")
            elif key in (ord("b"), ord("B")):
                mode = "bad"
                print(f"▶ Nagrywam ZŁĄ postawę... (próbki: {len(records)})")
            elif key == ord(" "):
                mode = "idle"
                print("⏸ Pauza")
            elif key in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()

    # ── zapis ─────────────────────────────────────────────────
    if records:
        df   = pd.DataFrame(records)
        good = (df["label"] == 0).sum()
        bad  = (df["label"] == 1).sum()
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n✅ Zapisano {len(df)} próbek → {OUTPUT_PATH}")
        print(f"   Dobra postawa (0): {good}")
        print(f"   Zła postawa  (1): {bad}")
    else:
        print("Brak danych.")


if __name__ == "__main__":
    main()

