import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import time
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "posture_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
CSV_PATH    = os.path.join(BASE_DIR, "data", "posture_data.csv")

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


def extract_features(landmarks) -> list:
    features = []
    for name, idx in LANDMARKS_TO_USE.items():
        lm = landmarks[idx]
        features.extend([lm.x, lm.y, lm.z])
    return features


def draw_ui(frame, label: int, confidence: float, corrections: int, flash: str = ""):
    # główny status
    if label == 0:
        text, color, bg = "DOBRA POSTAWA", (0, 220, 0), (0, 60, 0)
    else:
        text, color, bg = "ZLA POSTAWA!", (0, 0, 220), (0, 0, 60)

    cv2.rectangle(frame, (0, 0), (640, 70), bg, -1)
    cv2.putText(frame, text, (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    cv2.putText(frame, f"Pewnosc: {confidence:.1%}",
                (420, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # instrukcja klawiszy
    cv2.rectangle(frame, (0, frame.shape[0] - 80), (640, frame.shape[0]), (30, 30, 30), -1)
    cv2.putText(frame, "G = oznacz jako DOBRA  |  B = oznacz jako ZLA  |  Q = koniec",
                (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    cv2.putText(frame, f"Zapisanych korekt: {corrections}",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1)

    # flash komunikat (np. "ZAPISANO DOBRA!")
    if flash:
        cv2.putText(frame, flash, (150, frame.shape[0] - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)


def save_correction(features: list, correct_label: int, csv_path: str):
    """Dopisuje poprawioną próbkę do CSV."""
    columns = []
    for name in LANDMARKS_TO_USE.keys():
        columns += [f"{name}_x", f"{name}_y", f"{name}_z"]

    row = dict(zip(columns, features))
    row["label"] = correct_label

    df_new = pd.DataFrame([row])

    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, index=False)


def main():
    print("Wczytuję model...")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    print("✅ Model wczytany!\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    predictions  = []
    SMOOTH_N     = 10
    label        = 0
    confidence   = 1.0
    corrections  = 0
    flash_msg    = ""
    flash_until  = 0.0
    last_features = None   # ← zapamiętujemy ostatnie cechy

    with mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1
    ) as pose:

        print("📷 Uruchomiono!")
        print("Gdy model się myli — naciśnij G lub B aby zapisać poprawkę.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )

                features      = extract_features(results.pose_landmarks.landmark)
                last_features = features   # ← zapisz do późniejszego użycia

                X = np.array(features).reshape(1, -1)
                if scaler:
                    X = scaler.transform(X)

                pred       = model.predict(X)[0]
                proba      = model.predict_proba(X)[0]
                confidence = proba[pred]

                predictions.append(pred)
                if len(predictions) > SMOOTH_N:
                    predictions.pop(0)
                label = 1 if predictions.count(1) > SMOOTH_N // 2 else 0

            # flash wygasa po czasie
            flash_msg = flash_msg if time.time() < flash_until else ""

            draw_ui(frame, label, confidence, corrections, flash_msg)
            cv2.imshow("PostureGuard - Test + Korekcja", frame)

            key = cv2.waitKey(1) & 0xFF

            # ── ręczne oznaczanie ─────────────────────────────
            if key in (ord("g"), ord("G")) and last_features:
                save_correction(last_features, correct_label=0, csv_path=CSV_PATH)
                corrections += 1
                flash_msg    = "ZAPISANO: DOBRA!"
                flash_until  = time.time() + 1.5
                print(f"✅ Zapisano jako DOBRA  (łącznie korekt: {corrections})")

            elif key in (ord("b"), ord("B")) and last_features:
                save_correction(last_features, correct_label=1, csv_path=CSV_PATH)
                corrections += 1
                flash_msg    = "ZAPISANO: ZLA!"
                flash_until  = time.time() + 1.5
                print(f"✅ Zapisano jako ZLA  (łącznie korekt: {corrections})")

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    if corrections > 0:
        print(f"\n✅ Zapisano {corrections} korekt do {CSV_PATH}")
        print("Pamiętaj żeby wytrenować model jeszcze raz w Jupyter!")
    else:
        print("\nBrak korekt.")


if __name__ == "__main__":
    main()