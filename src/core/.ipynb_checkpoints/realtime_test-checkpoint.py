import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import time

# ── ścieżki ───────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "posture_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ── punkty ciała (identyczne jak w collect_data.py!) ──────────
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


def draw_prediction(frame, label: int, confidence: float, fps: float):
    """Rysuje wynik predykcji na ekranie."""
    if label == 0:
        text   = "DOBRA POSTAWA"
        color  = (0, 220, 0)
        bg     = (0, 60, 0)
    else:
        text   = "ZLA POSTAWA!"
        color  = (0, 0, 220)
        bg     = (0, 0, 60)

    # tło górnego paska
    cv2.rectangle(frame, (0, 0), (640, 70), bg, -1)

    # główny tekst
    cv2.putText(frame, text, (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

    # pewność modelu
    cv2.putText(frame, f"Pewnosc: {confidence:.1%}",
                (420, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # FPS w rogu
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


def main():
    # ── wczytaj model ─────────────────────────────────────────
    print("Wczytuję model...")
    model = joblib.load(MODEL_PATH)

    # scaler tylko jeśli zapisany (SVM go potrzebuje, RF nie)
    scaler = None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

    print("✅ Model wczytany!")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # zmienne do FPS i wygładzania predykcji
    fps_time    = time.time()
    fps         = 0.0
    predictions = []          # ostatnie N predykcji do wygładzania
    SMOOTH_N    = 10          # głosowanie z ostatnich 10 klatek

    label      = 0
    confidence = 1.0

    with mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1
    ) as pose:

        print("\n📷 Uruchomiono! Naciśnij Q aby wyjść.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # ── rysuj szkielet ────────────────────────────────
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )

                # ── predykcja ─────────────────────────────────
                features = extract_features(results.pose_landmarks.landmark)
                X = np.array(features).reshape(1, -1)

                if scaler is not None:
                    X = scaler.transform(X)

                pred       = model.predict(X)[0]
                proba      = model.predict_proba(X)[0]
                confidence = proba[pred]

                # wygładzanie — głosowanie z ostatnich SMOOTH_N klatek
                predictions.append(pred)
                if len(predictions) > SMOOTH_N:
                    predictions.pop(0)
                label = 1 if predictions.count(1) > SMOOTH_N // 2 else 0

                # log w terminalu co 2 sekundy
                now = time.time()
                if now - fps_time > 2:
                    status = "✅ DOBRA" if label == 0 else "❌ ZLA"
                    print(f"{status}  |  pewność: {confidence:.1%}")
                    fps_time = now

            # ── FPS ───────────────────────────────────────────
            fps = 1.0 / (time.time() - fps_time + 1e-9)

            draw_prediction(frame, label, confidence, fps)
            cv2.imshow("PostureGuard - Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\nZamknięto.")


if __name__ == "__main__":
    main()