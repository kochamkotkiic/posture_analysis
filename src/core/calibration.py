import cv2
import mediapipe as mp
import numpy as np
import time

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

CALIBRATION_SECONDS = 5   # ile sekund zbieramy próbki


def run_calibration(user_name: str) -> dict | None:
    """
    Uruchamia okno kalibracji.
    Zwraca słownik z uśrednionymi współrzędnymi bazowymi
    lub None jeśli użytkownik anulował.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    samples       = []   # zbierane próbki
    collecting    = False
    start_time    = None
    result        = None

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1
    ) as pose:

        print(f"\n🎯 Kalibracja dla: {user_name}")
        print("Usiądź prosto i naciśnij SPACJĘ aby rozpocząć...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # widoczność barków
            visibility = 0.0
            if results.pose_landmarks:
                lm        = results.pose_landmarks.landmark
                left_vis  = lm[11].visibility
                right_vis = lm[12].visibility
                visibility = (left_vis + right_vis) / 2.0

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )

                # zbieranie próbek
                if collecting and visibility > 0.6:
                    sample = {}
                    for name, idx in LANDMARKS_TO_USE.items():
                        p = lm[idx]
                        sample[f"{name}_x"] = p.x
                        sample[f"{name}_y"] = p.y
                        sample[f"{name}_z"] = p.z
                    samples.append(sample)

            # sprawdź czy czas kalibracji minął
            if collecting and start_time:
                elapsed  = time.time() - start_time
                progress = min(elapsed / CALIBRATION_SECONDS, 1.0)

                if elapsed >= CALIBRATION_SECONDS:
                    # zakończ kalibrację
                    if len(samples) >= 10:
                        result = _average_samples(samples)
                        print(f"✅ Zebrano {len(samples)} próbek kalibracyjnych!")
                    else:
                        print("❌ Za mało próbek — spróbuj ponownie!")
                        samples    = []
                        collecting = False
                        start_time = None
                        progress   = 0.0
                    if result:
                        break
            else:
                progress = 0.0

            # rysuj UI kalibracji
            _draw_calibration_ui(frame, user_name, collecting, progress, visibility)
            cv2.imshow("PostureGuard - Kalibracja", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and not collecting and visibility > 0.6:
                collecting = True
                start_time = time.time()
                samples    = []
                print("▶ Zbieranie próbek...")
            elif key == ord("q"):
                print("Anulowano kalibrację.")
                break

    cap.release()
    cv2.destroyAllWindows()
    return result


def _average_samples(samples: list[dict]) -> dict:
    """Uśrednia zebrane próbki kalibracyjne."""
    keys   = samples[0].keys()
    result = {}
    for key in keys:
        values      = [s[key] for s in samples]
        result[key] = float(np.mean(values))
    return result


def _draw_calibration_ui(frame, user_name: str, collecting: bool,
                          progress: float, visibility: float):
    h, w = frame.shape[:2]

    # górny pasek
    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.putText(frame, f"Kalibracja: {user_name}",
                (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    # widoczność barków
    vis_color = (0, 220, 0) if visibility > 0.7 else (0, 165, 255) if visibility > 0.4 else (0, 0, 220)
    vis_text  = "Widocznosc: OK ✓" if visibility > 0.7 else "Widocznosc: popraw pozycje!"
    cv2.putText(frame, vis_text, (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, vis_color, 2)

    if not collecting:
        # instrukcja
        cv2.putText(frame, "Usiądź PROSTO i patrz w kamerę",
                    (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, "Naciśnij SPACJĘ aby rozpocząć",
                    (15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    else:
        # pasek postępu
        bar_x, bar_y, bar_w, bar_h = 15, 140, w - 30, 30
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        filled = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), (0, 200, 0), -1)
        secs_left = CALIBRATION_SECONDS - int(progress * CALIBRATION_SECONDS)
        cv2.putText(frame, f"Nie ruszaj się... {secs_left}s",
                    (bar_x, bar_y + bar_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

    # dolna instrukcja
    cv2.rectangle(frame, (0, h - 35), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "Q = anuluj",
                (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)