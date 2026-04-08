import cv2
import os
import time

# --- KONFIGURACJA ---
SESSION_NAME = "test_naturalny_1"  # Nazwa sesji
OUTPUT_DIR = f"data/raw_frames/{SESSION_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def record_raw_frames():
    cap = cv2.VideoCapture(0)  # Ta sama kamera co w aplikacji

    # Ustawiamy rozdzielczość (upewnij się, że taka sama jak przy kalibracji!)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    print(f"Rozpoczynam nagrywanie do: {OUTPUT_DIR}")
    print("Naciśnij 'q', aby zakończyć.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Wyświetlamy podgląd, żebyś widziała co robisz
            cv2.imshow("NAGRYWANIE RAW (Pracuj naturalnie)", frame)

            # Zapisujemy klatkę jako JPG (jakość 100% - zero strat kompresji)
            frame_name = f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, frame_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Zakończono. Zapisano {frame_count} klatek.")


if __name__ == "__main__":
    record_raw_frames()