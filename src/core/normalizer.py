import numpy as np

LANDMARKS_TO_USE = [
    "nose", "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
]


def normalize_features(features: list[float], calibration: dict) -> list[float]:
    """
    Normalizuje wektor cech względem danych kalibracyjnych.
    Zamiast surowych współrzędnych model dostaje odchylenia od normy.
    """
    normalized = []
    for name in LANDMARKS_TO_USE:
        for axis in ["x", "y", "z"]:
            key       = f"{name}_{axis}"
            raw_val   = features[len(normalized)]
            base_val  = calibration.get(key, 0.0)
            normalized.append(raw_val - base_val)
    return normalized


def get_shoulder_distance(calibration: dict) -> float:
    """
    Zwraca bazową odległość między barkami z kalibracji.
    Używane do skalowania progów czułości.
    """
    lx = calibration.get("left_shoulder_x", 0.0)
    rx = calibration.get("right_shoulder_x", 1.0)
    return abs(lx - rx)