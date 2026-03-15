import json
import os
from datetime import datetime

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROFILES_DIR  = os.path.join(BASE_DIR, "data", "profiles")


def _ensure_dir():
    os.makedirs(PROFILES_DIR, exist_ok=True)


def get_all_profiles() -> list[dict]:
    """Zwraca listę wszystkich zapisanych profili."""
    _ensure_dir()
    profiles = []
    for fname in os.listdir(PROFILES_DIR):
        if fname.endswith(".json"):
            path = os.path.join(PROFILES_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                profiles.append(json.load(f))
    return sorted(profiles, key=lambda p: p.get("last_used", ""), reverse=True)


def get_profile(name: str) -> dict | None:
    """Wczytuje profil po nazwie. Zwraca None jeśli nie istnieje."""
    _ensure_dir()
    path = os.path.join(PROFILES_DIR, f"{name.lower()}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_profile(profile: dict):
    """Zapisuje profil do pliku JSON."""
    _ensure_dir()
    name = profile["name"].lower()
    path = os.path.join(PROFILES_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def create_profile(name: str) -> dict:
    """Tworzy nowy pusty profil (bez kalibracji)."""
    profile = {
        "name": name,
        "created":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "last_used":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "calibration": None,   # wypełni się po kalibracji
        "sensitivity": "medium"
    }
    save_profile(profile)
    return profile


def update_last_used(name: str):
    """Aktualizuje datę ostatniego użycia profilu."""
    profile = get_profile(name)
    if profile:
        profile["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        save_profile(profile)


def save_calibration(name: str, calibration_data: dict):
    """Zapisuje dane kalibracji do profilu."""
    profile = get_profile(name)
    if profile is None:
        raise ValueError(f"Profil '{name}' nie istnieje!")
    profile["calibration"] = calibration_data
    profile["calibrated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_profile(profile)
    print(f"✅ Kalibracja zapisana dla profilu: {name}")


def has_calibration(name: str) -> bool:
    """Sprawdza czy profil ma zapisaną kalibrację."""
    profile = get_profile(name)
    return profile is not None and profile.get("calibration") is not None


def delete_profile(name: str) -> bool:
    """Usuwa profil. Zwraca True jeśli się udało."""
    path = os.path.join(PROFILES_DIR, f"{name.lower()}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False