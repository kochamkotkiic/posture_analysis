import subprocess
import sys
import time

def main():
    print("🚀 Uruchamianie Kociego Strażnika Postawy...")

    # 1. Uruchomienie serwera AI w tle
    # Używamy sys.executable, aby mieć pewność, że używamy aktywnego środowiska (.venv)
    print("🧠 Włączanie serwera sztucznej inteligencji...")
    server_process = subprocess.Popen([sys.executable, "src/core/server.py"])

    # Dajemy serwerowi 2 sekundy na załadowanie modelu ML
    time.sleep(2)

    # 2. Uruchomienie interfejsu w Electronie
    print("🎨 Uruchamianie interfejsu graficznego...")
    # shell=True jest wymagane na systemie Windows, aby rozpoznać komendę 'npm'
    electron_process = subprocess.Popen("npm start", shell=True)

    try:
        # Ten kod zatrzymuje skrypt i czeka, aż użytkownik zamknie okienko Electrona
        electron_process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        # 3. Sprzątanie: Kiedy zamkniesz apkę, wyłączamy serwer w tle
        print("\n🛑 Zamykanie okna. Wyłączanie serwera AI...")
        server_process.terminate()
        server_process.wait()
        print("✨ Aplikacja zamknięta poprawnie. Do widzenia!")

if __name__ == "__main__":
    main()