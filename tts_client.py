import requests
import pyttsx3
import time
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class TTSClient:
    def __init__(self, server_url: str = "http://192.168.204.161:8000"):
        """Initialize the TTS client."""
        self.server_url = server_url
        self.engine = pyttsx3.init()
        self.last_spoken_object: Optional[str] = None
        
        # Configure TTS engine
        self.engine.setProperty('rate', 150)    # Speaking rate
        self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

        # Configure requests session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,  # number of retries
            backoff_factor=1,  # wait 1, 2, 4 seconds between retries
            status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry on
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_detection(self) -> Optional[dict]:
        """Get the latest detection from the server."""
        try:
            response = self.session.get(f"{self.server_url}/detection", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching detection: {e}")
            return None

    def speak_detection(self, detection: dict):
        """Speak the detected object if it's new."""
        if not detection or not detection["object"]:
            return

        current_object = detection["object"]
        if current_object != self.last_spoken_object:
            self.last_spoken_object = current_object
            confidence = detection["confidence"]
            text = f"I see a {current_object} with {confidence:.0%} confidence"
            print(f"Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()

    def run(self):
        """Main loop to continuously check for and speak new detections."""
        print(f"TTS Client started. Connecting to {self.server_url}")
        print("Press Ctrl+C to stop.")
        print("\nTroubleshooting tips:")
        print("1. Ensure the FastAPI server is running")
        print("2. Check if the port 8000 is open and accessible")
        print("3. Verify the server IP address is correct")
        print("4. Make sure both devices are on the same network")
        
        try:
            while True:
                detection = self.get_detection()
                if detection:
                    self.speak_detection(detection)
                time.sleep(0.5)  # Check every half second
        except KeyboardInterrupt:
            print("\nTTS Client stopped.")

if __name__ == "__main__":
    # Using the server's IP address
    client = TTSClient(server_url="http://192.168.204.161:8000")
    client.run() 