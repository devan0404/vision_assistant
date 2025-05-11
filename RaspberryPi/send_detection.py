import subprocess
import json
import requests
import time
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure the backend URL
BACKEND_URL = "http://192.168.204.161:8000/detections"

# Configure requests session with retry logic
session = requests.Session()
retry_strategy = Retry(
    total=3,  # number of retries
    backoff_factor=1,  # wait 1, 2, 4 seconds between retries
    status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry on
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

def run_detection():
    """Run the detection script and process its output."""
    logger.info("Starting detection script...")
    
    # Run the detection script
    process = subprocess.Popen(
        ["python3", "imx500_object_detection_demo.py", 
         "--model", "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )

    # Store the latest detection data
    latest_detection = None
    last_sent_time = 0
    SEND_INTERVAL = 4  # seconds

    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break

            # Skip progress bar and other non-JSON output
            if "Network Firmware Upload" in line or "[" in line and "]" not in line:
                continue

            # Try to parse JSON output
            try:
                if line.strip().startswith("[") and line.strip().endswith("]"):
                    detection_data = json.loads(line.strip())
                    if detection_data:  # Only update if we got actual detections
                        latest_detection = detection_data
                        logger.info(f"Received detection: {detection_data}")
            except json.JSONDecodeError:
                continue  # Skip lines that aren't valid JSON

            # Send to backend every SEND_INTERVAL seconds
            now = time.time()
            if latest_detection and now - last_sent_time >= SEND_INTERVAL:
                try:
                    response = session.post(
                        BACKEND_URL,
                        json=latest_detection,
                        timeout=5  # 5 second timeout
                    )
                    response.raise_for_status()
                    logger.info(f"Successfully sent detection to {BACKEND_URL}")
                    last_sent_time = now
                except requests.RequestException as e:
                    logger.error(f"Failed to send detection: {e}")
                    time.sleep(1)  # Wait a bit before retrying

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        logger.info("Detection script stopped")

if __name__ == "__main__":
    try:
        run_detection()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
