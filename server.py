from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import uvicorn
from datetime import datetime
import os
import pyttsx3
import threading
import queue
import time
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Slower speech rate for better clarity
tts_engine.setProperty('volume', 1.0)  # Maximum volume
tts_engine.setProperty('voice', tts_engine.getProperty('voices')[0].id)  # Use first available voice

# Create a queue for TTS messages
tts_queue = queue.Queue()
last_spoken = {}  # Track last spoken time for each detection
recent_detections = deque(maxlen=3)  # Reduced history size for faster processing
RELAXATION_TIME = 1.0  # Reduced to 1 second for more frequent announcements

def tts_worker():
    """Background worker to handle TTS announcements."""
    while True:
        try:
            message = tts_queue.get()
            if message:
                logger.info(f"TTS Queue: Announcing - {message}")
                tts_engine.say(message)
                tts_engine.runAndWait()
                logger.info(f"TTS Queue: Finished announcing - {message}")
            time.sleep(0.01)  # Minimal delay
        except Exception as e:
            logger.error(f"TTS Error: {str(e)}")
        finally:
            tts_queue.task_done()

# Start TTS worker thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store the latest detection
latest_detection = None
last_detection_time = None

class DetectionItem(BaseModel):
    label: str
    confidence: float
    x: Optional[float] = None
    y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None

def announce_detection(detection: DetectionItem):
    """Announce a detection immediately."""
    try:
        current_time = time.time()
        # Check if enough time has passed since last announcement
        if (detection.label not in last_spoken or 
            current_time - last_spoken[detection.label] >= RELAXATION_TIME):
            
            # Create announcement message
            confidence_percent = int(detection.confidence * 100)
            message = f"{detection.label}, {confidence_percent} percent"
            
            # Announce immediately
            logger.info(f"Announcing: {message}")
            tts_engine.say(message)
            tts_engine.runAndWait()
            
            # Update last spoken time
            last_spoken[detection.label] = current_time
            
    except Exception as e:
        logger.error(f"Error in announcement: {str(e)}")

@app.get("/")
async def read_root():
    """Serve the main HTML page."""
    return FileResponse('index.html')

@app.get("/detections")
async def get_detections():
    """Get the latest detection data."""
    if latest_detection is None:
        return {"message": "No detections yet"}
    return {
        "detections": latest_detection,
        "timestamp": last_detection_time
    }

@app.post("/detections")
async def receive_detections(detections: List[DetectionItem]):
    """Receive detection data from the camera."""
    global latest_detection, last_detection_time
    
    try:
        logger.info(f"Received {len(detections)} detections")
        # Process detections immediately
        for det in detections:
            logger.info(f"Processing detection: {det.label} ({det.confidence:.2f})")
            announce_detection(det)
        
        # Update the latest detection
        latest_detection = [det.model_dump() for det in detections]
        last_detection_time = datetime.now().isoformat()
        
        return {"status": "success", "message": "Detection received"}
    except Exception as e:
        logger.error(f"Error processing detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000) 