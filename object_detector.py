import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, camera_id: int = 0):
        """Initialize the object detector with the IMX500 camera."""
        self.camera_id = camera_id
        self.cap = None
        self.net = None
        self.classes = []
        logger.info(f"Initializing ObjectDetector with camera_id={camera_id}")
        self.initialize_camera()
        self.initialize_model()

    def initialize_camera(self):
        """Initialize the IMX500 camera."""
        try:
            logger.info("Opening camera...")
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")
            
            # Set camera properties for IMX500 - reduced resolution for better performance
            logger.info("Setting camera properties...")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced from 1920
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced from 1080
            self.cap.set(cv2.CAP_PROP_FPS, 15)  # Reduced from 30

            # Verify camera settings
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Camera initialized with resolution {actual_width}x{actual_height} at {actual_fps}fps")
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            raise

    def initialize_model(self):
        """Initialize the YOLO model for object detection."""
        try:
            logger.info("Loading YOLO model...")
            # Load YOLO model
            self.net = cv2.dnn.readNetFromDarknet(
                "yolov3.cfg",
                "yolov3.weights"
            )
            
            # Load class names
            logger.info("Loading class names...")
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(self.classes)} classes")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def detect_objects(self) -> Tuple[Optional[str], float]:
        """
        Detect objects in the current frame.
        Returns: (detected_object_label, confidence)
        """
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                return None, 0.0

            # Resize frame for faster processing
            frame = cv2.resize(frame, (416, 416))

            # Prepare image for detection
            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, (416, 416),
                swapRB=True, crop=False
            )
            
            # Perform detection
            self.net.setInput(blob)
            layer_names = self.net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            outputs = self.net.forward(output_layers)

            # Process detections
            height, width = frame.shape[:2]
            boxes = []
            confidences = []
            class_ids = []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Increased confidence threshold for better accuracy
                    if confidence > 0.7:  # Increased from 0.5
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression with stricter threshold
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.3)  # Increased confidence threshold

            if len(indices) > 0:
                # Get the highest confidence detection
                best_idx = indices[0]
                best_class_id = class_ids[best_idx]
                best_confidence = confidences[best_idx]
                detected_class = self.classes[best_class_id]
                
                # Only return detection if confidence is high enough
                if best_confidence > 0.7:  # Additional confidence check
                    logger.info(f"Detected {detected_class} with confidence {best_confidence:.2f}")
                    return detected_class, best_confidence

            return None, 0.0
        except Exception as e:
            logger.error(f"Error in detect_objects: {e}")
            return None, 0.0

    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            logger.info("Releasing camera resources")
            self.cap.release() 