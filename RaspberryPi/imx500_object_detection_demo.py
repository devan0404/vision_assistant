import argparse
import sys
from functools import lru_cache
import json
import time
import logging

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables with optimized values
last_detections = []
last_output_time = 0
OUTPUT_INTERVAL = 1.0  # Reduced from 2.5 to 1.0 for more frequent updates
MIN_CONFIDENCE = 0.65  # Minimum confidence threshold
MAX_DETECTIONS = 5     # Reduced from 10 to focus on most confident detections
FRAME_SKIP = 2        # Process every 3rd frame for better performance

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    try:
        bbox_normalization = intrinsics.bbox_normalization
        bbox_order = intrinsics.bbox_order
        threshold = max(args.threshold, MIN_CONFIDENCE)  # Ensure minimum confidence
        iou = args.iou
        max_detections = min(args.max_detections, MAX_DETECTIONS)  # Limit max detections

        np_outputs = imx500.get_outputs(metadata, add_batch=True)
        input_w, input_h = imx500.get_input_size()
        
        if np_outputs is None:
            logger.warning("No outputs received from model")
            return last_detections

        if intrinsics.postprocess == "nanodet":
            boxes, scores, classes = \
                postprocess_nanodet_detection(outputs=np_outputs[0], 
                                           conf=threshold, 
                                           iou_thres=iou,
                                           max_out_dets=max_detections)[0]
            from picamera2.devices.imx500.postprocess import scale_boxes
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            if bbox_normalization:
                boxes = boxes / input_h

            if bbox_order == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]
            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)

        # Filter and sort detections by confidence
        detections = [
            Detection(box, category, score, metadata)
            for box, score, category in zip(boxes, scores, classes)
            if score > threshold
        ]
        
        # Sort by confidence and take top N
        detections.sort(key=lambda x: x.conf, reverse=True)
        last_detections = detections[:max_detections]
        
        return last_detections
    except Exception as e:
        logger.error(f"Error in parse_detections: {str(e)}")
        return last_detections

@lru_cache
def get_labels():
    """Get and cache the labels."""
    try:
        labels = intrinsics.labels
        if intrinsics.ignore_dash_labels:
            labels = [label for label in labels if label and label != "-"]
        return labels
    except Exception as e:
        logger.error(f"Error getting labels: {str(e)}")
        return []

def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    try:
        detections = last_results
        if detections is None:
            return
            
        labels = get_labels()
        with MappedArray(request, stream) as m:
            for detection in detections:
                x, y, w, h = detection.box
                label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

                # Calculate text size and position
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = x + 5
                text_y = y + 15

                # Create a copy of the array to draw the background with opacity
                overlay = m.array.copy()

                # Draw the background rectangle on the overlay
                cv2.rectangle(overlay,
                            (text_x, text_y - text_height),
                            (text_x + text_width, text_y + baseline),
                            (255, 255, 255),  # Background color (white)
                            cv2.FILLED)

                alpha = 0.30
                cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

                # Draw text on top of the background
                cv2.putText(m.array, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Draw detection box
                cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)

            if intrinsics.preserve_aspect_ratio:
                b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
                color = (255, 0, 0)  # red
                cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))
    except Exception as e:
        logger.error(f"Error in draw_detections: {str(e)}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.65, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=5, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = get_args()
        logger.info("Initializing IMX500 camera and model...")

        # This must be called before instantiation of Picamera2
        imx500 = IMX500(args.model)
        intrinsics = imx500.network_intrinsics
        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"
        elif intrinsics.task != "object detection":
            logger.error("Network is not an object detection task")
            sys.exit(1)

        # Override intrinsics from args
        for key, value in vars(args).items():
            if key == 'labels' and value is not None:
                with open(value, 'r') as f:
                    intrinsics.labels = f.read().splitlines()
            elif hasattr(intrinsics, key) and value is not None:
                setattr(intrinsics, key, value)

        # Defaults
        if intrinsics.labels is None:
            with open("assets/coco_labels.txt", "r") as f:
                intrinsics.labels = f.read().splitlines()
        intrinsics.update_with_defaults()

        if args.print_intrinsics:
            print(intrinsics)
            sys.exit(0)

        picam2 = Picamera2(imx500.camera_num)
        config = picam2.create_preview_configuration(
            controls={"FrameRate": intrinsics.inference_rate}, 
            buffer_count=8  # Reduced buffer count for lower latency
        )

        imx500.show_network_fw_progress_bar()
        picam2.start(config, show_preview=True)

        if intrinsics.preserve_aspect_ratio:
            imx500.set_auto_aspect_ratio()

        last_results = None
        picam2.pre_callback = draw_detections
        frame_count = 0

        logger.info("Starting detection loop...")
        while True:
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            last_results = parse_detections(picam2.capture_metadata())
            now = time.time()
            
            if now - last_output_time >= OUTPUT_INTERVAL:
                try:
                    labels = get_labels()
                    output = []
                    for d in last_results:
                        if d.conf >= MIN_CONFIDENCE:  # Additional confidence check
                            output.append({
                                'label': labels[int(d.category)] if int(d.category) < len(labels) else str(d.category),
                                'confidence': float(d.conf),
                                'x': int(d.box[0]),
                                'y': int(d.box[1]),
                                'width': int(d.box[2]),
                                'height': int(d.box[3])
                            })
                    if output:  # Only print if we have detections
                        print(json.dumps(output), flush=True)
                        last_output_time = now
                except Exception as e:
                    logger.error(f"Error serializing detections: {str(e)}")

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        if 'picam2' in locals():
            picam2.stop()
        logger.info("Cleanup complete")
