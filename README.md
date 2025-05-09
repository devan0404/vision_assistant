# Raspberry Pi Object Detection System

This project implements a real-time object detection system using a Sony IMX500 camera on a Raspberry Pi 4. It includes a FastAPI server for exposing detection results and a TTS client for speaking detected objects.

## Prerequisites

- Raspberry Pi 4
- Sony IMX500 camera
- Wired earphones/speakers
- Python 3.7+
- YOLOv3 weights and configuration files

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd object-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLOv3 files:
```bash
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

4. Set up the systemd service:
```bash
sudo cp object-detection.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable object-detection
sudo systemctl start object-detection
```

## Usage

### Running the Server

The server runs automatically at startup. To manually start/stop:

```bash
sudo systemctl start object-detection
sudo systemctl stop object-detection
```

To check status:
```bash
sudo systemctl status object-detection
```

### Running the TTS Client

1. Connect earphones/speakers to the Raspberry Pi's audio jack
2. Run the TTS client:
```bash
python tts_client.py
```

The client will continuously query the server and speak detected objects through the connected audio device.

## API Endpoints

- `GET /detection`: Returns the latest detection result as JSON
  ```json
  {
    "object": "person",
    "confidence": 0.95,
    "timestamp": 1234567890.123
  }
  ```

## SSH Access

To access the Raspberry Pi via SSH:

1. Enable SSH on the Raspberry Pi:
```bash
sudo raspi-config
# Navigate to Interfacing Options > SSH > Enable
```

2. Connect via SSH:
```bash
ssh pi@<raspberry-pi-ip>
```

## Troubleshooting

1. Camera not detected:
   - Check camera connections
   - Verify camera is enabled in `raspi-config`
   - Check camera permissions

2. Audio not working:
   - Verify audio device is connected
   - Check volume settings
   - Test audio with `aplay` command

3. Server not starting:
   - Check logs: `sudo journalctl -u object-detection`
   - Verify all dependencies are installed
   - Check YOLOv3 files are present

## License

MIT License 