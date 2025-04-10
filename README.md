# PPE Detection Web App (YOLOv5 + Flask)

This is a real-time PPE (Personal Protective Equipment) detection system using YOLOv5 and Flask. It can detect PPE items like Hardhat, Mask, Safety Vest, and alert on violations.

## Features

✅ Video Upload Detection  
✅ Webcam Detection  
✅ Image Upload Detection  
✅ Real-time Bounding Box Overlay  
✅ Sound Alerts for Violations  
✅ Bootstrap UI for Responsiveness

## Setup Instructions

1. Clone or download the repository.
2. Place your trained `best.pt` model in the root directory.
3. Place `alert.wav` in the `static/` folder.
4. Run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

5. Launch the app:

```bash
source venv/bin/activate
python app.py
```

6. Open your browser at `http://localhost:5000`.

## Requirements

See [requirements.txt](requirements.txt)

## Notes

- Works with YOLOv5 models trained via [Ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- Ensure webcam access permission if using the webcam option
- Compatible with Python 3.8+

---

Enjoy a safer construction site with automated PPE detection!