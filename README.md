# Chess Piece Detection with OpenCV and YOLO

This project uses OpenCV and YOLOv8 to detect chess pieces in real-time from a camera feed.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the detector:**
   ```bash
   python chess_detector.py
   ```

## How It Works

- **OpenCV**: Captures video frames from your camera
- **YOLO (YOLOv8)**: Detects and classifies chess pieces in each frame
- The system draws bounding boxes around detected pieces with confidence scores

## Training Custom Model (Recommended)

To detect specific chess pieces, you'll need to train a custom YOLO model:

1. **Collect training data**: Take photos of chess pieces from different angles
2. **Annotate images**: Use tools like [Roboflow](https://roboflow.com/) or [LabelImg](https://github.com/heartexlabs/labelImg)
3. **Organize dataset** in YOLO format:
   ```
   dataset/
     images/
       train/
       val/
     labels/
       train/
       val/
   ```
4. **Create data.yaml**:
   ```yaml
   train: ./dataset/images/train
   val: ./dataset/images/val
   nc: 6  # number of classes
   names: ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
   ```
5. **Train the model**:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   model.train(data='data.yaml', epochs=100, imgsz=640)
   ```
6. **Use trained model**:
   ```python
   detector = ChessDetector('runs/detect/train/weights/best.pt')
   ```

## Controls

- Press **'q'** to quit the camera feed

## Next Steps

- Train a custom model on chess piece images for better accuracy
- Add piece color detection (white vs black)
- Implement board position tracking
- Add piece movement tracking
