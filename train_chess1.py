from ultralytics import YOLO
import os

# Change to script directory
os.chdir('/Users/yutian/Desktop/APCS/RoboVision')

# Load pre-trained YOLOv11 model
model = YOLO('yolo11n')  # Will auto-download if needed

# Train the model on Chess-pieces-1 dataset
results = model.train(
    data='Chess-Pieces-1/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='chess_pieces_1',
    device='mps',  # Use Apple Silicon GPU
    patience=100
)

print("Training complete!")
print(f"Best model saved at: {results.save_dir}/weights/best.pt")
