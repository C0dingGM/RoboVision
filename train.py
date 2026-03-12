from ultralytics import YOLO
from roboflow import Roboflow
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv('ROBOFLOW_API_KEY')

# Initialize Roboflow and download dataset
rf = Roboflow(api_key=api_key)
project = rf.workspace("wangs-workspace-ahzkf").project("chess-pieces-sywaj-f4o6a")
dataset = project.version(1).download("yolov8")

# Train the model with GPU acceleration
model = YOLO('yolov8n.pt')
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device='mps',  # Use Apple Silicon GPU
    name='chess_detection'
)

print("Training complete!")
print(f"Best model saved at: runs/detect/chess_detection/weights/best.pt")
