from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model on Chess-pieces-1 dataset
results = model.train(
    data='Chess-pieces-1/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='chess_pieces_1',
    device='mps',  # Use Apple Silicon GPU
    patience=100
)

print("Training complete!")
print(f"Best model saved at: {results.save_dir}/weights/best.pt")
