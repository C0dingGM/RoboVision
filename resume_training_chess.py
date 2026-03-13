from ultralytics import YOLO
import os

# Change to script directory
os.chdir('/Users/yutian/Desktop/APCS/RoboVision')

# Load the checkpoint from last training run
model = YOLO('runs/detect/chess_pieces_14/weights/last.pt')

# Resume training from checkpoint
results = model.train(
    resume=True
)

print("Training complete!")
print(f"Best model saved at: {results.save_dir}/weights/best.pt")
