from ultralytics import YOLO

# Resume training from last checkpoint
model = YOLO('runs/detect/chess_detection5/weights/last.pt')
results = model.train(resume=True)

print("Training complete!")
print(f"Best model saved at: runs/detect/chess_detection5/weights/best.pt")
